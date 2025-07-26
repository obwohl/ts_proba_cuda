# Dateipfad: ts_benchmark/baselines/duet/spliced_binned_pareto_standalone.py
# (DIES IST DIE VOLLSTÄNDIGE UND KORRIGIERTE VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

class SplicedBinnedPareto(Distribution):
    arg_constraints = {
        "lower_gp_xi": constraints.real, "lower_gp_beta": constraints.positive,
        "upper_gp_xi": constraints.real, "upper_gp_beta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, logits, lower_gp_xi, lower_gp_beta, upper_gp_xi, upper_gp_beta,
                 bins_lower_bound, bins_upper_bound, tail_percentile, validate_args=None):
        
        self.num_bins = logits.shape[-1]
        self.bins_lower_bound = bins_lower_bound
        self.bins_upper_bound = bins_upper_bound
        self.tail_percentile = tail_percentile
        
        (self.lower_gp_xi, self.lower_gp_beta, self.upper_gp_xi, self.upper_gp_beta) = broadcast_all(
            lower_gp_xi, lower_gp_beta, upper_gp_xi, upper_gp_beta)
        
        batch_shape = self.lower_gp_xi.shape
        event_shape = torch.Size()
        self.logits = logits.expand(batch_shape + (self.num_bins,))

        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        self._bin_width = (bins_upper_bound - bins_lower_bound) / self.num_bins
        self.bin_edges = torch.linspace(
            bins_lower_bound, bins_upper_bound, self.num_bins + 1, device=logits.device)
        
        # Diese Aufrufe verursachen den Fehler, wenn die icdf-Methode nicht korrekt ist.
        self.lower_threshold = self.icdf(torch.full_like(self.lower_gp_xi, self.tail_percentile), binned_only=True)
        self.upper_threshold = self.icdf(torch.full_like(self.upper_gp_xi, 1.0 - self.tail_percentile), binned_only=True)


    def icdf(self, quantiles: torch.Tensor, binned_only: bool = False) -> torch.Tensor:
        bin_probs = torch.softmax(self.logits, dim=-1)
        bin_cdfs = torch.cumsum(bin_probs, dim=-1)

        # Unterscheidet zwischen dem Aufruf aus crps_loss (Vektor) und __init__ (Tensor)
        is_vector_input = quantiles.dim() < len(self.batch_shape)
        
        if is_vector_input:
            # Pfad für crps_loss: quantiles ist ein Vektor, z.B. shape [99]
            # Dieser Pfad war bereits korrekt.
            cdfs_view = bin_cdfs.unsqueeze(-1)
            quantiles_view = quantiles.view((1,) * len(self.batch_shape) + (1, -1))
            
            bin_indices_exp = torch.sum(cdfs_view < quantiles_view, dim=-2)
            bin_indices = bin_indices_exp.squeeze(-2)
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            
            q_bcast = quantiles.view((1,) * len(self.batch_shape) + (-1,)).expand(self.batch_shape + (-1,))
            indices_for_gather = bin_indices
        else:
            # Pfad für __init__: quantiles hat die volle Batch-Form, z.B. shape [B, H, V]
            # HIER WAR DER FEHLER.
            q_bcast = quantiles
            
            q_exp = q_bcast.unsqueeze(-1)
            bin_indices = torch.sum(bin_cdfs < q_exp, dim=-1)
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            
            # Dies erzeugt einen Tensor der Form [B, H, V, 1]
            indices_for_gather = bin_indices.unsqueeze(-1)

        prob_in_bin = torch.gather(bin_probs, -1, indices_for_gather)
        cdf_start_padded = F.pad(bin_cdfs, (1, 0), 'constant', 0.0)[..., :-1]
        cdf_start_of_bin = torch.gather(cdf_start_padded, -1, indices_for_gather)
        lower_edge = self.bin_edges[indices_for_gather]

        # ##################################################################################
        # ## KORREKTURBLOCK ##
        # Im __init__-Pfad (wenn is_vector_input=False) hatten die gather-Operationen
        # eine überflüssige Dimension am Ende erzeugt ([B, H, V, 1]).
        # q_bcast hat aber die Form [B, H, V]. Dies führt zum Crash bei der Subtraktion.
        # Die Lösung: Wir entfernen diese letzte Dimension explizit mit .squeeze(-1).
        # ##################################################################################
        if not is_vector_input:
             prob_in_bin = prob_in_bin.squeeze(-1)
             cdf_start_of_bin = cdf_start_of_bin.squeeze(-1)
             lower_edge = lower_edge.squeeze(-1)

        # Die Subtraktion `q_bcast - cdf_start_of_bin` ist jetzt sicher.
        numerator = q_bcast - cdf_start_of_bin
        safe_prob_in_bin = torch.clamp(prob_in_bin, min=1e-9)
        frac_in_bin = torch.clamp(numerator / safe_prob_in_bin, 0.0, 1.0)
        icdf_binned = lower_edge + frac_in_bin * self._bin_width
        
        if binned_only:
            return icdf_binned

        # --- Tail Calculation ---
        lt = self.lower_threshold.unsqueeze(-1)
        ut = self.upper_threshold.unsqueeze(-1)
        lgx = self.lower_gp_xi.unsqueeze(-1)
        lgb = self.lower_gp_beta.unsqueeze(-1)
        ugx = self.upper_gp_xi.unsqueeze(-1)
        ugb = self.upper_gp_beta.unsqueeze(-1)
        
        in_lower_tail = q_bcast < self.tail_percentile
        in_upper_tail = q_bcast > (1.0 - self.tail_percentile)
        
        # Lower tail
        q_adj_lower = q_bcast / (self.tail_percentile + 1e-9)
        is_near_zero_lower = torch.abs(lgx) < 1e-6
        power_term_lower = torch.pow(q_adj_lower, -lgx)
        icdf_gpd_lower = lt - (lgb / lgx) * (power_term_lower - 1.0)
        icdf_exp_lower = lt - lgb * torch.log(q_adj_lower) # Korrigierte Form für Exponentialspezialfall
        icdf_lower = torch.where(is_near_zero_lower, icdf_exp_lower, icdf_gpd_lower)

        # Upper tail
        q_adj_upper = (1.0 - q_bcast) / (self.tail_percentile + 1e-9)
        is_near_zero_upper = torch.abs(ugx) < 1e-6
        power_term_upper = torch.pow(q_adj_upper, -ugx)
        icdf_gpd_upper = ut + (ugb / ugx) * (power_term_upper - 1.0)
        icdf_exp_upper = ut + ugb * torch.log(q_adj_upper) # Korrigierte Form für Exponentialspezialfall
        icdf_upper = torch.where(is_near_zero_upper, icdf_exp_upper, icdf_gpd_upper)
        
        value = torch.where(in_lower_tail, icdf_lower, icdf_binned)
        value = torch.where(in_upper_tail, icdf_upper, value)
        
        finfo = torch.finfo(value.dtype)
        return torch.nan_to_num(value, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    
    @property
    def mean(self) -> torch.Tensor:
        # Eine vernünftige Anzahl von Quantilen für eine stabile Mean-Approximation
        q = torch.linspace(0.005, 0.995, 199, device=self.logits.device)
        icdf_vals = self.icdf(q)
        return torch.mean(icdf_vals, dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.expand(self.batch_shape)
        
        # Binned Log Prob
        bin_indices = torch.clamp(torch.floor((value - self.bins_lower_bound) / self._bin_width), 0, self.num_bins - 1).long()
        bin_probs = torch.softmax(self.logits, dim=-1)
        prob_in_bin = torch.gather(bin_probs, -1, bin_indices.unsqueeze(-1)).squeeze(-1)
        log_prob_binned = torch.log(prob_in_bin / self._bin_width + 1e-9)

        # Tail Log Prob
        y = self.lower_threshold - value
        log1p_arg_lower = self.lower_gp_xi * y / self.lower_gp_beta
        is_near_zero_lower = torch.abs(self.lower_gp_xi) < 1e-6
        log_pdf_gpd_lower = -torch.log(self.lower_gp_beta) - (1.0 + 1.0 / self.lower_gp_xi) * torch.log1p(torch.clamp(log1p_arg_lower, max=1.0 - 1e-6))
        log_pdf_exp_lower = -torch.log(self.lower_gp_beta) - y / self.lower_gp_beta
        log_prob_lower = torch.where(is_near_zero_lower, log_pdf_exp_lower, log_pdf_gpd_lower)

        z = value - self.upper_threshold
        log1p_arg_upper = self.upper_gp_xi * z / self.upper_gp_beta
        is_near_zero_upper = torch.abs(self.upper_gp_xi) < 1e-6
        log_pdf_gpd_upper = -torch.log(self.upper_gp_beta) - (1.0 + 1.0 / self.upper_gp_xi) * torch.log1p(torch.clamp(log1p_arg_upper, max=1.0 - 1e-6))
        log_pdf_exp_upper = -torch.log(self.upper_gp_beta) - z / self.upper_gp_beta
        log_prob_upper = torch.where(is_near_zero_upper, log_pdf_exp_upper, log_pdf_gpd_upper)

        in_lower_tail = value < self.lower_threshold
        in_upper_tail = value > self.upper_threshold
        log_p = torch.where(in_lower_tail, log_prob_lower, torch.where(in_upper_tail, log_prob_upper, log_prob_binned))
        
        return torch.nan_to_num(log_p)

# ######################################################################
# ## VOLLSTÄNDIGER CODE: PROJEKTIONS-LAYER (UNVERÄNDERT) ##
# ######################################################################
class ProjectionResidualBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.norm(x + residual)
        return x

class MLPProjectionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers == 0:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.input_layer = nn.Linear(in_features, in_features)
            self.residual_blocks = nn.ModuleList(
                [ProjectionResidualBlock(d_model=in_features, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )
            self.final_layer = nn.Linear(in_features, out_features)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers > 0:
            x = self.input_layer(x)
            for block in self.residual_blocks:
                x = block(x)
            return self.final_layer(x)
        return self.projection(x)

class SplicedBinnedParetoOutput:
    def __init__(self, num_bins: int, bins_lower_bound: float, bins_upper_bound: float, tail_percentile: float,
                 projection_head_layers: int = 0, projection_head_dim_factor: int = 2, projection_head_dropout: float = 0.1):
        self.num_bins = num_bins
        self.bins_lower_bound = bins_lower_bound
        self.bins_upper_bound = bins_upper_bound
        self.tail_percentile = tail_percentile
        self.args_dim = num_bins + 4
        self.projection_head_layers = projection_head_layers
        self.projection_head_dim_factor = projection_head_dim_factor
        self.projection_head_dropout = projection_head_dropout

    def get_args_proj(self, in_features: int) -> nn.Module:
        hidden_dim = max(self.args_dim, in_features * self.projection_head_dim_factor)
        return MLPProjectionHead(
            in_features=in_features, out_features=self.args_dim,
            hidden_dim=hidden_dim, num_layers=self.projection_head_layers,
            dropout=self.projection_head_dropout
        )

    def distribution(self, distr_args: torch.Tensor) -> "SplicedBinnedPareto":
        logits_raw = distr_args[..., :self.num_bins]
        (lower_gp_xi_raw, lower_gp_beta_raw, 
         upper_gp_xi_raw, upper_gp_beta_raw) = [distr_args[..., i] for i in range(self.num_bins, self.num_bins + 4)]
        
        logits = 10.0 * torch.tanh(logits_raw / 10.0)
        BETA_FLOOR = 0.01
        XI_SCALE = 0.5
        lower_gp_beta = F.softplus(lower_gp_beta_raw) + BETA_FLOOR
        upper_gp_beta = F.softplus(upper_gp_beta_raw) + BETA_FLOOR
        lower_gp_xi = torch.tanh(lower_gp_xi_raw) * XI_SCALE
        upper_gp_xi = torch.tanh(upper_gp_xi_raw) * XI_SCALE

        return SplicedBinnedPareto(
            logits=logits, lower_gp_xi=lower_gp_xi, lower_gp_beta=lower_gp_beta,
            upper_gp_xi=upper_gp_xi, upper_gp_beta=upper_gp_beta,
            bins_lower_bound=self.bins_lower_bound,
            bins_upper_bound=self.bins_upper_bound,
            tail_percentile=self.tail_percentile,
        )