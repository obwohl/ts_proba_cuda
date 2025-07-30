# Dateipfad: ts_benchmark/baselines/duet/spliced_binned_pareto_standalone.py
# (DIES IST DIE VOLLSTÄNDIGE UND KORRIGIERTE VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

@torch.jit.script
def _log1p_div_x(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable implementation of log(1+x)/x.
    Uses the Taylor expansion log(1+x)/x ≈ 1 - x/2 for x close to 0.
    This is crucial for the GPD log-likelihood calculation to avoid
    the exploding term 1/xi when xi is close to zero.
    """
    is_small = torch.abs(x) < 1e-6
    # Avoid division by zero for the direct computation path
    safe_x = torch.where(is_small, torch.ones_like(x), x)
    
    taylor_approx = 1.0 - x / 2.0
    return torch.where(is_small, taylor_approx, torch.log1p(x) / safe_x)


@torch.jit.script
def _compiled_icdf_vector_loop(
    quantiles: torch.Tensor,
    bin_cdfs: torch.Tensor,
    bin_probs: torch.Tensor,
    bin_edges: torch.Tensor,
    bin_width: float,
    num_bins: int
) -> torch.Tensor:
    """
    JIT-kompilierte Funktion für die speichereffiziente, aber rechenintensive
    Schleife über die Quantile. Dies wird für die CRPS-Berechnung verwendet.
    Durch die Kompilierung wird der Python-Overhead pro Quantil eliminiert.
    """
    results_per_quantile = []
    cdf_start_padded = F.pad(bin_cdfs, (1, 0), 'constant', 0.0)[..., :-1]

    for q in quantiles:
        bin_indices = torch.sum(bin_cdfs < q, dim=-1)
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        
        indices_for_gather = bin_indices.unsqueeze(-1)

        prob_in_bin = torch.gather(bin_probs, -1, indices_for_gather).squeeze(-1)
        cdf_start_of_bin = torch.gather(cdf_start_padded, -1, indices_for_gather).squeeze(-1)
        lower_edge = bin_edges[indices_for_gather].squeeze(-1)
        
        numerator = q - cdf_start_of_bin
        safe_prob_in_bin = torch.clamp(prob_in_bin, min=1e-9)
        frac_in_bin = torch.clamp(numerator / safe_prob_in_bin, 0.0, 1.0)
        icdf_binned_for_q = lower_edge + frac_in_bin * bin_width
        results_per_quantile.append(icdf_binned_for_q)

    return torch.stack(results_per_quantile, dim=-1)

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

        # === KRITISCHER FIX: SCHLIESSE DIE THRESHOLD-LÜCKE ===
        # Das vorherige Verhalten hat die Schwellen dynamisch berechnet. Das Modell hat
        # gelernt, die Bins so breit zu machen, dass die Schwellen nach außen wandern
        # und keine Datenpunkte mehr als "im Tail" klassifiziert werden.
        # Die Lösung: Die Schwellen sind jetzt die festen, nicht-lernbaren Grenzen
        # des Binned-Bereichs. Das Modell kann diese nicht mehr manipulieren.
        self.lower_threshold = torch.full_like(self.lower_gp_xi, self.bins_lower_bound)
        self.upper_threshold = torch.full_like(self.upper_gp_xi, self.bins_upper_bound)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die kumulative Verteilungsfunktion (CDF), P(X <= value).
        """
        value = value.expand(self.batch_shape)

        # --- 1. CDF für den unteren Tail (value < lower_threshold) ---
        y = self.lower_threshold - value
        is_near_zero_lower = torch.abs(self.lower_gp_xi) < 1e-6
        # GPD-Fall
        log1p_arg_lower = self.lower_gp_xi * y / self.lower_gp_beta
        safe_log1p_arg_lower = torch.clamp(log1p_arg_lower, min=-(1.0 - 1e-6))
        surv_gpd_lower = torch.pow(1.0 + safe_log1p_arg_lower, -1.0 / self.lower_gp_xi)
        # Exponential-Fall (Grenzwert für xi=0)
        surv_exp_lower = torch.exp(-y / self.lower_gp_beta)
        # Kombinieren
        surv_lower = torch.where(is_near_zero_lower, surv_exp_lower, surv_gpd_lower)
        cdf_lower = self.tail_percentile * (1.0 - surv_lower)

        # --- 2. CDF für den oberen Tail (value > upper_threshold) ---
        z = value - self.upper_threshold
        is_near_zero_upper = torch.abs(self.upper_gp_xi) < 1e-6
        # GPD-Fall
        log1p_arg_upper = self.upper_gp_xi * z / self.upper_gp_beta
        safe_log1p_arg_upper = torch.clamp(log1p_arg_upper, min=-(1.0 - 1e-6))
        surv_gpd_upper = torch.pow(1.0 + safe_log1p_arg_upper, -1.0 / self.upper_gp_xi)
        # Exponential-Fall
        surv_exp_upper = torch.exp(-z / self.upper_gp_beta)
        # Kombinieren
        surv_upper = torch.where(is_near_zero_upper, surv_exp_upper, surv_gpd_upper)
        cdf_upper = (1.0 - self.tail_percentile) + self.tail_percentile * (1.0 - surv_upper)

        # --- 3. CDF für den gebinnten Bereich ---
        body_mass = 1.0 - 2.0 * self.tail_percentile
        bin_probs = torch.softmax(self.logits, dim=-1) * body_mass
        bin_cdfs_unscaled = torch.cumsum(bin_probs, dim=-1)
        cdf_start_of_bins = F.pad(bin_cdfs_unscaled, (1, 0), 'constant', 0.0)[..., :-1]
        bin_indices = torch.clamp(torch.floor((value - self.bins_lower_bound) / self._bin_width), 0, self.num_bins - 1).long()
        prob_of_bins_below = torch.gather(cdf_start_of_bins, -1, bin_indices.unsqueeze(-1)).squeeze(-1)
        prob_in_current_bin = torch.gather(bin_probs, -1, bin_indices.unsqueeze(-1)).squeeze(-1)
        fraction_in_current_bin = torch.clamp((value - self.bin_edges[bin_indices]) / self._bin_width, 0.0, 1.0)
        cdf_binned = self.tail_percentile + prob_of_bins_below + fraction_in_current_bin * prob_in_current_bin

        # --- 4. Kombiniere die drei Bereiche ---
        in_lower_tail = value < self.lower_threshold
        in_upper_tail = value > self.upper_threshold
        cdf_val = torch.where(in_lower_tail, cdf_lower, cdf_binned)
        cdf_val = torch.where(in_upper_tail, cdf_upper, cdf_val)
        
        return torch.clamp(cdf_val, 0.0, 1.0)

    def icdf(self, quantiles: torch.Tensor, binned_only: bool = False) -> torch.Tensor:
        bin_probs = torch.softmax(self.logits, dim=-1)
        bin_cdfs = torch.cumsum(bin_probs, dim=-1)

        # Unterscheidet zwischen dem Aufruf aus crps_loss (Vektor) und __init__ (Tensor)
        is_vector_input = quantiles.dim() < len(self.batch_shape)
        
        if is_vector_input:
            # --- OPTIMIERUNG: Rufe die JIT-kompilierte Funktion auf ---
            # Anstatt die Schleife in langsamem Python auszuführen, übergeben wir
            # alle notwendigen Tensoren an die kompilierte Funktion.
            icdf_binned = _compiled_icdf_vector_loop(
                quantiles,
                bin_cdfs,
                bin_probs,
                self.bin_edges,
                self._bin_width,
                self.num_bins
            )
            q_bcast = quantiles.view((1,) * len(self.batch_shape) + (-1,)).expand_as(icdf_binned)
            
        else:
            # Pfad für __init__: quantiles hat die volle Batch-Form, z.B. shape [B, H, V]
            # HIER WAR DER FEHLER bezüglich der Tensor-Dimensionen.
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
            # ## KORREKTURBLOCK: DIMENSIONS-MISMATCH ##
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
    
    def mean(self) -> torch.Tensor:
        # Eine vernünftige Anzahl von Quantilen für eine stabile Mean-Approximation
        q = torch.linspace(0.005, 0.995, 199, device=self.logits.device)
        icdf_vals = self.icdf(q)
        return torch.mean(icdf_vals, dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.expand(self.batch_shape)
        
        # Binned Log Prob
        # === DEFINITIVE FIX: The binned probability must be scaled by the body mass ===
        # The log_prob was previously using a raw softmax, implying the binned region had
        # a total probability of 1.0, which is incorrect. This created a mathematically
        # inconsistent distribution and was the final root cause of the learning failure.
        body_mass = 1.0 - 2.0 * self.tail_percentile
        bin_indices = torch.clamp(torch.floor((value - self.bins_lower_bound) / self._bin_width), 0, self.num_bins - 1).long()
        bin_probs = torch.softmax(self.logits, dim=-1) * body_mass
        prob_in_bin = torch.gather(bin_probs, -1, bin_indices.unsqueeze(-1)).squeeze(-1)
        log_prob_binned = torch.log(prob_in_bin / self._bin_width + 1e-9)

        # Tail Log Prob
        y = self.lower_threshold - value
        log1p_arg_lower = self.lower_gp_xi * y / self.lower_gp_beta
        # --- FIX: Clamp the argument to log1p to prevent NaNs, which kill gradients. ---
        # log1p(x) is only defined for x > -1. We clamp it to be safe.
        safe_log1p_arg_lower = torch.clamp(log1p_arg_lower, min=-(1.0 - 1e-6))
        
        # === FINAL FIX: Add the missing log-probability scaling for the tail mass ===
        # The log-probability must be scaled by the probability mass in the tail.
        log_tail_mass = torch.log(torch.tensor(self.tail_percentile, device=value.device))
        
        # --- Lower Tail ---
        y = self.lower_threshold - value
        log1p_arg_lower = self.lower_gp_xi * y / self.lower_gp_beta
        safe_log1p_arg_lower = torch.clamp(log1p_arg_lower, min=-(1.0 - 1e-6))
        # The numerically stable GPD log-pdf calculation. This formula now correctly
        # converges to the exponential PDF as xi -> 0, so the `torch.where` switch
        # is no longer needed and was the source of the zero-gradient trap.
        log_pdf_gpd_lower = -torch.log(self.lower_gp_beta) - torch.log1p(safe_log1p_arg_lower) - (y / self.lower_gp_beta) * _log1p_div_x(safe_log1p_arg_lower)
        log_prob_lower = log_tail_mass + log_pdf_gpd_lower

        # --- Upper Tail ---
        z = value - self.upper_threshold
        log1p_arg_upper = self.upper_gp_xi * z / self.upper_gp_beta
        safe_log1p_arg_upper = torch.clamp(log1p_arg_upper, min=-(1.0 - 1e-6))
        log_pdf_gpd_upper = -torch.log(self.upper_gp_beta) - torch.log1p(safe_log1p_arg_upper) - (z / self.upper_gp_beta) * _log1p_div_x(safe_log1p_arg_upper)
        log_prob_upper = log_tail_mass + log_pdf_gpd_upper

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
        
        # Die Logits werden leicht gedämpft, um extreme Wahrscheinlichkeiten zu vermeiden.
        logits = 10.0 * torch.tanh(logits_raw / 10.0)
        
        # Die Beta-Parameter (Skalierung der Tails) müssen positiv sein. Softplus stellt dies sicher.
        BETA_FLOOR = 0.01
        lower_gp_beta = F.softplus(lower_gp_beta_raw) + BETA_FLOOR
        upper_gp_beta = F.softplus(upper_gp_beta_raw) + BETA_FLOOR

        lower_gp_xi = lower_gp_xi_raw
        upper_gp_xi = upper_gp_xi_raw
        
        return SplicedBinnedPareto(
            logits=logits, lower_gp_xi=lower_gp_xi, lower_gp_beta=lower_gp_beta,
            upper_gp_xi=upper_gp_xi, upper_gp_beta=upper_gp_beta,
            bins_lower_bound=self.bins_lower_bound,
            bins_upper_bound=self.bins_upper_bound,
            tail_percentile=self.tail_percentile,
        )