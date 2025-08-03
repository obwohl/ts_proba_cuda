# ts_benchmark/baselines/duet/models/duet_prob_model.py
# (BASIEREND AUF DEM ORIGINALEN DUETMODEL, UMGEBAUT FÜR PROBABILISTISCHE VORHERSAGE)

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from einops import rearrange

# === CORE-KOMPONENTEN VON DUET ===
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer

# === NEUE PROBABILISTISCHE KOMPONENTEN ===
from ts_benchmark.baselines.duet.johnson_system import JohnsonOutput
from ts_benchmark.baselines.duet.layers.common import MLPProjectionHead
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert

class DenormalizingDistribution:
    """ 
    Wrapper für die Denormalisierung. 
    Nimmt eine Basis-Verteilung auf normalisierten Daten und die Statistik (mean, std)
    und gibt eine Verteilung zurück, deren Samples (z.B. via icdf) auf der Originalskala liegen.
    """
    def __init__(self, base_distribution: torch.distributions.Distribution, stats: torch.Tensor):
        self.base_dist = base_distribution
        # stats hat die Form: [B, N_vars, 2]
        # self.mean, self.std bekommen die Form: [B, 1, N_vars] für Broadcasting
        self.mean = stats[:, :, 0].unsqueeze(1)
        STD_FLOOR = 1e-6 # Sicherheits-Floor für die Standardabweichung
        self.std = torch.clamp(stats[:, :, 1], min=STD_FLOOR).unsqueeze(1)

    @property
    def loc(self) -> torch.Tensor:
        """
        Gibt den denormalisierten Median (loc) der Verteilung zurück.
        """
        # self.base_dist.loc hat die Form [B, N_vars, H]
        # self.mean/std haben die Form [B, 1, N_vars]
        
        # Passe die Dimensionen von mean/std für Broadcasting an
        mean_for_bcast = self.mean.squeeze(1).unsqueeze(-1) # [B, N_vars, 1]
        std_for_bcast = self.std.squeeze(1).unsqueeze(-1)   # [B, N_vars, 1]
        
        # Denormalisiere den loc-Parameter der Basis-Verteilung
        # [B, N_vars, H] * [B, N_vars, 1] + [B, N_vars, 1] -> [B, N_vars, H]
        return self.base_dist.loc * std_for_bcast + mean_for_bcast

    @property
    def batch_shape(self):
        # Definiert die "Größe" der Verteilung
        return self.base_dist.batch_shape

    @property
    def stddev(self) -> torch.Tensor:
        """
        Gibt die Standardabweichung der denormalisierten Verteilung zurück.
        Diese wird berechnet, indem die Standardabweichung der Basis-Verteilung
        mit dem Skalierungsfaktor multipliziert wird.
        """
        # self.std hat die Form [B, 1, N_vars]. Wir formen es zu [B, N_vars, 1] um, damit es mit base_dist.stddev ([B, N_vars, H]) broadcasted werden kann.
        return self.base_dist.stddev * self.std.permute(0, 2, 1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Erwartet `value` in [B, H, N_vars]
        value_norm = (value - self.mean) / self.std
        # base_dist.log_prob erwartet [B, N_vars, H], also permutieren
        log_p = self.base_dist.log_prob(value_norm.permute(0, 2, 1))
        # Korrekturterm (Log-Determinante der Jacobi-Matrix der Transformation)
        log_det_jacobian = torch.log(self.std).permute(0, 2, 1)
        return log_p - log_det_jacobian

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # base_dist.icdf gibt normalisierte Werte mit der Form [B, N_Vars, Horizon, ...] zurück.
        value_norm = self.base_dist.icdf(q)

        # KORREKTUR: Wir müssen die Form von mean/std an die von value_norm anpassen.
        # self.mean/std haben die Form [B, 1, N_vars].
        # Zielform für mean/std zum Broadcasten ist [B, N_vars, 1, ...].
        
        # 1. Entferne die mittlere Dimension: [B, 1, N_vars] -> [B, N_vars]
        mean_squeezed = self.mean.squeeze(1)
        std_squeezed = self.std.squeeze(1)
        
        # 2. Füge Dimension für den Horizont hinzu: [B, N_vars] -> [B, N_vars, 1]
        mean_for_bcast = mean_squeezed.unsqueeze(-1)
        std_for_bcast = std_squeezed.unsqueeze(-1)
        
        # 3. Wenn value_norm eine extra Quantil-Dimension hat, fügen wir eine weitere hinzu.
        # Form wird: [B, N_vars, 1] -> [B, N_vars, 1, 1]
        if value_norm.dim() > mean_for_bcast.dim():
            mean_for_bcast = mean_for_bcast.unsqueeze(-1)
            std_for_bcast = std_for_bcast.unsqueeze(-1)

        # Die Multiplikation funktioniert jetzt:
        # [B, N_Vars, Horizon, Q] * [B, N_Vars, 1, 1] -> [B, N_Vars, Horizon, Q]
        value_orig = value_norm * std_for_bcast + mean_for_bcast
        return value_orig

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalisiert einen externen Wert (z.B. den Zielwert) mit den Statistiken dieser Verteilung.
        Erwartet `value` in [B, H, N_vars].
        """
        return (value - self.mean) / self.std

# === DAS NEUE, PROBABILISTISCHE DUET MODELL ===

class DUETProbModel(nn.Module): # Umbenannt von DUETModel
    def __init__(self, config):
        super(DUETProbModel, self).__init__()

        # --- Kernkomponenten von DUET (bleiben erhalten) ---
        self.cluster = Linear_extractor_cluster(config)
        self.expert_embedding_dim = config.expert_embedding_dim
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.d_model = config.d_model
        self.horizon = config.horizon

        # Die Maske braucht die Anzahl der Variablen
        self.mask_generator = Mahalanobis_mask(config.seq_len, n_vars=self.n_vars)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        # --- NEU: Probabilistischer Kopf für Johnson-System ---
        # Initialisiere den Output-Layer mit den Kanaltypen aus der Konfiguration
        self.distr_output = JohnsonOutput(config.johnson_channel_types)

        # --- NEU: Ein einziger, vereinheitlichter Projektionskopf pro Kanal ---
        self.channel_names = list(config.channel_bounds.keys())
        self.projection_heads = nn.ModuleDict()

        in_features_per_channel = self.d_model
        # Output-Dimension: 4 Parameter (Johnson) * Horizontlänge
        out_features_per_channel = self.horizon * self.distr_output.args_dim

        hidden_dim_factor = getattr(config, 'projection_head_dim_factor', 2)
        hidden_dim = max(out_features_per_channel, in_features_per_channel // hidden_dim_factor)

        for name in self.channel_names:
            self.projection_heads[name] = MLPProjectionHead(
                in_features=in_features_per_channel,
                out_features=out_features_per_channel,
                hidden_dim=hidden_dim,
                num_layers=getattr(config, 'projection_head_layers', 0),
                dropout=getattr(config, 'projection_head_dropout', 0.1)
            )

        # --- MODIFIED: Conditionally store the user-defined channel adjacency prior ---
        self.channel_adjacency_prior = None
        if getattr(config, 'use_channel_adjacency_prior', False):
            prior_from_config = getattr(config, 'channel_adjacency_prior', None)
            if prior_from_config is not None:
                if not isinstance(prior_from_config, torch.Tensor):
                    self.channel_adjacency_prior = torch.tensor(prior_from_config, dtype=torch.float32)
                else:
                    self.channel_adjacency_prior = prior_from_config
                if self.channel_adjacency_prior.shape != (self.n_vars, self.n_vars):
                    raise ValueError(
                        f"channel_adjacency_prior shape mismatch. "
                        f"Expected ({self.n_vars}, {self.n_vars}), but got {self.channel_adjacency_prior.shape}"
                    )

    def forward(self, input_x: torch.Tensor):
        # input_x: [Batch, SeqLen, NVars]
        
        # --- NEW DIAGNOSTIC LOGGING ---
        if self.training: # Only log during training
            print(f"\n[DIAGNOSTIC LOG | DUETProbModel] Before RevIN:")
            print(f"  input_x mean: {input_x.mean().item():.6f}, std: {input_x.std().item():.6f}")
            print(f"  input_x min: {input_x.min().item():.6f}, max: {input_x.max().item():.6f}")
            print(f"  input_x has nan: {torch.isnan(input_x).any().item()}")
        # --- END NEW DIAGNOSTIC LOGGING ---

        x_for_main_model, stats = self.cluster.revin(input_x, 'norm')

        # --- NEW DIAGNOSTIC LOGGING ---
        if self.training: # Only log during training
            print(f"\n[DIAGNOSTIC LOG | DUETProbModel] After RevIN (before nan_to_num):")
            print(f"  x_for_main_model mean: {x_for_main_model.mean().item():.6f}, std: {x_for_main_model.std().item():.6f}")
            print(f"  x_for_main_model min: {x_for_main_model.min().item():.6f}, max: {x_for_main_model.max().item():.6f}")
            print(f"  x_for_main_model has nan: {torch.isnan(x_for_main_model).any().item()}")
        # --- END NEW DIAGNOSTIC LOGGING ---

        x_for_main_model = torch.nan_to_num(x_for_main_model)

        # --- NEW DIAGNOSTIC LOGGING ---
        if self.training: # Only log during training
            print(f"\n[DIAGNOSTIC LOG | DUETProbModel] After RevIN (after nan_to_num):")
            print(f"  x_for_main_model mean: {x_for_main_model.mean().item():.6f}, std: {x_for_main_model.std().item():.6f}")
        # --- END NEW DIAGNOSTIC LOGGING ---

        temporal_feature, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, clean_logits, noisy_logits = self.cluster(x_for_main_model)

        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        
        p_learned, p_final = None, None

        if self.n_vars > 1:
            changed_input = rearrange(x_for_main_model, 'b l n -> b n l')
            channel_mask, p_learned, p_final = self.mask_generator(
                changed_input,
                channel_adjacency_prior=self.channel_adjacency_prior
            )
            channel_group_feature, _ = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)
        else:
            channel_group_feature = temporal_feature
            if self.n_vars == 1:
                p_learned = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)
                p_final = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)


        distr_params_list = []
        for i, name in enumerate(self.channel_names):
            channel_feature = channel_group_feature[:, i, :]
            raw_params = self.projection_heads[name](channel_feature)

            # NEU: Stabilisierung der Ausgabe mit tanh an der korrekten Stelle.
            # Dies quetscht die Parameter in den stabilen Bereich [-1, 1] und
            # verhindert die numerische Instabilität in der Johnson-Verteilung.
            flat_params = torch.tanh(raw_params)
            
            reshaped_params = rearrange(flat_params, 'b (h p) -> b h p', h=self.horizon, p=self.distr_output.args_dim)
            distr_params_list.append(reshaped_params)


        distr_params = torch.stack(distr_params_list, dim=1)
        distr_params = torch.nan_to_num(distr_params, nan=0.0, posinf=1e4, neginf=-1e4)

        base_distr = self.distr_output.distribution(distr_params)
        final_distr = DenormalizingDistribution(base_distr, stats)

        if self.training: # Only log during training
            print(f"\n[DEBUG DUETProbModel.forward] expert_selection_counts (before return): {expert_selection_counts}")
            if expert_selection_counts is not None:
                print(f"  Shape: {expert_selection_counts.shape}, Dtype: {expert_selection_counts.dtype}, Device: {expert_selection_counts.device}")
                print(f"  Values: {expert_selection_counts.cpu().numpy()}")
        if self.training: # Only log during training
            print(f"\n[DEBUG DUETProbModel.forward] expert_selection_counts (before return): {expert_selection_counts}")
            if expert_selection_counts is not None:
                print(f"  Shape: {expert_selection_counts.shape}, Dtype: {expert_selection_counts.dtype}, Device: {expert_selection_counts.device}")
                print(f"  Values: {expert_selection_counts.cpu().numpy()}")
        return final_distr, base_distr, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, p_learned, p_final, clean_logits, noisy_logits

    def get_parameter_groups(self):
        """
        NEU: Kapselt die Logik zur Identifizierung von Parametergruppen für den Optimizer.
        Dies macht den Trainingscode in duet_prob.py sauberer und robuster.

        Returns:
            tuple: Ein Tupel mit drei Listen von Parametern:
                   (esn_uni_readout_params, esn_multi_readout_params, other_params)
        """
        esn_uni_readout_params = []
        esn_multi_readout_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if 'cluster.experts.' in name and '.readout.' in name:
                try:
                    expert_idx = int(name.split('cluster.experts.')[1].split('.')[0])
                    expert_module = self.cluster.experts[expert_idx]
                    if isinstance(expert_module, UnivariateReservoirExpert):
                        esn_uni_readout_params.append(param)
                    elif isinstance(expert_module, MultivariateReservoirExpert):
                        esn_multi_readout_params.append(param)
                    else: # z.B. lineare Experten haben keinen 'readout' Layer, aber zur Sicherheit
                        other_params.append(param)
                except (ValueError, IndexError):
                    other_params.append(param)
            else:
                other_params.append(param)
        
        return esn_uni_readout_params, esn_multi_readout_params, other_params