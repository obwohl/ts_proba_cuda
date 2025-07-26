# ts_benchmark/baselines/duet/models/duet_prob_model.py
# (BASIEREND AUF DEM ORIGINALEN DUETMODEL, UMGEBAUT FÜR PROBABILISTISCHE VORHERSAGE)

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from einops import rearrange

# === CORE-KOMPONENTEN VON DUET ===
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer

# === PROBABILISTISCHE KOMPONENTEN ===
from ts_benchmark.baselines.duet.spliced_binned_pareto_standalone import SplicedBinnedParetoOutput, MLPProjectionHead
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
    def batch_shape(self):
        # Definiert die "Größe" der Verteilung
        return self.base_dist.batch_shape

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

        # --- Probabilistischer Kopf (ersetzt den alten `linear_head`) ---
        # Helfer, um die Dimensionen der Verteilungsparameter zu bekommen
        self.distr_output_helper = SplicedBinnedParetoOutput(
            num_bins=getattr(config, 'num_bins', 100),
            bins_lower_bound=-1e6, # Nur Platzhalter
            bins_upper_bound=1e6,  # Nur Platzhalter
            tail_percentile=getattr(config, 'tail_percentile', 0.05)
        )
        
        # --- DEFINITIVE FIX: Use Per-Channel Projection Heads ---
        # The previous shared projection head was re-mixing channel information after
        # the Channel_transformer had carefully separated it. This created a difficult
        # optimization problem where the MLP had to learn two different functions
        # (one simple, one complex) with a single set of weights, leading to
        # conflicting gradients. By creating a dedicated head for each channel,
        # we ensure that the final distribution parameters for a channel are derived
        # ONLY from its own (masked) feature representation.
        self.channel_names = list(config.channel_bounds.keys())
        self.args_proj = nn.ModuleDict()
        in_features_per_channel = self.d_model
        out_features_per_channel = self.horizon * self.distr_output_helper.args_dim
        hidden_dim_factor = getattr(config, 'projection_head_dim_factor', 2)
        hidden_dim = max(self.distr_output_helper.args_dim, in_features_per_channel // hidden_dim_factor)

        for name in self.channel_names:
            self.args_proj[name] = MLPProjectionHead(
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
                # The prior can be passed as a list of lists, so we ensure it's a tensor.
                if not isinstance(prior_from_config, torch.Tensor):
                    self.channel_adjacency_prior = torch.tensor(prior_from_config, dtype=torch.float32)
                else:
                    self.channel_adjacency_prior = prior_from_config

                # Basic validation to prevent common errors
                if self.channel_adjacency_prior.shape != (self.n_vars, self.n_vars):
                    raise ValueError(
                        f"channel_adjacency_prior shape mismatch. "
                        f"Expected ({self.n_vars}, {self.n_vars}), but got {self.channel_adjacency_prior.shape}"
                    )

        # --- PERFORMANCE-OPTIMIERUNG: Verteilungs-Setup ---
        # Der Bug war hier: Die Verteilungs-Köpfe wurden mit den Grenzen der
        # Original-Daten initialisiert, obwohl das Modell auf normalisierten Daten
        # operiert. Dies führte zu einer doppelten Denormalisierung.
        #
        # Die Lösung: Wir initialisieren die Köpfe mit festen, sinnvollen Grenzen
        # für normalisierte Daten (z.B. [-10, 10]). Die `channel_bounds` aus der
        # Konfiguration werden hier nicht mehr benötigt, da die Denormalisierung
        # korrekt über den `DenormalizingDistribution`-Wrapper und die `stats`
        # aus der RevIN-Schicht erfolgt.
        # WICHTIG: Wir erstellen nur EINE Instanz, die vektorisiert über alle Kanäle arbeitet,
        # anstatt eines Dictionaries mit einer Instanz pro Kanal.
        self.distr_output = SplicedBinnedParetoOutput(
            num_bins=getattr(config, 'num_bins', 100), bins_lower_bound=-10.0, bins_upper_bound=10.0, tail_percentile=getattr(config, 'tail_percentile', 0.05)
        )

    def forward(self, input_x: torch.Tensor):
        # Der forward-Pass gibt jetzt ein Verteilungsobjekt und den MoE-Loss zurück.
        # input_x: [Batch, SeqLen, NVars]
        
        # --- 1. Normaler Modell-Pfad ---
        # RevIN (mit subtract_last) wird direkt auf den Original-Input angewendet.
        # RevIN gibt die normalisierten Daten und die Statistiken (mean, std) zurück.
        x_for_main_model, stats = self.cluster.revin(input_x, 'norm')
        x_for_main_model = torch.nan_to_num(x_for_main_model)

        # 2. Zeitliche Mustererkennung mit MoE (Linear_extractor_cluster)
        if self.CI:
            # Behandle jeden Kanal unabhängig
            channel_independent_input = rearrange(x_for_main_model, 'b l n -> (b n) l 1')
            reshaped_output, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) d 1 -> b d n', b=input_x.shape[0])
        else:
            # Das Cluster erhält die normalisierten Features direkt, da die Kanalinteraktion
            # später im Channel-Transformer stattfindet.
            temporal_feature, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts = self.cluster(x_for_main_model)

        # 3. Kanalübergreifende Interaktion mit Channel-Transformer
        # temporal_feature ist [B, D_Model, N_Vars] -> umformen zu [B, N_Vars, D_Model]
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        
        # Initialisiere die Matrizen, die zurückgegeben werden sollen
        p_learned, p_final = None, None

        if self.n_vars > 1:
            # --- DESIGN-VERBESSERUNG: Die Maske wird auf den un-gemischten Daten berechnet. ---
            # Die Mahalanobis-Maske soll die *intrinsische* Ähnlichkeit der Signale
            # bewerten. Der `pre_cluster_mixer` vermischt die Kanäle, was diese
            # Messung "verschmutzen" und zu widersprüchlichen Gradienten führen kann.
            # Wir übergeben daher die reinen, normalisierten Daten (`x_for_main_model`) an die Maske.
            changed_input = rearrange(x_for_main_model, 'b l n -> b n l')

            # --- BUG FIX: Removed noise addition. ---
            # The noise, intended for a past version of the distance metric, breaks the
            # shift-invariance of the FFT's magnitude, preventing the mask from correctly
            # identifying time-shifted signals. The current implementation does not need it.
            
            channel_mask, p_learned, p_final = self.mask_generator(
                changed_input,
                channel_adjacency_prior=self.channel_adjacency_prior
            )
            channel_group_feature, _ = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)
        else:
            channel_group_feature = temporal_feature
            # Erstelle Dummy-Matrizen für den Fall mit einer einzelnen Variable, um die Signatur konsistent zu halten.
            if self.n_vars == 1:
                p_learned = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)
                p_final = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)

        # 4. Erzeugung der Verteilungsparameter
        # channel_group_feature ist [B, N_Vars, D_Model]
        # --- DEFINITIVE FIX: Apply Per-Channel Heads ---
        all_distr_params = []
        for i, name in enumerate(self.channel_names):
            # Get features for this channel: [B, D_Model]
            channel_feature = channel_group_feature[:, i, :]
            
            # Pass through this channel's dedicated head
            channel_distr_params_flat = self.args_proj[name](channel_feature)
            
            # Prevent NaN/inf in the parameters
            channel_distr_params_flat = torch.nan_to_num(channel_distr_params_flat, nan=0.0, posinf=1e4, neginf=-1e4)

            # Reshape to [B, Horizon, N_Params]
            channel_distr_params = rearrange(
                channel_distr_params_flat,
                'b (h d) -> b h d',
                h=self.horizon,
                d=self.distr_output_helper.args_dim
            )
            all_distr_params.append(channel_distr_params)

        # Stack to get the final parameter tensor: [B, N_Vars, Horizon, N_Params]
        distr_params = torch.stack(all_distr_params, dim=1)

        # 5. Erstellung des finalen Verteilungsobjekts (VEKTORISIERT)
        # Anstatt über Kanäle zu loopen, übergeben wir den gesamten Parameter-Tensor
        # an eine einzige Verteilungsinstanz, die die Kanal-Dimension intern verarbeitet.
        # Dies ist ein massiver Performance-Gewinn.
        base_distr = self.distr_output.distribution(distr_params)
        final_distr = DenormalizingDistribution(base_distr, stats)

        # Der alte `denorm`-Schritt am Ende entfällt, da dies jetzt im Wrapper passiert.
        # KORREKTUR: Gib die Wahrscheinlichkeitsmatrizen zurück, um die 9-Werte-Signatur zu erfüllen.
        return final_distr, base_distr, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, p_learned, p_final

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