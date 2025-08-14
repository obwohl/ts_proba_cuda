from typing import Dict, List, Optional
import torch
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
from ts_benchmark.baselines.duet.johnson_system import JohnsonOutput
from ts_benchmark.baselines.duet.gpd_system import ExtendedGPDOutput
from ts_benchmark.baselines.duet.bgev_distribution import BGEVDistribution
from ts_benchmark.baselines.duet.layers.common import MLPProjectionHead
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert

class BGEVOutput:
    def __init__(self, config):
        self.args_dim = 3

    def distribution(self, params, horizon):
        return BGEVDistribution(params[..., 0], torch.nn.functional.softplus(params[..., 1]), torch.sigmoid(params[..., 2]))

class DenormalizingDistribution:
    """
    Wrapper for denormalization.
    Takes a base distribution on normalized data and the statistics (mean, std)
    and returns a distribution whose samples (e.g., via icdf) are on the original scale.
    """
    def __init__(self, base_distribution: torch.distributions.Distribution, stats: torch.Tensor):
        self.base_dist = base_distribution
        # stats has the shape: [B, N_vars, 2]
        # self.mean, self.std get the shape: [B, 1, N_vars] for broadcasting
        self.mean = stats[:, :, 0].unsqueeze(1)
        STD_FLOOR = 1e-6  # Safety floor for the standard deviation
        self.std = torch.clamp(stats[:, :, 1], min=STD_FLOOR).unsqueeze(1)

    @property
    def loc(self) -> torch.Tensor:
        """
        Returns the denormalized median (loc) of the distribution.
        """
        # self.base_dist.loc has shape [B, N_vars, H]
        # self.mean/std have shape [B, 1, N_vars]

        # Adjust dimensions of mean/std for broadcasting
        mean_for_bcast = self.mean.squeeze(1).unsqueeze(-1)  # [B, N_vars, 1]
        std_for_bcast = self.std.squeeze(1).unsqueeze(-1)    # [B, N_vars, 1]

        # Denormalize the loc parameter of the base distribution
        # [B, N_vars, H] * [B, N_vars, 1] + [B, N_vars, 1] -> [B, N_vars, H]
        return self.base_dist.loc * std_for_bcast + mean_for_bcast

    @property
    def batch_shape(self):
        # Defines the "size" of the distribution
        return self.base_dist.batch_shape

    @property
    def stddev(self) -> torch.Tensor:
        """
        Returns the standard deviation of the denormalized distribution.
        This is calculated by multiplying the standard deviation of the base distribution
        with the scaling factor.
        """
        # self.std has shape [B, 1, N_vars]. We reshape it to [B, N_vars, 1] so it can be broadcast with base_dist.stddev ([B, N_vars, H]).
        return self.base_dist.stddev * self.std.permute(0, 2, 1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Expects `value` in [B, H, N_vars]
        value_norm = (value - self.mean) / self.std
        # base_dist.log_prob expects [B, N_vars, H], so we permute
        log_p = self.base_dist.log_prob(value_norm.permute(0, 2, 1))
        # Correction term (log-determinant of the Jacobian of the transformation)
        log_det_jacobian = torch.log(self.std).permute(0, 2, 1)
        return log_p - log_det_jacobian

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # base_dist.icdf returns normalized values with shape [B, N_Vars, Horizon, ...].
        value_norm = self.base_dist.icdf(q)

        # CORRECTION: We need to adapt the shape of mean/std to that of value_norm.
        # self.mean/std have shape [B, 1, N_vars].
        # Target shape for mean/std for broadcasting is [B, N_vars, 1, ...].

        # 1. Remove the middle dimension: [B, 1, N_vars] -> [B, N_vars]
        mean_squeezed = self.mean.squeeze(1)
        std_squeezed = self.std.squeeze(1)

        # 2. Add dimension for the horizon: [B, N_vars] -> [B, N_vars, 1]
        mean_for_bcast = mean_squeezed.unsqueeze(-1)
        std_for_bcast = std_squeezed.unsqueeze(-1)

        # 3. If value_norm has an extra quantile dimension, we add another one.
        # Shape becomes: [B, N_vars, 1] -> [B, N_vars, 1, 1]
        if value_norm.dim() > mean_for_bcast.dim():
            mean_for_bcast = mean_for_bcast.unsqueeze(-1)
            std_for_bcast = std_for_bcast.unsqueeze(-1)

        # The multiplication works now:
        # [B, N_Vars, Horizon, Q] * [B, N_Vars, 1, 1] -> [B, N_Vars, Horizon, Q]
        value_orig = value_norm * std_for_bcast + mean_for_bcast
        return value_orig

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalizes an external value (e.g., the target value) with the statistics of this distribution.
        Expects `value` in [B, H, N_vars].
        """
        return (value - self.mean) / self.std


class DUETProbModel(nn.Module):  # Renamed from DUETModel
    def __init__(self, config):
        super(DUETProbModel, self).__init__()
        self.config = config

        # --- Core components of DUET (are preserved) ---
        self.cluster = Linear_extractor_cluster(config)
        # NEW DEBUG PRINTS
        # if hasattr(self.cluster.gating_input_projection, 'weight'):
        #     print(f"DEBUG: DUETProbModel.__init__ - cluster.gating_input_projection.weight stats: mean={self.cluster.gating_input_projection.weight.mean().item():.6f}, std={self.cluster.gating_input_projection.weight.std().item():.6f}, min={self.cluster.gating_input_projection.weight.min().item():.6f}, max={self.cluster.gating_input_projection.weight.max().item():.6f}")
        # if hasattr(self.cluster.gating_input_projection, 'bias'):
        #     print(f"DEBUG: DUETProbModel.__init__ - cluster.gating_input_projection.bias stats: mean={self.cluster.gating_input_projection.bias.mean().item():.6f}, std={self.cluster.gating_input_projection.bias.std().item():.6f}, min={self.cluster.gating_input_projection.bias.min().item():.6f}, max={self.cluster.gating_input_projection.bias.max().item():.6f}")

        self.expert_embedding_dim = config.expert_embedding_dim
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.d_model = config.d_model
        self.horizon = config.horizon

        # The mask needs the number of variables
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

        # --- NEW: Probabilistic head selection ---
        if config.distribution_family == "Johnson":
            self.distr_output = JohnsonOutput(config.channel_types)
        elif config.distribution_family == "AutoGPD":
            self.distr_output = ExtendedGPDOutput(config.channel_types)
        elif config.distribution_family == "bgev":
            self.distr_output = BGEVOutput(config)
        else:
            raise ValueError(f"Unknown distribution_family: {config.distribution_family}")

        # --- NEW: A single, unified projection head per channel ---
        self.channel_names = list(config.channel_bounds.keys())
        self.projection_heads = nn.ModuleDict()

        in_features_per_channel = self.d_model
        # Output dimension: parameters * horizon length
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

        # print(f"DEBUG: DUETProbModel.forward - Input_x stats (before RevIN): mean={input_x.mean():.6f}, std={input_x.std():.6f}, min={input_x.min():.6f}, max={input_x.max():.6f}")
        x_for_main_model, stats = self.cluster.revin(input_x, 'norm')

        x_for_main_model = torch.nan_to_num(x_for_main_model)
        # print(f"DEBUG: DUETProbModel.forward - x_for_main_model stats (after RevIN): mean={x_for_main_model.mean():.6f}, std={x_for_main_model.std():.6f}, min={x_for_main_model.min():.6f}, max={x_for_main_model.max():.6f}")

        temporal_feature, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, clean_logits, noisy_logits = self.cluster(x_for_main_model)
        # print(f"DEBUG: temporal_feature | Shape: {temporal_feature.shape} | Mean: {temporal_feature.mean():.4f} | Std: {temporal_feature.std():.4f} | Min: {temporal_feature.min():.4f} | Max: {temporal_feature.max():.4f}")
        # print(f"DEBUG: clean_logits | Shape: {clean_logits.shape} | Mean: {clean_logits.mean():.4f} | Std: {clean_logits.std():.4f} | Min: {clean_logits.min():.4f} | Max: {clean_logits.max():.4f}")
        # print(f"DEBUG: noisy_logits | Shape: {noisy_logits.shape} | Mean: {noisy_logits.mean():.4f} | Std: {noisy_logits.std():.4f} | Min: {noisy_logits.min():.4f} | Max: {noisy_logits.max():.4f}")

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
        # print(f"DEBUG: channel_group_feature | Shape: {channel_group_feature.shape} | Mean: {channel_group_feature.mean():.4f} | Std: {channel_group_feature.std():.4f} | Min: {channel_group_feature.min():.4f} | Max: {channel_group_feature.max():.4f}")


        distr_params_list = []
        for i, name in enumerate(self.channel_names):
            channel_feature = channel_group_feature[:, i, :]
            raw_params = self.projection_heads[name](channel_feature)
            # print(f"DEBUG: raw_params (Channel {name}) | Shape: {raw_params.shape} | Mean: {raw_params.mean():.4f} | Std: {raw_params.std():.4f} | Min: {raw_params.min():.4f} | Max: {raw_params.max():.4f}")

            # The tanh activation is removed. Parameter constraints are now handled
            # by the distribution-specific output classes (e.g., JohnsonOutput)
            # or within the distribution itself (e.g., ZIEGPD).
            
            reshaped_params = rearrange(raw_params, 'b (h p) -> b h p', h=self.horizon, p=self.distr_output.args_dim)
            distr_params_list.append(reshaped_params)


        distr_params = torch.stack(distr_params_list, dim=1)
        distr_params = torch.nan_to_num(distr_params, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.config.distribution_family == "AutoGPD":
            # --- NEW: Dynamically adjust for non-zero-inflated channels ---
            is_zi_flags = getattr(self.config, 'is_channel_zero_inflated', None)
            if is_zi_flags:
                for i, is_zero_inflated in enumerate(is_zi_flags):
                    if not is_zero_inflated:
                        # This channel is not zero-inflated.
                        # Force the logit of the zero probability ('pi_raw') to a large negative
                        # number, which will result in pi -> 0 after the sigmoid function.
                        # The parameter index for pi_raw is 0.
                        distr_params[:, i, :, 0] = -1e9  # A large negative value

            # For AutoGPD, the distribution handles normalization internally.
            # We pass the stats directly to it.
            base_distr = self.distr_output.distribution(distr_params, self.horizon, stats=stats)
            final_distr = base_distr
        else:
            # For other families, we use the denormalizing wrapper.
            base_distr = self.distr_output.distribution(distr_params, self.horizon)
            final_distr = DenormalizingDistribution(base_distr, stats)

        return final_distr, base_distr, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, p_learned, p_final, clean_logits, noisy_logits, distr_params

    def get_parameter_groups(self):
        """
        NEW: Encapsulates the logic for identifying parameter groups for the optimizer.
        This makes the training code in duet_prob.py cleaner and more robust.

        Returns:
            tuple: A tuple with three lists of parameters:
                   (esn_uni_readout_params, esn_multi_readout_params, other_params)
        """
        esn_uni_readout_params = []
        esn_multi_readout_params = []
        projection_head_params = []
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
                    else:  # e.g. linear experts do not have a 'readout' layer, but for safety
                        other_params.append(param)
                except (ValueError, IndexError):
                    other_params.append(param)
            elif 'projection_heads.' in name:
                projection_head_params.append(param)
            else:
                other_params.append(param)
        
        return esn_uni_readout_params, esn_multi_readout_params, projection_head_params, other_params
