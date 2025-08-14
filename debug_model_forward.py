import torch
import pandas as pd
import numpy as np
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel, DenormalizingDistribution
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
from optuna_full_search import FIXED_PARAMS # To get data_file and other fixed params
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig # To get the config class

# --- Use Hyperparameters from the Log ---
# These are the parameters that caused the "all zeros" issue.
debug_params = {
    "loss_coef": 1.1662624100261456,
    "seq_len": 480,
    "norm_mode": "subtract_median",
    "lr": 0.0033736163537270948,
    "d_model": 128,
    "d_ff": 128,
    "e_layers": 4,
    "moving_avg": 24,
    "n_heads": 2,
    "dropout": 0.09437562414314524,
    "fc_dropout": 0.08103079198045399,
    "batch_size": 512,
    "use_agc": True,
    "agc_lambda": 0.006530677488552578,
    "num_linear_experts": 4,
    "num_univariate_esn_experts": 3,
    "num_multivariate_esn_experts": 1,
    "k": 3,
    "hidden_size": 128,
    "reservoir_size_uni": 256,
    "spectral_radius_uni": 0.7548701849631492,
    "sparsity_uni": 0.1866297722284434,
    "leak_rate_uni": 0.6145288008229557,
    "input_scaling_uni": 0.9551863204908785,
    "esn_uni_weight_decay": 1.6471261855333192e-05,
    "reservoir_size_multi": 128,
    "spectral_radius_multi": 1.3387152069455168,
    "sparsity_multi": 0.10371867769424395,
    "leak_rate_multi": 0.9674377793033458,
    "input_scaling_multi": 0.11445688723397782,
    "esn_multi_weight_decay": 3.173135311966274e-06,
    "noise_epsilon": 0.002108695165991788,
    "projection_head_layers": 0,
    "loss_target_clip": None,
    "use_channel_adjacency_prior": False,
    "horizon": FIXED_PARAMS["horizon"],
}

if __name__ == "__main__":
    # --- Load Data ---
    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}'...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully.")

    # --- Prepare Config ---
    # Combine FIXED_PARAMS and debug_params
    combined_params = {**FIXED_PARAMS, **debug_params}

    # Need to add enc_in, dec_in, c_out based on data
    temp_config = TransformerConfig(**combined_params)
    temp_config.enc_in = temp_config.dec_in = temp_config.c_out = data.shape[1]
    temp_config.channel_bounds = {}
    for col in data.columns:
        min_val, max_val = data[col].min(), data[col].max()
        buffer = 0.1 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
        temp_config.channel_bounds[col] = {"lower": min_val - buffer, "upper": max_val + buffer}
    temp_config.channel_names = list(data.columns)

    print(f"DEBUG: temp_config.distribution_family = {temp_config.distribution_family}")

    # --- Instantiate Model ---
    model = DUETProbModel(temp_config)
    model.eval() # Set to evaluation mode
    device = torch.device("cpu") # Force CPU
    model.to(device)

    # --- Prepare Data Loader ---
    train_data, valid_data = train_val_split(data, FIXED_PARAMS["train_ratio_in_tv"], temp_config.seq_len)
    train_dataset, train_data_loader = forecasting_data_provider(train_data, temp_config, timeenc=1, batch_size=temp_config.batch_size, shuffle=False, drop_last=True)

    print("\nStarting detailed forward pass analysis...")
    
    # --- Iterate through a few batches and log intermediate tensor stats ---
    num_batches_to_check = 5 # Check first 5 batches
    for i, batch in enumerate(train_data_loader):
        if i >= num_batches_to_check:
            break

        input_data, target, _, _ = batch
        input_data = input_data.to(device)
        target = target.to(device)

        print(f"\n--- Batch {i} ---")
        print(f"Input x stats: mean={input_data.mean():.6f}, std={input_data.std():.6f}, min={input_data.min():.6f}, max={input_data.max():.6f}")
        print(f"Target y stats: mean={target.mean():.6f}, std={target.std():.6f}, min={target.min():.6f}, max={target.max():.6f}")

        with torch.no_grad():
            # Manually call parts of the forward pass to inspect intermediate tensors
            # 1. RevIN
            x_for_main_model, stats = model.cluster.revin(input_data, 'norm')
            x_for_main_model = torch.nan_to_num(x_for_main_model)
            print(f"  RevIN Output (x_for_main_model) stats: mean={x_for_main_model.mean():.6f}, std={x_for_main_model.std():.6f}, min={x_for_main_model.min():.6f}, max={x_for_main_model.max():.6f}")
            print(f"  RevIN Stats (mean, std) stats: mean={stats.mean():.6f}, std={stats.std():.6f}, min={stats.min():.6f}, max={stats.max():.6f}")

            # 2. Cluster (MoE Gating)
            temporal_feature, L_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, clean_logits, noisy_logits = model.cluster(x_for_main_model)
            print(f"  Cluster Output (temporal_feature) stats: mean={temporal_feature.mean():.6f}, std={temporal_feature.std():.6f}, min={temporal_feature.min():.6f}, max={temporal_feature.max():.6f}")
            print(f"  Loss Importance (L_importance): {L_importance.item():.6f}")
            if clean_logits is not None: print(f"  Clean Logits stats: mean={clean_logits.mean():.6f}, std={clean_logits.std():.6f}, min={clean_logits.min():.6f}, max={clean_logits.max():.6f}")
            if noisy_logits is not None: print(f"  Noisy Logits stats: mean={noisy_logits.mean():.6f}, std={noisy_logits.std():.6f}, min={noisy_logits.min():.6f}, max={noisy_logits.max():.6f}")
            if avg_gate_weights_linear is not None: print(f"  Avg Linear Gate Weights stats: mean={avg_gate_weights_linear.mean():.6f}, std={avg_gate_weights_linear.std():.6f}")
            if avg_gate_weights_uni_esn is not None: print(f"  Avg Uni ESN Gate Weights stats: mean={avg_gate_weights_uni_esn.mean():.6f}, std={avg_gate_weights_uni_esn.std():.6f}")
            if avg_gate_weights_multi_esn is not None: 
                # Handle potential single-element tensor for std
                multi_esn_std = avg_gate_weights_multi_esn.std() if avg_gate_weights_multi_esn.numel() > 1 else torch.tensor(0.0)
                print(f"  Avg Multi ESN Gate Weights stats: mean={avg_gate_weights_multi_esn.mean():.6f}, std={multi_esn_std:.6f}")

            # 3. Channel Transformer (if n_vars > 1)
            p_learned, p_final = None, None
            if model.n_vars > 1:
                from einops import rearrange
                changed_input = rearrange(x_for_main_model, 'b l n -> b n l')
                channel_mask, p_learned, p_final = model.mask_generator(
                    changed_input,
                    channel_adjacency_prior=model.channel_adjacency_prior
                )
                channel_group_feature, _ = model.Channel_transformer(x=rearrange(temporal_feature, 'b d n -> b n d'), attn_mask=channel_mask)
                print(f"  Channel Transformer Output (channel_group_feature) stats: mean={channel_group_feature.mean():.6f}, std={channel_group_feature.std():.6f}, min={channel_group_feature.min():.6f}, max={channel_group_feature.max():.6f}")
                if p_learned is not None: print(f"  P Learned stats: mean={p_learned.mean():.6f}, std={p_learned.std():.6f}")
                if p_final is not None: print(f"  P Final stats: mean={p_final.mean():.6f}, std={p_final.std():.6f}")
            else:
                channel_group_feature = temporal_feature
                if model.n_vars == 1:
                    p_learned = torch.ones(input_data.shape[0], 1, 1, device=input_data.device)
                    p_final = torch.ones(input_data.shape[0], 1, 1, device=input_data.device)

            # 4. Projection Heads and Distribution Parameters
            distr_params_list = []
            for j, name in enumerate(model.channel_names):
                channel_feature = channel_group_feature[:, j, :]
                raw_params = model.projection_heads[name](channel_feature)
                distr_params_list.append(raw_params)
                print(f"  Channel {j} Raw Params stats: mean={raw_params.mean():.6f}, std={raw_params.std():.6f}, min={raw_params.min():.6f}, max={raw_params.max():.6f}")

            distr_params = torch.stack(distr_params_list, dim=1)
            distr_params = torch.nan_to_num(distr_params, nan=0.0, posinf=1e4, neginf=-1e4)
            print(f"  Stacked Distr Params stats: mean={distr_params.mean():.6f}, std={distr_params.std():.6f}, min={distr_params.min():.6f}, max={distr_params.max():.6f}")

            # 5. Final Distribution
            if model.config.distribution_family == "AutoGPD":
                base_distr = model.distr_output.distribution(distr_params, model.horizon, stats=stats)
                final_distr = base_distr
            else:
                base_distr = model.distr_output.distribution(distr_params, model.horizon)
                final_distr = DenormalizingDistribution(base_distr, stats)
            
            print(f"  Final Distribution Mean stats: mean={final_distr.mean.mean():.6f}, std={final_distr.mean.std():.6f}")
            # Check stddev only if it's implemented and won't crash
            try:
                final_distr_stddev = final_distr.stddev
                print(f"  Final Distribution StdDev stats: mean={final_distr_stddev.mean():.6f}, std={final_distr_stddev.std():.6f}")
            except NotImplementedError:
                print("  Final Distribution StdDev not implemented for this distribution family.")

            # 6. Calculate NLL Loss (similar to how it's done in forecast_fit)
            target_horizon = target[:, -model.config.horizon:, :]
            if model.config.distribution_family == "Johnson":
                norm_target = final_distr.normalize_value(target_horizon).permute(0, 2, 1)
                log_probs = base_distr.log_prob(norm_target)
                nll_loss = -log_probs.mean()
            else:
                log_probs = final_distr.log_prob(target_horizon)
                nll_loss = -log_probs.mean()
            
            print(f"  NLL Loss: {nll_loss.item():.6f}")
            print(f"  Total Loss (NLL + Importance): {nll_loss.item() + model.config.loss_coef * L_importance.item():.6f}")

    print("\nDetailed forward pass analysis complete.")