import torch
import optuna
import inspect
import os
import torch.profiler
import time
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import sys
import io
import contextlib
from PIL import Image
from tqdm import tqdm

# === Korrekte Imports für das neue Modell und die Utilities ===
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel, DenormalizingDistribution
from ts_benchmark.baselines.duet.utils.tools import adjust_learning_rate, EarlyStopping
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
# === NEUER IMPORT FÜR DIE FENSTER-SUCHE ===
from ts_benchmark.baselines.duet.utils.window_search import find_interesting_windows
# === NEUER IMPORT FÜR EXPERTEN-TYPEN ===
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert
# === NEUER IMPORT FÜR JOHNSON-SYSTEM ===
from ts_benchmark.baselines.duet.johnson_system import get_best_johnson_fit
from ...models.model_base import ModelBase

# === NEU: In-Memory-Cache für die Ergebnisse der Fenstersuche ===
# Dies verhindert, dass die teure Suche in jedem Optuna-Trial neu ausgeführt wird.
WINDOW_SEARCH_CACHE = {}

def calculate_cvar(losses: np.ndarray, alpha: float) -> float:
    """Berechnet den Conditional Value at Risk (CVaR)."""
    if not isinstance(losses, np.ndarray):
        losses = np.array(losses)
    
    if losses.size == 0:
        return float('nan')

    # 1. Finde den Schwellenwert (VaR)
    var = np.quantile(losses, 1 - alpha)
    # 2. Berechne den Durchschnitt aller Verluste, die größer oder gleich dem VaR sind
    tail_losses = losses[losses >= var]

    # Handle edge case where no losses are in the tail (e.g., empty input)
    if tail_losses.size == 0:
        return float('nan')

    return float(tail_losses.mean())

def adaptive_clip_grad_(parameters, clip_factor=0.01, eps=1e-3):
    """
    Implementiert Adaptive Gradient Clipping (AGC) wie in "High-Performance Large-Scale Image Recognition Without Normalization" beschrieben.
    Clippt die Gradienten basierend auf dem Verhältnis der Gradientennorm zur Parameternorm.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    for p in filter(lambda p: p.grad is not None, parameters):
        # Berechne die Normen für den Parameter und seinen Gradienten
        p_norm = torch.norm(p.detach(), p=2.0)
        grad_norm = torch.norm(p.grad.detach(), p=2.0)
        
        # Berechne den maximal erlaubten Gradienten-Norm
        max_norm = p_norm * clip_factor
        
        if grad_norm > max_norm + eps:
            p.grad.detach().mul_(max_norm / (grad_norm + eps))

class TransformerConfig:
    """
    Konfigurationsklasse. Kombiniert Defaults mit übergebenen Argumenten.
    Bereinigt und auf die Bedürfnisse des neuen Modells zugeschnitten.
    """
    def __init__(self, **kwargs):
        defaults = {
            # --- Core Architecture ---
            "d_model": 512, "d_ff": 2048, "n_heads": 8, "e_layers": 2,
            "factor": 3, "activation": "gelu", "dropout": 0.1, "fc_dropout": 0.1,
            "output_attention": False,
            
            # --- MoE Parameters (General) ---
            "noisy_gating": True, "hidden_size": 256,
            "loss_coef": 1.0, # MoE loss coefficient
            
            # --- MoE Parameters (Expert Configuration) ---
            "num_linear_experts": 2,
            "num_univariate_esn_experts": 1,
            "num_multivariate_esn_experts": 1,
            "k": 2,

            # --- ESN Expert Default Parameters ---
            "reservoir_size_uni": 256,
            "spectral_radius_uni": 0.99,
            "sparsity_uni": 0.1,
            "leak_rate_uni": 1.0,
            "input_scaling_uni": 1.0,

            "reservoir_size_multi": 256,
            "spectral_radius_multi": 0.99,
            "sparsity_multi": 0.1,
            "leak_rate_multi": 1.0,
            "input_scaling_multi": 0.5,
            
            # --- NEW: ESN Readout Regularization ---
            "esn_uni_weight_decay": 0.0,
            "esn_multi_weight_decay": 0.0,
            
            # --- Training / Optimization ---
            "lr": 1e-4,
            "lradj": "cosine_warmup", "num_epochs": 100,
            "accumulation_steps": 1,
            "batch_size": 128, "patience": 10,
            "num_workers": 4,

            # --- NEW: Tier 2 Training Strategies ---
            "use_agc": False,
            "agc_lambda": 0.01,

            # --- Data & Miscellaneous ---
            "moving_avg": 25, "CI": False, "freq": "h",
            "quantiles": [0.1, 0.5, 0.9],
            "norm_mode": "subtract_median",

            # --- NEW: Projection Head Configuration ---
            "projection_head_layers": 0,
            "projection_head_dim_factor": 2,
            "projection_head_dropout": 0.1,

            # --- NEW: Interim Validation ---
            "interim_validation_seconds": None,

            # --- NEW: Performance Profiling ---
            "profile_epoch": None,
            "tqdm_update_freq": 10,
            "tqdm_min_interval": 1.0,
        }

        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Abgeleitete Werte
        if hasattr(self, 'seq_len'):
            setattr(self, "input_size", self.seq_len)
            setattr(self, "label_len", self.seq_len // 2) 
        else:
            raise AttributeError("Konfiguration muss 'seq_len' enthalten.")
        
        if hasattr(self, 'horizon'):
            setattr(self, "pred_len", self.horizon)
        else:
            raise AttributeError("Konfiguration muss 'horizon' enthalten.")
            
        total_experts = (getattr(self, "num_linear_experts", 0) +
                         getattr(self, "num_univariate_esn_experts", 0) +
                         getattr(self, "num_multivariate_esn_experts", 0))
        if total_experts > 0:
            setattr(self, "k", min(getattr(self, "k", 1), total_experts))


class DUETProb(ModelBase):
    def __init__(self, **kwargs):
        super(DUETProb, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.seq_len = self.config.seq_len
        self.model: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.checkpoint_path: Optional[str] = None
        self.interesting_window_indices: Optional[Dict] = None
        
    @property
    def model_name(self) -> str:
        return "DUET-Prob-Johnson-v1"

    @staticmethod
    def required_hyper_params() -> dict:
        return {"seq_len": "input_chunk_length", "horizon": "output_chunk_length"}

    def _build_model(self):
        """
        Initialisiert das zugrundeliegende PyTorch-Modell (DUETProbModel)
        basierend auf der aktuellen Konfiguration.
        """
        if not hasattr(self.config, 'enc_in'):
            raise AttributeError("Model configuration must have 'enc_in' set before building the model. Call _tune_hyper_params() first.")
        self.model = DUETProbModel(self.config)


    def _tune_hyper_params(self, train_valid_data: pd.DataFrame):
        """
        Setzt datenabhängige Konfigurationswerte, einschließlich Frequenz, Kanalanzahl
        und des Johnson-Verteilungstyps für jeden Kanal.
        """
        # --- Frequenz-Erkennung ---
        freq = pd.infer_freq(train_valid_data.index)
        if freq is None:
            self.config.freq = 's'
        else:
            freq_upper = freq.upper()
            if 'MIN' in freq_upper or freq_upper.startswith('T'):
                self.config.freq = 'min'
            elif freq_upper.startswith('M'):
                self.config.freq = 'ME'
            else:
                match = re.search(r"[a-zA-Z]", freq)
                if match:
                    self.config.freq = match.group(0).lower()
                else:
                    self.config.freq = 's'
        
        # --- Kanalanzahl ---
        column_num = train_valid_data.shape[1]
        self.config.enc_in = self.config.dec_in = self.config.c_out = column_num

        # --- NEU: Bestimmung des Johnson-Verteilungstyps pro Kanal ---
        train_data_for_fit, _ = train_val_split(train_valid_data, 0.9, self.config.seq_len)
        channel_types = []
        print("--- Analyzing data distribution to select Johnson system type for each channel... ---")
        for col_name in train_data_for_fit.columns:
            best_fit = get_best_johnson_fit(train_data_for_fit[col_name].values)
            channel_types.append(best_fit)
            print(f"  -> Channel '{col_name}': Best fit is Johnson {best_fit}")
        setattr(self.config, 'johnson_channel_types', channel_types)
        print("--- Johnson system analysis complete. ---\n")

        # --- Berechnung der Verteilungsgrenzen (wird für Johnson SB benötigt) ---
        channel_bounds = {}
        for col in train_data_for_fit.columns:
            min_val, max_val = train_data_for_fit[col].min(), train_data_for_fit[col].max()
            buffer = 0.1 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
            channel_bounds[col] = {"lower": min_val - buffer, "upper": max_val + buffer}
        setattr(self.config, 'channel_bounds', channel_bounds)

    def _find_interesting_windows(self, valid_data: pd.DataFrame):
        """
        Sucht einmalig die "schwierigsten" Fenster im Validierungsdatensatz.
        Das Ergebnis wird in self.interesting_window_indices gespeichert.
        """
        cache_key = self.config.seq_len
        if cache_key in WINDOW_SEARCH_CACHE:
            print("--- Loading interesting windows from cache... ---")
            self.interesting_window_indices = WINDOW_SEARCH_CACHE[cache_key]
            return

        print("\n--- Searching for interesting windows for diagnostic plots (one-time search for this seq_len)... ---")
        try:
            found_indices = find_interesting_windows(
                valid_data, self.config.horizon, self.config.seq_len
            )
            self.interesting_window_indices = found_indices
            WINDOW_SEARCH_CACHE[cache_key] = found_indices
            print("--- Found and cached interesting windows. ---\n")
        except Exception as e:
            print(f"WARNING: Could not find/cache interesting windows. Plotting will be skipped. Error: {e}")
            self.interesting_window_indices = None

    def _log_interesting_window_plots(self, epoch: int, writer: SummaryWriter, valid_dataset: Any):
        """
        Führt Inferenz auf den gefundenen "schwierigen" Fenstern durch und loggt
        die Plots in TensorBoard.
        """
        if self.config.horizon <= 1:
            if not hasattr(self, '_logged_horizon_skip_warning'):
                print("\n[INFO] Diagnostic window plotting is skipped for horizon <= 1 as plots would not be meaningful.")
                self._logged_horizon_skip_warning = True
            return

        if not self.interesting_window_indices:
            return
        
        with torch.no_grad():
            device = next(self.model.parameters()).device
            self.model.eval()

            for channel_name, methods in self.interesting_window_indices.items():
                for method_name, window_start_idx in methods.items():
                    forecast_start_idx = window_start_idx + self.config.horizon
                    sample_idx = forecast_start_idx - self.config.seq_len

                    if not (0 <= sample_idx < len(valid_dataset)):
                        continue

                    input_sample_tensor, target_sample_tensor, _, _ = valid_dataset[sample_idx]
                    actuals_data_tensor = target_sample_tensor[-self.config.horizon:, :]

                    input_data = input_sample_tensor.float().unsqueeze(0).to(device)
                    actuals_data = actuals_data_tensor.float().unsqueeze(0).to(device)
                    
                    denorm_distr, _, _, _, _, _, _, _, _ = self.model(input_data)

                    nll_per_point = -denorm_distr.log_prob(actuals_data)
                            
                    try:
                        channel_names = list(self.config.channel_bounds.keys())
                        channel_idx = channel_names.index(channel_name)
                        nll_val = nll_per_point[:, channel_idx, :].mean().item()
                    except (ValueError, AttributeError):
                        nll_val = nll_per_point.mean().item()
                        
                    avg_stddev = denorm_distr.stddev.mean().item()

                    fig = self._create_window_plot(
                        history=input_sample_tensor.cpu().numpy(),
                        actuals=actuals_data_tensor.cpu().numpy(),
                        prediction_dist=denorm_distr,
                        channel_name=channel_name,
                        title=f'{channel_name} | {method_name} | NLL: {nll_val:.2f} | AvgStdDev: {avg_stddev:.2f}'
                    )
                            
                    tag = f"Hard_Windows | {channel_name} | {method_name}"
                    writer.add_figure(tag, fig, global_step=epoch)
                    plt.close(fig)

        self.model.train()

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float, trial: Optional[Any] = None) -> "ModelBase":
        self._tune_hyper_params(train_valid_data)
        config = self.config

        target_channel_check = getattr(config, 'optimization_target_channel', None)
        if target_channel_check:
            channel_names = list(config.channel_bounds.keys())
            if target_channel_check not in channel_names:
                raise ValueError(
                    f"FATAL (Pre-Flight Check): Der Optimierungs-Zielkanal '{target_channel_check}' wurde in der Konfiguration angegeben, "
                    f"aber nicht in der Liste der verfügbaren Datenkanäle gefunden: {channel_names}. "
                    "Der Trial wird sofort abgebrochen, um Ressourcen zu sparen."
                )
            else:
                print(f"--- INFO: Optimierungs-Zielkanal '{target_channel_check}' wurde in den Datenkanälen gefunden und wird für die Validierung verwendet. ---")

        log_dir = getattr(config, 'log_dir', f'runs/{self.model_name}_{int(time.time())}')
        setattr(config, 'log_dir', log_dir)
        writer = SummaryWriter(log_dir)

        model_save_path = os.path.join(log_dir, 'best_model.pt')
        
        self._build_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        
        print(f"--- Model Analysis ---Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        
        if getattr(config, 'enable_diagnostic_plots', True) and valid_data is not None and not valid_data.empty:
            self._find_interesting_windows(valid_data)

        print("--- Preparing data for training... ---")
        print("INFO: Creating training sequences. This may take a moment for large datasets...")
        train_dataset, train_data_loader = forecasting_data_provider(train_data, config, timeenc=1, batch_size=config.batch_size, shuffle=True, drop_last=True)
        print(f"INFO: Training data prepared with {len(train_dataset)} samples.")
        
        valid_data_loader = None
        if valid_data is not None and not valid_data.empty:
            print("INFO: Creating validation sequences...")
            valid_dataset, valid_data_loader = forecasting_data_provider(valid_data, config, timeenc=1, batch_size=config.batch_size, shuffle=False, drop_last=False)
            print(f"INFO: Validation data prepared with {len(valid_dataset)} samples.")
        print("--- Data preparation complete. Starting training loop. ---")

        print("--- Setting up optimizer with targeted weight decay for ESN readouts... ---")
        
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        esn_uni_readout_params, esn_multi_readout_params, other_params = model_ref.get_parameter_groups()
        optimizer_grouped_parameters = [{'params': other_params, 'weight_decay': 0.0}]

        if esn_uni_readout_params and config.esn_uni_weight_decay > 0:
            optimizer_grouped_parameters.append({'params': esn_uni_readout_params, 'weight_decay': config.esn_uni_weight_decay})
            print(f"  -> Applying weight_decay={config.esn_uni_weight_decay:.2e} to {len(esn_uni_readout_params)} univariate ESN readout parameters.")

        if esn_multi_readout_params and config.esn_multi_weight_decay > 0:
            optimizer_grouped_parameters.append({'params': esn_multi_readout_params, 'weight_decay': config.esn_multi_weight_decay})
            print(f"  -> Applying weight_decay={config.esn_multi_weight_decay:.2e} to {len(esn_multi_readout_params)} multivariate ESN readout parameters.")
        
        print(f"  -> {len(other_params)} other parameters will have no weight decay.")
        print("--------------------------------------------------------------------")

        optimizer = Adam(optimizer_grouped_parameters, lr=config.lr)
        
        scheduler = None
        if config.lradj == "plateau" and valid_data_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=config.patience // 2, factor=0.5)

        self.early_stopping = EarlyStopping(
            patience=config.patience,
            verbose=True,
            delta=getattr(config, 'early_stopping_delta', 0),
            path=model_save_path
        )

        start_time = time.time()
        last_validation_time = start_time
        epoch_start_time = start_time
        last_memory_check_time = start_time
        initial_memory_check_seconds = 5
        regular_memory_check_interval_seconds = 60
        current_memory_check_interval = initial_memory_check_seconds

        max_training_time = getattr(config, 'max_training_time', float('inf'))
        global_step = 0
        
        accumulation_steps = config.accumulation_steps

        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        trace_handler = torch.profiler.tensorboard_trace_handler(
            os.path.join(config.log_dir, 'profiler')
        )
        
        profiler_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

        try:
            for epoch in range(config.num_epochs):
                epoch_start_time = time.time()
                self.model.train()
                
                epoch_total_losses, epoch_importance_losses, epoch_normalized_losses, epoch_denorm_losses_per_channel = [], [], [], []

                total_experts = config.num_linear_experts + config.num_univariate_esn_experts + config.num_multivariate_esn_experts
                sum_gate_weights_linear = torch.zeros(config.num_linear_experts, device=device) if config.num_linear_experts > 0 else None
                sum_gate_weights_uni_esn = torch.zeros(config.num_univariate_esn_experts, device=device) if config.num_univariate_esn_experts > 0 else None
                sum_gate_weights_multi_esn = torch.zeros(config.num_multivariate_esn_experts, device=device) if config.num_multivariate_esn_experts > 0 else None
                sum_selection_counts = torch.zeros(total_experts, device=device) if total_experts > 0 else None
                
                expert_metrics_batch_count = 0
                
                n_vars = self.config.c_out
                sum_p_learned = torch.zeros((n_vars, n_vars), device=device)
                sum_p_final = torch.zeros((n_vars, n_vars), device=device)
                num_batches_processed = 0
                
                optimizer.zero_grad()

                epoch_loop = tqdm(
                    train_data_loader,
                    desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                    leave=False,
                    file=sys.stdout,
                    mininterval=config.tqdm_min_interval
                )

                epoch_total_loss_sum = 0.0
                epoch_norm_loss_sum = 0.0

                with torch.profiler.profile(
                    activities=profiler_activities,
                    schedule=profiler_schedule,
                    on_trace_ready=trace_handler,
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True
                ) if config.profile_epoch == epoch else contextlib.nullcontext() as prof:
                    for i, batch in enumerate(epoch_loop):
                        global_step += 1
                        input_data, target, _, _ = batch
                        input_data = input_data.to(device)
                        target = target.to(device)
                        
                        denorm_distr, base_distr, loss_importance, batch_gate_weights_linear, batch_gate_weights_uni_esn, batch_gate_weights_multi_esn, batch_selection_counts, p_learned, p_final = self.model(input_data)
                        
                        target_horizon = target[:, -config.horizon:, :]
                        
                        norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)

                        log_probs = base_distr.log_prob(norm_target)
                        normalized_loss = -log_probs.mean()

                        total_loss = normalized_loss + config.loss_coef * loss_importance

                        with torch.no_grad():
                            denorm_loss_per_sample = -denorm_distr.log_prob(target_horizon).mean(dim=2)
                            epoch_denorm_losses_per_channel.append(denorm_loss_per_sample.cpu().numpy())

                        scaled_loss = total_loss / accumulation_steps
                        scaled_loss.backward()

                        if (i + 1) % accumulation_steps == 0:
                            if config.use_agc:
                                adaptive_clip_grad_(self.model.parameters(), clip_factor=config.agc_lambda)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        epoch_total_losses.append(total_loss.item())                        
                        epoch_normalized_losses.append(normalized_loss.item())
                        epoch_importance_losses.append(loss_importance.item())

                        if sum_gate_weights_linear is not None and batch_gate_weights_linear.numel() > 0:
                            sum_gate_weights_linear += batch_gate_weights_linear.detach().mean(dim=0)
                        if sum_gate_weights_uni_esn is not None and batch_gate_weights_uni_esn.numel() > 0:
                            sum_gate_weights_uni_esn += batch_gate_weights_uni_esn.detach().mean(dim=0)
                        if sum_gate_weights_multi_esn is not None and batch_gate_weights_multi_esn.numel() > 0:
                            sum_gate_weights_multi_esn += batch_gate_weights_multi_esn.detach().mean(dim=0)
                        if sum_selection_counts is not None and batch_selection_counts.numel() > 0:
                            sum_selection_counts += batch_selection_counts.detach().mean(dim=0)
                        
                        expert_metrics_batch_count += 1
                        sum_p_learned += p_learned.detach().sum(dim=0)
                        sum_p_final += p_final.detach().sum(dim=0)
                        num_batches_processed += input_data.size(0)
                        
                        if prof:
                            prof.step()

                        epoch_total_loss_sum += total_loss.item()
                        epoch_norm_loss_sum += normalized_loss.item()

                        if (i + 1) % config.tqdm_update_freq == 0 or (i + 1) == len(train_data_loader):
                            avg_epoch_loss = epoch_total_loss_sum / (i + 1)
                            avg_epoch_norm_loss = epoch_norm_loss_sum / (i + 1)
                            epoch_loop.set_postfix(loss=f"{avg_epoch_loss:.4f}", nll=f"{avg_epoch_norm_loss:.4f}")

                        current_time = time.time()
                        if trial and (current_time - last_memory_check_time) > current_memory_check_interval:
                            last_memory_check_time = current_time
                            max_memory_gb = getattr(config, 'max_memory_gb', None)
                            if max_memory_gb is not None:
                                allocated_gb = 0
                                device_type = device.type

                                if device_type == 'cuda':
                                    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                                    limit_type = "CUDA"
                                elif device_type == 'mps':
                                    allocated_gb = torch.mps.driver_allocated_memory() / (1024**3)
                                    limit_type = "MPS"
                                
                                if allocated_gb > 0:
                                    tqdm.write(
                                        f"[Memory Check] Epoch {epoch+1}, Batch {i+1}: "
                                        f"Allocated ({limit_type}): {allocated_gb:.2f} GB / Limit: {max_memory_gb:.2f} GB"
                                    )
                                    if allocated_gb > max_memory_gb:
                                        tqdm.write(f"  -> ☠️ PRUNING: Memory limit exceeded. Trial will be stopped.")
                                        raise optuna.exceptions.TrialPruned(f"Exceeded {limit_type} memory limit of {max_memory_gb} GB.")
                            
                            current_memory_check_interval = regular_memory_check_interval_seconds


                avg_train_total_loss = np.mean(epoch_total_losses)
                avg_train_norm_loss = np.mean(epoch_normalized_losses)
                avg_train_importance_loss = np.mean(epoch_importance_losses)
                
                writer.add_scalar("A) Overall Loss (for Optimizer) | Train_Total_Loss (NLL+MoE)", avg_train_total_loss, epoch)
                writer.add_scalar("A) Overall Loss (for Optimizer) | Train_Normalized_NLL", avg_train_norm_loss, epoch)
                writer.add_scalar("A) Overall Loss (for Optimizer) | Train_MoE_Importance_Loss", avg_train_importance_loss, epoch)

                writer.add_scalar("D) Normalized NLL (for Debugging) | Train", avg_train_norm_loss, epoch)

                if epoch_denorm_losses_per_channel:
                    all_train_losses_denorm = np.concatenate(epoch_denorm_losses_per_channel, axis=0)
                    
                    avg_train_loss_all_channels = np.mean(all_train_losses_denorm)
                    writer.add_scalar("B) Denormalized NLL (Evaluation) | Train_Avg_AllChannels", avg_train_loss_all_channels, epoch)

                    avg_train_loss_per_channel = np.mean(all_train_losses_denorm, axis=0)
                    channel_names = list(self.config.channel_bounds.keys())
                    for i, name in enumerate(channel_names):
                        writer.add_scalar(f"C) Denormalized NLL per Channel | {name} | Train", avg_train_loss_per_channel[i], epoch)

                if expert_metrics_batch_count > 0:
                    avg_gate_weights_linear = sum_gate_weights_linear / expert_metrics_batch_count if sum_gate_weights_linear is not None else None
                    avg_gate_weights_uni_esn = sum_gate_weights_uni_esn / expert_metrics_batch_count if sum_gate_weights_uni_esn is not None else None
                    avg_gate_weights_multi_esn = sum_gate_weights_multi_esn / expert_metrics_batch_count if sum_gate_weights_multi_esn is not None else None
                    avg_selection_counts = sum_selection_counts / expert_metrics_batch_count if sum_selection_counts is not None else torch.tensor([])
                
                model_to_log = self.model.module if hasattr(self.model, 'module') else self.model
                
                if avg_gate_weights_linear is not None:
                    for i, weight in enumerate(avg_gate_weights_linear):
                        writer.add_scalar(f"E) Expert Gating (Train) | Weights | Linear_{i}", weight.item(), epoch)
                if avg_gate_weights_uni_esn is not None:
                    for i, weight in enumerate(avg_gate_weights_uni_esn):
                        writer.add_scalar(f"E) Expert Gating (Train) | Weights | ESN_univariate_{i}", weight.item(), epoch)
                if avg_gate_weights_multi_esn is not None:
                    for i, weight in enumerate(avg_gate_weights_multi_esn):
                        writer.add_scalar(f"E) Expert Gating (Train) | Weights | ESN_multivariate_{i}", weight.item(), epoch)

                if avg_selection_counts.numel() > 0:
                    expert_idx = 0
                    for i in range(config.num_linear_experts):
                        writer.add_scalar(f"E) Expert Gating (Train) | Selection_Counts | Linear_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1
                    for i in range(config.num_univariate_esn_experts):
                        writer.add_scalar(f"E) Expert Gating (Train) | Selection_Counts | ESN_univariate_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1
                    for i in range(config.num_multivariate_esn_experts):
                        writer.add_scalar(f"E) Expert Gating (Train) | Selection_Counts | ESN_multivariate_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1

                if num_batches_processed > 0:
                    avg_p_learned = sum_p_learned / num_batches_processed
                    avg_p_final = sum_p_final / num_batches_processed

                metric_for_optimization = float('nan')

                if valid_data_loader is not None:
                    all_window_losses_denorm, all_window_losses_norm = self.validate(valid_data_loader, writer, epoch, device, desc=f"Epoch {epoch+1} Validation")
                    
                    channel_names = list(self.config.channel_bounds.keys())
                    for i, name in enumerate(channel_names):
                        avg_channel_loss = np.mean(all_window_losses_denorm[:, i])
                        writer.add_scalar(f"C) Denormalized NLL per Channel | {name} | Validation", avg_channel_loss, epoch)

                    target_channel = getattr(self.config, 'optimization_target_channel', None)
                    optimization_target_name_for_log = ""

                    if target_channel:
                        if target_channel not in channel_names:
                            raise ValueError(
                                f"FATAL: Der Optimierungs-Zielkanal '{target_channel}' wurde angegeben, "
                                f"aber nicht in der Liste der verfügbaren Datenkanäle gefunden: {channel_names}. "
                                "Bitte auf Tippfehler oder Probleme in der Daten-Pipeline prüfen."
                            )
                        
                        target_idx = channel_names.index(target_channel)
                        losses_for_optimization = all_window_losses_denorm[:, target_idx]
                        optimization_target_name_for_log = f"Kanal '{target_channel}'"
                    else:
                        losses_for_optimization = all_window_losses_denorm.mean(axis=1)
                        optimization_target_name_for_log = "Durchschnitt aller Kanäle"

                    avg_metric_for_opt = np.mean(losses_for_optimization)
                    cvar_metric_for_opt = calculate_cvar(losses_for_optimization, self.config.cvar_alpha)

                    overall_avg_nll = np.mean(all_window_losses_denorm)
                    overall_cvar_nll = calculate_cvar(all_window_losses_denorm.mean(axis=1), self.config.cvar_alpha)
                    if target_channel and target_channel in channel_names:
                        writer.add_scalar(f"Loss_Target_{target_channel} | Validation_Avg_NLL", avg_metric_for_opt, epoch)
                        writer.add_scalar(f"Loss_Target_{target_channel} | Validation_CVaR_NLL", cvar_metric_for_opt, epoch)


                    if self.config.optimization_metric == 'cvar':
                        metric_for_optimization = cvar_metric_for_opt
                        metric_name_for_logging = f"CVaR@{self.config.cvar_alpha:.2f}"
                    else:
                        metric_for_optimization = avg_metric_for_opt
                        metric_name_for_logging = "Avg NLL"

                    writer.add_scalar(f"Loss_Optimized | Validation_Metric", metric_for_optimization, epoch)

                    log_msg = (f"Epoch {epoch + 1} Validation | Overall Avg NLL: {overall_avg_nll:.6f} | "
                               f"Optimierungs-Metrik ({optimization_target_name_for_log}): {metric_name_for_logging} = {metric_for_optimization:.6f} --> Für Early Stopping & Pruning verwendet.")
                    tqdm.write(log_msg)

                    old_best_loss = self.early_stopping.val_loss_min
                    checkpoint_to_save = {
                        'config_dict': self.config.__dict__,
                        'model_state_dict': self.model.state_dict()
                    }
                    self.early_stopping(metric_for_optimization, checkpoint_to_save)

                    if trial:
                        actual_target = target_channel if target_channel else "all_channels_mean"
                        trial.set_user_attr("optimization_target_channel", actual_target)
                        trial.report(metric_for_optimization, epoch)
                        if trial.should_prune():
                            tqdm.write(f"  -> Trial pruned by {trial.study.pruner.__class__.__name__} after epoch {epoch + 1}.")
                            raise optuna.exceptions.TrialPruned()
                    
                    if getattr(config, 'enable_diagnostic_plots', False):
                        if self.early_stopping.val_loss_min < old_best_loss:
                            if valid_dataset and self.interesting_window_indices:
                                self._log_interesting_window_plots(epoch, writer, valid_dataset)
                            
                            if 'avg_p_learned' in locals() and 'avg_p_final' in locals():
                                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                                prior_matrix = model_ref.channel_adjacency_prior

                                fig_dependencies = self._log_dependency_heatmaps(
                                    prior_matrix=prior_matrix,
                                    learned_matrix=avg_p_learned.cpu().numpy(),
                                    final_matrix=avg_p_final.cpu().numpy()
                                )
                                writer.add_figure("Channel_Dependencies | Combined_View", fig_dependencies, global_step=epoch)
                                plt.close(fig_dependencies)

                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break
                    
                    if scheduler: scheduler.step(metric_for_optimization)

                if config.lradj != "plateau":
                    adjust_learning_rate(optimizer, epoch + 1, config, verbose=True)
                
                if (time.time() - start_time) > max_training_time:
                    print(f"Trial timed out after {(time.time() - start_time):.2f}s (max: {max_training_time}s).")
                    break
                
                writer.add_scalar("Misc | Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
                writer.flush()

                epoch_duration = time.time() - epoch_start_time
                summary_metrics = {
                    "epoch": epoch + 1,
                    "duration_s": epoch_duration,
                    "lr": optimizer.param_groups[0]['lr'],
                    "train_loss_total": avg_train_total_loss,
                    "train_loss_norm": avg_train_norm_loss,
                    "train_loss_importance": avg_train_importance_loss,
                    "validation_metric": metric_for_optimization,
                }
                self._log_epoch_summary_to_file(summary_metrics)

        finally:
            print("--- Finalizing run: closing writer. ---")
            if hasattr(self, "early_stopping") and self.early_stopping.path and os.path.exists(self.early_stopping.path):
                print(f"""
{'='*20} BEST MODEL AVAILABLE {'='*20}
>>> Best model was saved to {self.early_stopping.path} <<<
{'='*58}""")
            else:
                print("No checkpoint was saved during training.")
            writer.close()

        return self

    def _log_epoch_summary_to_file(self, metrics: Dict[str, Any]):
        """Schreibt eine formatierte Zusammenfassung der Epochen-Metriken in eine Log-Datei."""
        log_file_path = os.path.join(self.config.log_dir, 'training_summary.log')

        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write(f"Training Summary for {self.model_name}\n")
                f.write(f"Log started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        summary_str = f"""
============================== EPOCH {metrics['epoch']:<4} COMPLETE ==============================
--- Performance ---
Duration                : {metrics['duration_s']:.2f} s
Learning Rate           : {metrics['lr']:.2e}
Train Loss (Total)      : {metrics['train_loss_total']:.6f}
Validation Metric       : {metrics['validation_metric']:.6f}  (Lower is better)

--- Loss Components (Train) ---
Normalized NLL          : {metrics['train_loss_norm']:.6f}  (Main objective)
Importance Loss         : {metrics['train_loss_importance']:.6f}  (MoE balance)
"""
        with open(log_file_path, 'a') as f:
            f.write(summary_str)

    def validate(self, valid_data_loader, writer: Optional[SummaryWriter], epoch: Optional[int], device: torch.device, desc: str = "Validation") -> tuple[np.ndarray, np.ndarray]:
        all_denorm_losses, all_norm_losses = [], []
        self.model.eval()
        with torch.no_grad():
            validation_loop = tqdm(
                valid_data_loader,
                desc=desc,
                leave=False,
                file=sys.stdout,
                mininterval=self.config.tqdm_min_interval
            )
            for batch in validation_loop:
                input_data, target, _, _ = batch
                input_data = input_data.to(device)
                target = target.to(device)
                
                target_horizon = target[:, -self.config.horizon:, :]

                denorm_distr, base_distr, _, _, _, _, _, _, _ = self.model(input_data)
                
                denorm_loss_per_sample = -denorm_distr.log_prob(target_horizon).mean(dim=2)
                all_denorm_losses.append(denorm_loss_per_sample.cpu().numpy())

                norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
                norm_loss_per_sample = -base_distr.log_prob(norm_target).mean(dim=2)
                all_norm_losses.append(norm_loss_per_sample.cpu().numpy())

                if all_denorm_losses:
                    running_mean_denorm = np.mean(np.concatenate(all_denorm_losses))
                    running_mean_norm = np.mean(np.concatenate(all_norm_losses))
                    validation_loop.set_postfix(
                        denorm_nll=running_mean_denorm,
                        norm_nll=running_mean_norm
                    )

        self.model.train()
        return np.concatenate(all_denorm_losses), np.concatenate(all_norm_losses)

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call forecast_fit() first.")

        if hasattr(self, "early_stopping") and self.early_stopping.path and os.path.exists(self.early_stopping.path):
            print(f"--- Loading best model for forecast from {self.early_stopping.path} ---")
            device = next(self.model.parameters()).device
            self.model.load_state_dict(torch.load(self.early_stopping.path, map_location=device))
        elif self.early_stopping and self.early_stopping.best_score is None:
            print("WARNING: Forecasting with a model that has not been improved by EarlyStopping.")
            
        input_data = train.iloc[-self.seq_len:, :].values
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        self.model.eval()
        with torch.no_grad():
            distr, _, _, _, _, _, _, _, _ = self.model(input_tensor)
            
            q_list = self.config.quantiles
            q_tensor = torch.tensor(q_list, device=device, dtype=torch.float32)
            
            quantile_preds = distr.icdf(q_tensor)
        
        output_array = quantile_preds.squeeze(0).permute(1, 0, 2).cpu().numpy()
        return output_array

    def load(self, checkpoint_path: str) -> "ModelBase":
        """
        Loads a model's state_dict from a checkpoint file.
        This method assumes the model architecture has already been initialized.
        It does NOT load the configuration from the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
        if self.model is None:
            raise RuntimeError("Cannot load state_dict into a non-existent model. Please initialize the model first by calling _build_model().")

        device = next(self.model.parameters()).device
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        print(f"Model state_dict successfully loaded from {checkpoint_path}.")
        return self

    def _create_window_plot(self, history, actuals, prediction_dist, channel_name, title):
        """
        Erstellt eine Matplotlib-Figur für ein einzelnes Fenster.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        try:
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            channel_idx = model_ref.channel_names.index(channel_name)
        except (ValueError, AttributeError, IndexError):
            channel_idx = 0

        horizon_len = actuals.shape[0]
        
        history_to_plot = history[-horizon_len:, :]
        
        history_x = np.arange(horizon_len)
        forecast_x = np.arange(horizon_len, horizon_len + horizon_len)
        
        history_y = history_to_plot[:, channel_idx]
        actuals_y = actuals[:, channel_idx]
        
        quantiles = self.config.quantiles
        device = prediction_dist.mean.device
        q_tensor = torch.tensor(q_list, device=device, dtype=torch.float32)

        quantile_preds_full = prediction_dist.icdf(q_tensor).squeeze(0).cpu().numpy()
        
        quantile_preds_sliced = quantile_preds_full[:, :horizon_len, :]

        preds_y = quantile_preds_sliced[channel_idx, :, :]
        
        try:
            median_idx = quantiles.index(0.5)
        except (ValueError, AttributeError):
            median_idx = len(quantiles) // 2

        ax.plot(history_x, history_y, label="History", color="gray")
        
        ax.axvline(x=history_x[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start')

        ax.plot(forecast_x, actuals_y, label="Actual", color="black", linewidth=2, zorder=10)
        
        num_ci_levels = len(quantiles) // 2
        base_alpha = 0.1
        alpha_step = 0.15

        for i in range(num_ci_levels):
            lower_q_idx = i
            upper_q_idx = len(quantiles) - 1 - i
            
            current_alpha = base_alpha + (i * alpha_step)

            ax.fill_between(
                forecast_x, preds_y[:, lower_q_idx], preds_y[:, upper_q_idx],
                alpha=current_alpha, color='C0',
                label=f"CI {quantiles[lower_q_idx]}-{quantiles[upper_q_idx]}"
            )
            
        ax.plot(forecast_x, preds_y[:, median_idx], label="Median Forecast", color="blue", linestyle='--', zorder=11)
        
        ax.set_title(title)
        ax.set_ylabel("Time Series Value")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left')

        plt.tight_layout()
        
        return fig

    def _plot_single_heatmap(self, ax, matrix, title, channel_names, vmin=0, vmax=1):
        """Helper to draw a single heatmap on a given Matplotlib axis."""
        im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)

        if channel_names and len(channel_names) == matrix.shape[0]:
            ax.set_xticks(np.arange(len(channel_names)))
            ax.set_yticks(np.arange(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha="right")
            ax.set_yticklabels(channel_names)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = "w" if matrix[i, j] < 0.6 * vmax else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color)

        ax.set_title(title)
        return im

    def _log_dependency_heatmaps(self, prior_matrix, learned_matrix, final_matrix):
        """Creates a 3-panel matplotlib figure showing the dependency matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
        fig.suptitle("Channel Dependency Analysis (Epoch Average)", fontsize=16)

        channel_names = list(getattr(self.config, 'channel_bounds', {}).keys())
        n_vars = len(channel_names)

        if prior_matrix is not None:
            prior_np = prior_matrix.cpu().numpy()
        else:
            prior_np = np.ones((n_vars, n_vars))

        im1 = self._plot_single_heatmap(axes[0], prior_np, "User Prior", channel_names, vmin=0, vmax=1)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = self._plot_single_heatmap(axes[1], learned_matrix, "Learned (Unconstrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = self._plot_single_heatmap(axes[2], final_matrix, "Effective (Constrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        return fig