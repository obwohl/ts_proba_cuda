import torch
import optuna
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
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.utils.crps import crps_loss
from ts_benchmark.baselines.duet.utils.tools import adjust_learning_rate, EarlyStopping
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
# === NEUER IMPORT FÜR DIE FENSTER-SUCHE ===
from ts_benchmark.baselines.duet.utils.window_search import find_interesting_windows
# === NEUER IMPORT FÜR EXPERTEN-TYPEN ===
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert
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
            # NEU: Hybride Experten-Konfiguration
            "num_linear_experts": 2,
            "num_univariate_esn_experts": 1,
            "num_multivariate_esn_experts": 1,
            "k": 2,            # Default, will be overwritten below

            # --- ESN Expert Default Parameters ---
            # Univariate ESN
            "reservoir_size_uni": 256,
            "spectral_radius_uni": 0.99,
            "sparsity_uni": 0.1,
            "leak_rate_uni": 1.0,
            "input_scaling_uni": 1.0,

            # Multivariate ESN
            "reservoir_size_multi": 256,
            "spectral_radius_multi": 0.99,
            "sparsity_multi": 0.1,
            "leak_rate_multi": 1.0,
            "input_scaling_multi": 0.5, # Already separate
            
            # --- NEW: ESN Readout Regularization ---
            "esn_uni_weight_decay": 0.0,
            "esn_multi_weight_decay": 0.0,
            
            # --- Training / Optimization ---
            "lr": 1e-4,
            "lradj": "cosine_warmup", "num_epochs": 100,
            "accumulation_steps": 1, # NEU: Für Gradienten-Akkumulation
            "batch_size": 128, "patience": 10,
            "num_workers": 4,  # <<< HIER HINZUFÜGEN

            # --- NEW: Tier 2 Training Strategies ---
            "loss_function": "crps", # 'crps' or 'gfl'
            "gfl_gamma": 2.0,        # Focusing parameter for GFL
            "use_agc": False,        # Use Adaptive Gradient Clipping
            "agc_lambda": 0.01,      # Clipping factor for AGC

            # --- Data & Miscellaneous ---
            "moving_avg": 25, "CI": False, "freq": "h",
            "quantiles": [0.1, 0.5, 0.9], # Für die Inferenz
            "num_bins": 100, "tail_percentile": 0.05,
            "norm_mode": "subtract_median", # Preferred normalization mode

            # --- NEW: Projection Head Configuration ---
            "projection_head_layers": 0,       # Default to 0 for original behavior (single linear layer)
            "projection_head_dim_factor": 2,   # Hidden dim = in_features / factor
            "projection_head_dropout": 0.1,

            # --- NEW: Jerk Regularization for Xi ---
            "jerk_loss_coef": 0.0, # Default to 0 (disabled)

            # --- NEW: Interim Validation ---
            "interim_validation_seconds": None, # Default: disabled. Set to e.g. 300 for 5-min validation.

            # --- NEW: Performance Profiling ---
            "profile_epoch": None, # Set to an epoch number (e.g., 2) to enable profiling for that epoch.
            # NEU: Frequenz für die Aktualisierung der tqdm-Fortschrittsanzeige (jeder n-te Batch)
            "tqdm_update_freq": 10,
            # NEU: Mindestintervall in Sekunden für die Aktualisierung des tqdm-Balkens, um I/O-Spam zu reduzieren
            "tqdm_min_interval": 1.0,
        }

        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Abgeleitete Werte
        if hasattr(self, 'seq_len'):
            # Diese Werte werden von manchen Sub-Modulen erwartet
            setattr(self, "input_size", self.seq_len)
            setattr(self, "label_len", self.seq_len // 2) 
        else:
            raise AttributeError("Konfiguration muss 'seq_len' enthalten.")
        
        if hasattr(self, 'horizon'):
            setattr(self, "pred_len", self.horizon)
        else:
            raise AttributeError("Konfiguration muss 'horizon' enthalten.")
            
        # 'k' muss kleiner oder gleich der Gesamtanzahl Experten sein.
        # Wir setzen es hier sicherheitshalber nach der Experten-Definition.
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
        self.interesting_window_indices: Optional[Dict] = None # Für die neuen Plots
        
        # --- NEU: Cache für statische GFL-Tensoren zur Performance-Optimierung ---
        self._cached_gfl_bins_lower = None
        self._cached_gfl_bin_widths = None

    @property
    def model_name(self) -> str:
        return "DUET-Prob-CRPS-v2"

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
        und Verteilungsgrenzen (channel_bounds).
        """
        # --- Frequenz-Erkennung ---
        freq = pd.infer_freq(train_valid_data.index)
        if freq is None:
            # Fallback für unregelmäßige Daten
            self.config.freq = 's'
        else:
            # KORREKTUR: Die vorherige Logik hat Frequenzen wie 'min' (Minuten) fälschlicherweise
            # zu 'm' verkürzt, was Pandas als veralteten Code für Monate interpretiert und
            # eine DeprecationWarning auslöst. Diese neue Logik behandelt die Fälle explizit,
            # um die Mehrdeutigkeit zu vermeiden und die Warnung endgültig zu beheben.
            freq_upper = freq.upper()
            
            if 'MIN' in freq_upper or freq_upper.startswith('T'):
                self.config.freq = 'min'  # 'min' ist der eindeutige Code für Minuten.
            elif freq_upper.startswith('M'):
                self.config.freq = 'ME' # 'ME' ist der moderne Code für Monate (Month-End).
            else:
                # Für andere Frequenzen (H, D, S) funktioniert die ursprüngliche Logik,
                # bei der der erste Buchstabe genommen wird.
                match = re.search(r"[a-zA-Z]", freq)
                if match:
                    self.config.freq = match.group(0).lower()
                else:
                    self.config.freq = 's'
        
        # --- Kanalanzahl ---
        column_num = train_valid_data.shape[1]
        self.config.enc_in = self.config.dec_in = self.config.c_out = column_num

        # --- NEU: Berechnung der Verteilungsgrenzen ---
        # Wir berechnen die Grenzen aus dem Trainingsanteil der Daten.
        # Wir nehmen hier an, dass die Aufteilung dieselbe ist wie im Haupt-Trainingslauf (90/10).
        train_data_for_bounds, _ = train_val_split(train_valid_data, 0.9, self.config.seq_len)
        channel_bounds = {}
        for col in train_data_for_bounds.columns:
            min_val, max_val = train_data_for_bounds[col].min(), train_data_for_bounds[col].max()
            buffer = 0.1 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
            channel_bounds[col] = {"lower": min_val - buffer, "upper": max_val + buffer}
        setattr(self.config, 'channel_bounds', channel_bounds)

    def _find_interesting_windows(self, valid_data: pd.DataFrame):
        """
        Sucht einmalig die "schwierigsten" Fenster im Validierungsdatensatz.
        Das Ergebnis wird in self.interesting_window_indices gespeichert.
        """
        # Der Cache-Schlüssel ist die Sequenzlänge, da die Validierungsdaten davon abhängen.
        cache_key = self.config.seq_len

        if cache_key in WINDOW_SEARCH_CACHE:
            print("--- Loading interesting windows from cache... ---")
            self.interesting_window_indices = WINDOW_SEARCH_CACHE[cache_key]
            return

        print("\n--- Searching for interesting windows for diagnostic plots (one-time search for this seq_len)... ---")
        try:
            # Die Funktion `find_interesting_windows` gibt bereits ein Dictionary
            # mit den Kanalnamen als Schlüssel zurück. Wir können es direkt verwenden.
            found_indices = find_interesting_windows(
                valid_data, self.config.horizon, self.config.seq_len
            )
            self.interesting_window_indices = found_indices

            # Speichere das Ergebnis im Cache für zukünftige Trials mit derselben seq_len
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
        # --- NEUER FIX: Diese Plots sind für einen Horizont von 1 nicht aussagekräftig. ---
        if self.config.horizon <= 1:
            # Wir loggen diese Nachricht nur einmal pro Training, um das Terminal nicht zu überfluten.
            if not hasattr(self, '_logged_horizon_skip_warning'):
                print("\n[INFO] Diagnostic window plotting is skipped for horizon <= 1 as plots would not be meaningful.")
                self._logged_horizon_skip_warning = True
            return

        if not self.interesting_window_indices:
            return
        
        # === KRITISCHER FIX GEGEN SPEICHERLECKS ===
        # 1. Das gesamte Plotting ist eine Inferenz-Operation. Wir packen alles in `torch.no_grad()`,
        #    um die Erstellung von Berechnungs-Graphen zu verhindern. Das ist die Hauptursache des Lecks.
        # 2. Wir schalten das Modell explizit in den `eval()`-Modus, um Layer wie Dropout zu deaktivieren,
        #    und schalten es danach wieder in den `train()`-Modus für das weitere Training.
        with torch.no_grad():
            # Hole das Gerät vom Modell und schalte in den Evaluationsmodus
            device = next(self.model.parameters()).device
            self.model.eval()

            for channel_name, methods in self.interesting_window_indices.items():
                for method_name, window_start_idx in methods.items():
                    # window_start_idx ist der Beginn des "Vorher"-Fensters in den rohen Validierungsdaten.
                    # Wir wollen das "Nachher"-Fenster vorhersagen, das bei `window_start_idx + horizon` beginnt.
                    forecast_start_idx = window_start_idx + self.config.horizon
                    
                    # Der Input für diese Vorhersage ist das Fenster der Länge `seq_len`, das bei `forecast_start_idx` endet.
                    # Der `forecasting_data_provider` erstellt Samples, wobei das Sample `j` dem Input `raw_data[j : j + seq_len]` entspricht.
                    # Daher ist der Index des Samples, das wir benötigen, `forecast_start_idx - seq_len`.
                    sample_idx = forecast_start_idx - self.config.seq_len

                    # Sicherheitsabfrage: Liegt der Index innerhalb der Grenzen des Datasets?
                    if not (0 <= sample_idx < len(valid_dataset)):
                        continue

                    # KORREKTUR: Greife auf das Sample über die __getitem__-Methode zu,
                    # die ein Tupel (seq_x, seq_y, seq_x_mark, seq_y_mark) zurückgibt.
                    # Die Elemente sind bereits Tensoren, nicht NumPy-Arrays.
                    input_sample_tensor, target_sample_tensor, _, _ = valid_dataset[sample_idx]

                    # Wir brauchen den Teil des Targets, der dem Horizont entspricht.
                    # Das Target aus dem Dataset enthält auch den label_len-Teil.
                    actuals_data_tensor = target_sample_tensor[-self.config.horizon:, :]

                    # Füge die Batch-Dimension hinzu und stelle sicher, dass die Daten auf dem richtigen Gerät sind.
                    input_data = input_sample_tensor.float().unsqueeze(0).to(device)
                    actuals_data = actuals_data_tensor.float().unsqueeze(0).to(device)
                    
                    # Unpack 9 values to match the new model signature, even if most are unused here.
                    denorm_distr, base_distr, _, _, _, _, _, _, _ = self.model(input_data)

                    # === KORREKTUR: CRPS pro Kanal berechnen, nicht den globalen Durchschnitt ===
                    # crps_loss gibt einen Tensor der Form [B, N_vars, H] zurück.
                    crps_per_point = crps_loss(denorm_distr, actuals_data.permute(0, 2, 1))
                        
                    # Finde den Index des aktuellen Kanals, um den spezifischen Loss zu extrahieren.
                    try:
                        channel_names = list(self.config.channel_bounds.keys())
                        channel_idx = channel_names.index(channel_name)
                        # Berechne den mittleren CRPS für DIESEN Kanal.
                        crps_val = crps_per_point[:, channel_idx, :].mean().item()
                    except (ValueError, AttributeError):
                        # Fallback, falls der Kanal nicht gefunden wird (sollte nicht passieren).
                        # In diesem Fall wird der Plot mit dem Gesamt-CRPS beschriftet.
                        crps_val = crps_per_point.mean().item()
                        
                    # === NEU: DIAGNOSE-CODE ZUR ÜBERPRÜFUNG DER UNSICHERHEIT ===
                    # Berechne die mittlere Standardabweichung der denormalisierten Verteilung.
                    # Dies ist der entscheidende Wert, um die Breite der Vorhersage zu quantifizieren.
                    # Wir nehmen den Durchschnitt über den Horizont und alle Kanäle für eine einzelne Zahl.
                    avg_stddev = denorm_distr.std.mean().item()
                    # === ENDE DIAGNOSE-CODE ===

                    # Plot erstellen
                    fig = self._create_window_plot(
                        history=input_sample_tensor.cpu().numpy(),
                        actuals=actuals_data_tensor.cpu().numpy(),
                        prediction_dist=denorm_distr,
                        channel_name=channel_name, # NEU
                        title=f'{channel_name} | {method_name} | CRPS: {crps_val:.2f} | AvgStdDev: {avg_stddev:.2f}'
                    )
                        
                    # Plot in TensorBoard loggen
                    tag = f"Hard_Windows/{channel_name}/{method_name}"
                    writer.add_figure(tag, fig, global_step=epoch)
                    # WICHTIG: Schließe die Figur, um Speicherlecks zu verhindern.
                    # Ohne dies sammelt matplotlib Referenzen an, was zu massivem RAM- und Swap-Verbrauch führt.
                    plt.close(fig)

        # Wichtig: Schalte das Modell wieder in den Trainingsmodus für die nächste Epoche.
        self.model.train()

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float, trial: Optional[Any] = None) -> "ModelBase":
        self._tune_hyper_params(train_valid_data)
        config = self.config

        # Priorisiere einen existierenden log_dir aus der Konfiguration (vom Benchmark-Runner gesetzt).
        # Wenn nicht vorhanden, erstelle einen Standard-Ordner. Dies zentralisiert die Ausgabe.
        log_dir = getattr(config, 'log_dir', f'runs/{self.model_name}_{int(time.time())}')
        setattr(config, 'log_dir', log_dir) # Stelle sicher, dass er für die spätere Verwendung (z.B. Checkpoints) gesetzt ist
        writer = SummaryWriter(log_dir)
        
        # Initialisiere das Modell. `_tune_hyper_params` hat bereits alles Nötige gesetzt.
        self._build_model()

        
        # Priorisiere CUDA > MPS > CPU. Das ist die logischere Reihenfolge.
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        
        print(f"--- Model Analysis ---\nTotal trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # DataParallel-Unterstützung
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # Daten-Setup: Wir brauchen keinen externen Scaler, da RevIN im Modell ist.
        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        
        # === NEU: Einmalige Suche nach interessanten Fenstern für die Plots ===
        # Führe die teure Suche nur aus, wenn die Plots auch aktiviert sind.
        if getattr(config, 'enable_diagnostic_plots', True) and valid_data is not None and not valid_data.empty:
            self._find_interesting_windows(valid_data)

        print("\n--- Preparing data for training... ---")
        # Der Daten-Provider erwartet timeenc=1 oder 2. Wir verwenden 1.
        # Er wird die 'date'-Spalte jetzt finden und korrekt verarbeiten.
        print("INFO: Creating training sequences. This may take a moment for large datasets...")
        train_dataset, train_data_loader = forecasting_data_provider(train_data, config, timeenc=1, batch_size=config.batch_size, shuffle=True, drop_last=True)
        print(f"INFO: Training data prepared with {len(train_dataset)} samples.")
        
        valid_data_loader = None
        if valid_data is not None and not valid_data.empty:
            print("INFO: Creating validation sequences...")
            valid_dataset, valid_data_loader = forecasting_data_provider(valid_data, config, timeenc=1, batch_size=config.batch_size, shuffle=False, drop_last=False)
            print(f"INFO: Validation data prepared with {len(valid_dataset)} samples.")
        print("--- Data preparation complete. Starting training loop. ---\n")

        # --- NEU: Erstelle Parametergruppen für gezieltes Weight Decay auf ESN-Readouts ---
        print("--- Setting up optimizer with targeted weight decay for ESN readouts... ---")
        
        # Hole eine Referenz zum eigentlichen Modell, auch wenn es in DataParallel verpackt ist.
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Rufe die neue Methode des Modells auf, um die Parametergruppen zu erhalten.
        # Dies kapselt die Logik und macht den Code hier sauberer.
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

        self.early_stopping = EarlyStopping(patience=config.patience, verbose=True)

        # --- NEU: Zeitmessung für Optuna ---
        start_time = time.time()
        last_validation_time = start_time # Initialisiere mit der Startzeit
        # NEU: Zeitmessung für die Speicherüberwachung
        last_memory_check_time = start_time
        # NEU: Zweistufiges Intervall für die Speicherprüfung
        initial_memory_check_seconds = 5    # Erste, schnelle Prüfung
        regular_memory_check_interval_seconds = 60 # Reguläres Intervall danach
        current_memory_check_interval = initial_memory_check_seconds

        # Hole max_resource aus der Konfiguration.
        max_training_time = getattr(config, 'max_training_time', float('inf'))
        global_step = 0
        
        # --- NEU: Hole die Akkumulationsschritte aus der Konfiguration ---
        accumulation_steps = config.accumulation_steps

        # --- NEU: Vorbereitung für den PyTorch Profiler ---
        # Definiere die Aktivitäten basierend auf dem verfügbaren Gerät.
        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Hinweis: Für MPS wird die GPU-Zeit unter der CPU-Aktivität erfasst.
        
        # Definiere den Trace-Handler, der die Ergebnisse für TensorBoard speichert.
        trace_handler = torch.profiler.tensorboard_trace_handler(
            os.path.join(config.log_dir, 'profiler')
        )
        
        # Definiere einen Zeitplan, um nur wenige Batches zu profilieren und Warm-up-Kosten zu ignorieren.
        # Wir warten 1 Batch, machen 1 Batch Warm-up und zeichnen dann 3 Batches auf.
        profiler_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

        # --- Ende der Profiler-Vorbereitung ---

        try:
            for epoch in range(config.num_epochs):
                self.model.train()
                
                # Initialisiere Listen zum Sammeln von Metriken über alle Batches einer Epoche
                # === FIX: Speichere nur Python-Skalare (.item()), keine Tensoren, um Speicherlecks zu verhindern ===
                epoch_total_losses, epoch_crps_losses, epoch_importance_losses, epoch_normalized_losses = [], [], [], []
                epoch_channel_losses = {name: [] for name in getattr(config, 'channel_bounds', {}).keys()} # Denormalisiert
                epoch_normalized_channel_losses = {name: [] for name in getattr(config, 'channel_bounds', {}).keys()} # Normalisiert

                # === FINALER FIX FÜR SPEICHERLECKS: Laufende Summen für Experten-Metriken ===
                # Anstatt Listen von Tensoren zu sammeln, summieren wir die Werte auf.
                # Initialisiere die Summen-Tensoren auf dem richtigen Gerät.
                total_experts = config.num_linear_experts + config.num_univariate_esn_experts + config.num_multivariate_esn_experts
                sum_gate_weights_linear = torch.zeros(config.num_linear_experts, device=device) if config.num_linear_experts > 0 else None
                sum_gate_weights_uni_esn = torch.zeros(config.num_univariate_esn_experts, device=device) if config.num_univariate_esn_experts > 0 else None
                sum_gate_weights_multi_esn = torch.zeros(config.num_multivariate_esn_experts, device=device) if config.num_multivariate_esn_experts > 0 else None
                sum_selection_counts = torch.zeros(total_experts, device=device) if total_experts > 0 else None
                
                # Zähler für die Mittelwertbildung
                expert_metrics_batch_count = 0
                
                # === FIX: Verwende eine laufende Summe für große Matrizen statt sie in einer Liste zu sammeln ===
                # Dies verhindert die Allokation eines riesigen Tensors am Ende der Epoche.
                # === FINALE KORREKTUR: Hole `n_vars` zuverlässig aus der Konfiguration des Wrappers (`self.config`), ===
                # anstatt auf ein unbekanntes Attribut des inneren Modells (`self.model`) zu raten.
                # `self.config.c_out` wird in `_tune_hyper_params` garantiert gesetzt.
                n_vars = self.config.c_out
                sum_p_learned = torch.zeros((n_vars, n_vars), device=device)
                sum_p_final = torch.zeros((n_vars, n_vars), device=device)
                num_batches_processed = 0
                
                # Setze den Gradienten-Speicher vor der Epoche zurück
                optimizer.zero_grad()

                # --- VERBESSERUNG: tqdm für eine informative Fortschrittsanzeige ---
                # Wir umwickeln den DataLoader mit tqdm, um einen Fortschrittsbalken zu erhalten.
                epoch_loop = tqdm(
                    train_data_loader,
                    desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                    leave=False, # Verhindert, dass für jede Epoche eine Zeile übrig bleibt
                    file=sys.stdout, # Stellt sicher, dass die Ausgabe in der Konsole landet
                    mininterval=config.tqdm_min_interval # NEU: Kontrolliert die Update-Frequenz des Balkens
                )

                # --- VERBESSERUNG: Laufende Summen für den tqdm-Fortschrittsbalken ---
                # Wir verwenden einfache Floats, um den Durchschnitt über die Epoche zu verfolgen.
                epoch_total_loss_sum = 0.0
                epoch_norm_loss_sum = 0.0

                # --- NEU: Profiler-Kontext-Manager ---
                # Wenn die profile_epoch gesetzt ist, wird der Profiler für diese Epoche aktiviert.
                # Er verwendet den oben definierten Zeitplan und Trace-Handler.
                with torch.profiler.profile(
                    activities=profiler_activities,
                    schedule=profiler_schedule,
                    on_trace_ready=trace_handler,
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True
                ) if config.profile_epoch == epoch else contextlib.nullcontext() as prof:
                    for i, batch in enumerate(epoch_loop):
                        global_step += 1 # Inkrementiere bei jedem Batch
                        # Der Provider gibt jetzt 4 Elemente zurück: (input, target, input_mark, target_mark).
                        # Wir brauchen nur die ersten beiden und ignorieren die Zeit-Features.
                        input_data, target, _, _ = batch
                        input_data = input_data.to(device)
                        target = target.to(device)
                        
                        # Modell-Forward-Pass. Gibt jetzt 9 Werte zurück.
                        denorm_distr, base_distr, loss_importance, batch_gate_weights_linear, batch_gate_weights_uni_esn, batch_gate_weights_multi_esn, batch_selection_counts, p_learned, p_final = self.model(input_data)
                        
                        # Zielhorizont für die Loss-Berechnung
                        target_horizon = target[:, -config.horizon:, :] # Shape: [B, H, V]
                        
                        # === 1. Loss-Berechnung für die Optimierung (auf normalisierter Skala) ===
                        # Der Loss für die Backpropagation wird auf der normalisierten Verteilung berechnet,
                        # um Skalenunabhängigkeit zu gewährleisten.
                        norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
                        
                        # Wende optionales Clipping auf die normalisierten Ziele an.
                        loss_clip_value = getattr(config, 'loss_target_clip', None)
                        if loss_clip_value is not None:
                            norm_target_for_loss = torch.clamp(norm_target, -loss_clip_value, loss_clip_value)
                        else:
                            norm_target_for_loss = norm_target

                        # === NEU: Wähle die Loss-Funktion basierend auf der Konfiguration ===
                        if config.loss_function == 'gfl':
                            # --- OPTIMIERUNG: Cache für statische Tensoren ---
                            # Dieser Code wird nur einmal zu Beginn des Trainings ausgeführt.
                            if self._cached_gfl_bins_lower is None:
                                tqdm.write("[PERFORMANCE HINT] GFL-Cache: Erstelle statische Tensoren zum ersten Mal.")
                                # Greife direkt auf die skalaren Attribute des einzelnen Verteilungsobjekts zu.
                                bins_lower_bound_scalar = base_distr.bins_lower_bound
                                bin_width_scalar = base_distr._bin_width
                                
                                # Wandle sie in Tensoren um und forme sie für Broadcasting.
                                # Die Form (1, 1, 1) ist kompatibel mit dem Ziel-Tensor [B, N_Vars, H].
                                self._cached_gfl_bins_lower = torch.tensor(bins_lower_bound_scalar, device=device).view(1, 1, 1)
                                self._cached_gfl_bin_widths = torch.tensor(bin_width_scalar, device=device).view(1, 1, 1)

                            # Hole gecachte Tensoren, Anzahl der Bins und Logits direkt vom Objekt.
                            all_bins_lower = self._cached_gfl_bins_lower
                            all_bin_widths = self._cached_gfl_bin_widths
                            num_bins = base_distr.num_bins
                            all_logits = base_distr.logits # Shape: [B, N_Vars, H, N_Bins]

                            # --- Vollständig vektorisierte GFL-Implementierung ---
                            # Diese Logik bleibt im Kern gleich, arbeitet aber jetzt mit den direkt
                            # abgerufenen, korrekt geformten Tensoren.
                            target_cont_idx = (norm_target_for_loss - all_bins_lower) / all_bin_widths
                            target_cont_idx = torch.clamp(target_cont_idx, 0, num_bins - 1)
                            idx_lower = torch.floor(target_cont_idx).long()
                            idx_upper = torch.ceil(target_cont_idx).long()
                            idx_lower = torch.clamp(idx_lower, 0, num_bins - 1)
                            idx_upper = torch.clamp(idx_upper, 0, num_bins - 1)
                            weight_upper = target_cont_idx - idx_lower.float()
                            weight_lower = 1.0 - weight_upper
                            idx_lower_g = idx_lower.unsqueeze(-1)
                            idx_upper_g = idx_upper.unsqueeze(-1)
                            log_probs = F.log_softmax(all_logits, dim=-1)
                            log_prob_lower = torch.gather(log_probs, -1, idx_lower_g).squeeze(-1)
                            log_prob_upper = torch.gather(log_probs, -1, idx_upper_g).squeeze(-1)
                            loss_dfl = - (weight_lower * log_prob_lower + weight_upper * log_prob_upper)
                            probs = torch.softmax(all_logits, dim=-1)
                            prob_lower = torch.gather(probs, -1, idx_lower_g).squeeze(-1)
                            prob_upper = torch.gather(probs, -1, idx_upper_g).squeeze(-1)
                            pt = torch.where(idx_lower == idx_upper, prob_lower, prob_lower + prob_upper)
                            focal_weight = (1 - pt).pow(config.gfl_gamma)
                            gfl_loss_tensor = focal_weight * loss_dfl
                            normalized_loss = gfl_loss_tensor.mean()

                        else: # Fallback auf CRPS
                            normalized_loss = crps_loss(base_distr, norm_target_for_loss).mean()

                        # === NEU: Jerk-Penalty für die Xi-Parameter ===
                        jerk_loss = torch.tensor(0.0, device=device)
                        if config.jerk_loss_coef > 0:
                            # Greife direkt auf die Tensoren zu. Shape: [Batch, Anzahl_Variablen, Horizont]
                            lower_xi = base_distr.lower_gp_xi
                            upper_xi = base_distr.upper_gp_xi

                            # Permutiere die Dimensionen, um die Form [Batch, Horizont, Anzahl_Variablen] zu erhalten.
                            lower_xi_permuted = lower_xi.permute(0, 2, 1)
                            upper_xi_permuted = upper_xi.permute(0, 2, 1)

                            # Konkateniere entlang der letzten Dimension, um alle Parameter zu kombinieren.
                            # Ergebnis-Shape: [Batch, Horizont, Anzahl_Variablen * 2]
                            all_xi = torch.cat([lower_xi_permuted, upper_xi_permuted], dim=-1)
                            
                            # Wende die Finite-Differenzen-Formel für den "Jerk" entlang der Horizont-Dimension (dim=1) an.
                            # Dies erfordert eine minimale Horizontlänge von 4.
                            if all_xi.shape[1] >= 4:
                                jerk = all_xi[:, 3:, :] - 3 * all_xi[:, 2:-1, :] + 3 * all_xi[:, 1:-2, :] - all_xi[:, :-3, :]
                                jerk_loss = torch.mean(jerk**2)

                        total_loss = normalized_loss + config.loss_coef * loss_importance + config.jerk_loss_coef * jerk_loss

                        # --- NEU: Skaliere den Loss und führe Backward-Pass aus ---
                        scaled_loss = total_loss / accumulation_steps
                        scaled_loss.backward()

                        # --- NEU: Führe den Optimizer-Schritt nur alle `accumulation_steps` aus ---
                        if (i + 1) % accumulation_steps == 0:
                            if config.use_agc:
                                adaptive_clip_grad_(self.model.parameters(), clip_factor=config.agc_lambda)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        # === 2. Metriken für das Logging (ohne Gradienten) ===
                        with torch.no_grad():
                            denorm_crps_per_point = crps_loss(denorm_distr, target_horizon.permute(0, 2, 1))
                            denorm_crps_loss = denorm_crps_per_point.mean()
                            normalized_crps_per_point = crps_loss(base_distr, norm_target)

                        # Losses für die Epochen-Statistik sammeln
                        # === FIX: Konvertiere sofort zu .item(), um den Tensor-Speicher freizugeben ===
                        epoch_total_losses.append(total_loss.item())
                        epoch_crps_losses.append(denorm_crps_loss.item())
                        epoch_normalized_losses.append(normalized_loss.item())
                        epoch_importance_losses.append(loss_importance.item())

                        # === FIX: Addiere zur laufenden Summe hinzu, anstatt Tensoren zu speichern ===
                        if sum_gate_weights_linear is not None and batch_gate_weights_linear.numel() > 0:
                            sum_gate_weights_linear += batch_gate_weights_linear.detach().mean(dim=0)
                        if sum_gate_weights_uni_esn is not None and batch_gate_weights_uni_esn.numel() > 0:
                            sum_gate_weights_uni_esn += batch_gate_weights_uni_esn.detach().mean(dim=0)
                        if sum_gate_weights_multi_esn is not None and batch_gate_weights_multi_esn.numel() > 0:
                            sum_gate_weights_multi_esn += batch_gate_weights_multi_esn.detach().mean(dim=0)
                        if sum_selection_counts is not None and batch_selection_counts.numel() > 0:
                            sum_selection_counts += batch_selection_counts.detach().mean(dim=0)
                        
                        expert_metrics_batch_count += 1
                        # === FIX: Addiere zur laufenden Summe hinzu, anstatt die Matrix zu speichern ===
                        sum_p_learned += p_learned.detach().sum(dim=0) # Summiere über die Batch-Dimension
                        sum_p_final += p_final.detach().sum(dim=0)
                        num_batches_processed += input_data.size(0) # Addiere die Anzahl der Samples im Batch
                        
                        # Kanal-spezifische Losses
                        denorm_loss_per_channel = denorm_crps_per_point.mean(dim=(0, 2))
                        norm_loss_per_channel = normalized_crps_per_point.mean(dim=(0, 2))
                        channel_names = self.model.module.channel_names if hasattr(self.model, 'module') else self.model.channel_names
                        for idx, name in enumerate(channel_names):
                            epoch_channel_losses[name].append(denorm_loss_per_channel[idx].item())
                            epoch_normalized_channel_losses[name].append(norm_loss_per_channel[idx].item())
                        
                        # --- KORREKTUR: Dem Profiler mitteilen, dass der Step beendet ist ---
                        if prof:
                            prof.step()

                        # --- VERBESSERUNG: tqdm-Fortschrittsbalken in jedem Batch aktualisieren ---
                        # Wir berechnen und zeigen den laufenden Durchschnitt des Losses für die aktuelle Epoche an.
                        epoch_total_loss_sum += total_loss.item()
                        epoch_norm_loss_sum += normalized_loss.item()

                        # NEU: Aktualisiere die Fortschrittsanzeige nur alle `tqdm_update_freq` Batches
                        # oder beim letzten Batch der Epoche, um den finalen Stand zu sehen.
                        if (i + 1) % config.tqdm_update_freq == 0 or (i + 1) == len(train_data_loader):
                            avg_epoch_loss = epoch_total_loss_sum / (i + 1)
                            avg_epoch_norm_loss = epoch_norm_loss_sum / (i + 1)
                            epoch_loop.set_postfix(loss=f"{avg_epoch_loss:.4f}", norm_loss=f"{avg_epoch_norm_loss:.4f}")

                        # === KORREKTUR: Verallgemeinerte Speicherüberwachung für CUDA und MPS ===
                        current_time = time.time()
                        if trial and (current_time - last_memory_check_time) > current_memory_check_interval:
                            last_memory_check_time = current_time
                            max_memory_gb = getattr(config, 'max_memory_gb', None)
                            if max_memory_gb is not None:
                                allocated_gb = 0
                                device_type = device.type # Hole den Typ des aktiven Geräts

                                if device_type == 'cuda':
                                    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                                    limit_type = "CUDA"
                                elif device_type == 'mps':
                                    allocated_gb = torch.mps.driver_allocated_memory() / (1024**3)
                                    limit_type = "MPS"
                                
                                if allocated_gb > 0: # Nur loggen, wenn eine Überwachung stattfindet
                                    tqdm.write(
                                        f"[Memory Check] Epoch {epoch+1}, Batch {i+1}: "
                                        f"Allocated ({limit_type}): {allocated_gb:.2f} GB / Limit: {max_memory_gb:.2f} GB"
                                    )
                                    if allocated_gb > max_memory_gb:
                                        tqdm.write(f"  -> 🚨 PRUNING: Memory limit exceeded. Trial will be stopped.")
                                        raise optuna.exceptions.TrialPruned(f"Exceeded {limit_type} memory limit of {max_memory_gb} GB.")
                            
                            # Nach der ersten Prüfung auf das reguläre Intervall umschalten
                            current_memory_check_interval = regular_memory_check_interval_seconds


                # --- Logging am Ende der Epoche ---
                # Die tqdm-Schleife wird am Ende der Epoche automatisch geschlossen und aufgeräumt.
                # === FIX: Berechne den Mittelwert aus der Liste von Python-Floats ===
                avg_train_total_loss = np.mean(epoch_total_losses)
                avg_train_crps_loss = np.mean(epoch_crps_losses)
                avg_train_norm_loss = np.mean(epoch_normalized_losses)
                avg_train_importance_loss = np.mean(epoch_importance_losses)

                writer.add_scalar("Loss/Train_Total", avg_train_total_loss, epoch)
                writer.add_scalar("Loss_Normalized/Train_Loss", avg_train_norm_loss, epoch)
                writer.add_scalar("Loss/Train_CRPS", avg_train_crps_loss, epoch)
                writer.add_scalar("Loss/Train_Importance", avg_train_importance_loss, epoch)
                if config.jerk_loss_coef > 0:
                    writer.add_scalar("Loss/Train_Jerk", jerk_loss.item(), epoch)
                
                for name, losses in epoch_channel_losses.items():
                    writer.add_scalar(f"Loss_per_Channel/Train_{name}", np.mean(losses), epoch)
                for name, losses in epoch_normalized_channel_losses.items():
                    writer.add_scalar(f"Loss_Normalized_per_Channel/Train_{name}", np.mean(losses), epoch)

                # --- Mittelung der Experten-Metriken über die Epoche ---
                # === FIX: Berechne den Durchschnitt aus den laufenden Summen ===
                if expert_metrics_batch_count > 0:
                    avg_gate_weights_linear = sum_gate_weights_linear / expert_metrics_batch_count if sum_gate_weights_linear is not None else None
                    avg_gate_weights_uni_esn = sum_gate_weights_uni_esn / expert_metrics_batch_count if sum_gate_weights_uni_esn is not None else None
                    avg_gate_weights_multi_esn = sum_gate_weights_multi_esn / expert_metrics_batch_count if sum_gate_weights_multi_esn is not None else None
                    avg_selection_counts = sum_selection_counts / expert_metrics_batch_count if sum_selection_counts is not None else torch.tensor([])
                
                # --- NEUE, ROBUSTERE LOGGING-LOGIK FÜR EXPERTEN ---
                model_to_log = self.model.module if hasattr(self.model, 'module') else self.model
                
                # Log Gating-Gewichte
                if avg_gate_weights_linear is not None:
                    for i, weight in enumerate(avg_gate_weights_linear):
                        writer.add_scalar(f"Expert_Gating_Weights/Linear_{i}", weight.item(), epoch)
                if avg_gate_weights_uni_esn is not None:
                    for i, weight in enumerate(avg_gate_weights_uni_esn):
                        writer.add_scalar(f"Expert_Gating_Weights/ESN_univariate_{i}", weight.item(), epoch)
                if avg_gate_weights_multi_esn is not None:
                    for i, weight in enumerate(avg_gate_weights_multi_esn):
                        writer.add_scalar(f"Expert_Gating_Weights/ESN_multivariate_{i}", weight.item(), epoch)

                # Log Selection Counts
                if avg_selection_counts.numel() > 0:
                    expert_idx = 0
                    # Log linear experts
                    for i in range(config.num_linear_experts):
                        writer.add_scalar(f"Expert_Selection_Counts/Linear_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1
                    # Log univariate ESN experts
                    for i in range(config.num_univariate_esn_experts):
                        writer.add_scalar(f"Expert_Selection_Counts/ESN_univariate_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1
                    # Log multivariate ESN experts
                    for i in range(config.num_multivariate_esn_experts):
                        writer.add_scalar(f"Expert_Selection_Counts/ESN_multivariate_{i}", avg_selection_counts[expert_idx].item(), epoch)
                        expert_idx += 1

                # --- Mittelung und Logging der Channel-Masken ---
                # === FIX: Berechne den Durchschnitt aus der laufenden Summe ===
                if num_batches_processed > 0:
                    avg_p_learned = sum_p_learned / num_batches_processed
                    avg_p_final = sum_p_final / num_batches_processed

                # --- Validierung ---
                if valid_data_loader is not None:
                    # 1. Validiere einmal, um die Verluste pro Kanal und pro Fenster zu erhalten.
                    # all_window_losses_denorm hat jetzt die Form [num_windows, num_channels]
                    all_window_losses_denorm, all_window_losses_norm = self.validate(valid_data_loader, writer, epoch, device, desc=f"Epoch {epoch+1} Validation")
                    
                    # --- NEU: Logge die denormalisierten Verluste pro Kanal ---
                    channel_names = list(self.config.channel_bounds.keys())
                    for i, name in enumerate(channel_names):
                        avg_channel_loss = np.mean(all_window_losses_denorm[:, i])
                        writer.add_scalar(f"Loss_per_Channel/Validation_{name}", avg_channel_loss, epoch)

                    # --- NEU: Logik zur Auswahl der Optimierungsmetrik ---
                    target_channel = getattr(self.config, 'optimization_target_channel', None)
                    losses_for_optimization = None
                    
                    if target_channel and target_channel in channel_names:
                        try:
                            target_idx = channel_names.index(target_channel)
                            losses_for_optimization = all_window_losses_denorm[:, target_idx]
                        except ValueError:
                            tqdm.write(f"WARNUNG: Zielkanal '{target_channel}' nicht gefunden. Nutze Durchschnitt.")
                            target_channel = None # Reset to trigger fallback
                    
                    if losses_for_optimization is None: # Fallback oder Standardverhalten
                        losses_for_optimization = all_window_losses_denorm.mean(axis=1)

                    # --- Berechne die Metriken basierend auf den ausgewählten Verlusten ---
                    avg_metric_for_opt = np.mean(losses_for_optimization)
                    cvar_metric_for_opt = calculate_cvar(losses_for_optimization, self.config.cvar_alpha)

                    # --- Logge die globalen Metriken weiterhin zur Beobachtung ---
                    overall_avg_crps = np.mean(all_window_losses_denorm)
                    writer.add_scalar("Loss/Validation_Overall_Avg_CRPS", overall_avg_crps, epoch)
                    writer.add_scalar("Loss_Normalized/Validation_Avg_CRPS", np.mean(all_window_losses_norm), epoch)

                    # 2. Wähle die Metrik für die Optimierung basierend auf der Konfiguration.
                    if self.config.optimization_metric == 'cvar':
                        metric_for_optimization = cvar_metric_for_opt
                        metric_name_for_logging = f"CVaR@{self.config.cvar_alpha}"
                    else:  # Standard ist der Durchschnitts-CRPS
                        metric_for_optimization = avg_metric_for_opt
                        metric_name_for_logging = "Avg CRPS"

                    # Logge die tatsächlich verwendete Optimierungsmetrik
                    writer.add_scalar(f"Loss_Optimized/Validation_Metric", metric_for_optimization, epoch)

                    # 3. Gib die Metriken in der Konsole aus und hebe die aktive hervor.
                    log_msg = f"Epoch {epoch + 1} Validation | Overall Avg CRPS: {overall_avg_crps:.6f} | "
                    if target_channel:
                        log_msg += f"Target ('{target_channel}') {metric_name_for_logging}: {metric_for_optimization:.6f} | "
                    else:
                        log_msg += f"Global {metric_name_for_logging}: {metric_for_optimization:.6f} | "
                    log_msg += "--> Using this for Early Stopping."
                    tqdm.write(log_msg)

                    # Speichere den alten besten Loss, um eine Verbesserung zu erkennen
                    old_best_loss = self.early_stopping.val_loss_min
                    # 4. Verwende die ausgewählte Metrik für Early Stopping.
                    self.early_stopping(metric_for_optimization, self.model.state_dict())

                    # 5. Verwende die ausgewählte Metrik für das Optuna Pruning.
                    if trial:
                        trial.set_user_attr("optimization_target_channel", target_channel)
                        # NEU: Melde die Metrik mit der aktuellen Epochennummer als Schritt.
                        trial.report(metric_for_optimization, epoch)
                        if trial.should_prune():
                            tqdm.write(f"  -> Trial pruned by {trial.study.pruner.__class__.__name__} after epoch {epoch + 1}.")
                            raise optuna.exceptions.TrialPruned()
                    
                    # === NEU: Führe speicherintensive Plots nur aus, wenn explizit aktiviert ===
                    # Während der Optuna-Suche ist dies deaktiviert, um den Speicherverbrauch niedrig zu halten.
                    if getattr(config, 'enable_diagnostic_plots', False):
                        if self.early_stopping.val_loss_min < old_best_loss:
                            if valid_dataset and self.interesting_window_indices:
                                self._log_interesting_window_plots(epoch, writer, valid_dataset)
                            
                            # Logge die Channel-Abhängigkeiten nur, wenn sich das Modell verbessert hat.
                            if 'avg_p_learned' in locals() and 'avg_p_final' in locals():
                                # Hole den Prior vom Modell
                                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                                prior_matrix = model_ref.channel_adjacency_prior

                                # Erstelle den kombinierten 3-Panel-Plot
                                fig_dependencies = self._log_dependency_heatmaps(
                                    prior_matrix=prior_matrix,
                                    learned_matrix=avg_p_learned.cpu().numpy(),
                                    final_matrix=avg_p_final.cpu().numpy()
                                )
                                writer.add_figure("Channel_Dependencies/Combined_View", fig_dependencies, global_step=epoch)
                                plt.close(fig_dependencies)

                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break
                    
                    if scheduler: scheduler.step(metric_for_optimization)

                if config.lradj != "plateau":
                    # Silencing the learning rate update to reduce terminal clutter
                    adjust_learning_rate(optimizer, epoch + 1, config, verbose=False)
                
                # === END-OF-EPOCH OPTUNA REPORTING (REMOVED) ===
                # The trial.report() call at the end of the epoch is now redundant.
                # The new, time-based intermediate validation check is the single
                # source of truth for the pruner, which prevents warnings and is more robust.
                
                # Füge einen harten Timeout hinzu, um sicherzustellen, dass kein Trial die max_training_time überschreitet.
                if (time.time() - start_time) > max_training_time:
                        print(f"Trial timed out after {(time.time() - start_time):.2f}s (max: {max_training_time}s).")
                        break # Beendet die Epochen-Schleife
                
                writer.add_scalar("Misc/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        finally:
            # Lade den besten Zustand vom EarlyStopping und speichere ihn
            if self.early_stopping and self.early_stopping.check_point:
                self.checkpoint_path = os.path.join(config.log_dir, 'best_model.pt')
                os.makedirs(config.log_dir, exist_ok=True)
                
                checkpoint_to_save = {
                    'model_state_dict': self.early_stopping.check_point,
                    'config_dict': self.config.__dict__
                }
                torch.save(checkpoint_to_save, self.checkpoint_path)
                print(f"\n--- Finalizing run ---\n>>> Best model saved to {self.checkpoint_path} <<<")
            writer.close()
        
        # Lade das beste Modell in den Speicher, um es für die Vorhersage zu verwenden
        if self.checkpoint_path:
            self.load(self.checkpoint_path)
        return self

    def validate(self, valid_data_loader, writer: Optional[SummaryWriter], epoch: Optional[int], device: torch.device, desc: str = "Validation") -> tuple[np.ndarray, np.ndarray]:
        all_denorm_losses, all_norm_losses = [], []
        self.model.eval()
        with torch.no_grad():
            validation_loop = tqdm(
                valid_data_loader,
                desc=desc,
                leave=False,
                file=sys.stdout,
                mininterval=self.config.tqdm_min_interval # NEU: Kontrolliert die Update-Frequenz des Balkens
            )
            for batch in validation_loop:
                # Der Provider gibt jetzt 4 Elemente zurück. Wir ignorieren die Zeit-Features.
                input_data, target, _, _ = batch
                input_data = input_data.to(device)
                target = target.to(device)
                
                target_horizon = target[:, -self.config.horizon:, :]

                # Unpack to match new signature (9 values)
                denorm_distr, base_distr, _, _, _, _, _, _, _ = self.model(input_data)
                
                # Berechne den denormalisierten Loss für EarlyStopping/Optuna
                # crps_loss erwartet target in [B, N_vars, H]
                # Wir berechnen den mittleren CRPS pro Kanal pro Sample im Batch -> [B, N_vars]
                denorm_loss_per_sample = crps_loss(denorm_distr, target_horizon.permute(0, 2, 1)).mean(dim=2)
                all_denorm_losses.append(denorm_loss_per_sample.cpu().numpy())

                # Berechne den normalisierten Loss für das Logging
                norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
                norm_loss_per_sample = crps_loss(base_distr, norm_target).mean(dim=2)
                all_norm_losses.append(norm_loss_per_sample.cpu().numpy())

                # Update tqdm bar with the running mean of all collected losses so far
                if all_denorm_losses:
                    # Berechne den Gesamt-Durchschnitt über alle Fenster und Kanäle
                    running_mean_denorm = np.mean(np.concatenate(all_denorm_losses)) 
                    running_mean_norm = np.mean(np.concatenate(all_norm_losses))
                    validation_loop.set_postfix(
                        denorm_crps=running_mean_denorm,
                        norm_crps=running_mean_norm
                    )

        self.model.train()
        # Wir geben die vollständigen Arrays aller Fenster-Losses zurück
        return np.concatenate(all_denorm_losses), np.concatenate(all_norm_losses)

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call forecast_fit() first.")
            
        input_data = train.iloc[-self.seq_len:, :].values
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        self.model.eval()
        with torch.no_grad():
            # Unpack 9 values to match the new model signature. We only need the first one.
            distr, _, _, _, _, _, _, _, _ = self.model(input_tensor)
            
            # Hole Quantil-Vorhersagen
            q_list = self.config.quantiles
            q_tensor = torch.tensor(q_list, device=device, dtype=torch.float32)
            
            # distr.icdf gibt bei mehreren Quantilen [B, H, V, Q] zurück
            quantile_preds = distr.icdf(q_tensor)
        
        # Permutiere für das erwartete Output-Format [V, H, Q]
        # Wir nehmen das erste (und einzige) Element aus dem Batch
        output_array = quantile_preds.squeeze(0).permute(1, 0, 2).cpu().numpy()
        return output_array

    def load(self, checkpoint_path: str) -> "ModelBase":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Lade die Konfiguration aus dem Checkpoint, um das Modell neu zu erstellen
        config_dict = checkpoint['config_dict']
        self.config = TransformerConfig(**config_dict)
        
        self.model = DUETProbModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model successfully loaded from {checkpoint_path}.")
        return self

    def _create_window_plot(self, history, actuals, prediction_dist, channel_name, title):
        """Erstellt eine Matplotlib-Figur für ein einzelnes Fenster, inkl. Tail-Parameter-Analyse."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Finde den Index des zu plottenden Kanals
        try:
            # VERBESSERUNG: Hole die Kanalreihenfolge direkt vom Modell, das ist robuster.
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            channel_idx = model_ref.channel_names.index(channel_name)
        except (ValueError, AttributeError, IndexError):
             # Fallback, wenn die Namen nicht übereinstimmen
            # KORREKTUR: Tippfehler 'channel' zu 'channel_idx'
            channel_idx = 0

        # Daten vorbereiten
        horizon_len = actuals.shape[0]
        
        # === KORREKTUR: Zeige nur `horizon` Schritte der Historie an ===
        # Das entspricht dem "Vorher"-Fenster aus der `find_interesting_windows`-Suche.
        history_to_plot = history[-horizon_len:, :]
        
        # Zeitachsen-Indizes für den Plot
        history_x = np.arange(horizon_len)
        forecast_x = np.arange(horizon_len, horizon_len + horizon_len)
        
        # History und Actuals für den spezifischen Kanal
        history_y = history_to_plot[:, channel_idx]
        actuals_y = actuals[:, channel_idx]
        
        # Quantile aus der Konfiguration holen
        quantiles = self.config.quantiles
        device = prediction_dist.mean.device
        q_tensor = torch.tensor(quantiles, device=device, dtype=torch.float32)

        # --- DATEN VOM MODELL HOLEN (haben den langen Horizont, z.B. 96) ---
        # Shape ist [Anzahl_Variablen, H_pred, Anzahl_Quantile]
        quantile_preds_full = prediction_dist.icdf(q_tensor).squeeze(0).cpu().numpy()
        base_dist = prediction_dist.base_dist
        # Shape ist [H_pred]
        lower_tail_xi_full = base_dist.lower_gp_xi.squeeze(0)[channel_idx, :].cpu().numpy()
        upper_tail_xi_full = base_dist.upper_gp_xi.squeeze(0)[channel_idx, :].cpu().numpy()

        # --- ENDGÜLTIGE KORREKTUR: An der korrekten Dimension zuschneiden ---
        # Schneide an Dimension 1 (Horizont) statt an Dimension 0 (Variablen)
        quantile_preds_sliced = quantile_preds_full[:, :horizon_len, :]
        lower_tail_xi = lower_tail_xi_full[:horizon_len]
        upper_tail_xi = upper_tail_xi_full[:horizon_len]

        # --- Weiterverarbeitung für den Plot ---
        # Wähle den relevanten Kanal aus dem gesliceten Tensor aus.
        # Shape von preds_y ist jetzt [H_true, Q]
        preds_y = quantile_preds_sliced[channel_idx, :, :]
        
        # Median-Index finden
        try:
            median_idx = quantiles.index(0.5)
        except (ValueError, AttributeError):
            median_idx = len(quantiles) // 2

        # --- Plotting ---
        # History
        ax.plot(history_x, history_y, label="History", color="gray")
        
        ax.axvline(x=history_x[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start')

        # Actuals
        ax.plot(forecast_x, actuals_y, label="Actual", color="black", linewidth=2, zorder=10)
        
        # Confidence Intervals
        num_ci_levels = len(quantiles) // 2
        # Define a base alpha. Wider CIs (i=0) will be more transparent.
        base_alpha = 0.1
        alpha_step = 0.15

        for i in range(num_ci_levels):
            lower_q_idx = i
            upper_q_idx = len(quantiles) - 1 - i
            
            # The narrowest interval (largest i) will be the most opaque.
            current_alpha = base_alpha + (i * alpha_step)

            ax.fill_between(
                forecast_x, preds_y[:, lower_q_idx], preds_y[:, upper_q_idx],
                alpha=current_alpha, color='C0', # Use a consistent color for all intervals
                label=f"CI {quantiles[lower_q_idx]}-{quantiles[upper_q_idx]}"
            )
            
        # Median Forecast
        ax.plot(forecast_x, preds_y[:, median_idx], label="Median Forecast", color="blue", linestyle='--', zorder=11)
        
        # --- NEU: Zweite Y-Achse für die Tail-Parameter (xi) ---
        ax2 = ax.twinx()
        
        # Plot der xi-Parameter
        ax2.plot(forecast_x, upper_tail_xi, label='Upper Tail ξ (xi)', color='darkorange', linestyle='none', marker='^', markersize=4, zorder=20)
        ax2.plot(forecast_x, lower_tail_xi, label='Lower Tail ξ (xi)', color='purple', linestyle='none', marker='v', markersize=4, zorder=20)
        
        # Bereiche für die Tail-Typen hervorheben
        # Finde einen sinnvollen y-Bereich für die Achse
        min_xi = min(np.min(lower_tail_xi), np.min(upper_tail_xi))
        max_xi = max(np.max(lower_tail_xi), np.max(upper_tail_xi))
        plot_min = min(-0.5, min_xi - 0.1)
        plot_max = max(0.5, max_xi + 0.1)
        ax2.set_ylim(plot_min, plot_max)

        # Fréchet-Domain (gefährlich)
        # Weibull-Domain (harmlos)
        # Gumbel-Schwelle
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)

        ax2.set_ylabel("Tail Shape Parameter (ξ)", color="darkorange")
        ax2.tick_params(axis='y', labelcolor="darkorange")
        ax2.grid(False) # Deaktiviere das Grid für die zweite Achse, um es sauber zu halten

        # Layout
        ax.set_title(title)
        ax.set_ylabel("Time Series Value")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Kombinierte Legende für beide Achsen
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        
        # WICHTIG: Die Figur zurückgeben, damit sie geloggt werden kann
        return fig

    def _plot_single_heatmap(self, ax, matrix, title, channel_names, vmin=0, vmax=1):
        """Helper to draw a single heatmap on a given Matplotlib axis."""
        im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)

        if channel_names and len(channel_names) == matrix.shape[0]:
            ax.set_xticks(np.arange(len(channel_names)))
            ax.set_yticks(np.arange(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha="right")
            ax.set_yticklabels(channel_names)

        # Annotate cells with values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Choose text color based on background brightness for better readability
                text_color = "w" if matrix[i, j] < 0.6 * vmax else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color)

        ax.set_title(title)
        return im # Return the image object for the colorbar

    def _log_dependency_heatmaps(self, prior_matrix, learned_matrix, final_matrix):
        """Creates a 3-panel matplotlib figure showing the dependency matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
        fig.suptitle("Channel Dependency Analysis (Epoch Average)", fontsize=16)

        channel_names = list(getattr(self.config, 'channel_bounds', {}).keys())
        n_vars = len(channel_names)

        # --- Panel 1: User Prior ---
        if prior_matrix is not None:
            prior_np = prior_matrix.cpu().numpy()
        else:
            # If no prior is given, it's equivalent to a matrix of all ones.
            prior_np = np.ones((n_vars, n_vars))

        im1 = self._plot_single_heatmap(axes[0], prior_np, "User Prior", channel_names, vmin=0, vmax=1)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # --- Panel 2: Learned (Unconstrained) ---
        im2 = self._plot_single_heatmap(axes[1], learned_matrix, "Learned (Unconstrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # --- Panel 3: Effective (Constrained) ---
        im3 = self._plot_single_heatmap(axes[2], final_matrix, "Effective (Constrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        return fig