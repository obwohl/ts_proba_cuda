# optuna_run_heuristic_search.py
import optuna
import os
import subprocess
import json
import logging
import numpy as np
import sys
import torch
import torch.multiprocessing
import time
import gc
from optuna.pruners import HyperbandPruner
import pandas as pd

# Direkter Import der notwendigen Klassen
# Importiere die Modellklasse UND die CVaR-Hilfsfunktion aus der Modelldatei
from ts_benchmark.baselines.duet.duet_prob import DUETProb, calculate_cvar
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split

# --- FIX: "Too many open files" Error ---
# Ändert die Strategie, wie Worker-Prozesse Daten teilen.
# 'file_system' ist robuster für langlaufende Experimente als der Standard 'file_descriptor'.
torch.multiprocessing.set_sharing_strategy('file_system')

# --- 1. Logging-Konfiguration ---
logging.getLogger("optuna").setLevel(logging.INFO)
# --- WICHTIG: Ändere den Namen, um die Ergebnisse dieses Experiments zu isolieren ---
STUDY_NAME = "eisbach_airtemp_pressure_focus_auf_wassertemp" 
STORAGE_NAME = "sqlite:///optuna_study.db"

# --- NEU: Konfiguration für den automatisierten Warm-Start ---
# Setze auf den Namen der alten Studie, von der die besten Trials übernommen werden sollen.
# Auf `None` setzen, um diese Funktion zu deaktivieren.
WARM_START_FROM_OLD_STUDY = None 
# Wie viele der besten Trials aus der alten Studie sollen übernommen werden?
NUM_WARM_START_TRIALS_FROM_STUDY = 5
# Schalter für den alten JSON-basierten Warm-Start. Wird ignoriert, wenn WARM_START_FROM_OLD_STUDY gesetzt ist.
USE_WARM_START_FROM_JSON = True 

# --- 2. Feste Trainingsparameter für die lange, intensive Suche ---
FIXED_PARAMS = {
    "data_file": "eisbach_airtemp96_pressure96.csv", 
    "horizon": 96,
    "train_ratio_in_tv": 0.8, # NEU: Split-Verhältnis explizit gemacht
    # --- NEU: Wähle die zu optimierende Metrik ---
    # 'cvar': Conditional Value at Risk (Durchschnitt der schlechtesten 5% Fehler) -> robustere Modelle
    # 'avg_crps': Durchschnittlicher CRPS-Fehler über alle Fenster -> beste Durchschnitts-Performance
    "optimization_metric": "cvar",
    # --- NEU: Gezielte Optimierung auf einen einzelnen Kanal ---
    # Setze hier einen Kanalnamen (z.B. "wassertemp"), um den Validierungs-Loss nur für diesen Kanal zu berechnen.
    # Setze auf `None`, um den Durchschnitt über alle Kanäle zu verwenden (Standardverhalten).
    "optimization_target_channel": "wassertemp",
    "num_epochs": 50,
    "patience": 5,    
    # NEU: num_workers wird über Umgebungsvariable gesteuert, um Parallelisierung zu erleichtern.
    # Standardwert ist 8, wenn die Variable nicht gesetzt ist.
    "num_workers": int(os.getenv("TRIAL_WORKERS", "4")),
    "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], # <-- HIER ERWEITERN
    "lradj": "constant", # Angepasst für bessere Performance
    # --- NEU: Konfiguration für epochenbasiertes Pruning ---
    # Die minimale Anzahl an Epochen, bevor ein Trial abgebrochen werden kann.
    "min_epochs_for_pruning": 3,
    # PyTorch Profiler ist deaktiviert. Setze eine Epochennummer (z.B. 1), um ihn zu aktivieren.
    # NEU: Hartes Speicherlimit in Gigabyte für MPS-Geräte. Auf `None` setzen, um zu deaktivieren.
    "cvar_alpha": 0.05, # Wir wollen den mittleren Fehler der 5% schlechtesten Fälle optimieren
    "max_memory_gb": None,
    "profile_epoch": None,
    # NEU: Schalter zum Deaktivieren der speicherintensiven Plots während der Optuna-Suche.
    "enable_diagnostic_plots": False,
    # "channel_adjacency_prior": [
    #     [1, 1, 0, 0, 0, 0],  
    #     [1, 1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1], 
    # ] für isarpegel
    "channel_adjacency_prior": [
        [1, 1, 1],  #wassertemp
        [0, 1, 1], #airtemp96
        [0, 0, 1], #pressure96
    ]
}

def get_suggested_params(trial: optuna.Trial) -> dict:
    """Schlägt einen Satz von Hyperparametern vor."""
    params = {}
    params["seq_len"] = trial.suggest_categorical("seq_len", [48, 96, 192, 384])
    params["norm_mode"] = trial.suggest_categorical("norm_mode", ["subtract_last", "subtract_median"])
    params["lr"] = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    params["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512])
    params["d_ff"] = trial.suggest_categorical("d_ff", [32, 64, 128, 256, 512])
    params["e_layers"] = trial.suggest_int("e_layers", 1, 3)

    # --- NEU: moving_avg als kategorialer Hyperparameter ---
    # Gib hier sinnvolle Werte basierend auf der bekannten Periodizität deiner Daten an.
    params["moving_avg"] = trial.suggest_categorical("moving_avg", [5,49, 97, 193]) # für wasserpegel schwierig zu raten, stündlich, halbtäglich, täglich, oder zweitäglich? soll optuna rausfinden.

    # --- KORREKTUR: Statischer Suchraum für n_heads ---
    # 1. Schlage n_heads immer aus der vollen Liste vor, um den Suchraum statisch zu halten.
    params["n_heads"] = trial.suggest_categorical("n_heads", [1, 2, 4, 8])

    # 2. Prüfe die Gültigkeit der Kombination und prune den Trial, wenn sie ungültig ist.
    if params["d_model"] % params["n_heads"] != 0:
        raise optuna.exceptions.TrialPruned(f"d_model ({params['d_model']}) is not divisible by n_heads ({params['n_heads']}).")
        
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    params["fc_dropout"] = trial.suggest_float("fc_dropout", 0.0, 0.5)
    
    # Optuna schlägt die exakte Batch-Größe vor, die verwendet werden soll.
    params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    params["loss_coef"] = trial.suggest_float("loss_coef", 0.1, 2.0, log=True)
    
    # --- NEU: Loss-Funktion als Hyperparameter ---
    params["loss_function"] = trial.suggest_categorical("loss_function", ["gfl", "crps"])
    if params["loss_function"] == 'gfl':
        params["gfl_gamma"] = trial.suggest_float("gfl_gamma", 0.5, 5.0)

    params["use_agc"] = trial.suggest_categorical("use_agc", [True, False])
    if params["use_agc"]:
        params["agc_lambda"] = trial.suggest_float("agc_lambda", 0.001, 0.1, log=True)

    # --- NEU: Hybride Experten-Konfiguration ---
    params["num_linear_experts"] = trial.suggest_int("num_linear_experts", 0, 8)
    params["num_univariate_esn_experts"] = trial.suggest_int("num_univariate_esn_experts", 0, 8)
    params["num_multivariate_esn_experts"] = trial.suggest_int("num_multivariate_esn_experts", 0, 8)
    
    # Stelle sicher, dass mindestens ein Experte vorhanden ist.
    total_experts = (
        params["num_linear_experts"] + 
        params["num_univariate_esn_experts"] + 
        params["num_multivariate_esn_experts"]
    )
    if total_experts == 0: 
        raise optuna.exceptions.TrialPruned("Total number of experts is zero.")
    
    params["k"] = trial.suggest_int("k", 1, total_experts)

    params["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])

    # Schlage univariaten ESN-Parameter vor, wenn diese Experten verwendet werden.
    if params["num_univariate_esn_experts"] > 0:
        params["reservoir_size_uni"] = trial.suggest_categorical("reservoir_size_uni", [16, 32, 64, 128, 256])
        params["spectral_radius_uni"] = trial.suggest_float("spectral_radius_uni", 0.6, 1.4)
        params["sparsity_uni"] = trial.suggest_float("sparsity_uni", 0.01, 0.5)
        params["leak_rate_uni"] = trial.suggest_float("leak_rate_uni", 0.1, 1.0)
        params["input_scaling_uni"] = trial.suggest_float("input_scaling_uni", 0.1, 2.0, log=True)
        # NEU: Weight Decay (L2-Regularisierung) für die univariaten ESN-Readout-Schichten
        params["esn_uni_weight_decay"] = trial.suggest_float("esn_uni_weight_decay", 1e-6, 1e-1, log=True)

    # Schlage multivariaten ESN-Parameter vor, wenn diese Experten verwendet werden.
    if params["num_multivariate_esn_experts"] > 0:
        params["reservoir_size_multi"] = trial.suggest_categorical("reservoir_size_multi", [16, 32, 64, 128, 256])
        params["spectral_radius_multi"] = trial.suggest_float("spectral_radius_multi", 0.6, 1.4)
        params["sparsity_multi"] = trial.suggest_float("sparsity_multi", 0.01, 0.5)
        params["leak_rate_multi"] = trial.suggest_float("leak_rate_multi", 0.1, 1.0)
        params["input_scaling_multi"] = trial.suggest_float("input_scaling_multi", 0.01, 1.0, log=True)
        # NEU: Weight Decay (L2-Regularisierung) für die multivariaten ESN-Readout-Schichten
        params["esn_multi_weight_decay"] = trial.suggest_float("esn_multi_weight_decay", 1e-6, 1e-1, log=True)

    params["projection_head_layers"] = trial.suggest_int("projection_head_layers", 0, 4)
    if params["projection_head_layers"] > 0:
        params["projection_head_dim_factor"] = trial.suggest_categorical("projection_head_dim_factor", [1, 2, 4, 8])
        params["projection_head_dropout"] = trial.suggest_float("projection_head_dropout", 0.0, 0.5)

    params["loss_target_clip"] = trial.suggest_categorical("loss_target_clip", [None, 5.0, 10.0, 15.0])

    # --- NEU: Jerk-Penalty für glatte Xi-Trajektorien ---
    params["jerk_loss_coef"] = trial.suggest_float("jerk_loss_coef", 1e-5, 1.0, log=True)

    # --- NEU: Channel Adjacency Prior an/ausschalten ---
    params["use_channel_adjacency_prior"] = trial.suggest_categorical("use_channel_adjacency_prior", [True, False])

    return params

def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    """Führt einen Trainingslauf durch und gibt die beiden Zielmetriken (avg_crps, cvar_crps) zurück."""
    trial_num = trial.number
    print(f"\n\n{'='*20} STARTING TRIAL #{trial_num} {'='*20}")
    
    suggested_params = get_suggested_params(trial)
    model_hyper_params = {**FIXED_PARAMS, **suggested_params}

    print("Testing Parameters:")
    for key, value in model_hyper_params.items():
        print(f"  - {key}: {value}")

    save_dir = f"results/optuna_heuristic/{STUDY_NAME}/trial_{trial_num}"
    os.makedirs(save_dir, exist_ok=True)
    model_hyper_params['log_dir'] = save_dir

    model = None # Ensure model is defined in the outer scope for the finally block
    try:
        # 1. Initialisiere das Modell mit den geprüften Parametern
        model = DUETProb(**model_hyper_params)

        # 2. Führe das Training aus.
        model.forecast_fit(data, train_ratio_in_tv=model_hyper_params["train_ratio_in_tv"], trial=trial)

        # 3. Extrahiere Metadaten nach dem Training
        total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        trial.set_user_attr("total_trainable_parameters", total_params)

        # 4. Berechne die finalen Metriken auf dem besten Modell
        # Lade das beste Modell und führe eine finale Validierung durch
        model.model.load_state_dict(model.early_stopping.check_point)
        _, valid_data = train_val_split(data, model_hyper_params["train_ratio_in_tv"], model_hyper_params["seq_len"])
        _, valid_loader = forecasting_data_provider(valid_data, model.config, timeenc=1, batch_size=model.config.batch_size, shuffle=False, drop_last=False)
        device = next(model.model.parameters()).device
        
        # Die `validate`-Funktion gibt jetzt ein Array aller Fenster-Losses zurück
        all_window_losses, _ = model.validate(valid_loader, None, None, device, "Final Validation")

        if not np.all(np.isfinite(all_window_losses)):
             print(f"TRIAL #{trial_num} resulted in invalid losses. Pruning.")
             raise optuna.exceptions.TrialPruned("Training did not produce a valid finite loss.")

        # Berechne die beiden Zielmetriken
        avg_crps = float(np.mean(all_window_losses))
        cvar_crps = calculate_cvar(all_window_losses, model_hyper_params["cvar_alpha"])

        # Speichere beide Metriken als User-Attribute, damit sie immer verfügbar sind
        trial.set_user_attr("avg_crps", avg_crps)
        trial.set_user_attr("cvar_crps", cvar_crps)

        # Logge die finalen Metriken übersichtlich
        cvar_alpha = model_hyper_params["cvar_alpha"]
        print("\n" + "-"*25 + f" TRIAL #{trial_num} FINAL METRICS " + "-"*25)
        print(f"  -> Avg CRPS : {avg_crps:.6f}")
        print(f"  -> CVaR@{cvar_alpha} CRPS: {cvar_crps:.6f}")
        print("-"*(50 + len(str(trial_num)) + 18) + "\n")

        # Gib die ausgewählte Zielmetrik für die Optimierung zurück
        optimized_metric = model_hyper_params.get("optimization_metric", "cvar")
        if optimized_metric == "cvar":
            return cvar_crps
        else: # 'avg_crps' oder Fallback
            return avg_crps

    except optuna.exceptions.TrialPruned:
        # Wichtig: Wenn ein Trial pruned wird (entweder durch den Pruner oder manuell),
        # muss die Exception weitergereicht werden, damit Optuna den Status korrekt als "PRUNED" setzt.
        print(f"TRIAL #{trial_num} was pruned by Optuna during training.")
        raise
    except Exception as e:
        import traceback
        print(f"TRIAL #{trial_num} FAILED with an unexpected exception during the main training run.")
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned(f"Full training run failed: {e}")
    finally:
        # WICHTIG: Explizite Speicherbereinigung, um Ressourcenlecks zu verhindern.
        # Dies ist entscheidend, um den "Too many open files"-Fehler zu beheben, der durch
        # nicht geschlossene DataLoader-Worker-Prozesse verursacht wird.
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()
        print(f"--- TRIAL #{trial_num} CLEANUP COMPLETE ---")

def setup_logging(study_name: str):
    """Konfiguriert das Logging, um für jeden Prozess eine eigene Log-Datei zu erstellen."""
    log_dir = f"logs/{study_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Erstelle einen eindeutigen Log-Dateinamen mit der Prozess-ID (PID)
    pid = os.getpid()
    log_file_path = os.path.join(log_dir, f"trial_worker_{pid}.log")

    # Konfiguriere das Root-Logging
    # Dies fängt Logs von Optuna und anderen Bibliotheken ab
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w' # 'w' für überschreiben, 'a' für anhängen
    )

    # Leite stdout und stderr in die Log-Datei um, um auch print() und Fehler abzufangen
    sys.stdout = open(log_file_path, 'a')
    sys.stderr = open(log_file_path, 'a')
    print(f"Logging for PID {pid} is set up to write to {log_file_path}")
    
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize", # Wir optimieren jetzt eine einzelne Metrik
        load_if_exists=True, # Wichtig, um die Studie fortzusetzen
        pruner=HyperbandPruner(
            min_resource=FIXED_PARAMS["min_epochs_for_pruning"],
            max_resource=FIXED_PARAMS["num_epochs"], # Die maximale Ressource ist die Gesamtzahl der Epochen
            reduction_factor=3,
        )
    )

    # --- NEU: Logging für diesen Worker-Prozess einrichten ---
    setup_logging(STUDY_NAME)

    # Die Warm-Start-Logik wurde in das Skript 'prepare_study.py' verschoben,
    # um Race Conditions beim parallelen Start zu vermeiden.

    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}' once before starting the study...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully. Starting optimization...")

    study.optimize(lambda trial: objective(trial, data), n_trials=100)

    print("\n\n" + "="*50 + "\nHEURISTIC SEARCH FINISHED\n" + "="*50)
    try:
        # Da wir jetzt eine Single-Objective-Optimierung durchführen, gibt es genau einen besten Trial.
        best_trial = study.best_trial
        optimized_metric_name = FIXED_PARAMS.get("optimization_metric", "cvar").upper()

        print(f"Best trial found (optimized for {optimized_metric_name}):")
        print(f"  - Optuna Trial Number: {best_trial.number}")
        print(f"  - Optimized Value ({optimized_metric_name}): {best_trial.value:.6f}")
        
        # Gib die anderen Metriken aus den User-Attributen aus
        print("  - All Metrics:")
        avg_crps_val = best_trial.user_attrs.get('avg_crps', 'N/A')
        cvar_crps_val = best_trial.user_attrs.get('cvar_crps', 'N/A')

        avg_crps_str = f"{avg_crps_val:.6f}" if isinstance(avg_crps_val, float) else avg_crps_val
        cvar_crps_str = f"{cvar_crps_val:.6f}" if isinstance(cvar_crps_val, float) else cvar_crps_val
        print(f"    - Avg CRPS: {avg_crps_str}")
        print(f"    - CVaR CRPS: {cvar_crps_str}")
        
        print(f"  - Best Hyperparameters: {best_trial.params}")
        print(f"  - Full User Attributes: {best_trial.user_attrs}")
    except ValueError:
        print("No successful trials were completed.")
    print("\nTo analyze the results, use the 'analyse_study.py' script.")
    print("To visualize the Pareto front, use: optuna.visualization.plot_pareto_front(study)")