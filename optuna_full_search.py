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
from ts_benchmark.baselines.duet.duet_prob import DUETProb
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split

# --- FIX: "Too many open files" Error ---
# Ändert die Strategie, wie Worker-Prozesse Daten teilen.
# 'file_system' ist robuster für langlaufende Experimente als der Standard 'file_descriptor'.
torch.multiprocessing.set_sharing_strategy('file_system')

# --- 1. Logging-Konfiguration ---
logging.getLogger("optuna").setLevel(logging.INFO)
# --- WICHTIG: Ändere den Namen, um die Ergebnisse dieses Experiments zu isolieren ---
STUDY_NAME = "isar1" 
STORAGE_NAME = "sqlite:///optuna_study.db"
USE_WARM_START = True # NEU: Schalter, um das Einreihen von Start-Trials zu steuern

# --- 2. Feste Trainingsparameter für die lange, intensive Suche ---
FIXED_PARAMS = {
    "data_file": "isar_2005_2020_train.csv", 
    "horizon": 96,
    "train_ratio_in_tv": 0.8, # NEU: Split-Verhältnis explizit gemacht
    "num_epochs": 1000,
    "patience": 5,
    "num_workers": 0, # HIER FESTLEGEN
    "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], # <-- HIER ERWEITERN
    "lradj": "constant", # Angepasst für bessere Performance
    "loss_function": "gfl",   # Festgelegte Loss-Funktion. Alternative: "crps"
    # NEU: Intervall in Sekunden für die Zwischen-Validierung.
    "interim_validation_seconds": None, # 300s = 5min. Auf `None` setzen, um zu deaktivieren.
    "min_training_time": 600, 
    "max_training_time": 5400,
    # PyTorch Profiler ist deaktiviert. Setze eine Epochennummer (z.B. 1), um ihn zu aktivieren.
    # NEU: Hartes Speicherlimit in Gigabyte für MPS-Geräte. Auf `None` setzen, um zu deaktivieren.
    "max_memory_gb": 14.0,
    "profile_epoch": None,
    # NEU: Schalter zum Deaktivieren der speicherintensiven Plots während der Optuna-Suche.
    "enable_diagnostic_plots": False,
    "channel_adjacency_prior": [
        [1, 1, 0, 0, 0, 0],  
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1], 
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
    params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    params["loss_coef"] = trial.suggest_float("loss_coef", 0.1, 2.0, log=True)
    
    # --- NEU: Tier 2 Trainingsstrategien ---
    # Loss-Funktion ist in FIXED_PARAMS auf 'gfl' gesetzt.
    # Daher wird der gfl_gamma-Parameter immer vorgeschlagen.
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
    """Führt einen Trainingslauf durch und gibt den Validierungs-Loss zurück."""
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

        # 4. Gib den besten Validierungs-Loss zurück
        best_valid_loss = model.early_stopping.val_loss_min
        if not np.isfinite(best_valid_loss) or best_valid_loss is None:
             print(f"TRIAL #{trial_num} resulted in an invalid loss ({best_valid_loss}). Pruning.")
             raise optuna.exceptions.TrialPruned("Training did not produce a valid finite loss.")

        print(f"TRIAL #{trial_num} COMPLETED. Best validation loss: {best_valid_loss:.6f}, Total Params: {total_params:,}")
        return best_valid_loss

    except optuna.exceptions.TrialPruned:
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

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        load_if_exists=True,
        pruner=HyperbandPruner(
            min_resource=FIXED_PARAMS["min_training_time"],
            max_resource=FIXED_PARAMS["max_training_time"],
            reduction_factor=3,
        )
    )

    # --- NEU: Saubere Logik für den Warm-Start aus einer JSON-Datei ---
    if USE_WARM_START and len(study.get_trials(deepcopy=False)) == 0:
        warm_start_file = "optuna_warm_starts.json"
        print(f"Attempting to warm-start the study from '{warm_start_file}'...")
        try:
            with open(warm_start_file, 'r') as f:
                warm_start_configs = json.load(f)
            
            print(f"Enqueuing {len(warm_start_configs)} initial trials from config file...")
            for i, params in enumerate(warm_start_configs):
                study.enqueue_trial(params, skip_if_exists=True)
                print(f"  -> Enqueued trial config #{i+1}")
        except FileNotFoundError:
            print(f"  -> WARNING: Warm-start file '{warm_start_file}' not found. Starting without initial trials.")
        except json.JSONDecodeError as e:
            print(f"  -> WARNING: Could not parse '{warm_start_file}'. Check for syntax errors. Error: {e}")
            print("  -> Starting without initial trials.")

    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}' once before starting the study...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully. Starting optimization...")

    study.optimize(lambda trial: objective(trial, data), n_trials=100)

    print("\n\n" + "="*50 + "\nHEURISTIC SEARCH FINISHED\n" + "="*50)
    try:
        print(f"Best trial: #{study.best_trial.number}")
        print(f"  Value (min valid CRPS): {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
        print(f"  User Attributes: {study.best_trial.user_attrs}")
    except ValueError:
        print("No successful trials were completed.")
    print("\nTo analyze the results, use the 'analyse_study.py' script.")