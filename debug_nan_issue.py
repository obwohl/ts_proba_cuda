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

from ts_benchmark.baselines.duet.duet_prob import DUETProb, calculate_cvar
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
from optuna_full_search import objective, FIXED_PARAMS, get_suggested_params, setup_logging, analyze_johnson_systems

# --- Function to redirect all output to a file ---
def setup_file_logging():
    """Redirects stdout and stderr to a file."""
    log_file = "debug_output.txt"
    # Overwrite the file
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout
    print(f"--- All output is redirected to {log_file} ---")

# --- Mock Trial Object for Debugging ---
class DebugTrial:
    def __init__(self, params, trial_number=0):
        self.params = params
        self.number = trial_number
        self.user_attrs = {}
        self.study = lambda: None # Mock study attribute
        self.study.pruner = optuna.pruners.NopPruner()


    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, low)

    def suggest_int(self, name, low, high):
        return self.params.get(name, low)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value
        print(f"DEBUG_TRIAL: Set user attr '{key}' to '{value}'")

    def report(self, value, step):
        print(f"DEBUG_TRIAL: Reported value {value} for step {step}")

    def should_prune(self):
        return False

if __name__ == "__main__":
    # --- Set up logging ---
    # This will now write all output to 'debug_output.txt'
    setup_file_logging()

    STUDY_NAME = "debug_nan_issue"
    # The original logging to a separate file per PID is still useful for parallel runs,
    # but for this single debug script, having one consolidated file is better.
    # We still call it to maintain compatibility.
    setup_logging(STUDY_NAME)

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
        "use_channel_adjacency_prior": False
    }

    # --- Create a mock trial ---
    trial = DebugTrial(debug_params)

    # --- Load Data ---
    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}'...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully.")

    # --- Modify FIXED_PARAMS for a short run ---
    FIXED_PARAMS['num_epochs'] = 1
    # This is a new parameter I will add to the model to limit the number of batches
    FIXED_PARAMS['debug_max_batches'] = 1

    print("Running a single short trial for debugging...")

    # --- Call the objective function directly ---
    try:
        # NEU: Führe die Johnson-Analyse einmalig vor dem Aufruf aus
        johnson_map = analyze_johnson_systems(data)
        objective(trial, data, STUDY_NAME, johnson_map)
    except Exception as e:
        logging.exception("An error occurred during the objective function call.")
        raise

    print("\n\n" + "="*50 + "\nDEBUG TRIAL FINISHED\n" + "="*50)

    # --- You can inspect the results or logs generated in the 'logs/debug_short_trial' directory ---
    log_dir = os.path.join("logs", STUDY_NAME)
    print(f"Check logs in: {log_dir}")
    # Find the latest log file in the directory
    try:
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"Latest log file: {latest_log}")
            with open(latest_log, 'r') as f:
                print("\n--- LATEST LOG CONTENT ---")
                print(f.read())
    except FileNotFoundError:
        print(f"Log directory not found: {log_dir}")
    except Exception as e:
        print(f"Error reading log file: {e}")
