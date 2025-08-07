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
from optuna_full_search import objective, FIXED_PARAMS, get_suggested_params


if __name__ == "__main__":
    STUDY_NAME = "single_trial"
    STORAGE_NAME = "sqlite:///optuna_study.db"

    try:
        optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        print(f"Study '{STUDY_NAME}' deleted.")
    except:
        pass

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        load_if_exists=False,
    )

    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}' once before starting the study...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully. Starting optimization...")

    print("Running a single trial for debugging...")
    trial = study.ask()
    objective(trial, data, STUDY_NAME)

    print("\n\n" + "="*50 + "\nSINGLE TRIAL FINISHED\n" + "="*50)
    try:
        best_trial = study.best_trial
        optimized_metric_name = FIXED_PARAMS.get("optimization_metric", "cvar").upper()

        print(f"Best trial found (optimized for {optimized_metric_name}):")
        print(f"  - Optuna Trial Number: {best_trial.number}")
        print(f"  - Optimized Value ({optimized_metric_name}): {best_trial.value:.6f}")

        # Gib die anderen Metriken aus den User-Attributen aus
        print("  - All Metrics:")
        avg_nll_val = best_trial.user_attrs.get('avg_nll', 'N/A')
        cvar_nll_val = best_trial.user_attrs.get('cvar_nll', 'N/A')

        avg_nll_str = f"{avg_nll_val:.6f}" if isinstance(avg_nll_val, float) else avg_nll_val
        cvar_nll_str = f"{cvar_nll_val:.6f}" if isinstance(cvar_nll_val, float) else cvar_nll_val
        print(f"    - Avg NLL: {avg_nll_str}")
        print(f"    - CVaR NLL: {cvar_nll_str}")

        print(f"  - Best Hyperparameters: {best_trial.params}")
        print(f"  - Full User Attributes: {best_trial.user_attrs}")
    except ValueError:
        print("No successful trials were completed.")
