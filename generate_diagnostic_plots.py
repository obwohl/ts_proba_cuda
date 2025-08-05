import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# === IMPORTS AUS DEM PROJEKT ===
# Annahme: Das Skript wird vom Projekt-Root-Verzeichnis ausgeführt.
from ts_benchmark.baselines.duet.duet_prob import DUETProb, TransformerConfig
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
from ts_benchmark.baselines.duet.utils.window_search import find_interesting_windows
from ts_benchmark.data.data_source import LocalForecastingDataSource

# ==============================================================================
#                 POST-HOC SKRIPT ZUR ERZEUGUNG DIAGNOSTISCHER PLOTS
# ==============================================================================
#
# Workflow:
# 1. Führen Sie Ihre Optuna-Studie mit deaktivierten Plots durch.
# 2. Identifizieren Sie die besten Trial-Nummern (z.B. über das Optuna-Dashboard).
# 3. Passen Sie die Konfiguration unten an (STUDY_NAME, TRIAL_NUMBERS_TO_PLOT).
# 4. Starten Sie das Skript: python generate_diagnostic_plots.py
#
# Das Skript generiert dann nur für die ausgewählten Top-Trials die
# speicher- und rechenintensiven "Hard Window"-Plots.
#
# ==============================================================================

# --- 1. ZENTRALE KONFIGURATION ---
STUDY_NAME = "preci_short_egpd"
TRIAL_NUMBERS_TO_PLOT = [10]  # Tragen Sie hier die gewünschten Trial-Nummern ein

# --- Pfade und Datenkonfiguration (muss mit optuna_full_search.py übereinstimmen) ---
BASE_RESULTS_DIR = Path(f"results/optuna_heuristic/{STUDY_NAME}")
DATA_FILE_PATH = "preci_test.csv" # KORREKTUR: Muss mit der in optuna_full_search.py verwendeten Datei übereinstimmen.
TRAIN_RATIO_IN_TV = 0.9 # Das Split-Verhältnis aus dem Training


def load_model_and_config(trial_path: Path) -> tuple[DUETProbModel | None, TransformerConfig | None]:
    """Lädt das Modell und die Konfiguration aus einem Trial-Verzeichnis."""
    checkpoint_path = trial_path / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"  -> WARNUNG: Kein Checkpoint 'best_model.pt' in {trial_path} gefunden. Überspringe.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = TransformerConfig(**checkpoint['config_dict'])
        model = DUETProbModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"  -> Modell aus {checkpoint_path.name} erfolgreich geladen.")
        return model, config
    except Exception as e:
        print(f"  -> FEHLER beim Laden des Modells aus {checkpoint_path}: {e}")
        return None, None

def get_validation_data(config: TransformerConfig):
    """Lädt die Daten und wendet exakt den gleichen Split wie im Training an."""
    print("  -> Lade und splitte Daten, um den Validierungsdatensatz zu erhalten...")
    data_source = LocalForecastingDataSource()
    full_data = data_source._load_series(DATA_FILE_PATH)
    _, valid_data_df = train_val_split(full_data, TRAIN_RATIO_IN_TV, config.seq_len)
    valid_dataset, _ = forecasting_data_provider(valid_data_df, config, timeenc=1, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return valid_data_df, valid_dataset

def find_or_load_interesting_windows(study_path: Path, valid_data_df: pd.DataFrame, config: TransformerConfig) -> dict | None:
    """
    Sucht nach interessanten Fenstern oder lädt sie aus dem Cache.
    Der Cache ist spezifisch für die Sequenzlänge.
    """
    cache_filename = study_path / f"interesting_windows_cache_seq{config.seq_len}.json"
    
    if cache_filename.exists():
        print(f"  -> Lade 'interessante Fenster' aus dem Cache: {cache_filename.name}")
        with open(cache_filename, 'r') as f:
            return json.load(f)
    else:
        print("  -> Keine Cache-Datei gefunden. Starte einmalige Suche nach 'interessanten Fenstern'...")
        try:
            indices = find_interesting_windows(valid_data_df, config.horizon, config.seq_len)
            with open(cache_filename, 'w') as f:
                json.dump(indices, f, indent=4)
            print(f"  -> Suche abgeschlossen und Ergebnisse in {cache_filename.name} gespeichert.")
            return indices
        except Exception as e:
            print(f"  -> FEHLER bei der Suche nach Fenstern: {e}")
            return None

def create_window_plot(history, actuals, prediction_dist, channel_name, title, quantiles, channel_names_in_order):
    """Erstellt eine Matplotlib-Figur für ein einzelnes Fenster (aus duet_prob.py kopiert)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    try:
        # KORREKTUR: Verwende die übergebene, korrekte Liste der Kanalnamen.
        channel_idx = channel_names_in_order.index(channel_name)
    except (ValueError, AttributeError, IndexError):
        channel_idx = 0

    horizon_len = actuals.shape[0]
    history_to_plot = history[-horizon_len:, :]
    history_x = np.arange(horizon_len)
    forecast_x = np.arange(horizon_len, horizon_len * 2)
    history_y = history_to_plot[:, channel_idx]
    actuals_y = actuals[:, channel_idx]
    
    device = prediction_dist.mean.device
    q_tensor = torch.tensor(quantiles, device=device, dtype=torch.float32)
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
    base_alpha, alpha_step = 0.1, 0.15
    for i in range(num_ci_levels):
        lower_q_idx, upper_q_idx = i, len(quantiles) - 1 - i
        current_alpha = base_alpha + (i * alpha_step)
        ax.fill_between(forecast_x, preds_y[:, lower_q_idx], preds_y[:, upper_q_idx], alpha=current_alpha, color='C0', label=f"CI {quantiles[lower_q_idx]}-{quantiles[upper_q_idx]}")
            
    ax.plot(forecast_x, preds_y[:, median_idx], label="Median Forecast", color="blue", linestyle='--', zorder=11)
    ax.set_title(title)
    ax.set_ylabel("Time Series Value")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

def main():
    """Hauptfunktion zur Orchestrierung des Plotting-Prozesses."""
    print(f"Starte diagnostische Plot-Generierung für Studie: '{STUDY_NAME}'")

    # --- NEU: Haupt-Plot-Verzeichnis erstellen ---
    main_plot_output_dir = BASE_RESULTS_DIR / "diagnostic_plots"
    main_plot_output_dir.mkdir(exist_ok=True)
    print(f"Alle Plots werden im Verzeichnis '{main_plot_output_dir}' gespeichert.")

    # --- NEU: Trials nach seq_len gruppieren, um Datenaufbereitung zu minimieren ---
    trials_by_seq_len = defaultdict(list)
    print("\n--- Gruppiere Trials nach Sequenzlänge (seq_len) ---")
    for trial_num in TRIAL_NUMBERS_TO_PLOT:
        trial_path = BASE_RESULTS_DIR / f"trial_{trial_num}"
        _, config = load_model_and_config(trial_path)
        if config:
            trials_by_seq_len[config.seq_len].append(trial_num)
            print(f"  -> Trial #{trial_num} hat seq_len={config.seq_len}")
        else:
            print(f"  -> WARNUNG: Konnte Konfiguration für Trial #{trial_num} nicht laden. Wird übersprungen.")

    # --- NEU: Schleife über die `seq_len`-Gruppen ---
    for seq_len, trial_numbers in trials_by_seq_len.items():
        print(f"\n{'='*20} Verarbeite Gruppe mit seq_len={seq_len} {'='*20}")

        # --- Einmalige Vorbereitung pro Gruppe (Daten laden, Fenster finden) ---
        first_trial_path_in_group = BASE_RESULTS_DIR / f"trial_{trial_numbers[0]}"
        _, config_for_group = load_model_and_config(first_trial_path_in_group)
        if not config_for_group:
            print(f"FEHLER: Konnte die Konfiguration von Trial #{trial_numbers[0]} nicht laden. Überspringe Gruppe.")
            continue

        valid_data_df, valid_dataset = get_validation_data(config_for_group)
        interesting_windows = find_or_load_interesting_windows(BASE_RESULTS_DIR, valid_data_df, config_for_group)
        if not interesting_windows:
            print("FEHLER: Konnte keine 'interessanten Fenster' finden oder laden. Überspringe Gruppe.")
            continue

        # --- Schleife über die Trials innerhalb der Gruppe ---
        for trial_num in tqdm(trial_numbers, desc=f"Verarbeite Trials für seq_len={seq_len}", leave=False):
            trial_path = BASE_RESULTS_DIR / f"trial_{trial_num}"
            
            # Zielverzeichnis für die Plots dieses Trials
            output_plot_dir = main_plot_output_dir / f"trial_{trial_num}"
            
            # --- NEU: Überspringen, wenn Plots bereits existieren und der Ordner nicht leer ist ---
            if output_plot_dir.exists() and any(output_plot_dir.iterdir()):
                tqdm.write(f"  -> Plots für Trial #{trial_num} existieren bereits in '{output_plot_dir}'. Überspringe.")
                continue

            output_plot_dir.mkdir(exist_ok=True)

            model, config = load_model_and_config(trial_path)
            if not model:
                continue

            device = next(model.parameters()).device

            # --- Plotting-Logik (adaptiert von _log_interesting_window_plots) ---
            with torch.no_grad():
                for channel_name, methods in tqdm(interesting_windows.items(), desc=f"Plots für Trial #{trial_num}", leave=False):
                    for method_name, window_start_idx in methods.items():
                        forecast_start_idx = window_start_idx + config.horizon
                        sample_idx = forecast_start_idx - config.seq_len

                        if not (0 <= sample_idx < len(valid_dataset)):
                            continue

                        input_sample, target_sample, _, _ = valid_dataset[sample_idx]
                        actuals_data = target_sample[-config.horizon:, :]
                        input_data = input_sample.float().unsqueeze(0).to(device)
                        
                        denorm_distr, _, _, _, _, _, _, _, _, _, _, _ = model(input_data)
                        
                        # NLL für den Titel berechnen
                        actuals_tensor = actuals_data.float().unsqueeze(0).to(device)
                        nll_per_point = -denorm_distr.log_prob(actuals_tensor)
                        try:
                            channel_names_in_order = list(config.channel_bounds.keys())
                            channel_idx = channel_names_in_order.index(channel_name)
                            nll_val = nll_per_point[:, channel_idx, :].mean().item()
                        except (ValueError, AttributeError):
                            nll_val = nll_per_point.mean().item()

                        fig_title = f'Trial {trial_num} | {channel_name} | {method_name} | NLL: {nll_val:.2f}'
                        fig = create_window_plot(
                            history=input_sample.cpu().numpy(),
                            actuals=actuals_data.cpu().numpy(),
                            prediction_dist=denorm_distr,
                            channel_name=channel_name,
                            title=fig_title,
                            quantiles=config.quantiles,
                            channel_names_in_order=list(config.channel_bounds.keys())
                        )
                        
                        plot_filename = f"{channel_name.replace(' ', '_')}_{method_name}.png"
                        fig.savefig(output_plot_dir / plot_filename)
                        plt.close(fig)
            
            tqdm.write(f"  -> ✅ Alle Plots für Trial #{trial_num} in '{output_plot_dir}' gespeichert.")

    print("\n🎉 Alle ausgewählten Trials wurden erfolgreich verarbeitet.")

if __name__ == "__main__":
    main()