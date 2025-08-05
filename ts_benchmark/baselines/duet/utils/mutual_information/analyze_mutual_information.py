import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import sys
import os
import warnings
from numba import jit

# This script is intended to be run from the project root.
# To ensure that the ts_benchmark module can be found, we add the project root to the sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ts_benchmark.utils.statistical_tests import is_significantly_zero_inflated

# ========================== KONFIGURATION ==========================
# Pfad zur CSV-Datei mit den Zeitreihendaten
DATA_FILE_PATH = "dataset/forecasting/preci_large.csv"

# --- Vorhersagehorizont ---
# Dieser Wert MUSS mit dem `horizon` des Modells übereinstimmen.
HORIZON = 24

# --- Suchbereich für die Fenstergröße (moving_avg) ---
WINDOW_SIZES_TO_TEST = range(6, 96, 4)

# --- Diskretisierung ---
# Anzahl der Bins für die statistischen Merkmale (Mittelwert, Standardabweichung)
NUM_BINS = 50

# --- Top N Ergebnisse ---
TOP_N_RESULTS = 10

# --- Plot-Konfiguration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PLOT_OUTPUT_FILENAME = os.path.join(script_dir, "mutual_information_analysis.png")
CSV_OUTPUT_FILENAME = os.path.join(script_dir, "mutual_information_analysis.csv")
PLOT_INDIVIDUAL_MI_TERMS = True # Neu: Steuert, ob individuelle MI-Terme geplottet werden
# ===================================================================

@jit(nopython=True)
def _numba_percentile(arr, q):
    if len(arr) == 0:
        return np.nan
    sorted_arr = np.sort(arr)
    idx = int(np.ceil(q / 100.0 * len(sorted_arr))) - 1
    if idx < 0:
        idx = 0
    if idx >= len(sorted_arr):
        idx = len(sorted_arr) - 1
    return sorted_arr[idx]

@jit(nopython=True)
def _calculate_strided_stats_numba(data_array, window_size, horizon, is_channel_zero_inflated):
    total_windows = len(data_array) - window_size - horizon + 1
    if total_windows <= 0:
        return (np.empty(0, dtype=data_array.dtype),
                np.empty(0, dtype=data_array.dtype),
                np.empty(0, dtype=data_array.dtype),
                np.empty(0, dtype=data_array.dtype),
                np.empty(0, dtype=data_array.dtype),
                np.empty(0, dtype=data_array.dtype))

    past_strided = np.empty((total_windows, window_size), dtype=data_array.dtype)
    future_strided = np.empty((total_windows, horizon), dtype=data_array.dtype)

    for i in range(total_windows):
        past_strided[i] = data_array[i : i + window_size]
        future_strided[i] = data_array[i + window_size : i + window_size + horizon]

    past_means = np.empty(total_windows, dtype=data_array.dtype)
    past_stds = np.empty(total_windows, dtype=data_array.dtype)

    for i in range(total_windows):
        past_means[i] = np.mean(past_strided[i])
        past_stds[i] = np.std(past_strided[i])

    if is_channel_zero_inflated:

        non_zero_mean = np.empty(total_windows, dtype=data_array.dtype)
        non_zero_std = np.empty(total_windows, dtype=data_array.dtype)
        non_zero_90th_percentile = np.empty(total_windows, dtype=data_array.dtype)
        zero_proportion = np.empty(total_windows, dtype=data_array.dtype)

        for i in range(total_windows):
            f = future_strided[i]
            non_zeros = f[f > 0]
            if len(non_zeros) > 0:
                non_zero_mean[i] = np.mean(non_zeros)
                non_zero_std[i] = np.std(non_zeros)
                non_zero_90th_percentile[i] = _numba_percentile(non_zeros, 90)
            else:
                non_zero_mean[i] = np.nan
                non_zero_std[i] = np.nan
                non_zero_90th_percentile[i] = np.nan
            zero_proportion[i] = np.sum(f == 0) / horizon
        return past_means, past_stds, non_zero_mean, non_zero_std, non_zero_90th_percentile, zero_proportion
    else:
        mean = np.empty(total_windows, dtype=data_array.dtype)
        std = np.empty(total_windows, dtype=data_array.dtype)
        percentile_90th = np.empty(total_windows, dtype=data_array.dtype)

        for i in range(total_windows):
            f = future_strided[i]
            mean[i] = np.mean(f)
            std[i] = np.std(f)
            percentile_90th[i] = _numba_percentile(f, 90)
        return past_means, past_stds, mean, std, percentile_90th, np.full(total_windows, np.nan, dtype=data_array.dtype)

def robust_binning(data, num_bins):
    """
    Eine robuste Binning-Funktion, die mit NaN-Werten und Zero-Inflation umgehen kann.
    Erstellt einen separaten Bin für Nullen und binned den Rest der Daten.
    """
    # Konvertiere zu Pandas Series, um .dropna() und Indexing zu nutzen
    data_series = pd.Series(data)
    data_clean = data_series.dropna()

    if data_clean.empty:
        return pd.Series(np.nan, index=data_series.index)

    # Identifiziere Nullen und Nicht-Nullen
    is_zero = (data_clean == 0)
    non_zero_data = data_clean[~is_zero]

    # Initialisiere das Ergebnis-Series mit NaN, basierend auf dem Original-Index
    binned_data = pd.Series(np.nan, index=data_series.index)

    # Bin für Nullen: Weise allen Nullen den Wert 0 zu
    if is_zero.any():
        binned_data.loc[is_zero.index[is_zero]] = 0

    # Binning für Nicht-Null-Werte
    if not non_zero_data.empty:
        # Reduziere die Anzahl der Bins für Nicht-Null-Werte um 1 (da 0 schon einen Bin hat)
        # Stelle sicher, dass mindestens 1 Bin für Nicht-Null-Werte übrig bleibt
        num_bins_for_non_zero = max(1, num_bins - 1)

        try:
            # Versuche pd.qcut für Nicht-Null-Werte
            binned_non_zero = pd.qcut(non_zero_data, q=num_bins_for_non_zero, labels=False, duplicates='drop')
        except ValueError:
            # Fallback zu pd.cut, falls qcut fehlschlägt
            try:
                binned_non_zero, _ = pd.cut(non_zero_data, bins=num_bins_for_non_zero, labels=False, duplicates='drop', retbins=True)
            except ValueError:
                # Wenn auch cut fehlschlägt (z.B. nur ein einzigartiger Nicht-Null-Wert),
                # weise allen Nicht-Null-Werten einen einzigen Bin-Wert zu (z.B. 1)
                binned_non_zero = pd.Series(1, index=non_zero_data.index)

        # Verschiebe die Bin-Werte der Nicht-Null-Daten um 1, damit sie nicht mit dem Null-Bin kollidieren
        binned_data.loc[non_zero_data.index] = binned_non_zero + 1

    # Konvertiere zu Integer, aber nur wenn es keine NaNs gibt, sonst bleibt es float
    return binned_data.astype(int, errors='ignore') # Use errors='ignore' to keep NaNs as float


def calculate_mi_for_series(series: pd.Series, window_size: int, horizon: int, num_bins: int, is_channel_zero_inflated: bool) -> dict:
    """
    Berechnet die Mutual Information zwischen den statistischen Merkmalen (mean, std)
    des Vergangenheits-Fensters und verschiedenen Proxies der Zukunftsverteilung.
    Die Proxies werden je nach Zero-Inflation-Status des Kanals angepasst.
    """
    series_clean = series.dropna()
    if len(series_clean) < window_size + horizon:
        return {'mutual_information': np.nan}

    # Use the numba-optimized function for core numerical calculations
    if is_channel_zero_inflated:
        past_means, past_stds, non_zero_mean, non_zero_std, non_zero_90th_percentile, zero_proportion = \
            _calculate_strided_stats_numba(series_clean.values, window_size, horizon, is_channel_zero_inflated)
        future_proxies = {
            'non_zero_mean': non_zero_mean,
            'non_zero_std': non_zero_std,
            'non_zero_90th_percentile': non_zero_90th_percentile,
            'zero_proportion': zero_proportion
        }
    else:
        past_means, past_stds, mean, std, percentile_90th, _ = \
            _calculate_strided_stats_numba(series_clean.values, window_size, horizon, is_channel_zero_inflated)
        future_proxies = {
            'mean': mean,
            'std': std,
            '90th_percentile': percentile_90th
        }

    mi_results = {} # Dictionary zum Speichern der individuellen MI-Terme

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if is_channel_zero_inflated:
            try:
                past_means_binned = robust_binning(past_means, num_bins)
                past_stds_binned = robust_binning(past_stds, num_bins)
                future_proxies_binned = {k: robust_binning(v, num_bins) for k, v in future_proxies.items()}
            except ValueError:
                return {'mutual_information': 0.0}

            df_data = {
                'past_mean': past_means_binned,
                'past_std': past_stds_binned,
            }
            df_data.update(future_proxies_binned)
            df = pd.DataFrame(df_data).dropna()

            if len(df) < 2:
                return {'mutual_information': np.nan}

            mi_results['mi_past_mean_vs_non_zero_mean'] = mutual_info_score(df['past_mean'], df['non_zero_mean'])
            mi_results['mi_past_std_vs_non_zero_mean'] = mutual_info_score(df['past_std'], df['non_zero_mean'])
            mi_results['mi_past_mean_vs_non_zero_std'] = mutual_info_score(df['past_mean'], df['non_zero_std'])
            mi_results['mi_past_std_vs_non_zero_std'] = mutual_info_score(df['past_std'], df['non_zero_std'])
            mi_results['mi_past_mean_vs_non_zero_90th_percentile'] = mutual_info_score(df['past_mean'], df['non_zero_90th_percentile'])
            mi_results['mi_past_std_vs_non_zero_90th_percentile'] = mutual_info_score(df['past_std'], df['non_zero_90th_percentile'])
            mi_results['mi_past_mean_vs_zero_proportion'] = mutual_info_score(df['past_mean'], df['zero_proportion'])
            mi_results['mi_past_std_vs_zero_proportion'] = mutual_info_score(df['past_std'], df['zero_proportion'])

        else:
            try:
                past_means_binned = robust_binning(past_means, num_bins)
                past_stds_binned = robust_binning(past_stds, num_bins)
                future_proxies_binned = {k: robust_binning(v, num_bins) for k, v in future_proxies.items()}
            except ValueError:
                return {'mutual_information': 0.0}

            df_data = {
                'past_mean': past_means_binned,
                'past_std': past_stds_binned,
            }
            df_data.update(future_proxies_binned)
            df = pd.DataFrame(df_data).dropna()

            if len(df) < 2:
                return {'mutual_information': np.nan}

            mi_results['mi_past_mean_vs_mean'] = mutual_info_score(df['past_mean'], df['mean'])
            mi_results['mi_past_std_vs_mean'] = mutual_info_score(df['past_std'], df['mean'])
            mi_results['mi_past_mean_vs_std'] = mutual_info_score(df['past_mean'], df['std'])
            mi_results['mi_past_std_vs_std'] = mutual_info_score(df['past_std'], df['std'])
            mi_results['mi_past_mean_vs_90th_percentile'] = mutual_info_score(df['past_mean'], df['90th_percentile'])
            mi_results['mi_past_std_vs_90th_percentile'] = mutual_info_score(df['past_std'], df['90th_percentile'])

    mi_results['mutual_information'] = sum(mi_results.values())
    return mi_results

if __name__ == "__main__":
    print(f"Lade Daten aus '{DATA_FILE_PATH}'...")
    df_long = pd.read_csv(DATA_FILE_PATH)
    df_long['date'] = pd.to_datetime(df_long['date'])
    channel_names = df_long['cols'].unique()

    all_results = []

    print(f"Analysiere {len(channel_names)} Kanäle mit {len(WINDOW_SIZES_TO_TEST)} verschiedenen Fenstergrößen...")

    # Pre-extract channel series and their zero-inflation status
    channel_data = {}
    channel_zero_inflated_status = {}
    print("Führe Zero-Inflation-Tests für jeden Kanal durch...")
    for channel in tqdm(channel_names, desc="Teste auf Zero-Inflation"):
        series = df_long[df_long['cols'] == channel].set_index('date')['data']
        channel_data[channel] = series
        # Verwende den neuen, robusten Test aus der zentralen Utils-Datei
        # Das Ergebnis wird für die spätere Verwendung zwischengespeichert
        channel_zero_inflated_status[channel] = is_significantly_zero_inflated(series, verbose=True)
        print(f"Kanal '{channel}' ist Zero-Inflated: {channel_zero_inflated_status[channel]}")

    for size in tqdm(WINDOW_SIZES_TO_TEST, desc="Teste Fenstergrößen"):
        for channel in channel_names:
            series = channel_data[channel]
            is_channel_zero_inflated = channel_zero_inflated_status[channel]
            mi_scores_dict = calculate_mi_for_series(series, window_size=size, horizon=HORIZON, num_bins=NUM_BINS, is_channel_zero_inflated=is_channel_zero_inflated)
            
            if not np.isnan(mi_scores_dict['mutual_information']):
                result_entry = {
                    'window_size': size,
                    'channel': channel,
                }
                result_entry.update(mi_scores_dict)
                all_results.append(result_entry)

    if not all_results:
        print("\nFEHLER: Keine Ergebnisse berechnet. Überprüfe die Daten und die Konfiguration.", file=sys.stderr)
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CSV_OUTPUT_FILENAME, index=False, float_format='%.6f')
    print(f"\nDetaillierte Ergebnisse wurden in '{CSV_OUTPUT_FILENAME}' gespeichert.")

    # --- Berechne die verschiedenen aggregierten Metriken ---
    print("\nBerechne aggregierte Metriken...")
    # 1. Mittelwert (Average)
    avg_mi_df = results_df.groupby('window_size')['mutual_information'].mean().reset_index()
    avg_mi_df = avg_mi_df.rename(columns={'mutual_information': 'mean_mi'})

    # 2. Median
    median_mi_df = results_df.groupby('window_size')['mutual_information'].median().reset_index()
    median_mi_df = median_mi_df.rename(columns={'mutual_information': 'median_mi'})

    # 3. Konsistenz-Score (Mean / Std Dev)
    std_mi_df = results_df.groupby('window_size')['mutual_information'].std().reset_index()
    std_mi_df = std_mi_df.rename(columns={'mutual_information': 'std_mi'})
    consistency_df = pd.merge(avg_mi_df, std_mi_df, on='window_size')
    # Füge einen kleinen Epsilon-Wert hinzu, um Division durch Null zu vermeiden
    consistency_df['consistency_score'] = consistency_df['mean_mi'] / (consistency_df['std_mi'] + 1e-9)

    # 4. Worst-Case Score (Max-Min)
    min_mi_df = results_df.groupby('window_size')['mutual_information'].min().reset_index()
    min_mi_df = min_mi_df.rename(columns={'mutual_information': 'min_mi'})

    # 5. Durchschnittlicher Rang
    results_df['rank'] = results_df.groupby('channel')['mutual_information'].rank(method='dense', ascending=False)
    avg_rank_df = results_df.groupby('window_size')['rank'].mean().reset_index()
    avg_rank_df = avg_rank_df.rename(columns={'rank': 'avg_rank'})

    # --- Gib die Top-Kandidaten für jede Metrik aus ---
    metrics = {
        "Mittelwert (Mean)": (avg_mi_df, 'mean_mi', False),
        "Median": (median_mi_df, 'median_mi', False),
        "Konsistenz (Mean/Std)": (consistency_df, 'consistency_score', False),
        "Bester schlechtester Fall (Max-Min)": (min_mi_df, 'min_mi', False),
        "Bester Durchschnitts-Rang": (avg_rank_df, 'avg_rank', True) # True, da niedrigerer Rang besser ist
    }

    for name, (df, col, asc) in metrics.items():
        print("\n" + "="*50)
        print(f"  TOP {TOP_N_RESULTS} KANDIDATEN NACH: {name}")
        print("="*50)
        best_candidates = df.sort_values(by=col, ascending=asc).head(TOP_N_RESULTS)
        print(best_candidates.to_string(index=False))
        print("="*50)

    # --- Ergebnisse plotten ---
    print(f"\nErstelle Plot und speichere als '{PLOT_OUTPUT_FILENAME}'...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    pivot_df = results_df.pivot(index='window_size', columns='channel', values='mutual_information')

    for channel in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[channel], label=channel, alpha=0.4, marker='.')

    # Plotte Mittelwert und Median
    ax.plot(avg_mi_df['window_size'], avg_mi_df['mean_mi'], label='Durchschnitt (Mean)', color='black', linewidth=2.5, linestyle='--')
    ax.plot(median_mi_df['window_size'], median_mi_df['median_mi'], label='Median', color='firebrick', linewidth=2.5, linestyle=':')

    ax.set_title("Mutual Information Analyse für verschiedene Fenstergrößen", fontsize=16, weight='bold')
    ax.set_xlabel("Fenstergröße (Stunden) - Kandidat für 'moving_avg'", fontsize=12)
    ax.set_ylabel("Mutual Information (Höher ist besser)", fontsize=12)
    ax.legend(title='Kanäle / Metriken', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Markiere die besten Punkte auf der Durchschnittslinie
    best_points_mean = avg_mi_df.sort_values(by='mean_mi', ascending=False).head(TOP_N_RESULTS)
    ax.scatter(best_points_mean['window_size'], best_points_mean['mean_mi'], color='blue', s=120, zorder=5, label=f'Top {TOP_N_RESULTS} (Mean)', marker='D')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(PLOT_OUTPUT_FILENAME)
    plt.close(fig)

    print("\nAnalyse abgeschlossen.")

    # --- Plotten individueller MI-Terme ---
    if PLOT_INDIVIDUAL_MI_TERMS:
        print("\nErstelle Plots für individuelle MI-Terme...")
        # Identifiziere alle MI-Term-Spalten (beginnen mit 'mi_')
        mi_term_columns = [col for col in results_df.columns if col.startswith('mi_')]

        for mi_col in mi_term_columns:
            fig_ind, ax_ind = plt.subplots(figsize=(18, 10))
            pivot_df_ind = results_df.pivot(index='window_size', columns='channel', values=mi_col)

            for channel in pivot_df_ind.columns:
                ax_ind.plot(pivot_df_ind.index, pivot_df_ind[channel], label=channel, alpha=0.4, marker='.')
            
            # Plot mean of individual MI term
            avg_ind_mi_df = results_df.groupby('window_size')[mi_col].mean().reset_index()
            ax_ind.plot(avg_ind_mi_df['window_size'], avg_ind_mi_df[mi_col], label='Durchschnitt', color='black', linewidth=2.5, linestyle='--')

            ax_ind.set_title(f"Mutual Information Analyse für {mi_col.replace('mi_', '').replace('_', ' ').title()} (Individuell)", fontsize=16, weight='bold')
            ax_ind.set_xlabel("Fenstergröße (Stunden)", fontsize=12)
            ax_ind.set_ylabel("Mutual Information (Höher ist besser)", fontsize=12)
            ax_ind.legend(title='Kanäle / Metriken', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax_ind.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(os.path.join(script_dir, f"individual_mi_{mi_col}.png"))
            plt.close(fig_ind)