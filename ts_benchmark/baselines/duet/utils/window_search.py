import numpy as np
import numba
from scipy.stats import ks_2samp
import pandas as pd
from tqdm import tqdm


@numba.jit(nopython=True)
def _calculate_slope(y: np.ndarray) -> float:
    """
    Berechnet die Steigung einer einfachen linearen Regression (y = mx + b)
    für eine gegebene Zeitreihe y. Numba-optimiert.
    """
    n = len(y)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x * sum_x

    if np.abs(denominator) < 1e-9:
        return 0.0

    return numerator / denominator


def find_interesting_windows(
    data: pd.DataFrame | np.ndarray,
    horizon: int,
    seq_len: int
) -> dict:
    """
    Durchsucht einen DataFrame oder NumPy-Array nach den "interessantesten" aufeinanderfolgenden
    Fenstern basierend auf vier verschiedenen Metriken.
    Die Suche wird auf den Bereich beschränkt, für den eine Vorhersage möglich ist.

    Args:
        data (pd.DataFrame | np.ndarray): Die Validierungsdaten.
        horizon (int): Die Länge des Vorhersagefensters.

    Returns:
        dict: Ein Dictionary mit den Indizes der gefundenen Fenster für jeden Kanal und jede Methode.
    """
    if isinstance(data, pd.DataFrame):
        channel_names = data.columns.tolist()
        data_np = data.values
    else:
        data_np = data
        channel_names = [f'channel_{i}' for i in range(data.shape[1])]

    num_timesteps, num_channels = data_np.shape
    # Sicherheitsprüfung: Wenn die Daten zu kurz sind, um auch nur ein Fensterpaar zu bilden, leeres Dict zurückgeben.
    # Wir brauchen mindestens `seq_len` an Historie und `horizon` für die Vorhersage.
    if num_timesteps < seq_len + horizon:
        return {}

    print("\nINFO: Searching for interesting windows in validation data for later analysis...")

    results = {i: {} for i in range(num_channels)}

    # --- KORREKTUR: Suche auf den plotbaren Bereich einschränken ---
    # Wir können nur Fenster analysieren, für die wir auch eine Vorhersage machen können.
    # Eine Vorhersage für ein Fenster, das bei `t` beginnt, benötigt Daten von `t - seq_len` bis `t`.
    # Das "Nachher"-Fenster, das wir vorhersagen wollen, beginnt bei `window_start_idx + horizon`.
    # Also muss `window_start_idx + horizon >= seq_len` sein.
    # Der früheste `window_start_idx` ist somit `seq_len - horizon`.
    # Wir nennen diesen Offset `search_offset`.
    search_offset = seq_len - horizon
    search_data = data_np[search_offset:]
    search_num_timesteps = search_data.shape[0]

    # Sicherheitsprüfung nach der Einschränkung
    if search_num_timesteps < 2 * horizon:
        return {}

    # Erstelle eine "View" auf alle möglichen Fensterpaare, ohne Daten zu kopieren.
    # WICHTIG: Auf `search_data` anwenden!
    strides = (search_data.strides[0], search_data.strides[0], search_data.strides[1])
    shape = (search_num_timesteps - 2 * horizon + 1, 2 * horizon, num_channels)
    all_windows = np.lib.stride_tricks.as_strided(search_data, shape=shape, strides=strides)

    # Teile in Vorher- und Nachher-Fenster auf
    windows_before = all_windows[:, :horizon, :]
    windows_after = all_windows[:, horizon:, :]

    # --- 1. Mittelwert- & Varianz-Differenz (vektorisiert mit NumPy) ---
    mean_diffs = np.abs(np.mean(windows_after, axis=1) - np.mean(windows_before, axis=1))
    var_diffs = np.abs(np.var(windows_after, axis=1) - np.var(windows_before, axis=1))

    # Finde die Indizes der maximalen Differenzen für jeden Kanal
    # WICHTIG: Den Offset wieder zum Ergebnis addieren!
    for i in range(num_channels):
        results[i]['max_mean_diff_idx'] = np.argmax(mean_diffs[:, i]) + search_offset
        results[i]['max_var_diff_idx'] = np.argmax(var_diffs[:, i]) + search_offset

    # --- 2. Trendumkehr & KS-Distanz (mit Numba-optimierten Schleifen) ---
    num_windows = shape[0]

    # Initialisiere Arrays für die Ergebnisse
    trend_diffs = np.zeros((num_windows, num_channels))
    ks_dists = np.zeros((num_windows, num_channels))

    @numba.jit(nopython=True, parallel=True)
    def calculate_trend_diffs(wb, wa, out_arr):
        for i in numba.prange(num_windows):
            for j in range(num_channels):
                slope_before = _calculate_slope(wb[i, :, j])
                slope_after = _calculate_slope(wa[i, :, j])
                out_arr[i, j] = np.abs(slope_after - slope_before)

    print(" -> Analyzing windows for trend reversals (Numba)...")
    calculate_trend_diffs(windows_before, windows_after, trend_diffs)

    # Die KS-Distanz verwendet Scipy und kann nicht direkt in Numba nopython=True
    # ausgeführt werden. Eine reine Python-Schleife ist hier immer noch schnell genug,
    # da sie nur einmal pro Trainingslauf ausgeführt wird.
    for i in tqdm(range(num_windows), desc=" -> Analyzing windows for distribution shifts (KS-test)", leave=False):
        for j in range(num_channels):
            # ks_2samp gibt (statistik, p-wert) zurück. Wir brauchen nur die Statistik.
            ks_stat, _ = ks_2samp(windows_before[i, :, j], windows_after[i, :, j])
            
            # SICHERHEITSMASSNAHME: Wenn ks_stat NaN ist (z.B. weil ein Fenster
            # nur aus NaNs besteht), behandeln wir die Distanz als 0.
            # Das verhindert, dass NaNs in das Ergebnis-Array gelangen und
            # np.argmax zu unerwartetem Verhalten oder Fehlern führt.
            if np.isnan(ks_stat):
                ks_dists[i, j] = 0.0
            else:
                ks_dists[i, j] = ks_stat


    # Finde die Indizes der maximalen Differenzen für jeden Kanal
    # WICHTIG: Den Offset wieder zum Ergebnis addieren!
    for i in range(num_channels):
        results[i]['max_trend_rev_idx'] = np.argmax(trend_diffs[:, i]) + search_offset
        results[i]['max_ks_dist_idx'] = np.argmax(ks_dists[:, i]) + search_offset

    # Konvertiere die Kanal-Indizes (0, 1, 2...) in die tatsächlichen Kanalnamen
    final_results = {}

    for i, name in enumerate(channel_names):
        final_results[name] = {
            'max_mean_diff_idx': int(results[i]['max_mean_diff_idx']),
            'max_var_diff_idx': int(results[i]['max_var_diff_idx']),
            'max_trend_rev_idx': int(results[i]['max_trend_rev_idx']),
            'max_ks_dist_idx': int(results[i]['max_ks_dist_idx']),
        }

    # --- NEU: Hinzufügen von 6 festen, aber zufälligen, nicht-überlappenden Fenstern ---
    print(" -> Searching for 6 additional random, non-overlapping windows for general analysis...")
    NUM_RANDOM_WINDOWS = 6
    RANDOM_SEED = 1337  # Fester Seed für Reproduzierbarkeit über verschiedene Studienläufe hinweg
    np.random.seed(RANDOM_SEED)

    # 1. Sammle alle bereits belegten Zeiträume aus den "harten" Fenstern.
    # Ein Fenster belegt den Raum von [Input-Start] bis [Output-Ende].
    occupied_indices = set()
    all_hard_indices = set()
    for channel_results in final_results.values():
        for idx in channel_results.values():
            all_hard_indices.add(idx)

    for window_start_idx in all_hard_indices:
        # Der Input für die Vorhersage beginnt bei `window_start_idx + horizon - seq_len`.
        # Das "Nachher"-Fenster endet bei `window_start_idx + 2 * horizon`.
        sample_start_idx = window_start_idx + horizon - seq_len
        sample_end_idx = window_start_idx + 2 * horizon
        occupied_indices.update(range(sample_start_idx, sample_end_idx))

    # 2. Erstelle einen Pool an gültigen, freien Start-Indizes.
    possible_start_indices = []
    # Die Schleife durchläuft alle möglichen Startpunkte im durchsuchbaren Bereich.
    for i in range(num_windows):
        potential_start_idx = i + search_offset
        
        # Definiere den Raum, den dieses potenzielle Fenster belegen würde.
        potential_sample_start = potential_start_idx + horizon - seq_len
        potential_sample_end = potential_start_idx + 2 * horizon
        potential_range = set(range(potential_sample_start, potential_sample_end))

        # Prüfe, ob es eine Überschneidung mit bereits belegten Bereichen gibt.
        if potential_range.isdisjoint(occupied_indices):
            possible_start_indices.append(potential_start_idx)

    # 3. Wähle 6 zufällige Fenster aus dem Pool (ohne Zurücklegen).
    if len(possible_start_indices) < NUM_RANDOM_WINDOWS:
        print(f"    WARNUNG: Weniger als {NUM_RANDOM_WINDOWS} nicht-überlappende zufällige Fenster gefunden. "
              f"Verwende {len(possible_start_indices)} stattdessen.")
        num_to_sample = len(possible_start_indices)
    else:
        num_to_sample = NUM_RANDOM_WINDOWS

    if num_to_sample > 0:
        random_indices = np.random.choice(possible_start_indices, size=num_to_sample, replace=False)
        # 4. Füge die zufälligen Fenster zu den Ergebnissen hinzu.
        for i, random_idx in enumerate(random_indices):
            method_name = f'random_window_{i+1}_idx'
            for channel_name in final_results:
                final_results[channel_name][method_name] = int(random_idx)

    return final_results