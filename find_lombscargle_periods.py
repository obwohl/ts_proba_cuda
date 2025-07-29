import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
from tqdm import tqdm

# --- WICHTIG: Wir verwenden jetzt die hochoptimierte astropy-Implementierung ---
# Diese ist oft deutlich schneller und robuster als die von scipy.
# Falls nicht installiert: pip install astropy
try:
    from astropy.timeseries import LombScargle
except ImportError:
    print("FEHLER: Das 'astropy'-Paket wurde nicht gefunden. Bitte installieren Sie es mit: pip install astropy", file=sys.stderr)
    sys.exit(1)

# ========================== KONFIGURATION ==========================
# Pfad zur CSV-Datei mit den Zeitreihendaten
DATA_FILE_PATH = "dataset/forecasting/eisbach_airtemp96_pressure96.csv"

# --- Konfiguration für den Lomb-Scargle-Test ---
# Signifikanzniveau für die "False Alarm Probability" (FAP).
# 0.01 bedeutet: Wir akzeptieren nur Peaks, bei denen die Wahrscheinlichkeit,
# dass sie durch reines Rauschen entstanden sind, unter 1% liegt.
FAP_LEVEL = 0.0001

# --- Filter für die Periodenlänge ---
# Plausibler Bereich für Perioden (in Stunden), die wir überhaupt betrachten.
MIN_PERIOD_HOURS = 4
# Setze dies auf die längste `seq_len`, die im Optuna-Suchraum vorkommt.
MAX_PERIOD_HOURS = 384
# ===================================================================
# --- NEU: Konfiguration für die faire, kanalübergreifende Auswahl ---
# Wie viele der global besten Perioden sollen am Ende ausgewählt werden?
# Dies ist eine Obergrenze für die Anzahl der Experten, die wir trainieren wollen.
MAX_FINAL_PERIODS = 15

# Schwellenwert, um nur die "lokal dominanten" Peaks pro Kanal zu behalten.
# Ein Peak muss mindestens X% der Stärke des stärksten Peaks in seinem Kanal haben.
# 0.1 = 10%. Dies filtert kleine, aber signifikante Rausch-Peaks heraus.
LOCAL_PEAK_THRESHOLD_FACTOR = 0.1


def find_significant_periods_lombscargle(series: pd.Series, channel_name: str, create_plot: bool = False) -> pd.DataFrame:
    """
    Findet statistisch signifikante Perioden in einer Zeitreihe mittels des
    robusten Lomb-Scargle-Periodogramms und dessen False-Alarm-Probability.

    Gibt eine Liste der gefundenen signifikanten Perioden zurück.
    """
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return pd.DataFrame()

    # 1. Zeitachse vorbereiten (in Stunden seit Beginn)
    time_hours = (series_clean.index - series_clean.index[0]).total_seconds() / 3600.0
    values = series_clean.values

    # Gute Praxis: Linearen Trend entfernen, um die Periodogramm-Qualität zu verbessern
    p = np.polyfit(time_hours, values, 1)
    values_detrended = values - np.polyval(p, time_hours)

    # 2. Frequenzen definieren, die wir testen wollen
    # Wir erstellen ein feines Raster von Frequenzen, das unseren Perioden entspricht.
    min_freq = 1.0 / MAX_PERIOD_HOURS
    max_freq = 1.0 / MIN_PERIOD_HOURS
    # Die Anzahl der Frequenzen bestimmt die Auflösung. 10000 ist ein guter Kompromiss.
    frequencies = np.linspace(min_freq, max_freq, 10000) 

    # 3. Lomb-Scargle-Periodogramm berechnen
    # Wir verwenden die astropy-Implementierung, die deutlich schneller ist.
    ls = LombScargle(time_hours, values_detrended, normalization='standard')
    power = ls.power(frequencies, method='fast')

    # 4. Signifikanzschwelle berechnen
    # Die 'baluev' Methode ist eine schnelle analytische Approximation.
    # WICHTIGER FIX: Wir casten das Ergebnis explizit zu einem float.
    # `find_peaks` erwartet entweder einen Skalar oder ein Array der gleichen Länge wie die Daten.
    # `astropy` kann manchmal ein Array mit einem Element zurückgeben, was den Fehler verursacht.
    fap_threshold = float(ls.false_alarm_level(FAP_LEVEL, method='baluev'))

    # 5. Finde alle Peaks, die ÜBER dieser statistischen Schwelle liegen
    # KORREKTUR: Verwende den neuen Konfigurationsparameter für einen sinnvollen Abstand.
    significant_peaks, _ = find_peaks(power, height=fap_threshold, distance=MIN_PEAK_DISTANCE_SAMPLES)

    if len(significant_peaks) == 0:
        return pd.DataFrame()

    # 6. Ergebnisse aufbereiten und ausgeben
    found_periods = 1.0 / frequencies[significant_peaks]
    found_powers = power[significant_peaks]
    
    results_df = pd.DataFrame({
        'Periode (Stunden)': found_periods,
        'Stärke (Power)': found_powers
    }).sort_values(by='Stärke (Power)', ascending=False).reset_index(drop=True)

    if create_plot:
        print(f"  -> Signifikante Periode(n) für '{channel_name}' gefunden. Erstelle Plot...")

        # --- Visualisierung ---
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_periods = 1.0 / frequencies
        
        ax.plot(plot_periods, power, label='Lomb-Scargle Periodogramm', color='C0', zorder=2)
        ax.axhline(y=fap_threshold, color='red', linestyle='--', label=f'Signifikanzschwelle (FAP={FAP_LEVEL})', zorder=1)
        
        ax.plot(final_periods, final_powers, "x", color='green', markersize=10, mew=2, label='Lokale Champions (nach Filter)', zorder=3)
        
        for period, pwr in zip(final_periods, final_powers):
            ax.text(period, pwr, f' {period:.1f}h', verticalalignment='bottom', color='green', weight='bold')

        ax.set_title(f"Lomb-Scargle Periodogramm für '{channel_name}'", fontsize=16)
        ax.set_xlabel("Periode (in Stunden)")
        ax.set_ylabel("Normalisierte Leistung (Power)")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, which='both', linestyle='--')

        plt.tight_layout()
        output_filename = f"lombscargle_periods_{channel_name}.png"
        plt.savefig(output_filename)
        print(f"  -> Plot für '{channel_name}' gespeichert als: '{output_filename}'")
        plt.close(fig)

    return results_df

if __name__ == "__main__":
    df_long = pd.read_csv(DATA_FILE_PATH)
    df_long['date'] = pd.to_datetime(df_long['date'])
    channel_names = df_long['cols'].unique()

    all_results_dfs = []

    print(f"Analysiere {len(channel_names)} Kanäle aus '{DATA_FILE_PATH}'...")
    # Hier ist die tqdm-Fortschrittsanzeige!
    for channel in tqdm(channel_names, desc="Analysiere Kanäle"):
        series = df_long[df_long['cols'] == channel].set_index('date')['data']
        
        # KORREKTUR: Erstelle einen Plot für JEDEN Kanal.
        results_df = find_significant_periods_lombscargle(series, channel, create_plot=True)
        
        if not results_df.empty:
            results_df['channel'] = channel
            all_results_dfs.append(results_df)

    print("\n" + "="*50)
    print("  ZUSAMMENFASSUNG ALLER SIGNIFIKANTEN PERIODEN")
    print("="*50)

    if not all_results_dfs:
        print("Keine signifikanten Perioden in allen Kanälen gefunden.")
    else:
        # KORREKTUR: Erstelle ein sauberes, kombiniertes DataFrame und speichere es als CSV.
        summary_df = pd.concat(all_results_dfs, ignore_index=True)
        summary_df = summary_df[['channel', 'Periode (Stunden)', 'Stärke (Power)']]
        summary_df = summary_df.sort_values(by=['channel', 'Stärke (Power)'], ascending=[True, False])
        
        output_csv_path = "significant_periods_summary.csv"
        summary_df.to_csv(output_csv_path, index=False, float_format='%.2f')
        
        print(f"Analyse abgeschlossen. Eine Zusammenfassung aller gefundenen Perioden wurde in der Datei gespeichert:\n  -> {output_csv_path}")
        print("\nZusätzlich wurden detaillierte Plots für jeden Kanal als 'lombscargle_periods_<Kanalname>.png' gespeichert.")
    
    print("\nAnalyse abgeschlossen.")