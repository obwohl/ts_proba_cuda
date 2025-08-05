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
DATA_FILE_PATH = "dataset/forecasting/preci_large.csv"

# --- Konfiguration für den Lomb-Scargle-Test ---
# Signifikanzniveau für die "False Alarm Probability" (FAP).
# 0.0001 bedeutet: Wir akzeptieren nur Peaks, bei denen die Wahrscheinlichkeit,
# dass sie durch reines Rauschen entstanden sind, unter 0.01% liegt.
FAP_LEVEL = 0.01

# --- Filter für die Periodenlänge ---
# Plausibler Bereich für Perioden (in Stunden), die wir überhaupt betrachten.
MIN_PERIOD_HOURS = 4
# Setze dies auf die längste `seq_len`, die im Optuna-Suchraum vorkommt.
MAX_PERIOD_HOURS = 384
# Minimaler Abstand zwischen Peaks in den Frequenz-Samples. Verhindert, dass
# sehr nahe beieinander liegende Peaks als separate Perioden erkannt werden.
# Ein Wert von 10 ist bei 10000 Frequenz-Samples oft ein guter Start.
MIN_PEAK_DISTANCE_SAMPLES = 10

# --- Konfiguration für die faire, kanalübergreifende Auswahl ---
# Wie viele der global besten Perioden sollen am Ende ausgewählt werden?
# Dies ist eine Obergrenze für die Anzahl der Experten, die wir trainieren wollen.
MAX_FINAL_PERIODS = 15
# ===================================================================

def plot_lombscargle_periodogram(channel_name: str, results_df: pd.DataFrame, plot_data: dict):
    """
    Erstellt und speichert einen Plot des Lomb-Scargle-Periodogramms.
    """
    print(f"  -> Signifikante Periode(n) für '{channel_name}' gefunden. Erstelle Plot...")

    # Plotte alle statistisch signifikanten Peaks für diesen Kanal
    significant_periods = results_df['Periode (Stunden)'].values
    significant_powers = results_df['Stärke (Power)'].values

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_periods = 1.0 / plot_data['frequencies']
    
    ax.plot(plot_periods, plot_data['power'], label='Lomb-Scargle Periodogramm', color='C0', zorder=2)
    ax.axhline(y=plot_data['fap_threshold'], color='red', linestyle='--', label=f'Signifikanzschwelle (FAP={FAP_LEVEL})', zorder=1)
    ax.plot(significant_periods, significant_powers, "x", color='green', markersize=10, mew=2, label='Signifikante Peaks', zorder=4)
    
    for period, pwr in zip(significant_periods, significant_powers):
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

def find_significant_periods_lombscargle(series: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Findet statistisch signifikante Perioden in einer Zeitreihe mittels des
    robusten Lomb-Scargle-Periodogramms und dessen False-Alarm-Probability.
    """
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return pd.DataFrame(), {}

    # 1. Zeitachse vorbereiten (in Stunden seit Beginn)
    time_hours = (series_clean.index - series_clean.index[0]).total_seconds() / 3600.0
    values = series_clean.values

    # Gute Praxis: Linearen Trend entfernen, um die Periodogramm-Qualität zu verbessern
    p = np.polyfit(time_hours, values, 1)
    values_detrended = values - np.polyval(p, time_hours)

    # 2. Frequenzen definieren, die wir testen wollen
    min_freq = 1.0 / MAX_PERIOD_HOURS
    max_freq = 1.0 / MIN_PERIOD_HOURS
    frequencies = np.linspace(min_freq, max_freq, 10000) 

    # 3. Lomb-Scargle-Periodogramm berechnen
    # Die Normalisierung 'standard' ist entscheidend für ein vergleichbares SNR.
    # Die Power gibt an, welcher Anteil der Gesamtvarianz durch die Frequenz erklärt wird.
    ls = LombScargle(time_hours, values_detrended, normalization='standard')
    power = ls.power(frequencies, method='fast')

    # 4. Signifikanzschwelle berechnen
    fap_threshold = float(ls.false_alarm_level(FAP_LEVEL, method='baluev'))

    # 5. Finde alle Peaks, die ÜBER dieser statistischen Schwelle liegen
    significant_peaks, _ = find_peaks(power, height=fap_threshold, distance=MIN_PEAK_DISTANCE_SAMPLES)

    if len(significant_peaks) == 0:
        return pd.DataFrame(), {}

    # 6. Ergebnisse aufbereiten
    found_periods = 1.0 / frequencies[significant_peaks]
    found_powers = power[significant_peaks]

    results_df = pd.DataFrame({'Periode (Stunden)': found_periods, 'Stärke (Power)': found_powers})
    
    # NEU: Berechne das Signal-Rausch-Verhältnis (SNR) als vergleichbares Maß.
    # Die 'standard' normalisierte Power (P) ist der Anteil der Varianz, der durch die
    # Periode erklärt wird. SNR = (erklärte Varianz) / (unerklärte Varianz) = P / (1 - P).
    # Dieses Maß ist über alle Kanäle hinweg fair vergleichbar.
    power_col = results_df['Stärke (Power)']
    # Vermeide Division durch Null, falls Power jemals exakt 1.0 sein sollte.
    results_df['SNR'] = np.divide(power_col, 1 - power_col, out=np.full_like(power_col, np.inf), where=(1 - power_col) != 0)

    # Sortiere nach dem neuen, besseren SNR-Maß, um die stärksten Signale zuerst zu haben.
    results_df = results_df.sort_values(by='SNR', ascending=False).reset_index(drop=True)

    plot_data = {
        'frequencies': frequencies,
        'power': power,
        'fap_threshold': fap_threshold,
    }
    return results_df, plot_data


if __name__ == "__main__":
    df_long = pd.read_csv(DATA_FILE_PATH)
    df_long['date'] = pd.to_datetime(df_long['date'])
    channel_names = df_long['cols'].unique()

    all_results_dfs = []

    print(f"Analysiere {len(channel_names)} Kanäle aus '{DATA_FILE_PATH}'...")
    for channel in tqdm(channel_names, desc="Analysiere Kanäle"):
        series = df_long[df_long['cols'] == channel].set_index('date')['data']
        results_df, plot_data = find_significant_periods_lombscargle(series)
        
        if not results_df.empty:
            plot_lombscargle_periodogram(channel, results_df, plot_data)
            results_df['channel'] = channel
            all_results_dfs.append(results_df)

    print("\n" + "="*50)
    print("  ZUSAMMENFASSUNG ALLER SIGNIFIKANTEN PERIODEN")
    print("="*50)

    if not all_results_dfs:
        print("Keine signifikanten Perioden in allen Kanälen gefunden.")
    else:
        # Schritt 1: Erstelle ein sauberes, kombiniertes DataFrame
        summary_df = pd.concat(all_results_dfs, ignore_index=True)
        
        # Sortiere die CSV-Ausgabe nach Kanal und dem neuen SNR für bessere Lesbarkeit
        output_csv_path = "significant_periods_summary_all.csv"
        summary_df.sort_values(by=['channel', 'SNR'], ascending=[True, False]).to_csv(
            output_csv_path, index=False, float_format='%.6f'
        )
        
        print(f"Analyse abgeschlossen. Eine Zusammenfassung aller lokal dominanten Perioden wurde gespeichert:\n  -> {output_csv_path}")
        print("\nZusätzlich wurden detaillierte Plots für jeden Kanal als 'lombscargle_periods_<Kanalname>.png' gespeichert.")

        # Schritt 2: Führe die globale Auswahl der besten Perioden durch (JETZT BASIEREND AUF SNR)
        print("\n" + "="*50)
        print(f"  GLOBALE AUSWAHL DER TOP {MAX_FINAL_PERIODS} PERIODEN (NACH SNR)")
        print("="*50)

        # Sortiere alle gefundenen Peaks global nach ihrem Signal-Rausch-Verhältnis (SNR)
        globally_best_df = summary_df.sort_values(by='SNR', ascending=False)

        # Wähle die besten Perioden aus, aber stelle sicher, dass sie einzigartig sind.
        # Wir runden sie, um sehr ähnliche Werte (z.B. 23.9h und 24.1h) zu gruppieren.
        final_periods_list = []
        for period in globally_best_df['Periode (Stunden)']:
            rounded_period = int(round(period))
            if rounded_period not in final_periods_list:
                final_periods_list.append(rounded_period)
            if len(final_periods_list) >= MAX_FINAL_PERIODS:
                break
        
        final_periods_list.sort()

        print("Die folgenden Perioden (in Stunden) wurden als die global stärksten (höchstes Signal-Rausch-Verhältnis) und relevantesten identifiziert.")
        print("Diese eignen sich als `seq_len` (oder `input_chunk_length`) für Ihr Modell:\n")
        print(f"Empfohlene `seq_len` Werte: {final_periods_list}")
    
    print("\nAnalyse abgeschlossen.")