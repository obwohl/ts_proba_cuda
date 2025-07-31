import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pandas as pd

# --- 1. Korrekte Imports (KEINE Redundanz) ---
# Die Modell- und Konfigurationsklassen werden direkt aus dem Projekt importiert.
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig

class MedianForecastWrapper(nn.Module):
    def __init__(self, model, target_channel_idx=0):
        super().__init__()
        self.model = model.eval()
        self.target_channel_idx = target_channel_idx
        # Hole das Gerät direkt vom Modell, um Konsistenz zu gewährleisten
        self.device = next(model.parameters()).device

    def forward(self, x):
        # x ist hier ein Batch von synthetischen Zeitreihen (Numpy Array)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)

        with torch.no_grad():
            # Das Modell gibt eine denormalisierte Verteilung und weitere Werte zurück
            dist, *_ = self.model(x)
            # Wir extrahieren den Median (loc) für den Zielkanal
            median = dist.loc[:, self.target_channel_idx] # Shape: [Batch, Horizon]
        
        # Gib den gesamten Horizont zurück, damit SHAP jeden Zeitschritt erklären kann
        return median.cpu().numpy() # Shape: [Batch, Horizon]

def plot_shap_over_time(shap_values, channel_names, horizon, target_channel, output_filename, plot_absolute=True):
    """
    Erstellt einen gestapelten Balken-Plot der SHAP-Werte über den Zeithorizont
    und speichert ihn als Datei.

    Args:
        shap_values (np.ndarray): Array der SHAP-Werte mit Shape (num_features, horizon).
        channel_names (list): Liste der Feature-Namen.
        horizon (int): Länge des Vorhersagehorizonts.
        target_channel (str): Name des Zielkanals für den Plot-Titel.
        output_filename (str): Dateiname für den zu speichernden Plot.
        plot_absolute (bool): Wenn True, werden absolute SHAP-Werte geplottet.
                              Wenn False, werden relative (mit Vorzeichen) Werte geplottet.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    horizon_steps = np.arange(horizon)
    num_features = shap_values.shape[0]
    
    if plot_absolute:
        # --- Plot für absolute Werte ---
        data_to_plot = np.abs(shap_values)
        bottom = np.zeros(horizon)
        for i in range(num_features):
            ax.bar(horizon_steps, data_to_plot[i, :], bottom=bottom, label=channel_names[i], width=1.0)
            bottom += data_to_plot[i, :]
        ax.set_ylabel("Absoluter SHAP-Wert (Feature Importance)")
        ax.set_title(f"Absolute Feature Importance für '{target_channel}' über den Horizont")
    else:
        # --- Plot für relative Werte (mit Vorzeichen) ---
        data_to_plot = shap_values
        bottom_positive = np.zeros(horizon)
        bottom_negative = np.zeros(horizon)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(num_features):
            values = data_to_plot[i, :]
            color = colors[i % len(colors)]
            ax.bar(horizon_steps, np.maximum(0, values), bottom=bottom_positive, color=color, label=channel_names[i], width=1.0)
            ax.bar(horizon_steps, np.minimum(0, values), bottom=bottom_negative, color=color, width=1.0)
            bottom_positive += np.maximum(0, values)
            bottom_negative += np.minimum(0, values)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("SHAP-Wert (Beitrag zur Vorhersage)")
        ax.set_title(f"Relativer Feature-Einfluss für '{target_channel}' über den Horizont")

    ax.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlabel("Zeitschritt im Vorhersagehorizont")
    ax.set_xlim(-0.5, horizon - 0.5)
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Mache Platz für die Legende außerhalb des Plots
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Plot gespeichert unter: {output_filename}")

def main():
    """
    Hauptfunktion zum Laden des Modells, der Daten und zur Ausführung der SHAP-Analyse.
    """
    # --- 2. Konfiguration (vorher argparse) ---
    # Hier können die Pfade und der Zielkanal direkt im Skript festgelegt werden.
    CHECKPOINT_PATH = Path("results/optuna_heuristic/eisbach_96_studentt_mean_nll_loss_eisbach/trial_153/best_model.pt")
    DATA_FILE_PATH = Path("dataset/forecasting/combo_96.csv")
    TARGET_CHANNEL = "wassertemp"

    device = torch.device("cpu") # SHAP läuft auf der CPU
    
    print(f"Lade Checkpoint von: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        config_dict = checkpoint['config_dict']
        model_state_dict = checkpoint['model_state_dict']
    except FileNotFoundError:
        print(f"FEHLER: Checkpoint-Datei nicht gefunden unter {CHECKPOINT_PATH}")
        return

    config = TransformerConfig(**config_dict)
    model = DUETProbModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    print("✅ Modell erfolgreich geladen.")

    # --- 3. Parameter und Daten automatisch vorbereiten ---
    print("\nLese Parameter aus Modell-Konfiguration und lade Daten...")
    seq_len = config.seq_len
    horizon = config.horizon
    channel_names = list(config.channel_bounds.keys())
    n_features = len(channel_names)
    print(f"  - Sequenzlänge (seq_len): {seq_len}")
    print(f"  - Vorhersagehorizont (horizon): {horizon}")
    print(f"  - Kanäle im Modell: {channel_names}")

    try:
        target_channel_idx = channel_names.index(TARGET_CHANNEL)
        print(f"  - Zielkanal für Erklärung: '{TARGET_CHANNEL}' (Index: {target_channel_idx})")
    except ValueError:
        print(f"❌ FEHLER: Der Zielkanal '{TARGET_CHANNEL}' wurde nicht in den Kanälen des Modells gefunden.")
        return

    # Lade die Daten und bereite sie vor
    df_long = pd.read_csv(DATA_FILE_PATH)
    df_wide = df_long.pivot(index='date', columns='cols', values='data')[channel_names]
    df_wide.ffill(inplace=True); df_wide.bfill(inplace=True)

    if len(df_wide) < seq_len:
        print(f"❌ FEHLER: Nicht genügend Daten in '{DATA_FILE_PATH}'. Benötigt: {seq_len}, Vorhanden: {len(df_wide)}")
        return
    
    # Wir nehmen die letzte mögliche Sequenz aus den Daten als die Instanz, die wir erklären wollen.
    instance_to_explain = df_wide.iloc[-seq_len:].values[np.newaxis, ...].astype(np.float32)

    # HINWEIS: Dies ist der von dir gewünschte Debugging-Schritt.
    # Für eine korrekte Analyse muss dies durch repräsentative Daten ersetzt werden,
    # z.B. eine Stichprobe von 100 Fenstern aus den Trainingsdaten.
    print("\nWARNUNG: Verwende einen Null-Vektor als Hintergrunddaten (nur zum Debuggen).")
    background_data = np.zeros_like(instance_to_explain)

    print(f"✅ Daten vorbereitet. Shape der zu erklärenden Instanz: {instance_to_explain.shape}")

    # --- 4. SHAP-Wrapper und Explainer einrichten ---
    # Erstelle den Modell-Wrapper, der den Median für den ausgewählten Zielkanal zurückgibt
    wrapped_model = MedianForecastWrapper(model, target_channel_idx=target_channel_idx)

    # Dies ist die "Übersetzungsfunktion", die die Logik aus deinem MyTimeExplainer kapselt.
    def shap_wrapper(coalition_vector):
        # coalition_vector (z') hat die Form (num_coalitions, num_features) und besteht aus 0en und 1en.
        num_coalitions = coalition_vector.shape[0]
        num_features = coalition_vector.shape[1]
        
        # Erstelle einen Batch von Hintergrund-Samples. Da unser Hintergrund nur aus einem
        # Null-Vektor besteht, wiederholen wir diesen einfach.
        # Bei echten Hintergrunddaten würde man hier zufällig sampeln.
        synth_batch = np.repeat(background_data, num_coalitions, axis=0)

        # Ersetze die Kanäle der "anwesenden" Features mit den Werten aus der Original-Instanz
        for i in range(num_coalitions):
            for j in range(num_features):
                if coalition_vector[i, j] == 1:
                    # Ersetze den gesamten Kanal j in der i-ten synthetischen Probe
                    synth_batch[i, :, j] = instance_to_explain[0, :, j]
        
        # Führe das Modell auf dem Batch synthetischer Daten aus
        return wrapped_model.forward(synth_batch)

    # Der KernelExplainer erwartet einen 2D-Hintergrund, um die Anzahl der Features zu kennen.
    # Da unser Wrapper den Hintergrund bereits selbst verwaltet, können wir einen Dummy übergeben.
    background_summary = np.zeros((1, n_features))
    
    # --- KORREKTUR FÜR ALTE SHAP-VERSION: Verwende den direkten `KernelExplainer` ---
    # Deine `shap`-Version ist veraltet und kennt die moderne `shap.Explainer`-Klasse nicht.
    # Wir müssen den `KernelExplainer` direkt aufrufen, wie es in älteren Versionen üblich war.
    explainer = shap.KernelExplainer(shap_wrapper, background_summary)

    # Die zu erklärende Instanz für SHAP ist ein Vektor aus 1en, der sagt: "Alle Features sind anwesend".
    instance_for_shap = np.ones((1, n_features))

    print("\nBerechne SHAP-Werte (dies kann eine Weile dauern)...")
    # Die alte API verwendet die `.shap_values()`-Methode und gibt eine Liste von Arrays zurück
    # (ein Array für jeden Zeitschritt im Horizont).
    shap_values = explainer.shap_values(instance_for_shap, nsamples='auto')
    # Die Ausgabe ist eine Liste, die ein Array der Form (num_features, num_outputs) enthält.
    print(f"✅ Berechnung abgeschlossen. SHAP-Werte-Struktur: Liste der Länge {len(shap_values)}, inneres Array hat Shape {shap_values[0].shape}")

    # --- 5. Ergebnisse anzeigen ---
    # KORREKTUR: Wir müssen die SHAP-Werte korrekt mitteln.
    # `shap_values` ist eine Liste, die ein Array der Form (num_features, horizon_outputs) enthält.
    # Wir nehmen das erste Element, das ist das Array, das wir brauchen.
    shap_values_array = shap_values[0]  # Shape: (10, 96)

    # KORREKTUR 2: `summary_plot` erwartet eine Matrix (num_samples, num_features).
    # Unsere SHAP-Werte haben die Form (num_features, horizon_steps).
    # Wir behandeln die Zeitschritte des Horizonts als "Samples" und transponieren
    # das Array, um die erwartete Form (horizon_steps, num_features) zu erhalten.
    shap_values_for_plot = shap_values_array.T  # Shape wird zu (96, 10)

    # --- Plot 1: Durchschnittliche Wichtigkeit (bestehend, jetzt gespeichert) ---
    print("\nVisualisiere die durchschnittliche Feature-Wichtigkeit über den gesamten Horizont...")
    shap.summary_plot(
        shap_values_for_plot,  # Übergib die transponierte Matrix
        feature_names=channel_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"Durchschnittliche Feature Importance für '{TARGET_CHANNEL}'\n(gemittelt über {horizon} Vorhersageschritte)")
    plt.xlabel("mittlerer |SHAP-Wert| (Einfluss auf die Modellausgabe)")
    plt.tight_layout()
    summary_plot_filename = "shap_summary_plot.png"
    plt.savefig(summary_plot_filename)
    plt.close()
    print(f"✅ Plot gespeichert unter: {summary_plot_filename}")

    # --- Plot 2: Absolute Wichtigkeit über die Zeit ---
    plot_shap_over_time(
        shap_values=shap_values_array,
        channel_names=channel_names,
        horizon=horizon,
        target_channel=TARGET_CHANNEL,
        output_filename="shap_absolute_importance_over_time.png",
        plot_absolute=True
    )

    # --- Plot 3: Relative Wichtigkeit über die Zeit ---
    plot_shap_over_time(
        shap_values=shap_values_array,
        channel_names=channel_names,
        horizon=horizon,
        target_channel=TARGET_CHANNEL,
        output_filename="shap_relative_importance_over_time.png",
        plot_absolute=False
    )

if __name__ == "__main__":
    main()