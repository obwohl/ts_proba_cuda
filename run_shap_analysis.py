import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import sliding_window_view


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

def apply_plotting_style():
    """
    Wendet das duet_proba-Plotting-Theme basierend auf dem Styleguide an.
    """
    style_dict = {
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": "#231F20",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#231F20",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.labelsize": 10,
        "axes.labelweight": "normal",
        "axes.labelcolor": "#231F20",
        "axes.prop_cycle": plt.cycler(color=["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]),
        "xtick.major.size": 2,
        "xtick.minor.size": 1,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.major.top": True,
        "xtick.major.bottom": True,
        "xtick.minor.top": True,
        "xtick.minor.bottom": True,
        "xtick.color": "#231F20",
        "ytick.major.size": 2,
        "ytick.minor.size": 1,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.color": "#231F20",
        "grid.color": "#231F20",
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF",
        "figure.dpi": 150, # Erhöhe die DPI für schärfere Plots
    }
    plt.rcParams.update(style_dict)

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

def plot_instance_vs_background(instance_data, background_data, channel_names, target_channel_idx, seq_len, output_filename):
    """
    Erstellt einen Vergleichsplot zwischen der zu erklärenden Instanz und dem
    nächstgelegenen Hintergrund-Zentroiden für den Zielkanal.

    Args:
        instance_data (np.ndarray): Die zu erklärende Instanz mit Shape (1, seq_len, n_features).
        background_data (np.ndarray): Der nächstgelegene Hintergrund-Zentroid mit Shape (seq_len, n_features).
        channel_names (list): Liste der Kanalnamen.
        target_channel_idx (int): Index des Zielkanals.
        seq_len (int): Länge der Sequenz.
        output_filename (str): Dateiname für den zu speichernden Plot.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    target_channel_name = channel_names[target_channel_idx]
    
    # Extrahiere die Zeitreihen für den Zielkanal
    instance_series = instance_data[0, :, target_channel_idx]
    background_series = background_data[:, target_channel_idx]
    
    x_axis = np.arange(seq_len)
    
    # Plotte beide Zeitreihen
    ax.plot(x_axis, instance_series, label=f"Tatsächlicher Input für '{target_channel_name}'", color="#4E79A7", linewidth=2)
    ax.plot(x_axis, background_series, label=f"Nächstgelegener Hintergrund (Zentroid) für '{target_channel_name}'", color="#F28E2B", linestyle='--', linewidth=2)
    
    ax.set_title(f"Vergleich: Tatsächlicher Input vs. SHAP-Hintergrund\nfür Kanal '{target_channel_name}'")
    ax.set_xlabel("Zeitschritt im Input-Fenster")
    ax.set_ylabel("Wert")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Vergleichsplot gespeichert unter: {output_filename}")

def main():
    """
    Hauptfunktion zum Laden des Modells, der Daten und zur Ausführung der SHAP-Analyse.
    """
    # --- NEU: Wende das konsistente Plotting-Theme an ---
    apply_plotting_style()

    # --- 2. Konfiguration (vorher argparse) ---
    # Hier können die Pfade und der Zielkanal direkt im Skript festgelegt werden.
    CHECKPOINT_PATH = Path("results/optuna_heuristic/eisbach_96_studentt_mean_nll_loss_eisbach/trial_153/best_model.pt")
    DATA_FILE_PATH = Path("dataset/forecasting/combo_96.csv")
    
    # --- NEU: Wähle den Modus für die Hintergrunddaten ---
    # 'median': Konstante Median-Linie pro Kanal (schnell, aber einfach).
    # 'kmeans': Repräsentative Fenster aus den Trainingsdaten (Best Practice, langsamer).
    BACKGROUND_MODE = 'kmeans'
    TARGET_CHANNEL = "wassertemp"

    # --- NEU: Priorisiere CUDA für die Inferenz, mit CPU als Fallback ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Verwende Gerät für die Inferenz: {device}")
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

    # Wir definieren die "Trainingsdaten" als alle Daten außer der zu erklärenden Instanz.
    df_train = df_wide.iloc[:-seq_len]

    # --- Hintergrunddaten erstellen ---
    if BACKGROUND_MODE == 'median':
        print("\nErstelle Hintergrunddaten (Modus 1) basierend auf dem Kanal-Median der Trainingsdaten...")
        # Berechne den Median für jeden Kanal über die Trainingsdaten.
        channel_medians = df_train.median().values # Ergibt einen Vektor der Form (n_features,)
        # Erstelle das Hintergrund-Fenster, indem der Median-Vektor für jeden Zeitschritt wiederholt wird.
        background_data = np.full_like(instance_to_explain, channel_medians)
        print(f"  - Hintergrunddaten mit Shape {background_data.shape} erstellt.")

    elif BACKGROUND_MODE == 'kmeans':
        N_BACKGROUND_SAMPLES = 200  # Anzahl der Cluster für K-Means
        MAX_WINDOWS_FOR_KMEANS = 20000  # Teilmenge für bessere Performance
        print(f"\nErstelle Hintergrunddaten (Modus 2) basierend auf K-Means Clustering ({N_BACKGROUND_SAMPLES} Zentren)...")
        
        # Erstelle alle möglichen gleitenden Fenster aus den Trainingsdaten
        train_windows_view = sliding_window_view(df_train.values, window_shape=(seq_len, n_features))
        train_windows = np.squeeze(train_windows_view, axis=1)
        num_windows = train_windows.shape[0]
        print(f"  - {num_windows} Trainingsfenster gefunden.")

        # Wähle eine repräsentative Teilmenge aus, wenn die Daten zu groß sind
        if num_windows > MAX_WINDOWS_FOR_KMEANS:
            print(f"  - Wähle zufällig {MAX_WINDOWS_FOR_KMEANS} Fenster für K-Means aus...")
            random_indices = np.random.choice(num_windows, MAX_WINDOWS_FOR_KMEANS, replace=False)
            windows_for_kmeans = train_windows[random_indices]
        else:
            windows_for_kmeans = train_windows

        # Forme die Fenster für K-Means um (2D) und führe Clustering durch
        windows_flat = windows_for_kmeans.reshape(windows_for_kmeans.shape[0], -1)
        print("  - Führe K-Means Clustering aus (dies kann einen Moment dauern)...")
        kmeans = KMeans(n_clusters=N_BACKGROUND_SAMPLES, random_state=42, n_init='auto').fit(windows_flat)
        
        # Forme die Cluster-Zentren zurück in die Fensterform (3D). Dies sind unsere Hintergrund-Samples.
        background_data = kmeans.cluster_centers_.reshape(N_BACKGROUND_SAMPLES, seq_len, n_features).astype(np.float32)
        print(f"  - Hintergrunddaten mit Shape {background_data.shape} erstellt.")

        # --- NEU: Finde den nächstgelegenen Zentroiden und plotte den Vergleich ---
        print("\nSuche den nächstgelegenen Hintergrund-Zentroiden für die Visualisierung...")
        # Forme die Daten für die Abstandsberechnung um
        instance_flat = instance_to_explain.reshape(1, -1)
        background_flat = background_data.reshape(N_BACKGROUND_SAMPLES, -1)
        
        # Berechne die euklidischen Abstände
        distances = np.linalg.norm(background_flat - instance_flat, axis=1)
        closest_centroid_idx = np.argmin(distances)
        closest_centroid = background_data[closest_centroid_idx]
        print(f"  - Zentroid #{closest_centroid_idx} ist der nächstgelegene.")
        
        # Erstelle den Vergleichsplot
        plot_instance_vs_background(
            instance_data=instance_to_explain,
            background_data=closest_centroid,
            channel_names=channel_names,
            target_channel_idx=target_channel_idx,
            seq_len=seq_len,
            output_filename="shap_instance_vs_background.png"
        )

    # --- NEU: Analyse der Baseline-Vorhersage ---
    # Wir berechnen, was das Modell vorhersagt, wenn es nur die Hintergrunddaten als Input erhält.
    # Dies gibt uns eine "neutrale" Referenzvorhersage.
    print("\n--- Analysiere die Baseline-Vorhersage (Modell-Output für Hintergrunddaten) ---")
    # Wichtig: Wir verwenden das Originalmodell, nicht den SHAP-Wrapper,
    # da wir die Vorhersagen für ALLE Kanäle wollen.
    with torch.no_grad():
        background_tensor = torch.from_numpy(background_data).to(device)
        # Das Modell gibt eine Verteilung zurück. Wir extrahieren den Median (loc).
        dist, *_ = model(background_tensor)
        # dist.loc hat die Form [Batch, N_Vars, Horizon]
        baseline_predictions = dist.loc.cpu().numpy()

    if BACKGROUND_MODE == 'median':
        # baseline_predictions hat die Form (1, n_features, horizon).
        # Wir entfernen die Batch-Dimension und transponieren für das DataFrame.
        preds_for_df = baseline_predictions.squeeze(0).T # Shape -> (horizon, n_features)
        df_baseline = pd.DataFrame(preds_for_df, columns=channel_names)
        df_baseline.index.name = "Horizon_Step"
        output_filename = "baseline_prediction_median_background.csv"
        df_baseline.to_csv(output_filename)
        print(f"✅ Vorhersage für Median-Hintergrund gespeichert: {output_filename}")

    elif BACKGROUND_MODE == 'kmeans':
        # baseline_predictions hat die Form (n_samples, n_features, horizon).
        # Wir berechnen Statistiken über die `n_samples` Vorhersagen.
        mean_preds = np.mean(baseline_predictions, axis=0).T
        median_preds = np.median(baseline_predictions, axis=0).T
        std_preds = np.std(baseline_predictions, axis=0).T

        # Kombiniere die Statistiken in einem DataFrame mit Multi-Level-Spalten
        df_baseline_stats = pd.concat([pd.DataFrame(d, columns=channel_names) for d in [mean_preds, median_preds, std_preds]], axis=1, keys=['mean_prediction', 'median_prediction', 'std_dev_prediction'])
        df_baseline_stats.index.name = "Horizon_Step"
        output_filename = "baseline_prediction_kmeans_background.csv"
        df_baseline_stats.to_csv(output_filename)
        print(f"✅ Statistiken der Vorhersagen für K-Means-Hintergründe gespeichert: {output_filename}")

    print(f"✅ Daten vorbereitet. Shape der zu erklärenden Instanz: {instance_to_explain.shape}")

    # --- 4. SHAP-Wrapper und Explainer einrichten ---
    # Erstelle den Modell-Wrapper, der den Median für den ausgewählten Zielkanal zurückgibt
    wrapped_model = MedianForecastWrapper(model, target_channel_idx=target_channel_idx)

    # Dies ist die "Übersetzungsfunktion", die die Logik aus deinem MyTimeExplainer kapselt.
    def shap_wrapper(coalition_vector):
        # coalition_vector (z') hat die Form (num_coalitions, num_features) und besteht aus 0en und 1en.
        num_coalitions = coalition_vector.shape[0]
        num_features = coalition_vector.shape[1]

        # --- NEU: Dynamische Erstellung des Hintergrund-Batches ---
        # Diese Logik funktioniert für beide Modi (Median und K-Means).
        if background_data.shape[0] == 1:
            # Modus 1: Median. Wir haben nur ein Hintergrund-Sample.
            synth_batch = np.repeat(background_data, num_coalitions, axis=0)
        else:
            # Modus 2: K-Means. Wir haben mehrere Hintergrund-Samples (Zentren).
            # Wähle für jede Koalition zufällig ein Hintergrund-Sample aus.
            background_indices = np.random.choice(background_data.shape[0], num_coalitions, replace=True)
            synth_batch = background_data[background_indices]
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