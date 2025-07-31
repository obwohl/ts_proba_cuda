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

def plot_instance_vs_background(instance_data, background_data, prediction_for_instance, prediction_for_background, channel_names, target_channel_idx, seq_len, horizon, output_filename):
    """
    Erstellt einen erweiterten Vergleichsplot: Input und Vorhersage für die
    tatsächliche Instanz vs. den nächstgelegenen Hintergrund-Zentroiden.

    Args:
        instance_data (np.ndarray): Die zu erklärende Instanz mit Shape (1, seq_len, n_features).
        background_data (np.ndarray): Der nächstgelegene Hintergrund-Zentroid mit Shape (seq_len, n_features).
        prediction_for_instance (np.ndarray): Vorhersage für die Instanz, Shape (1, n_features, horizon).
        prediction_for_background (np.ndarray): Vorhersage für den Hintergrund, Shape (1, n_features, horizon).
        channel_names (list): Liste der Kanalnamen.
        target_channel_idx (int): Index des Zielkanals.
        seq_len (int): Länge der Sequenz.
        horizon (int): Länge der Vorhersage-Sequenz.
        output_filename (str): Dateiname für den zu speichernden Plot.
    """
    fig, ax = plt.subplots(figsize=(20, 7))
    
    target_channel_name = channel_names[target_channel_idx]
    
    # --- 1. Daten extrahieren ---
    # Inputs
    instance_series = instance_data[0, :, target_channel_idx]
    background_series = background_data[:, target_channel_idx]
    # Predictions
    instance_prediction_series = prediction_for_instance[0, target_channel_idx, :]
    background_prediction_series = prediction_for_background[0, target_channel_idx, :]
    
    # --- 2. Zeitachsen definieren ---
    x_axis_input = np.arange(seq_len)
    x_axis_prediction = np.arange(seq_len, seq_len + horizon)
    
    # --- 3. Plotten ---
    ax.plot(x_axis_input, instance_series, label=f"Tatsächlicher Input ('{target_channel_name}')", color="#4E79A7", linewidth=2)
    ax.plot(x_axis_input, background_series, label=f"Hintergrund-Input (Zentroid)", color="#F28E2B", linestyle='--', linewidth=2)
    ax.plot(x_axis_prediction, instance_prediction_series, label=f"Tatsächliche Vorhersage", color="#4E79A7", linewidth=2, marker='.')
    ax.plot(x_axis_prediction, background_prediction_series, label=f"Hintergrund-Vorhersage (Baseline)", color="#F28E2B", linestyle='--', linewidth=2, marker='.')
    ax.axvline(x=seq_len - 0.5, color='black', linestyle=':', linewidth=1.5, label='Forecast Start')
    ax.set_title(f"Vergleich: Input & Vorhersage (Instanz vs. SHAP-Hintergrund)\nfür Kanal '{target_channel_name}'")
    ax.set_xlabel("Zeitschritt (Input-Fenster | Vorhersage-Horizont)")
    ax.set_ylabel("Wert")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Vergleichsplot gespeichert unter: {output_filename}")

def plot_shap_additivity_check(prediction_for_instance_target, shap_base_value, shap_values_sum, horizon, output_filename):
    """
    Erstellt einen Plot, um die Additivitätseigenschaft von SHAP zu verifizieren.
    Vergleicht die Lücke zwischen Vorhersage und Baseline mit der Summe der SHAP-Werte.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Berechne die Lücke basierend auf der übergebenen Baseline
    prediction_gap = prediction_for_instance_target - shap_base_value

    x_axis = np.arange(horizon)
    
    ax.plot(x_axis, prediction_gap, label='Lücke (Vorhersage - Baseline)', color="#4E79A7", linewidth=2.5)
    ax.plot(x_axis, shap_values_sum, label='Summe aller SHAP-Werte', color="#F28E2B", linestyle='--', linewidth=2.5)
    
    difference = np.mean(np.abs(prediction_for_instance_target - (shap_base_value + shap_values_sum)))
    
    ax.set_title(f"SHAP Additivitäts-Check\n(Mittlere absolute Abweichung: {difference:.4g})")
    ax.set_xlabel("Zeitschritt im Vorhersage-Horizont")
    ax.set_ylabel("Wert")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Additivitäts-Check Plot gespeichert unter: {output_filename}")

def plot_shap_waterfall(actual_prediction, base_value, shap_values, channel_names, horizon, output_filename):
    """
    Erstellt einen "Wasserfall"-Plot, der zeigt, wie die SHAP-Werte die Vorhersage
    von der Baseline zur tatsächlichen Vorhersage "schieben".

    Args:
        actual_prediction (np.ndarray): Die tatsächliche Vorhersage (shape: [horizon,]).
        base_value (np.ndarray): Die Basis-Vorhersage von SHAP (shape: [horizon,]).
        shap_values (np.ndarray): Die SHAP-Werte (shape: [num_features, horizon]).
        channel_names (list): Liste der Kanalnamen.
        horizon (int): Länge des Vorhersagehorizonts.
        output_filename (str): Dateiname für den zu speichernden Plot.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    x_axis = np.arange(horizon)

    # Starte den "Wasserfall" bei der Basis-Vorhersage
    current_bottom = base_value.copy()

    # Iteriere durch jedes Feature und füge seinen Beitrag als gefüllte Fläche hinzu
    for i in range(len(channel_names)):
        values = shap_values[i, :]
        ax.fill_between(x_axis, current_bottom, current_bottom + values, label=f'{channel_names[i]} (Beitrag)', step='post')
        current_bottom += values

    # Plotte die Start- und Endpunkte zur Verdeutlichung
    ax.plot(x_axis, base_value, color='black', linestyle='--', linewidth=2, label='SHAP Baseline (expected_value)', zorder=10)
    ax.plot(x_axis, actual_prediction, color='red', linestyle='-', linewidth=2.5, label='Tatsächliche Vorhersage', zorder=10)

    ax.set_title("SHAP Erklärung: Wie die Feature-Beiträge die Vorhersage formen", fontsize=16)
    ax.set_xlabel("Zeitschritt im Vorhersage-Horizont")
    ax.set_ylabel("Vorhergesagter Wert")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ SHAP-Wasserfall-Plot gespeichert unter: {output_filename}")

def plot_shap_force_over_time(actual_prediction, base_value, shap_values, channel_names, horizon, output_filename):
    """
    Erstellt einen "Force Plot" über den Zeithorizont, der positive und negative
    SHAP-Beiträge getrennt voneinander visualisiert.

    Args:
        actual_prediction (np.ndarray): Die tatsächliche Vorhersage (shape: [horizon,]).
        base_value (np.ndarray): Die Basis-Vorhersage von SHAP (shape: [horizon,]).
        shap_values (np.ndarray): Die SHAP-Werte (shape: [num_features, horizon]).
        channel_names (list): Liste der Kanalnamen.
        horizon (int): Länge des Vorhersagehorizonts.
        output_filename (str): Dateiname für den zu speichernden Plot.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    x_axis = np.arange(horizon)

    # Initialisiere die "Böden" für positive und negative Stapel bei der Baseline
    bottom_positive = base_value.copy()
    bottom_negative = base_value.copy()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(len(channel_names)):
        values = shap_values[i, :]
        color = colors[i % len(colors)]
        # Positive Beiträge werden auf den positiven Stapel addiert
        ax.fill_between(x_axis, bottom_positive, bottom_positive + np.maximum(0, values), color=color, step='post', label=f'{channel_names[i]} (Beitrag)')
        bottom_positive += np.maximum(0, values)
        # Negative Beiträge werden vom negativen Stapel abgezogen
        ax.fill_between(x_axis, bottom_negative, bottom_negative + np.minimum(0, values), color=color, step='post')
        bottom_negative += np.minimum(0, values)

    ax.plot(x_axis, base_value, color='black', linestyle='--', linewidth=2, label='SHAP Baseline (expected_value)', zorder=10)
    ax.plot(x_axis, actual_prediction, color='red', linestyle='-', linewidth=2.5, label='Tatsächliche Vorhersage', zorder=10)

    ax.set_title("SHAP Force Plot: Positive vs. Negative Feature-Beiträge", fontsize=16)
    ax.set_xlabel("Zeitschritt im Vorhersage-Horizont")
    ax.set_ylabel("Vorhergesagter Wert")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ SHAP-Force-Plot gespeichert unter: {output_filename}")

def create_event_shap_wrapper(model_wrapper, instance_to_explain, background_data):
    """
    Erstellt eine Wrapper-Funktion für Event-Level-SHAP-Erklärungen.

    Diese Funktion gibt eine neue Funktion zurück, die von shap.KernelExplainer
    verwendet werden kann, um die Wichtigkeit von Zeitschritten (Events) zu bewerten.

    Args:
        model_wrapper (callable): Der bereits existierende Modell-Wrapper, der einen
                                  Batch von Sequenzen (N, L, C) entgegennimmt und
                                  Vorhersagen (N, H) zurückgibt.
        instance_to_explain (np.ndarray): Die zu erklärende Instanz mit Shape (1, seq_len, n_features).
        background_data (np.ndarray): Die Hintergrund-Daten mit Shape (n_samples, seq_len, n_features).

    Returns:
        callable: Eine Funktion, die einen Koalitionsvektor entgegennimmt und
                  Modellvorhersagen für die daraus erstellten synthetischen
                  Daten zurückgibt.
    """
    seq_len = instance_to_explain.shape[1]
    n_background_samples = background_data.shape[0]

    def event_shap_wrapper(coalition_vector):
        """
        Diese Funktion wird vom SHAP-Explainer aufgerufen.

        Args:
            coalition_vector (np.ndarray): Ein 2D-Array der Form (num_coalitions, seq_len)
                                           mit 0en und 1en. Eine 1 an Position (i, t)
                                           bedeutet, dass für die i-te Koalition der
                                           Zeitschritt t "anwesend" ist.
        """
        num_coalitions = coalition_vector.shape[0]

        # 1. Erstelle einen Batch von Hintergrund-Samples.
        # Wähle für jede zu testende Koalition zufällig ein Hintergrund-Fenster aus.
        background_indices = np.random.choice(n_background_samples, num_coalitions, replace=True)
        synth_batch = background_data[background_indices].copy() # .copy() ist wichtig!

        # 2. Perturbiere den Batch basierend auf dem Koalitionsvektor.
        # Iteriere durch jede synthetische Probe im Batch.
        for i in range(num_coalitions):
            # Finde die Indizes der "anwesenden" Zeitschritte für diese Koalition.
            present_timesteps = np.where(coalition_vector[i] == 1)[0]

            # Wenn Zeitschritte anwesend sind, ersetze sie mit den Werten
            # aus der Original-Instanz.
            if len(present_timesteps) > 0:
                # Dies ist der entscheidende Schritt: Wir ersetzen ganze Zeit-Scheiben.
                # synth_batch[i, t, :] wird durch instance_to_explain[0, t, :] ersetzt.
                synth_batch[i, present_timesteps, :] = instance_to_explain[0, present_timesteps, :]

        # 3. Führe das Modell auf dem synthetischen Batch aus.
        return model_wrapper.forward(synth_batch)

    return event_shap_wrapper

def plot_event_importance(instance_data, event_shap_values, target_channel_idx, channel_names, output_filename):
    """
    Visualisiert die Wichtigkeit von Zeitschritten (Events) für die Vorhersage.
    """
    seq_len = instance_data.shape[1]
    target_channel_name = channel_names[target_channel_idx]
    mean_abs_shap_per_step = np.mean(np.abs(event_shap_values), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    x_axis = np.arange(seq_len)
    ax1.plot(x_axis, instance_data[0, :, target_channel_idx], color="#4E79A7", label=f"Input-Verlauf '{target_channel_name}'")
    ax1.set_title(f"Event (Zeitschritt) Wichtigkeit für die Vorhersage von '{target_channel_name}'")
    ax1.set_ylabel("Wert")
    ax1.legend(loc="upper left")

    normalized_importance = (mean_abs_shap_per_step - mean_abs_shap_per_step.min()) / (mean_abs_shap_per_step.max() - mean_abs_shap_per_step.min())
    colors = plt.cm.viridis(normalized_importance)
    ax2.bar(x_axis, np.ones(seq_len), color=colors, width=1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("Zeitschritt im Input-Fenster")
    ax2.set_ylabel("Wichtigkeit", rotation=0, ha='right', va='center', labelpad=40)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=mean_abs_shap_per_step.min(), vmax=mean_abs_shap_per_step.max()))
    cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.1, pad=0.02)
    cbar.set_label('Mittlerer |SHAP-Wert|')

    plt.tight_layout(pad=0.1, h_pad=1.5)
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Event-Wichtigkeits-Plot gespeichert unter: {output_filename}")

def plot_all_channels_colored_by_importance(instance_data, event_shap_values, target_channel_name, channel_names, output_filename):
    """
    Erstellt einen Multi-Panel-Plot, der für jeden Input-Kanal den Zeitverlauf
    anzeigt, eingefärbt nach der Event-Wichtigkeit auf einer logarithmischen Skala.
    """
    # Importiere die notwendigen Module für den Plot
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    n_features = instance_data.shape[2]
    seq_len = instance_data.shape[1]

    # Erstelle eine Figur mit so vielen Subplots wie es Kanäle gibt
    fig, axes = plt.subplots(n_features, 1, figsize=(20, 5 * n_features), sharex=True, constrained_layout=True)
    fig.suptitle(f"Einfluss aller Input-Kanäle auf die Vorhersage von '{target_channel_name}'\n(Linienfarbe zeigt Wichtigkeit des Zeitpunkts)", fontsize=18, weight='bold')

    # 1. Berechne die Wichtigkeit EINMAL für alle Plots
    importance_per_step = np.mean(np.abs(event_shap_values), axis=1)

    # 2. Logarithmische Normalisierung, um kleine Unterschiede sichtbar zu machen
    # Füge eine winzige Konstante hinzu, um log(0) zu vermeiden.
    epsilon = 1e-10
    norm = LogNorm(vmin=np.maximum(importance_per_step.min(), epsilon), vmax=importance_per_step.max())
    cmap = plt.cm.viridis

    # Iteriere durch jeden Kanal und erstelle einen eigenen Subplot
    for i, ax in enumerate(axes):
        series_data = instance_data[0, :, i]
        x_axis = np.arange(seq_len)

        # Erstelle die Liniensegmente für die LineCollection
        points = np.array([x_axis, series_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Erstelle die LineCollection mit dickerer Linie
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        segment_colors = (importance_per_step[:-1] + importance_per_step[1:]) / 2
        lc.set_array(segment_colors + epsilon) # Füge auch hier epsilon hinzu
        lc.set_linewidth(3.0) # <-- DICKERE LINIE

        ax.add_collection(lc)
        ax.set_xlim(x_axis.min(), x_axis.max())
        ax.set_ylim(series_data.min() - 0.1 * np.ptp(series_data), series_data.max() + 0.1 * np.ptp(series_data))
        ax.set_title(f"Input-Verlauf: '{channel_names[i]}'", fontsize=12)
        ax.set_ylabel("Wert")

    # WUNSCH: Die Farbleiste wird entfernt, da sie als redundant empfunden wird.
    # Die Intuition "Gelb = wichtig" ist ausreichend.
    # fig.colorbar(lc, ax=axes.ravel().tolist(), label="Mittlerer |SHAP-Wert| (Log-Skala)", pad=0.01)
    axes[-1].set_xlabel("Zeitschritt im Input-Fenster") # Nur die unterste x-Achse beschriften
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Farbiger Multi-Kanal Zeitreihen-Plot gespeichert unter: {output_filename}")

def plot_all_channels_with_importance_bars(instance_data, event_shap_values, target_channel_name, channel_names, output_filename):
    """
    Erstellt einen Multi-Panel-Plot, der für jeden Kanal die Zeitreihe (schwarze Linie)
    und den gerichteten SHAP-Beitrag (farbige Balken) auf einer zweiten Y-Achse darstellt.
    Dies ist die ultimative Visualisierung, die den Verlauf mit seinem Einfluss kombiniert.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_features = instance_data.shape[2]
    seq_len = instance_data.shape[1]

    fig, axes = plt.subplots(n_features, 1, figsize=(20, 6 * n_features), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Zeitreihenverlauf vs. SHAP-Beitrag zur Vorhersage von '{target_channel_name}'",
        fontsize=18, weight='bold'
    )

    # 1. Berechne die gerichtete Wichtigkeit (mittlerer SHAP-Wert) EINMAL für alle Plots
    directional_importance = np.mean(event_shap_values, axis=1)
    x_axis = np.arange(seq_len)

    # 2. Definiere die Farben für die Balken EINMAL
    bar_colors = ['#E15759' if val > 0 else '#4E79A7' for val in directional_importance]

    # 3. Bestimme die Y-Achsen-Grenzen für die SHAP-Balken EINMAL, um Konsistenz zu gewährleisten
    max_abs_shap = np.max(np.abs(directional_importance)) * 1.15 # 15% Puffer
    shap_ylim = (-max_abs_shap, max_abs_shap) if max_abs_shap > 0 else (-1, 1)

    # Iteriere durch jeden Kanal und erstelle einen eigenen Subplot
    for i, ax in enumerate(axes):
        # --- Linke Y-Achse: Zeitreihenwert ---
        series_data = instance_data[0, :, i]
        ax.plot(x_axis, series_data, color='black', linewidth=1.5, zorder=10)
        ax.set_ylabel(f"Wert '{channel_names[i]}'", color='black', fontsize=12)
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_title(f"Input-Kanal: '{channel_names[i]}'", fontsize=14)
        # Sorge für etwas Platz
        ax.set_ylim(series_data.min() - 0.1 * np.ptp(series_data), series_data.max() + 0.1 * np.ptp(series_data))
    """
    Erstellt ein Balkendiagramm, das die gerichtete Wichtigkeit (mittlerer SHAP-Wert)
    jedes Input-Zeitschritts quantitativ darstellt, um die Beobachtungen aus dem
    Hintergrund-Plot zu verifizieren.
    """
    seq_len = len(directional_importance)
    x_axis = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(20, 7))

    # Verwende die Farben aus dem Plotting-Style für Konsistenz
    # Rot (positiv) und Blau (negativ)
    colors = ['#E15759' if val > 0 else '#4E79A7' for val in directional_importance]

    ax.bar(x_axis, directional_importance, color=colors, width=1.0)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"Quantitative Event-Wichtigkeit für '{target_channel_name}'\n(Mittlerer SHAP-Wert über den gesamten Horizont pro Input-Zeitschritt)")
    ax.set_xlabel("Zeitschritt im Input-Fenster")
    ax.set_ylabel("Mittlerer SHAP-Wert (Beitrag zur Vorhersage)")
    ax.set_xlim(-0.5, seq_len - 0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E15759', label='Positiver Beitrag (hebt Vorhersage an)'),
        Patch(facecolor='#4E79A7', label='Negativer Beitrag (senkt Vorhersage ab)')]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Quantitativer Wichtigkeits-Plot gespeichert unter: {output_filename}")

def plot_event_saliency_with_prediction(instance_data, prediction_data, event_shap_values, target_channel_idx, channel_names, output_filename):
    """
    Erstellt eine Saliency-Map-Visualisierung, die sich auf die 2D-Heatmap konzentriert.
    Der Kontext-Plot wurde entfernt, um Irreführung zu vermeiden.
    """
    seq_len = instance_data.shape[1]
    horizon = prediction_data.shape[0]
    target_channel_name = channel_names[target_channel_idx]

    # Nur noch ein einzelner Plot, keine geteilte Achse mehr.
    fig, ax = plt.subplots(figsize=(20, 8))

    # --- Die 2D Saliency Map (Input-Zeit vs. Output-Zeit) ---
    # VERBESSERUNG: Wir plotten die absoluten SHAP-Werte mit 'viridis', um die
    # Magnitude der Wichtigkeit hervorzuheben, was leichter zu interpretieren ist.
    saliency_map = np.abs(event_shap_values)
    im = ax.imshow(saliency_map.T, aspect='auto', cmap='viridis', interpolation='nearest')

    # Titel direkt auf die Achse setzen
    ax.set_title(f"Saliency Map für '{target_channel_name}': Einfluss von Input-Zeit (X) auf Output-Zeit (Y)", fontsize=16)
    ax.set_xlabel("Zeitschritt im Input-Fenster (t_input)")
    ax.set_ylabel("Zeitschritt im Vorhersage-Horizont (t_pred)")
    fig.colorbar(im, ax=ax, label="Absoluter SHAP-Wert (|Einfluss|)")

    # Die X-Achse muss nur noch die Input-Länge umfassen
    ax.set_xlim(-0.5, seq_len - 0.5)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Saliency-Map-Plot gespeichert unter: {output_filename}")

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
    # --- NEU: Konfigurierbare Anzahl an Samples für Event-SHAP ---
    # 'auto' ist schnell, kann aber zu instabilen Ergebnissen führen.
    # Ein höherer Wert (z.B. 2048) ist langsamer, aber deutlich robuster.
    EVENT_SHAP_NSAMPLES = 2048
    TARGET_CHANNEL = "wassertemp" # <-- HIER ÄNDERN, um die Vorhersage für Lufttemperatur zu erklären

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
        N_BACKGROUND_SAMPLES = 50  # Anzahl der Cluster für K-Means
        MAX_WINDOWS_FOR_KMEANS = 10000  # Teilmenge für bessere Performance
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

        # --- NEU: Finde den nächstgelegenen Zentroiden und berechne Vorhersagen für den Plot ---
        print("\nSuche den nächstgelegenen Hintergrund-Zentroiden für die Visualisierung...")
        instance_flat = instance_to_explain.reshape(1, -1)
        background_flat = background_data.reshape(N_BACKGROUND_SAMPLES, -1)
        distances = np.linalg.norm(background_flat - instance_flat, axis=1)
        closest_centroid_idx = np.argmin(distances)
        closest_centroid = background_data[closest_centroid_idx]
        print(f"  - Zentroid #{closest_centroid_idx} ist der nächstgelegene.")
        
        print("\nBerechne Vorhersagen für den Vergleichsplot...")
        with torch.no_grad():
            # 1. Vorhersage für die tatsächliche Instanz
            instance_tensor = torch.from_numpy(instance_to_explain).to(device)
            dist_instance, *_ = model(instance_tensor)
            prediction_for_instance = dist_instance.loc.cpu().numpy()
            # 2. Vorhersage für den nächstgelegenen Hintergrund-Zentroid
            closest_centroid_batch = closest_centroid[np.newaxis, ...].astype(np.float32)
            background_tensor = torch.from_numpy(closest_centroid_batch).to(device)
            dist_background, *_ = model(background_tensor)
            prediction_for_background = dist_background.loc.cpu().numpy()
        print("  - Vorhersagen berechnet.")
        
        plot_instance_vs_background(
            instance_data=instance_to_explain,
            background_data=closest_centroid,
            prediction_for_instance=prediction_for_instance,
            prediction_for_background=prediction_for_background,
            channel_names=channel_names,
            target_channel_idx=target_channel_idx,
            seq_len=seq_len,
            horizon=horizon,
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

    # --- NEU: Plot 4: Additivitäts-Check ---
    print("\nÜberprüfe die Additivitätseigenschaft von SHAP...")
    # 1. Hole die tatsächliche Vorhersage für die zu erklärende Instanz
    # prediction_for_instance hat die Form (1, n_features, horizon)
    prediction_for_instance_target = prediction_for_instance[0, target_channel_idx, :] # Shape: (horizon,)

    # 2. Berechne die Summe der SHAP-Werte über alle Features für jeden Zeitschritt
    # shap_values_array hat die Form (n_features, horizon)
    shap_values_sum = np.sum(shap_values_array, axis=0) # Shape: (horizon,)

    # 3. Hole die von SHAP intern berechnete Baseline.
    # `explainer.expected_value` ist ein Array mit der Länge des Horizonts.
    # Dies ist der y-Achsenabschnitt des von SHAP angenäherten Modells.
    shap_internal_baseline = explainer.expected_value
    print(f"  - SHAP interne Baseline (explainer.expected_value) hat die Form: {shap_internal_baseline.shape}")

    # 4. Erstelle den Verifikations-Plot mit der internen Baseline von SHAP.
    # Die Lücke sollte jetzt fast Null sein.
    plot_shap_additivity_check(
        prediction_for_instance_target=prediction_for_instance_target,
        shap_base_value=shap_internal_baseline,
        shap_values_sum=shap_values_sum,
        horizon=horizon,
        output_filename="shap_additivity_check.png"
    )

    # --- NEU: Plot 5: Der erklärende Wasserfall-Plot ---
    print("\nErstelle den SHAP-Wasserfall-Plot, um die Vorhersage zu erklären...")
    plot_shap_waterfall(
        actual_prediction=prediction_for_instance_target,
        base_value=shap_internal_baseline,
        shap_values=shap_values_array,
        channel_names=channel_names,
        horizon=horizon,
        output_filename="shap_waterfall_explanation.png"
    )

    # --- NEU: Plot 6: Der detaillierte Force-Plot über die Zeit ---
    print("\nErstelle den detaillierten SHAP-Force-Plot...")
    plot_shap_force_over_time(
        actual_prediction=prediction_for_instance_target,
        base_value=shap_internal_baseline,
        shap_values=shap_values_array,
        channel_names=channel_names,
        horizon=horizon,
        output_filename="shap_force_plot_over_time.png"
    )

    # --- NEUE SEKTION: Event-Level SHAP Analyse ---
    print("\n" + "="*25 + " EVENT-LEVEL SHAP ANALYSIS " + "="*25)

    # 1. Erstelle den spezifischen Wrapper für Event-Erklärungen
    event_wrapper_fn = create_event_shap_wrapper(
        model_wrapper=wrapped_model,
        instance_to_explain=instance_to_explain,
        background_data=background_data
    )

    # 2. Bereite die Inputs für den KernelExplainer vor (angepasst für Events)
    event_background_summary = np.zeros((1, seq_len))
    event_instance_for_shap = np.ones((1, seq_len))

    # 3. Initialisiere und führe den Explainer aus
    print("\nInitialisiere KernelExplainer für Events...")
    event_explainer = shap.KernelExplainer(event_wrapper_fn, event_background_summary)

    print("Berechne Event-SHAP-Werte (dies kann eine Weile dauern)...")
    # Die resultierenden SHAP-Werte haben die Form (seq_len, horizon)
    event_shap_values = event_explainer.shap_values(event_instance_for_shap, nsamples=EVENT_SHAP_NSAMPLES)[0]
    print(f"✅ Berechnung abgeschlossen. Event-SHAP-Werte haben Shape: {event_shap_values.shape}")

    # 4. Visualisiere die Event-Wichtigkeit
    print("\nVisualisiere Event-Wichtigkeit...")
    plot_event_importance(
        instance_data=instance_to_explain,
        event_shap_values=event_shap_values,
        target_channel_idx=target_channel_idx,
        channel_names=channel_names,
        output_filename="shap_event_importance.png"
    )

    # 5. Visualisiere die Saliency Map
    print("\nVisualisiere die detaillierte Saliency Map...")
    plot_event_saliency_with_prediction(
        instance_data=instance_to_explain,
        prediction_data=prediction_for_instance[0, target_channel_idx, :], # Die Vorhersage für den Zielkanal
        event_shap_values=event_shap_values,
        target_channel_idx=target_channel_idx,
        channel_names=channel_names,
        output_filename="shap_saliency_map.png"
    )

    # 6. Visualisiere den farbigen Zeitreihen-Plot (Magnitude der Wichtigkeit)
    print("\nVisualisiere den farbigen Multi-Kanal Zeitreihen-Plot...")
    plot_all_channels_colored_by_importance(
        instance_data=instance_to_explain,
        event_shap_values=event_shap_values,
        target_channel_name=TARGET_CHANNEL,
        channel_names=channel_names,
        output_filename="shap_colored_multichannel_importance.png"
    )

    # 7. NEU: Visualisiere den Zeitreihen-Plot mit eingefärbtem Hintergrund
    print("\nVisualisiere den Zeitreihen-Plot mit Wichtigkeits-Hintergrund (SymLog)...")
    plot_all_channels_with_background_importance(
        instance_data=instance_to_explain,
        event_shap_values=event_shap_values,
        target_channel_name=TARGET_CHANNEL,
        channel_names=channel_names,
        output_filename="shap_background_importance.png"
    )

    # 8. NEU: Quantitativer Plot der gerichteten Wichtigkeit als Balkendiagramm
    print("\nErstelle quantitativen Plot der gerichteten Wichtigkeit zur Verifikation...")
    # Die gerichtete Wichtigkeit ist der mittlere SHAP-Wert pro Input-Zeitschritt über den gesamten Horizont.
    # Wir berechnen sie hier, um sie an die neue Plot-Funktion zu übergeben.
    directional_importance = np.mean(event_shap_values, axis=1)
    plot_directional_importance_bars(
        directional_importance=directional_importance,
        target_channel_name=TARGET_CHANNEL,
        output_filename="shap_directional_importance_bars.png"
    )

if __name__ == "__main__":
    main()