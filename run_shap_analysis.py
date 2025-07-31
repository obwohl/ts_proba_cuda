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

    # 2. Bestimme die Y-Achsen-Grenzen für die SHAP-Balken EINMAL, um Konsistenz zu gewährleisten
    max_abs_shap = np.max(np.abs(directional_importance)) * 1.15 # 15% Puffer
    shap_ylim = (-max_abs_shap, max_abs_shap) if max_abs_shap > 0 else (-1, 1)

    # Iteriere durch jeden Kanal und erstelle einen eigenen Subplot
    for i, ax in enumerate(axes):
        # --- Linke Y-Achse: Zeitreihenwert ---
        series_data = instance_data[0, :, i]
        ax.plot(x_axis, series_data, color='black', linewidth=1.5, zorder=10, label=f"Wert '{channel_names[i]}'")
        ax.set_ylabel(f"Wert '{channel_names[i]}'", color='black', fontsize=12)
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_title(f"Input-Kanal: '{channel_names[i]}'", fontsize=14)
        y_min, y_max = series_data.min(), series_data.max()
        padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)

        # --- KORRIGIERT: Rechte Y-Achse für SHAP-Beiträge ---
        ax2 = ax.twinx()
        
        # Zeichne positive und negative Balken getrennt, um die korrekte Darstellung zu erzwingen
        ax2.bar(x_axis, np.maximum(0, directional_importance), color='#E15759', width=1.0, alpha=0.7)
        ax2.bar(x_axis, np.minimum(0, directional_importance), color='#4E79A7', width=1.0, alpha=0.7)
        
        # Verwende eine neutrale Farbe für die Achsenbeschriftung, um Verwirrung zu vermeiden
        ax2.set_ylabel("SHAP Beitrag", color='#555555', fontsize=12)
        ax2.tick_params(axis='y', colors='#555555')
        
        ax2.set_ylim(shap_ylim)
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax2.grid(False)

    # --- Globale Legende für die gesamte Figur ---
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Zeitreihen-Verlauf'),
        Patch(facecolor='#E15759', alpha=0.7, label='Positiver SHAP-Beitrag (hebt Vorhersage an)'),
        Patch(facecolor='#4E79A7', alpha=0.7, label='Negativer SHAP-Beitrag (senkt Vorhersage ab)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fancybox=True, shadow=True, borderaxespad=2)
    
    axes[-1].set_xlabel("Zeitschritt im Input-Fenster", fontsize=12)
    fig.subplots_adjust(bottom=0.1)

    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ Kombinierter Plot mit dualer Y-Achse (korrigiert) gespeichert unter: {output_filename}")


def main():
    """
    Hauptfunktion zum Laden des Modells, der Daten und zur Ausführung der SHAP-Analyse.
    """
    # --- Wende das konsistente Plotting-Theme an ---
    apply_plotting_style()

    # --- Konfiguration ---
    CHECKPOINT_PATH = Path("results/optuna_heuristic/eisbach_96_studentt_mean_nll_loss_eisbach/trial_153/best_model.pt")
    DATA_FILE_PATH = Path("dataset/forecasting/combo_96.csv")
    BACKGROUND_MODE = 'kmeans'
    EVENT_SHAP_NSAMPLES = 2048
    TARGET_CHANNEL = "wassertemp"

    # --- Inferenz-Gerät wählen ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Verwende Gerät für die Inferenz: {device}")
    
    # --- Modell laden ---
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

    # --- Parameter und Daten vorbereiten ---
    print("\nLese Parameter aus Modell-Konfiguration und lade Daten...")
    seq_len = config.seq_len
    horizon = config.horizon
    channel_names = list(config.channel_bounds.keys())
    n_features = len(channel_names)
    
    try:
        target_channel_idx = channel_names.index(TARGET_CHANNEL)
    except ValueError:
        print(f"❌ FEHLER: Der Zielkanal '{TARGET_CHANNEL}' wurde nicht in den Kanälen des Modells gefunden.")
        return

    df_long = pd.read_csv(DATA_FILE_PATH)
    df_wide = df_long.pivot(index='date', columns='cols', values='data')[channel_names]
    df_wide.ffill(inplace=True); df_wide.bfill(inplace=True)

    if len(df_wide) < seq_len:
        print(f"❌ FEHLER: Nicht genügend Daten in '{DATA_FILE_PATH}'. Benötigt: {seq_len}, Vorhanden: {len(df_wide)}")
        return
    
    instance_to_explain = df_wide.iloc[-seq_len:].values[np.newaxis, ...].astype(np.float32)
    df_train = df_wide.iloc[:-seq_len]

    # --- Hintergrunddaten erstellen ---
    if BACKGROUND_MODE == 'kmeans':
        N_BACKGROUND_SAMPLES = 50
        MAX_WINDOWS_FOR_KMEANS = 10000
        print(f"\nErstelle Hintergrunddaten (Modus: K-Means) mit {N_BACKGROUND_SAMPLES} Zentren...")
        
        train_windows_view = sliding_window_view(df_train.values, window_shape=(seq_len, n_features))
        train_windows = np.squeeze(train_windows_view, axis=1)
        
        windows_for_kmeans = train_windows
        if train_windows.shape[0] > MAX_WINDOWS_FOR_KMEANS:
            random_indices = np.random.choice(train_windows.shape[0], MAX_WINDOWS_FOR_KMEANS, replace=False)
            windows_for_kmeans = train_windows[random_indices]

        windows_flat = windows_for_kmeans.reshape(windows_for_kmeans.shape[0], -1)
        kmeans = KMeans(n_clusters=N_BACKGROUND_SAMPLES, random_state=42, n_init='auto').fit(windows_flat)
        background_data = kmeans.cluster_centers_.reshape(N_BACKGROUND_SAMPLES, seq_len, n_features).astype(np.float32)
        print(f"  - Hintergrunddaten mit Shape {background_data.shape} erstellt.")
    else: # Median-Modus
        channel_medians = df_train.median().values
        background_data = np.full_like(instance_to_explain, channel_medians)

    # --- Event-Level SHAP Analyse ---
    print("\n" + "="*25 + " EVENT-LEVEL SHAP ANALYSIS " + "="*25)
    wrapped_model = MedianForecastWrapper(model, target_channel_idx=target_channel_idx)

    event_wrapper_fn = create_event_shap_wrapper(
        model_wrapper=wrapped_model,
        instance_to_explain=instance_to_explain,
        background_data=background_data
    )

    event_background_summary = np.zeros((1, seq_len))
    event_instance_for_shap = np.ones((1, seq_len))

    event_explainer = shap.KernelExplainer(event_wrapper_fn, event_background_summary)
    print("Berechne Event-SHAP-Werte (dies kann eine Weile dauern)...")
    event_shap_values = event_explainer.shap_values(event_instance_for_shap, nsamples=EVENT_SHAP_NSAMPLES)[0]
    print(f"✅ Berechnung abgeschlossen. Event-SHAP-Werte haben Shape: {event_shap_values.shape}")

    # --- Finale Visualisierung ---
    print("\nErstelle den finalen kombinierten Plot mit dualer Y-Achse...")
    plot_all_channels_with_importance_bars(
        instance_data=instance_to_explain,
        event_shap_values=event_shap_values,
        target_channel_name=TARGET_CHANNEL,
        channel_names=channel_names,
        output_filename="shap_final_multichannel_contributions.png"
    )

if __name__ == "__main__":
    # Simplified main execution for clarity
    # NOTE: The full script includes many other plotting functions not shown here for brevity.
    # The essential part is the corrected plot_all_channels_with_importance_bars function.
    
    # Dummy functions to allow the script to run
    def create_event_shap_wrapper(model_wrapper, instance_to_explain, background_data):
        return lambda x: np.random.randn(x.shape[0], 96) # Dummy wrapper
    
    # We will call main to execute the logic
    main()