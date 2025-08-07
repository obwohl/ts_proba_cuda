import pandas as pd
import torch
import argparse
from pathlib import Path

# === IMPORTS FÜR MODELL ===
# Diese Pfade müssen relativ zum Ausführungsort des Skripts stimmen.
# Annahme: Das Skript liegt im Hauptverzeichnis des "Inferenz-Pakets".
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig
#test

# --- Feste Konfiguration ---
# Die Spaltenreihenfolge muss exakt der beim Training entsprechen!
SERIES_ORDER = ['wassertemp', 'airtemp_96', 'pressure_96']
# Die Quantile, die vorhergesagt werden sollen.
QUANTILES_TO_PREDICT = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

def run_forecast(checkpoint_path: Path, data_path: Path, output_path: Path):
    """
    Führt eine einzelne Vorhersage basierend auf den letzten Datenpunkten durch
    und speichert das Ergebnis als CSV-Datei.
    """
    # --- 1. Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- 2. Modell aus dem Checkpoint laden ---
    print(f"🔄 Loading model from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config_dict = checkpoint['config_dict']
        config_dict['distribution_family'] = 'ZIEGPD_M1'  # Override distribution family
        config = TransformerConfig(**config_dict)
        # Wichtig: Setze die Quantile in der Konfiguration, die das Modell für die Vorhersage verwenden soll.
        config.quantiles = QUANTILES_TO_PREDICT
        config.channel_types = SERIES_ORDER
        model = DUETProbModel(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model. Reason: {e}")
        return

    # --- 3. Daten laden und vorbereiten ---
    print(f"🔄 Loading and preparing data from '{data_path}'...")
    df_long = pd.read_csv(data_path)
    df_wide = df_long.pivot(index='date', columns='cols', values='data')[SERIES_ORDER]
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide.ffill(inplace=True)
    df_wide.bfill(inplace=True)
    
    # Überprüfen, ob genügend Daten für den Input vorhanden sind
    if len(df_wide) < config.seq_len:
        print(f"❌ ERROR: Not enough data. The model requires an input of length {config.seq_len}, but the data only has {len(df_wide)} rows.")
        return
    
    # Nimm die letzten `seq_len` Datenpunkte als Input für das Modell
    input_df = df_wide.iloc[-config.seq_len:]
    print(f"✅ Data prepared. Using last {config.seq_len} timestamps for input (from {input_df.index.min()} to {input_df.index.max()}).")

    # --- 4. Inferenz durchführen ---
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32).unsqueeze(0).to(device)

    print("🔄 Running inference...")
    with torch.no_grad():
        # Das Modell gibt eine Verteilung zurück.
        distr, _, _, _, _, _, _, _, _ = model(input_tensor)
        
        # Hole die Vorhersagen für die definierten Quantile
        q_tensor = torch.tensor(QUANTILES_TO_PREDICT, device=device, dtype=torch.float32)
        # distr.icdf gibt bei mehreren Quantilen [B, N_Vars, H, Q] zurück
        quantile_predictions_tensor = distr.icdf(q_tensor)

    # Form für die CSV-Ausgabe anpassen: [B, N, H, Q] -> [H, N, Q]
    prediction_array = quantile_predictions_tensor.squeeze(0).permute(1, 0, 2).cpu().numpy()
    print("✅ Inference complete.")

    # --- 5. Ergebnis als CSV speichern ---
    print(f"🔄 Saving forecast to '{output_path}'...")
    
    # Erstelle einen Zeitindex für die Vorhersage
    last_timestamp = input_df.index[-1]
    freq = pd.infer_freq(df_wide.index)
    forecast_index = pd.date_range(start=last_timestamp + pd.to_timedelta(1, unit=freq if freq else 'H'), periods=config.horizon, freq=freq)
    
    # Erstelle einen hierarchischen Multi-Index für die Spalten (Variable, Quantil)
    multi_header = pd.MultiIndex.from_product([SERIES_ORDER, QUANTILES_TO_PREDICT], names=['variable', 'quantile'])
    
    # Das Array hat die Form [Horizon, Num_Vars, Num_Quantiles]. Wir müssen es umformen.
    # Reshape zu [Horizon, Num_Vars * Num_Quantiles]
    reshaped_preds = prediction_array.reshape(config.horizon, -1)
    
    # Wandle den Multi-Index in einen einzelnen, flachen Header um (z.B. 'wassertemp_q0.5')
    flat_header = [f"{var}_q{q}" for var, q in multi_header]
    
    # Erstelle den finalen DataFrame
    forecast_df = pd.DataFrame(reshaped_preds, index=forecast_index, columns=flat_header)
    
    forecast_df.to_csv(output_path)
    print(f"🎉 Forecast successfully saved to {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run a single forecast with a trained DUET-Prob model and save to CSV.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.pt).")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument('--output-csv', type=str, required=True, help="Path to save the output forecast CSV file.")
    args = parser.parse_args()

    run_forecast(Path(args.checkpoint), Path(args.data_file), Path(args.output_csv))

if __name__ == '__main__':
    main()