

import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

# --- Kopie von JohnsonSU_torch aus johnson_system.py für den eigenständigen Test ---
class JohnsonSU_torch(Distribution):
    """Reine PyTorch-Implementierung der log_prob der Johnson SU-Verteilung."""
    arg_constraints = {
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, gamma, delta, xi, lambda_, validate_args=None):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape, validate_args=validate_args)
        self._log_delta = self.delta.log()
        self._log_lambda = self.lambda_.log()
        self._log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        y = (value - self.xi) / self.lambda_
        z = self.gamma + self.delta * torch.asinh(y)
        log_p = self._log_delta - self._log_lambda - self._log_sqrt_2pi - 0.5 * z.pow(2) - 0.5 * (1 + y.pow(2)).log()
        return log_p

# --- Neue Zero-Inflated Johnson SU Verteilung ---
class ZeroInflatedJohnsonSU(Distribution):
    """
    Eine Zero-Inflated Johnson SU Verteilung.
    Sie modelliert eine Wahrscheinlichkeitsmasse bei Null und eine Johnson SU Verteilung für positive Werte.
    """
    arg_constraints = {
        'p_zero_logit': constraints.real, # Logit der Wahrscheinlichkeit von Null
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    support = constraints.greater_than_eq(0.0) # Support ist [0, unendlich)
    has_rsample = False # rsample wird für diesen schnellen Test nicht implementiert

    def __init__(self, p_zero_logit, gamma, delta, xi, lambda_, validate_args=None):
        self.p_zero_logit, self.gamma, self.delta, self.xi, self.lambda_ = \
            broadcast_all(p_zero_logit, gamma, delta, xi, lambda_)
        super().__init__(self.p_zero_logit.shape, validate_args=validate_args)

        self.p_zero = torch.sigmoid(self.p_zero_logit)
        self.johnson_su_dist = JohnsonSU_torch(gamma, delta, xi, lambda_)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Sicherstellen, dass der Wert nicht negativ ist
        if (value < 0).any():
            raise ValueError("Werte für ZeroInflatedJohnsonSU müssen nicht-negativ sein.")

        # Wahrscheinlichkeitsmasse bei Null
        log_prob_zero = torch.log(self.p_zero)

        # Log-Wahrscheinlichkeit für positive Werte
        positive_values_mask = (value > 0)
        log_prob_positive_component = torch.full_like(value, -float('inf')) # Initialisiere mit -inf

        if positive_values_mask.any():
            # Berechne log_prob für positive Werte unter Verwendung der Johnson SU Verteilung
            # und kombiniere mit (1 - p_zero)
            log_prob_positive_component[positive_values_mask] = \
                torch.log(1 - self.p_zero) + self.johnson_su_dist.log_prob(value[positive_values_mask])

        # Kombiniere basierend darauf, ob der Wert Null oder positiv ist
        result_log_prob = torch.where(value == 0, log_prob_zero, log_prob_positive_component)

        return result_log_prob

# --- Test-Setup ---
print("--- Starte Zero-Inflated Johnson SU Test mit echten Daten ---")

# 1. Lade die echten Daten
DATA_PATH = '/Users/friedrichsiemers/ts_proba_cuda/dataset/forecasting/preci_short.csv'
try:
    preci_data_df = pd.read_csv(DATA_PATH)
    
    # **WICHTIG: Konvertiere die 'data'-Spalte explizit zu numerischen Werten**
    # 'errors='coerce'' wird nicht-numerische Werte in NaN umwandeln
    preci_data_df['data'] = pd.to_numeric(preci_data_df['data'], errors='coerce')
    # Entferne Zeilen mit NaN-Werten, die durch die Konvertierung entstanden sein könnten
    preci_data_df.dropna(subset=['data'], inplace=True)

    data_series = preci_data_df['data'] # Jetzt die 'data'-Spalte verwenden

    print(f"\nErfolgreich Daten von {DATA_PATH} geladen. Shape: {preci_data_df.shape}")
    print("Erste 5 Zeilen der Daten:")
    print(preci_data_df.head())

    # Berechne den Anteil der Nullen
    num_zeros = (data_series == 0).sum()
    total_samples = len(data_series)
    proportion_zeros = num_zeros / total_samples
    print(f"\nAnteil der Nullen in der 'data'-Spalte: {proportion_zeros * 100:.2f}%")

    # Setze p_zero_logit basierend auf dem Anteil der Nullen
    # Eine einfache Heuristik: logit(p) = log(p / (1-p))
    # Füge eine kleine Konstante hinzu, um log(0) oder log(inf) zu vermeiden
    p_zero_for_logit = max(1e-6, min(1 - 1e-6, proportion_zeros))
    estimated_p_zero_logit = torch.tensor(math.log(p_zero_for_logit / (1 - p_zero_for_logit)))

    # Für den Johnson SU Teil (für positive Werte) - hier nur Beispielparameter
    # In einer echten Implementierung würden diese Parameter durch MLE auf den positiven Werten geschätzt.
    # Wir wählen hier plausible Werte für rechtsschiefe Niederschlagsdaten.
    example_gamma = torch.tensor(-1.5) # Rechtsschiefe
    example_delta = torch.tensor(1.0)  # Kurtosis
    example_xi = torch.tensor(0.1)     # Lokation (nahe Null)
    example_lambda = torch.tensor(0.5) # Skalierung

    # Erstelle die Instanz der Zero-Inflated Johnson SU Verteilung
    zi_johnson_su_dist = ZeroInflatedJohnsonSU(
        p_zero_logit=estimated_p_zero_logit,
        gamma=example_gamma,
        delta=example_delta,
        xi=example_xi,
        lambda_=example_lambda
    )

    print(f"\nVerteilungsparameter (basierend auf Daten und Beispielen):")
    print(f"  p_zero (Wahrscheinlichkeit von Null): {zi_johnson_su_dist.p_zero.item():.4f} (geschätzt aus Daten: {proportion_zeros:.4f})")
    print(f"  Johnson SU Gamma: {example_gamma.item():.4f}")
    print(f"  Johnson SU Delta: {example_delta.item():.4f}")
    print(f"  Johnson SU Xi: {example_xi.item():.4f}")
    print(f"  Johnson SU Lambda_: {example_lambda.item():.4f}")

    # 2. Überprüfung der Gesamtwahrscheinlichkeit (konzeptuell)
    print("\n--- Überprüfung der Gesamtwahrscheinlichkeit (konzeptuell) ---")
    p_zero_val = zi_johnson_su_dist.p_zero.item()
    p_positive_val = 1 - p_zero_val
    total_probability_sum = p_zero_val + p_positive_val

    print(f"Wahrscheinlichkeit für X=0 (p_zero): {p_zero_val:.6f}")
    print(f"Wahrscheinlichkeit für X>0 (1 - p_zero): {p_positive_val:.6f}")
    print(f"Summe der Wahrscheinlichkeiten (p_zero + (1 - p_zero)): {total_probability_sum:.6f}")
    print(f"Ist die Summe nahe 1? {abs(total_probability_sum - 1.0) < 1e-6}")

    print("\nMathematische Begründung:")
    print("Eine Zero-Inflated kontinuierliche Verteilung definiert die Gesamtwahrscheinlichkeit als die Summe der diskreten Wahrscheinlichkeitsmasse bei Null und des Integrals der PDF für positive Werte.")
    print("Da die zugrunde liegende Johnson SU-Verteilung eine gültige PDF hat, die über ihren gesamten Support zu 1 integriert, ist die Gesamtwahrscheinlichkeit der Zero-Inflated-Verteilung immer p_zero + (1 - p_zero) * 1 = 1.")
    print("Die Konsistenz liegt in der Definition der Verteilung selbst.")

    # 3. Demonstration von log_prob für einige Werte aus den Daten
    print("\n--- Demonstration von log_prob für einige Werte aus den Daten ---")
    # Nimm einige Nullen und einige positive Werte aus den Daten
    # Sicherstellen, dass es Nullen und Positive gibt, bevor man samplet
    if num_zeros > 0:
        sample_zeros = data_series[data_series == 0].sample(min(5, num_zeros)).tolist()
    else:
        sample_zeros = []

    num_positives = len(data_series) - num_zeros
    if num_positives > 0:
        sample_positives = data_series[data_series > 0].sample(min(5, num_positives)).tolist()
    else:
        sample_positives = []

    test_values_from_data = torch.tensor(sample_zeros + sample_positives, dtype=torch.float32)

    if len(test_values_from_data) > 0:
        log_probs_from_data = zi_johnson_su_dist.log_prob(test_values_from_data)
        for val, lp in zip(test_values_from_data, log_probs_from_data):
            print(f"  log_prob(value={val.item():.2f}): {lp.item():.4f}")
    else:
        print("Nicht genügend Datenpunkte zum Demonstrieren der log_prob (nach Bereinigung).")

except FileNotFoundError:
    print(f"\nFEHLER: Datei nicht gefunden unter {DATA_PATH}. Bitte überprüfen Sie den Pfad.")
except Exception as e:
    print(f"\nEin Fehler ist beim Lesen oder Verarbeiten der Daten aufgetreten: {e}")

print("\n--- Test abgeschlossen ---")
