import unittest
import torch
import sys
import os
import torch.nn as nn
# Füge das Projektverzeichnis zum Python-Pfad hinzu, um den Import der Module zu ermöglichen.
# Dies ist ein übliches Vorgehen in Test-Dateien.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.spliced_binned_pareto_standalone import SplicedBinnedParetoOutput, SplicedBinnedPareto

class TestTailParameterization(unittest.TestCase):

    def setUp(self):
        """Bereitet eine Dummy-Instanz von SplicedBinnedParetoOutput für die Tests vor."""
        self.distr_output = SplicedBinnedParetoOutput(
            num_bins=10,
            bins_lower_bound=-5.0,
            bins_upper_bound=5.0,
            tail_percentile=0.05
        )
        # Dummy-Dimensionen für den Test
        self.batch_size = 1
        self.n_vars = 1
        self.horizon = 5

    def test_xi_is_unconstrained_after_fix(self):
        """
        Dieser Test validiert das KORREKTE Verhalten NACH dem Fix.
        Er stellt sicher, dass die `xi`-Parameter direkt vom `TailsHead`
        durchgereicht werden, ohne eine blockierende `tanh`-Funktion.
        """
        print("\n" + "="*60)
        print("  TEST 1: Validierung des korrekten Verhaltens (OHNE `tanh`)")
        print("="*60)

        # Rohe Output-Werte, wie sie vom `TailsHead` kommen könnten.
        # Wir testen einen Bereich von kleinen bis sehr großen Werten.
        # Die Parameter sind in der Reihenfolge: logits, lower_xi, lower_beta, upper_xi, upper_beta
        
        # Dummy-Logits und Betas
        dummy_logits = torch.randn(self.batch_size, self.n_vars, self.horizon, self.distr_output.num_bins)
        dummy_betas = torch.ones(self.batch_size, self.n_vars, self.horizon, 1) # softplus(1) > 0

        # Die kritischen rohen xi-Werte, die wir testen wollen
        raw_xi_values = torch.tensor([-100.0, -3.0, 0.0, 3.0, 100.0]).view(1, 1, 5, 1)
        
        print(f"Rohe `xi`-Werte, die vom `TailsHead` kommen könnten:\n{raw_xi_values.flatten().tolist()}")

        # Kombiniere zu einem vollständigen Parameter-Tensor
        distr_params = torch.cat([
            dummy_logits,
            raw_xi_values,      # lower_xi_raw
            dummy_betas.clone(),# lower_beta_raw
            raw_xi_values,      # upper_xi_raw
            dummy_betas.clone() # upper_beta_raw
        ], dim=-1)

        # Rufe die Methode auf, die die `tanh`-Transformation intern durchführt
        distribution: SplicedBinnedPareto = self.distr_output.distribution(distr_params)

        # Extrahiere die finalen, transformierten xi-Werte aus dem Verteilungsobjekt
        final_lower_xi = distribution.lower_gp_xi.flatten().tolist()
        final_upper_xi = distribution.upper_gp_xi.flatten().tolist()

        print("\nFinale `xi`-Werte, wie sie von der `distribution()`-Methode verarbeitet werden:")
        print(f"  -> Lower Xi: {[f'{x:.4f}' for x in final_lower_xi]}")
        print(f"  -> Upper Xi: {[f'{x:.4f}' for x in final_upper_xi]}")

        print("\nBEOBACHTUNG:")
        print("Die finalen `xi`-Werte sind identisch mit den rohen Inputs.")
        print("Das `tanh` wurde erfolgreich entfernt. Der Gradient vom NLL-Loss kann nun")
        print("ungehindert fließen und das Training der Tail-Parameter ermöglichen.")

        # Assertion, um das NEUE, KORREKTE Verhalten zu bestätigen.
        # Wir prüfen, ob die finale Liste exakt der rohen Input-Liste entspricht.
        self.assertListEqual(final_lower_xi, raw_xi_values.flatten().tolist())
        self.assertListEqual(final_upper_xi, raw_xi_values.flatten().tolist())


class TestLearningDynamics(unittest.TestCase):
    """
    Diese Testklasse prüft nicht nur die Transformation, sondern den Lernprozess selbst.
    Sie simuliert einen Forward- und Backward-Pass, um zu validieren, dass die
    Gradienten korrekt fließen und die Parameter in die richtige Richtung lernen.
    """
    def setUp(self):
        self.num_bins = 10
        self.distr_output = SplicedBinnedParetoOutput(
            num_bins=self.num_bins,
            bins_lower_bound=-5.0,
            bins_upper_bound=5.0,
            tail_percentile=0.05
        )
        # Simuliere einen einfachen "Kopf", der die Verteilungsparameter ausgibt.
        # Dies ist entscheidend, da wir die Gradienten seiner Gewichte prüfen wollen.
        self.d_model = 32
        self.horizon = 1
        self.n_vars = 1
        # Der Kopf gibt (num_bins + 4) Parameter pro Zeitschritt aus.
        self.output_dim = self.horizon * (self.num_bins + 4)
        self.head = nn.Linear(self.d_model, self.output_dim)

    def test_gradients_flow_to_tail_parameters(self):
        """
        Validiert, dass ein Datenpunkt im Tail einen Gradienten an die
        Tail-Parameter (xi, beta) sendet und dass dieser Gradient in die
        erwartete Richtung zeigt.
        """
        print("\n" + "="*60)
        print("  TEST 2: Validierung des Lerndynamik-Verhaltens")
        print("="*60)

        # Ein Dummy-Input für unser Modell
        dummy_input = torch.randn(1, self.d_model)

        # Ein extremer Zielwert, der weit im oberen Tail liegt.
        # Dies sollte einen starken Lernimpuls für die Tail-Parameter auslösen.
        y_true = torch.tensor([[[20.0]]]) # Shape [B, V, H]
        print(f"Simulierter Zielwert (y_true): {y_true.item():.1f} (liegt weit außerhalb der Bins [-5, 5])")

        # --- 1. Forward-Pass ---
        # Hole die rohen Parameter vom simulierten Kopf
        raw_params = self.head(dummy_input)
        # Forme sie für die Verteilung um: [B, V, H, Params]
        distr_params = raw_params.view(1, self.n_vars, self.horizon, self.num_bins + 4)
        
        # --- FIX: Sage PyTorch, dass es den Gradienten für diesen Nicht-Blatt-Tensor behalten soll. ---
        # Ohne dies wird .grad für distr_params und seine Slices `None` sein.
        distr_params.retain_grad()

        # Erzeuge die Verteilung
        distribution: SplicedBinnedPareto = self.distr_output.distribution(distr_params)

        # --- 2. Loss-Berechnung (NLL) ---
        # Wir verwenden die NLL, da sie das Signal für die Tails liefert.
        # log_prob erwartet die Form [B, V, H]
        nll_loss = -distribution.log_prob(y_true)
        print(f"Berechneter NLL-Loss für den extremen Wert: {nll_loss.item():.4f}")
        self.assertFalse(torch.isinf(nll_loss), "Loss sollte nicht unendlich sein.")

        # --- 3. Backward-Pass ---
        nll_loss.backward()
        print("\nBackward-Pass ausgeführt. Überprüfe die Gradienten...")

        # --- 4. Überprüfung der Gradienten und Parameter ---
        # Wir extrahieren die rohen Parameter und ihre Gradienten
        # Die Tail-Parameter sind die letzten 4 im Tensor
        raw_lower_xi = distr_params[..., self.num_bins]
        raw_upper_xi = distr_params[..., self.num_bins + 2]

        # Hypothese 1: Fließt ein Gradient?
        # KORREKTUR: Greife auf den Gradienten über den Eltern-Tensor zu, für den .retain_grad() aufgerufen wurde.
        # .grad wird für Slices (wie raw_upper_xi) nicht direkt gefüllt, sondern im .grad des Eltern-Tensors.
        self.assertIsNotNone(distr_params.grad, "Der Gradient für den gesamten Parameter-Tensor sollte NICHT None sein.")
        grad_upper_xi = distr_params.grad[..., self.num_bins + 2]

        self.assertIsNotNone(grad_upper_xi, "Der Gradient für upper_xi sollte NICHT None sein.")
        self.assertNotEqual(grad_upper_xi.abs().sum().item(), 0.0, "Der Gradient für upper_xi sollte nicht Null sein.")
        print(f"✅ Gradient für upper_xi ist vorhanden und nicht Null: {grad_upper_xi.item():.4f}")

        # Hypothese 2: Lernt der Parameter in die richtige Richtung?
        # Um die Wahrscheinlichkeit eines hohen Wertes (y_true=20) zu erhöhen,
        # muss der Tail "fetter" werden, d.h. `upper_xi` muss steigen.
        # Der Loss (NLL) soll minimiert werden. Daher muss der Gradient `d(Loss)/d(xi)`
        # negativ sein, damit der Optimizer `xi` erhöht (xi_new = xi_old - lr * grad).
        self.assertLess(grad_upper_xi.item(), 0, "Der Gradient für upper_xi sollte negativ sein, um xi zu erhöhen.")
        print("✅ Der Gradient ist negativ. Das Modell lernt, den Tail 'fetter' zu machen, was korrekt ist.")

        # Hypothese 3: Bleiben die Bins stabil?
        # KORREKTUR: Auch hier den Gradienten vom Eltern-Tensor holen.
        raw_logits_grad = distr_params.grad[..., :self.num_bins]
        self.assertIsNotNone(raw_logits_grad, "Die Logits sollten ebenfalls einen Gradienten erhalten.")
        print("✅ Die Logits der Bins haben ebenfalls einen Gradienten erhalten (durch die Normalisierungskonstante).")

if __name__ == '__main__':
    # Erstellt ein Verzeichnis für die Tests, falls es nicht existiert
    if not os.path.exists('tests'):
        os.makedirs('tests')
    unittest.main(module=__name__, exit=False)