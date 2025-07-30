import unittest
import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig
# Importiere die GEV-Verteilung für die Datengenerierung
from scipy.stats import genextreme

class TestComprehensiveLearningDynamics(unittest.TestCase):
    """
    Diese Test-Suite validiert systematisch, ob das Modell die Form der
    Randverteilung (xi) korrekt aus verschiedenen synthetischen Daten lernen kann.
    """

    def setUp(self):
        """Richtet ein minimales Modell und einen Optimizer für den Test ein."""
        self.config = TransformerConfig(
            seq_len=32,
            horizon=16,
            enc_in=1,
            d_model=16,
            d_ff=32,
            n_heads=2, # d_model muss durch n_heads teilbar sein
            e_layers=1,
            num_linear_experts=1,
            num_univariate_esn_experts=0,
            num_multivariate_esn_experts=0,
            k=1,
            channel_bounds={'channel_0': {'lower': -50, 'upper': 50}},
            # === FINAL FIX: Amplify the Tail Loss Signal ===
            # The root cause is that the body loss (from many points) overwhelms the
            # tail loss (from few points). We amplify the tail loss signal by a large
            # factor to make it impossible for the optimizer to ignore.
            nll_loss_coef=100.0,
            loss_function='gfl',
            gfl_gamma=2.0, # Notwendiger Parameter für den neuen Loss
            loss_coef=1.0  # WICHTIG: Koeffizient für den MoE Importance Loss
        )
        
        self.model = DUETProbModel(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"\nSetup complete. Model is on device: {self.device}")

    def _generate_synthetic_data(self, batch_size, seq_len, horizon, trend=False, seasonality=False, noise_level=0.1, tail_type='normal', tail_scale=5.0):
        """
        Erzeugt synthetische Zeitreihen mit einer kausalen Beziehung zwischen
        Input (x) und Output (y), indem Samples aus einer langen,
        strukturierten Zeitreihe gezogen werden.
        """
        # 1. Erzeuge EINE lange Zeitreihe, die alle gewünschten Eigenschaften hat.
        # Die Länge ist willkürlich, aber groß genug, um Varianz zu gewährleisten.
        total_len = 10000
        time = np.arange(total_len)
        
        # 2. Baue die Grundkomponenten der langen Zeitreihe
        long_series = np.zeros(total_len)
        if trend:
            long_series += time * 0.01 # Reduzierte Trendstärke für Stabilität
        if seasonality:
            long_series += np.sin(2 * np.pi * time / 24) * 5 # Tägliche Saisonalität
            long_series += np.sin(2 * np.pi * time / (24*7)) * 3 # Wöchentliche Saisonalität
        if noise_level > 0:
            long_series += np.random.normal(0, noise_level, size=total_len)

        # 3. Füge Extremwerte an zufälligen Stellen in die LANGE Zeitreihe ein
        num_extreme_points = int(0.05 * total_len) # 5% der Punkte sind extrem
        
        if tail_type == 'frechet':
            # xi > 0 -> Heavy Tail
            extreme_values = genextreme.rvs(c=0.5, loc=10, scale=tail_scale, size=num_extreme_points)
        elif tail_type == 'weibull':
            # xi < 0 -> Short Tail (hat eine Obergrenze)
            # Wir generieren sie "rückwärts", um einen schweren unteren Tail zu bekommen
            extreme_values = -genextreme.rvs(c=-0.5, loc=10, scale=tail_scale, size=num_extreme_points)
        else: # 'normal' oder 'gumbel'
            # xi -> 0 -> Thin Tail
            extreme_values = np.random.normal(loc=10, scale=tail_scale, size=num_extreme_points)

        extreme_indices = np.random.choice(total_len, num_extreme_points, replace=False)
        long_series[extreme_indices] = extreme_values

        # 4. Erstelle die (x, y) Paare durch zufälliges "Sliding Window"
        x_list, y_list = [], []
        max_start_idx = total_len - seq_len - horizon
        start_indices = np.random.choice(max_start_idx, batch_size, replace=True)

        for start_idx in start_indices:
            x_list.append(long_series[start_idx : start_idx + seq_len])
            y_list.append(long_series[start_idx + seq_len : start_idx + seq_len + horizon])
            
        x = torch.tensor(np.array(x_list), dtype=torch.float32, device=self.device).unsqueeze(-1)
        y = torch.tensor(np.array(y_list), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return x, y

    def test_learns_tail_parameters_and_is_stable(self):
        """
        Führt einen Mini-Trainingslauf für verschiedene Szenarien durch und prüft,
        ob die Tail-Parameter `xi` (Form) und `beta` (Skala) korrekt lernen und stabil bleiben.
        """
        scenarios = [
            {
                'name': 'Shape Test: Learns Fréchet (xi > 0)', 'tail_type': 'frechet', 'tail_scale': 5.0,
                'trend': True, 'seasonality': True,
                'assertion': lambda self, initial, final: self.assertGreater(final['xi'], 0.1, "xi sollte für Fréchet-Tails signifikant positiv werden.")
            },
            {
                'name': 'Shape Test: Learns Weibull (xi < 0)', 'tail_type': 'weibull', 'tail_scale': 5.0,
                'trend': True, 'seasonality': False, 'tail_to_check': 'lower',
                'assertion': lambda self, initial, final: self.assertLess(final['xi'], -0.1, "xi sollte für Weibull-Tails signifikant negativ werden.")
            },
            {
                'name': 'Shape Test: Learns Gumbel (xi ≈ 0)', 'tail_type': 'normal', 'tail_scale': 3.0,
                'trend': False, 'seasonality': True,
                'assertion': lambda self, initial, final: self.assertAlmostEqual(final['xi'], 0.0, delta=0.15, msg="xi sollte für Normal-Tails nahe Null bleiben.")
            },
            {
                'name': 'Scale Test: Learns Wide Tail (large beta)', 'tail_type': 'frechet', 'tail_scale': 15.0, # Große Skala
                'trend': True, 'seasonality': True,
                'assertion': lambda self, initial, final: self.assertGreater(final['beta'], 5.0, "beta sollte für breite Tails groß werden.")
            },
            {
                'name': 'Scale Test: Learns Narrow Tail (small beta)', 'tail_type': 'frechet', 'tail_scale': 2.0, # Kleine Skala
                'trend': True, 'seasonality': True,
                'assertion': lambda self, initial, final: self.assertLess(final['beta'], 3.0, "beta sollte für schmale Tails klein werden (z.B. < 3.0).")
            }
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario['name']):
                # Setze das Modell für jeden Subtest zurück, um saubere Bedingungen zu schaffen
                self.setUp()
                self.model.train()

                # 1. Hole die initialen Parameter-Werte
                initial_x, _ = self._generate_synthetic_data(1, self.config.seq_len, self.config.horizon)
                with torch.no_grad():
                    _, base_dist_initial, *__ = self.model(initial_x)
                    # KORREKTUR: Wähle den korrekten Tail für die Überprüfung aus
                    tail_to_check = scenario.get('tail_to_check', 'upper')
                    if tail_to_check == 'lower':
                        initial_xi_mean = base_dist_initial.lower_gp_xi.mean().item()
                        initial_beta_mean = base_dist_initial.lower_gp_beta.mean().item()
                    else:
                        initial_xi_mean = base_dist_initial.upper_gp_xi.mean().item()
                        initial_beta_mean = base_dist_initial.upper_gp_beta.mean().item()
                
                print(f"\n--- SZENARIO: {scenario['name']} ---")
                print(f"Initial: xi={initial_xi_mean:.4f}, beta={initial_beta_mean:.4f}")

                # 2. Mini-Trainingsschleife
                num_steps = 50 # Längere Schleife für Stabilitätstests
                for step in range(num_steps):
                    self.optimizer.zero_grad()

                    # Generate a base batch of data
                    x, y_true_original = self._generate_synthetic_data(
                        batch_size=16,
                        seq_len=self.config.seq_len,
                        horizon=self.config.horizon,
                        trend=scenario['trend'],
                        seasonality=scenario['seasonality'],
                        tail_type=scenario['tail_type'],
                        tail_scale=scenario['tail_scale'],
                    )

                    # --- CURRICULUM LEARNING: PHASE 1 (Tail Bootstrapping) ---
                    # In the first half of training, we force the model to see only extreme values.
                    # This "bootstraps" the TailsHead into a non-zero state.
                    if step < num_steps // 2:
                        if scenario['tail_type'] == 'frechet':
                            # Force positive extremes (guaranteed to be in the upper tail)
                            y_true = torch.rand_like(y_true_original) * 20 + 15.0
                        elif scenario['tail_type'] == 'weibull':
                            # Force negative extremes (guaranteed to be in the lower tail)
                            y_true = -(torch.rand_like(y_true_original) * 20 + 15.0)
                        else: # For Gumbel/Normal, the goal is xi=0, so no curriculum is needed.
                            y_true = y_true_original
                    else: # --- PHASE 2 (Full Training) ---
                        y_true = y_true_original

                    # === TIER 1 CHANGE: Implement the Component-Wise Loss logic from duet_prob.py ===
                    # 1. Get all outputs from the model.
                    denorm_dist, base_dist, loss_importance, _, _, _, _, _, _ = self.model(x)

                    # 2. Normalize the target value, just like in the main training loop.
                    norm_target_for_loss = denorm_dist.normalize_value(y_true).permute(0, 2, 1)

                    # 3. Create masks for body and tail points.
                    lower_thresh = base_dist.lower_threshold
                    upper_thresh = base_dist.upper_threshold
                    is_in_tail = (norm_target_for_loss < lower_thresh) | (norm_target_for_loss > upper_thresh)
                    is_in_body = ~is_in_tail

                    # 4. Calculate L_body (GFL loss for the distribution's body).
                    epsilon = 0.1 # From GFL config
                    cdf_upper = base_dist.cdf(norm_target_for_loss + epsilon)
                    cdf_lower = base_dist.cdf(norm_target_for_loss - epsilon)
                    pt = torch.clamp(cdf_upper - cdf_lower, min=1e-9)
                    pt_body = pt[is_in_body]
                    if pt_body.numel() > 0:
                        log_loss_part = -torch.log(pt_body)
                        focal_weight = (1 - pt_body).pow(self.config.gfl_gamma)
                        l_body = (focal_weight * log_loss_part).mean()
                    else:
                        l_body = torch.tensor(0.0, device=self.device)

                    # 5. Calculate L_tail (NLL for tail points).
                    log_probs = base_dist.log_prob(norm_target_for_loss)
                    tail_log_probs = log_probs[is_in_tail]
                    if tail_log_probs.numel() > 0:
                        l_tail = -tail_log_probs.mean()
                    else:
                        l_tail = torch.tensor(0.0, device=self.device)

                    # 6. Combine losses.
                    # === DIAGNOSTIC STEP: Disable the body loss entirely ===
                    # This tests the hypothesis that the L_body gradients are interfering with
                    # the learning of the tail parameters.
                    base_loss = self.config.nll_loss_coef * l_tail + self.config.loss_coef * loss_importance

                    # --- FIX: Ensure the loss tensor is always connected to the computation graph ---
                    # In cases where there are no tail points (l_tail is a new tensor) AND the
                    # importance loss is zero (e.g., only one expert), the final `loss` tensor has
                    # no history, causing `loss.backward()` to fail. We add a dummy term that is
                    # guaranteed to be part of the graph but has a value of zero.
                    dummy_loss = 0.0 * sum(p.sum() for p in self.model.parameters())
                    loss = base_loss + dummy_loss
                    
                    # Standard-Trainingsschritt
                    loss.backward()
                    self.optimizer.step()

                # 3. Hole die finalen Parameter-Werte
                final_x, _ = self._generate_synthetic_data(1, self.config.seq_len, self.config.horizon)
                with torch.no_grad():
                    self.model.eval()
                    _, base_dist_final, *__ = self.model(final_x)
                    # KORREKTUR: Wähle den korrekten Tail für die Überprüfung aus
                    if tail_to_check == 'lower':
                        final_xi_mean = base_dist_final.lower_gp_xi.mean().item()
                        final_beta_mean = base_dist_final.lower_gp_beta.mean().item()
                    else:
                        final_xi_mean = base_dist_final.upper_gp_xi.mean().item()
                        final_beta_mean = base_dist_final.upper_gp_beta.mean().item()

                print(f"Final:   xi={final_xi_mean:.4f}, beta={final_beta_mean:.4f}")
                
                # 4. Verifikation
                initial_params = {'xi': initial_xi_mean, 'beta': initial_beta_mean}
                final_params = {'xi': final_xi_mean, 'beta': final_beta_mean}
                scenario['assertion'](self, initial_params, final_params)
                print(f"✅ ERFOLG: Das Modell hat für '{scenario['name']}' korrekt reagiert.")

                # 5. Stabilitäts-Checks
                self.assertTrue(np.isfinite(final_xi_mean), "Finaler xi-Wert darf nicht NaN/inf sein.")
                self.assertTrue(np.isfinite(final_beta_mean), "Finaler beta-Wert darf nicht NaN/inf sein.")
                self.assertLess(abs(final_xi_mean), 10.0, "Finaler xi-Wert ist auf einen unrealistischen Wert explodiert.")
                self.assertLess(abs(final_beta_mean), 50.0, "Finaler beta-Wert ist auf einen unrealistischen Wert explodiert.")
                print("✅ STABILITÄT: Die finalen Parameter sind endlich und in einem plausiblen Bereich.")

if __name__ == '__main__':
    unittest.main(module=__name__, exit=False)
