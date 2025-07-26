import unittest
import torch
import torch.nn as nn
import warnings

# Die zu testende Fabrik-Funktion
from ts_benchmark.baselines.duet.layers.expert_factory import create_experts

# Die Experten-Klassen, die die Fabrik erstellen soll
from ts_benchmark.baselines.duet.layers.linear_pattern_extractor import Linear_extractor
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert

# Helferklasse für die Konfiguration
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestExpertFactory(unittest.TestCase):

    def setUp(self):
        """Erstellt eine hybride Standardkonfiguration für die Tests."""
        self.base_config = {
            # Allgemeine Parameter
            "d_model": 32,
            "seq_len": 64,
            "enc_in": 3, # Wichtig für multivariate Experten
            "moving_avg": 25, # Für Linear_extractor

            # Parameter für univariaten ESN
            "reservoir_size_uni": 64,
            "spectral_radius_uni": 0.9,
            "sparsity_uni": 0.1,
            "leak_rate_uni": 1.0,
            "input_scaling": 1.0, # Behält den alten Namen für Kompatibilität

            # Parameter für multivariaten ESN
            "reservoir_size_multi": 128,
            "spectral_radius_multi": 1.1,
            "sparsity_multi": 0.2,
            "leak_rate_multi": 0.5,
            "input_scaling_multi": 0.5,
        }

    def test_01_creates_correct_hybrid_expert_pool(self):
        """Test 1: Erstellt die Fabrik die korrekte Anzahl und die richtigen Typen von hybriden Experten?"""
        config = dotdict(self.base_config.copy())
        config.num_linear_experts = 3
        config.num_univariate_esn_experts = 2
        config.num_multivariate_esn_experts = 1
        
        experts_list = create_experts(config)

        # Überprüfe die Gesamtanzahl
        self.assertIsInstance(experts_list, nn.ModuleList, "Die Fabrik sollte eine nn.ModuleList zurückgeben.")
        self.assertEqual(len(experts_list), 6, "Die Gesamtanzahl der Experten ist falsch.")

        # Zähle die Typen
        num_linear = sum(isinstance(exp, Linear_extractor) for exp in experts_list)
        num_uni_esn = sum(isinstance(exp, UnivariateReservoirExpert) for exp in experts_list)
        num_multi_esn = sum(isinstance(exp, MultivariateReservoirExpert) for exp in experts_list)
        
        self.assertEqual(num_linear, 3, "Die Anzahl der linearen Experten ist falsch.")
        self.assertEqual(num_uni_esn, 2, "Die Anzahl der univariaten ESN-Experten ist falsch.")
        self.assertEqual(num_multi_esn, 1, "Die Anzahl der multivariaten ESN-Experten ist falsch.")

    def test_02_handles_zero_experts_of_one_type(self):
        """Test 2: Funktioniert die Fabrik auch, wenn ein Expertentyp nicht vorhanden ist?"""
        config = dotdict(self.base_config.copy())
        config.num_linear_experts = 5
        config.num_univariate_esn_experts = 0 # Kein univariater ESN
        config.num_multivariate_esn_experts = 2
        
        experts_list = create_experts(config)
        
        self.assertEqual(len(experts_list), 7)
        self.assertEqual(sum(isinstance(exp, Linear_extractor) for exp in experts_list), 5)
        self.assertEqual(sum(isinstance(exp, UnivariateReservoirExpert) for exp in experts_list), 0)
        self.assertEqual(sum(isinstance(exp, MultivariateReservoirExpert) for exp in experts_list), 2)

    def test_03_applies_correct_esn_params(self):
        """Test 3: Wendet die Fabrik die spezifischen Konfigurationen auf die korrekten ESN-Typen an?"""
        config = dotdict(self.base_config.copy())
        config.num_univariate_esn_experts = 1
        config.num_multivariate_esn_experts = 1

        experts_list = create_experts(config)
        
        # Extrahiere die ESN-Experten aus der Liste
        uni_esn = next(exp for exp in experts_list if isinstance(exp, UnivariateReservoirExpert))
        multi_esn = next(exp for exp in experts_list if isinstance(exp, MultivariateReservoirExpert))

        # Überprüfe, ob die univariaten Parameter korrekt gesetzt wurden
        self.assertEqual(uni_esn.reservoir_size, config.reservoir_size_uni)
        self.assertAlmostEqual(uni_esn.spectral_radius_target, config.spectral_radius_uni)
        self.assertAlmostEqual(uni_esn.sparsity, config.sparsity_uni)
        
        # Überprüfe, ob die multivariaten Parameter korrekt gesetzt wurden
        self.assertEqual(multi_esn.reservoir_size, config.reservoir_size_multi)
        self.assertAlmostEqual(multi_esn.spectral_radius_target, config.spectral_radius_multi)
        self.assertAlmostEqual(multi_esn.sparsity, config.sparsity_multi)

    def test_04_legacy_fallback_for_old_configs(self):
        """Test 4: Funktioniert der Fallback für alte Konfigurationen mit 'num_esn_experts'?"""
        config = dotdict(self.base_config.copy())
        config.num_esn_experts = 3 # Veralteter Parameter

        # Teste, ob eine DeprecationWarning ausgelöst wird
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Stelle sicher, dass alle Warnungen aufgezeichnet werden
            experts_list = create_experts(config)
            
            self.assertEqual(len(w), 1, "Es sollte genau eine DeprecationWarning ausgelöst werden.")
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        # Überprüfe, ob die Legacy-Experten als univariat erstellt wurden
        self.assertEqual(len(experts_list), 3)
        self.assertTrue(all(isinstance(exp, UnivariateReservoirExpert) for exp in experts_list))

if __name__ == '__main__':
    unittest.main()
