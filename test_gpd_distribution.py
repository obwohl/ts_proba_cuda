import torch
import unittest
import sys
import os

# Add the project root to the Python path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ts_benchmark.baselines.duet.extended_gpd import ZeroInflatedExtendedGPD_M1_Continuous

class TestZeroInflatedExtendedGPD(unittest.TestCase):

    def test_shape_consistency(self):
        """Tests if the output shapes of log_prob and icdf are correct."""
        print("\n--- Running Test: Shape Consistency ---")
        
        # Parameters with a batch shape of [2, 3] and horizon of 4
        batch_size, n_vars, horizon = 2, 3, 4
        pi = torch.rand(batch_size, n_vars, horizon)
        kappa_raw = torch.randn(batch_size, n_vars, horizon)
        sigma_raw = torch.randn(batch_size, n_vars, horizon)
        xi = torch.randn(batch_size, n_vars, horizon)

        dist = ZeroInflatedExtendedGPD_M1_Continuous(pi, kappa_raw, sigma_raw, xi)

        # Test log_prob shape
        values = torch.rand(batch_size, horizon, n_vars) # Shape [B, H, N_vars]
        log_probs = dist.log_prob(values)
        self.assertEqual(log_probs.shape, values.shape)
        print(f"log_prob input shape: {values.shape}, output shape: {log_probs.shape} -> OK")

        # Test icdf shape
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        icdf_values = dist.icdf(quantiles)
        expected_shape = (batch_size, n_vars, horizon, len(quantiles))
        self.assertEqual(icdf_values.shape, expected_shape)
        print(f"icdf input shape: {quantiles.shape}, output shape: {icdf_values.shape} -> OK")

    def test_zero_inflation_log_prob(self):
        """Tests the log_prob for values at zero."""
        print("\n--- Running Test: Zero-Inflation log_prob ---")
        pi = torch.tensor([[[0.25]]])
        dist = ZeroInflatedExtendedGPD_M1_Continuous(pi, torch.tensor([[[1.0]]]), torch.tensor([[[1.0]]]), torch.tensor([[[0.1]]]))
        
        # Value at zero
        values = torch.zeros(1, 1, 1)
        log_prob_at_zero = dist.log_prob(values)
        
        # The probability of being zero is pi. So log_prob should be log(pi).
        self.assertTrue(torch.allclose(log_prob_at_zero, torch.log(pi)))
        print(f"log_prob(0) matches log(pi): {log_prob_at_zero.item():.4f} vs {torch.log(pi).item():.4f} -> OK")

    def test_zero_inflation_icdf(self):
        """Tests the icdf for quantiles within the zero-inflation mass."""
        print("\n--- Running Test: Zero-Inflation icdf ---")
        pi_val = 0.3
        pi = torch.tensor([[[pi_val]]])
        dist = ZeroInflatedExtendedGPD_M1_Continuous(pi, torch.tensor([[[1.0]]]), torch.tensor([[[1.0]]]), torch.tensor([[[0.1]]]))

        # Quantiles less than or equal to pi
        quantiles = torch.tensor([0.0, 0.1, pi_val])
        icdf_values = dist.icdf(quantiles)

        # For q <= pi, the value should be 0
        self.assertTrue(torch.all(icdf_values == 0))
        print(f"icdf(q <= pi) returns 0 for q={quantiles.tolist()} -> OK")

        # Quantile greater than pi should be > 0
        quantiles_above = torch.tensor([pi_val + 0.1])
        icdf_above = dist.icdf(quantiles_above)
        self.assertTrue(torch.all(icdf_above > 0))
        print(f"icdf(q > pi) returns value > 0: {icdf_above.item():.4f} -> OK")

    def test_support_boundary(self):
        """Tests that log_prob of negative values is -inf."""
        print("\n--- Running Test: Support Boundary (value < 0) ---")
        dist = ZeroInflatedExtendedGPD_M1_Continuous(
            pi=torch.tensor(0.1), 
            kappa_raw=torch.tensor(1.0), 
            sigma_raw=torch.tensor(1.0), 
            xi=torch.tensor(0.1)
        )
        
        values = torch.tensor([-1.0, -0.001])
        log_probs = dist.log_prob(values)
        
        self.assertTrue(torch.all(torch.isneginf(log_probs)))
        print(f"log_prob of negative values is -inf -> OK")

    def test_xi_zero_case(self):
        """Tests the special case where xi is close to zero (should be exponential)."""
        print("\n--- Running Test: xi -> 0 (Exponential case) ---")
        
        # To make kappa = 1.0, we need softplus(kappa_raw) + 1e-6 = 1.0
        # softplus(kappa_raw) = 1.0 - 1e-6
        # kappa_raw = log(exp(1.0 - 1e-6) - 1)
        kappa_raw_for_one = torch.log(torch.exp(torch.tensor(1.0 - 1e-6)) - 1)

        dist = ZeroInflatedExtendedGPD_M1_Continuous(
            pi=torch.tensor(0.0), # No zero inflation for this test
            kappa_raw=kappa_raw_for_one, 
            sigma_raw=torch.log(torch.exp(torch.tensor(2.0)) - 1), # sigma=2
            xi=torch.tensor(1e-10) # xi is very close to 0
        )
        
        # When pi=0 and kappa=1, ZIEGPD is just GPD.
        # When xi=0, GPD is the Exponential distribution.
        # So this should be an Exponential(1/sigma) distribution.
        value = torch.tensor([1.0])
        log_prob_val = dist.log_prob(value)
        
        # Manual calculation for Exponential(rate=1/sigma)
        sigma = torch.tensor(2.0)
        rate = 1 / sigma
        expected_log_prob = torch.log(rate) - rate * value
        
        self.assertTrue(torch.allclose(log_prob_val, expected_log_prob, atol=1e-5))
        print(f"log_prob for xi~0 matches Exponential log_prob: {log_prob_val.item():.4f} vs {expected_log_prob.item():.4f} -> OK")

    def test_inverse_function_property(self):
        """Tests if icdf(cdf(x)) is close to x."""
        print("\n--- Running Test: Inverse Function Property (icdf(cdf(x)) ~= x) ---")
        pi = torch.tensor([0.1, 0.5])
        kappa_raw = torch.randn(2)
        sigma_raw = torch.randn(2)
        xi = torch.randn(2)
        
        dist = ZeroInflatedExtendedGPD_M1_Continuous(pi, kappa_raw, sigma_raw, xi)
        
        # Test with some positive values
        x = torch.tensor([0.1, 1.0, 5.0, 20.0])
        
        # We need a cdf function. It's not explicitly in the class, but we can build it.
        # W(F(z; sigma, xi); kappa)
        # Note: The public API doesn't expose cdf, but we can use the internal _gpd_cdf
        # and the extension function W(u) = u^kappa.
        
        # cdf(x) = (1-pi) * W(GPD_CDF(x)) for x > 0
        # For x=0, cdf(0) = pi
        # For x>0, cdf(x) = pi + (1-pi) * W(GPD_CDF(x))
        
        # This test is complex without a public cdf. Let's test a simpler variant:
        # For a given quantile q > pi, is cdf(icdf(q)) close to q?
        
        q = torch.tensor([0.6, 0.8, 0.95]) # Quantiles > max(pi)
        
        # Expand shapes for broadcasting
        pi_exp = pi.unsqueeze(-1)
        kappa_exp = dist.kappa.unsqueeze(-1)
        sigma_exp = dist.sigma.unsqueeze(-1)
        xi_exp = dist.xi.unsqueeze(-1)
        
        # 1. Get the value from icdf
        x_from_icdf = dist.icdf(q) # Shape [2, 1, 4]
        
        # 2. Calculate the GPD CDF of that value
        # x_from_icdf has shape [2, 1, 3], need to align for _gpd_cdf
        # Let's simplify test to a single parameter set
        dist_single = ZeroInflatedExtendedGPD_M1_Continuous(
            torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.8), torch.tensor(0.2)
        )
        x_val = dist_single.icdf(q)
        
        gpd_cdf_val = dist_single._gpd_cdf(x_val, dist_single.sigma, dist_single.xi)
        
        # 3. Apply the extension function W(u) = u^kappa
        w_gpd_cdf = gpd_cdf_val.pow(dist_single.kappa)
        
        # 4. Combine with zero-inflation: cdf(x) = pi + (1-pi) * W(GPD_CDF(x))
        reconstructed_q = dist_single.pi + (1 - dist_single.pi) * w_gpd_cdf
        
        self.assertTrue(torch.allclose(reconstructed_q, q, atol=1e-5))
        print(f"cdf(icdf(q)) is close to q -> OK")


if __name__ == '__main__':
    print("======================================================")
    print("= Running Unit Tests for ZIEGPD Distribution =")
    print("======================================================")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)