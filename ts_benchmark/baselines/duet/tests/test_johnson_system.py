
import unittest
import torch
import numpy as np
import scipy.stats
import pandas as pd

from ts_benchmark.baselines.duet.johnson_system import (
    JohnsonSU,
    JohnsonSB,
    JohnsonSL,
    JohnsonSN,
    get_best_johnson_fit,
    JohnsonOutput,
    CombinedJohnsonDistribution,
)
from ts_benchmark.baselines.duet.duet_prob import DUETProb, TransformerConfig, DUETProbModel, DenormalizingDistribution

class TestJohnsonDistributions(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.horizon = 3
        self.n_vars = 4  # One for each distribution type

    def test_johnson_su(self):
        """Test the JohnsonSU wrapper."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSU(
            gamma=torch.tensor([gamma] * self.batch_size, device=self.device),
            delta=torch.tensor([delta] * self.batch_size, device=self.device),
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.randn(self.batch_size, device=self.device)
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.johnsonsu.logpdf(value.cpu().numpy(), a=gamma, b=delta, loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

        q = torch.tensor([0.25, 0.5, 0.75], device=self.device)
        icdf_torch = dist.icdf(q)
        icdf_scipy = scipy.stats.johnsonsu.ppf(q.cpu().numpy(), a=gamma, b=delta, loc=xi, scale=lambda_)
        # Need to adjust for broadcasting
        self.assertTrue(np.allclose(icdf_torch.cpu().numpy(), np.tile(icdf_scipy, (self.batch_size, 1)), atol=1e-6))

    def test_johnson_sb(self):
        """Test the JohnsonSB wrapper."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSB(
            gamma=torch.tensor([gamma] * self.batch_size, device=self.device),
            delta=torch.tensor([delta] * self.batch_size, device=self.device),
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.rand(self.batch_size, device=self.device) # SB is bounded
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.johnsonsb.logpdf(value.cpu().numpy(), a=gamma, b=delta, loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

    def test_johnson_sb_bounds(self):
        """Test JohnsonSB log_prob at and outside bounds."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSB(
            gamma=torch.tensor([gamma], device=self.device),
            delta=torch.tensor([delta], device=self.device),
            xi=torch.tensor([xi], device=self.device),
            lambda_=torch.tensor([lambda_], device=self.device),
        )

        # Value exactly at lower bound
        value_at_lower = torch.tensor([xi], device=self.device)
        log_p_at_lower = dist.log_prob(value_at_lower)
        self.assertTrue(torch.isinf(log_p_at_lower).item() and log_p_at_lower.item() < 0) # Should be -inf

        # Value exactly at upper bound
        value_at_upper = torch.tensor([xi + lambda_], device=self.device)
        log_p_at_upper = dist.log_prob(value_at_upper)
        self.assertTrue(torch.isinf(log_p_at_upper).item() and log_p_at_upper.item() < 0) # Should be -inf

        # Value slightly below lower bound
        value_below = torch.tensor([xi - 0.001], device=self.device)
        log_p_below = dist.log_prob(value_below)
        self.assertTrue(torch.isinf(log_p_below).item() and log_p_below.item() < 0) # Should be -inf

        # Value slightly above upper bound
        value_above = torch.tensor([xi + lambda_ + 0.001], device=self.device)
        log_p_above = dist.log_prob(value_above)
        self.assertTrue(torch.isinf(log_p_above).item() and log_p_above.item() < 0) # Should be -inf

    def test_johnson_sb_small_lambda(self):
        """Test JohnsonSB with very small lambda (scale) to check for numerical stability."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1e-7 # Very small lambda
        dist = JohnsonSB(
            gamma=torch.tensor([gamma], device=self.device),
            delta=torch.tensor([delta], device=self.device),
            xi=torch.tensor([xi], device=self.device),
            lambda_=torch.tensor([lambda_], device=self.device),
        )
        value = torch.tensor([xi + lambda_ / 2], device=self.device) # Value within the tiny range
        log_p = dist.log_prob(value)
        self.assertFalse(torch.isnan(log_p).item()) # Should not be NaN
        self.assertFalse(torch.isinf(log_p).item()) # Should not be inf

    def test_johnson_sl(self):
        """Test the JohnsonSL (log-normal) wrapper."""
        delta, xi, lambda_ = 0.9, 0.0, 1.0 # s, loc, scale
        dist = JohnsonSL(
            delta=torch.tensor([delta] * self.batch_size, device=self.device),
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.rand(self.batch_size, device=self.device) + 0.1 # Must be positive
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.lognorm.logpdf(value.cpu().numpy(), s=delta, loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

    def test_johnson_sl_bounds(self):
        """Test JohnsonSL log_prob at and below xi (loc)."""
        delta, xi, lambda_ = 0.9, 0.0, 1.0
        dist = JohnsonSL(
            delta=torch.tensor([delta], device=self.device),
            xi=torch.tensor([xi], device=self.device),
            lambda_=torch.tensor([lambda_], device=self.device),
        )

        # Value exactly at xi
        value_at_xi = torch.tensor([xi], device=self.device)
        log_p_at_xi = dist.log_prob(value_at_xi)
        self.assertTrue(torch.isinf(log_p_at_xi).item() and log_p_at_xi.item() < 0) # Should be -inf

        # Value slightly below xi
        value_below_xi = torch.tensor([xi - 0.001], device=self.device)
        log_p_below_xi = dist.log_prob(value_below_xi)
        self.assertTrue(torch.isinf(log_p_below_xi).item() and log_p_below_xi.item() < 0) # Should be -inf

        # Value slightly above xi (should be finite)
        value_above_xi = torch.tensor([xi + 0.001], device=self.device)
        log_p_above_xi = dist.log_prob(value_above_xi)
        self.assertFalse(torch.isinf(log_p_above_xi).item()) # Should be finite

    def test_johnson_sl_small_lambda(self):
        """Test JohnsonSL with very small lambda (scale) to check for numerical stability."""
        delta, xi, lambda_ = 0.9, 0.0, 1e-7 # Very small lambda
        dist = JohnsonSL(
            delta=torch.tensor([delta], device=self.device),
            xi=torch.tensor([xi], device=self.device),
            lambda_=torch.tensor([lambda_], device=self.device),
        )
        value = torch.tensor([xi + 0.1], device=self.device) # Value well above xi
        log_p = dist.log_prob(value)
        self.assertFalse(torch.isnan(log_p).item()) # Should not be NaN
        self.assertFalse(torch.isinf(log_p).item()) # Should not be inf

    def test_johnson_sn(self):
        """Test the JohnsonSN (normal) wrapper."""
        xi, lambda_ = 0.0, 1.0 # loc, scale
        dist = JohnsonSN(
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.randn(self.batch_size, device=self.device)
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.norm.logpdf(value.cpu().numpy(), loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

class TestJohnsonFit(unittest.TestCase):
    def test_get_best_johnson_fit(self):
        """Test the fitting function with synthetic data."""
        np.random.seed(42)
        
        # SU data
        data_su = scipy.stats.johnsonsu.rvs(a=0.5, b=1.5, size=1000)
        self.assertEqual(get_best_johnson_fit(data_su), 'SU')

        # SB data
        data_sb = scipy.stats.johnsonsb.rvs(a=0.5, b=1.5, size=1000)
        self.assertEqual(get_best_johnson_fit(data_sb), 'SB')

        # SL (log-normal) data
        data_sl = scipy.stats.lognorm.rvs(s=0.9, size=1000)
        self.assertEqual(get_best_johnson_fit(data_sl), 'SL')

        # SN (normal) data
        data_sn = scipy.stats.norm.rvs(size=1000)
        self.assertEqual(get_best_johnson_fit(data_sn), 'SN')

        # Test edge case: small data
        self.assertEqual(get_best_johnson_fit(np.random.randn(5)), 'SU')
        
        # Test edge case: data with NaNs
        data_with_nan = np.concatenate([data_su, [np.nan]*10])
        self.assertEqual(get_best_johnson_fit(data_with_nan), 'SU')

class TestCombinedJohnson(unittest.TestCase):
    def setUp(self):
        """Set up for combined distribution tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.horizon = 5
        self.channel_types = ['SU', 'SB', 'SL', 'SN']
        self.n_vars = len(self.channel_types)

        # Create dummy parameters
        self.gamma = torch.randn(self.batch_size, self.n_vars, self.horizon, device=self.device)
        self.delta = torch.rand(self.batch_size, self.n_vars, self.horizon, device=self.device) + 0.1
        self.xi = torch.randn(self.batch_size, self.n_vars, self.horizon, device=self.device)
        self.lambda_ = torch.rand(self.batch_size, self.n_vars, self.horizon, device=self.device) + 0.1

        self.combined_dist = CombinedJohnsonDistribution(
            channel_types=self.channel_types,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_
        )

    def test_combined_masks(self):
        """Test if the masks for different types are created correctly."""
        self.assertTrue(torch.equal(self.combined_dist.su_mask, torch.tensor([True, False, False, False], device=self.device)))
        self.assertTrue(torch.equal(self.combined_dist.sb_mask, torch.tensor([False, True, False, False], device=self.device)))
        self.assertTrue(torch.equal(self.combined_dist.sl_mask, torch.tensor([False, False, True, False], device=self.device)))
        self.assertTrue(torch.equal(self.combined_dist.sn_mask, torch.tensor([False, False, False, True], device=self.device)))

    def test_combined_log_prob(self):
        """Test if log_prob dispatches correctly."""
        value = torch.randn(self.batch_size, self.n_vars, self.horizon, device=self.device)
        log_p_combined = self.combined_dist.log_prob(value)

        # Manual calculation
        log_p_manual = torch.zeros_like(log_p_combined)
        
        # SU
        su_dist = JohnsonSU(self.gamma[:, 0, :], self.delta[:, 0, :], self.xi[:, 0, :], self.lambda_[:, 0, :])
        log_p_manual[:, 0, :] = su_dist.log_prob(value[:, 0, :])
        
        # SB
        sb_dist = JohnsonSB(self.gamma[:, 1, :], self.delta[:, 1, :], self.xi[:, 1, :], self.lambda_[:, 1, :])
        log_p_manual[:, 1, :] = sb_dist.log_prob(value[:, 1, :])

        # SL
        sl_dist = JohnsonSL(self.delta[:, 2, :], self.xi[:, 2, :], self.lambda_[:, 2, :])
        log_p_manual[:, 2, :] = sl_dist.log_prob(value[:, 2, :])

        # SN
        sn_dist = JohnsonSN(self.xi[:, 3, :], self.lambda_[:, 3, :])
        log_p_manual[:, 3, :] = sn_dist.log_prob(value[:, 3, :])

        self.assertTrue(torch.allclose(log_p_combined, log_p_manual, atol=1e-5))

    def test_combined_log_prob_out_of_bounds(self):
        """Test if log_prob correctly handles out-of-bounds values for combined distributions."""
        # Create a value tensor with some out-of-bounds values
        value = torch.randn(self.batch_size, self.n_vars, self.horizon, device=self.device)

        # For SB (channel 1): Set some values outside [xi, xi + lambda_]
        # Ensure xi and lambda_ are positive for this test to make sense
        self.xi[:, 1, :] = 0.0
        self.lambda_[:, 1, :] = 1.0
        value[:, 1, 0] = -0.1  # Below lower bound
        value[:, 1, 1] = 1.1   # Above upper bound
        # Ensure remaining values are well within bounds for SB
        # For SB, values must be between xi and xi + lambda_
        value[:, 1, 2:] = self.xi[:, 1, 2:] + (self.lambda_[:, 1, 2:] * (torch.rand(self.batch_size, self.horizon - 2, device=self.device) * 0.8 + 0.1)) # (0.1, 0.9) of the range

        # For SL (channel 2): Set some values <= xi
        self.xi[:, 2, :] = 0.0
        value[:, 2, 0] = -0.1  # Less than xi
        value[:, 2, 1] = 0.0   # Equal to xi
        # Ensure remaining values are well within bounds for SL (i.e., > xi)
        # For SL, values must be > xi
        value[:, 2, 2:] = self.xi[:, 2, 2:] + (self.lambda_[:, 2, 2:] * (torch.rand(self.batch_size, self.horizon - 2, device=self.device) * 0.8 + 0.1)) # (0.1, 0.9) of the scale above xi

        log_p_combined = self.combined_dist.log_prob(value)

        # Assert -inf for out-of-bounds SB values
        self.assertTrue(torch.isinf(log_p_combined[:, 1, 0]).all() and (log_p_combined[:, 1, 0] < 0).all())
        self.assertTrue(torch.isinf(log_p_combined[:, 1, 1]).all() and (log_p_combined[:, 1, 1] < 0).all())

        # Assert -inf for out-of-bounds SL values
        self.assertTrue(torch.isinf(log_p_combined[:, 2, 0]).all() and (log_p_combined[:, 2, 0] < 0).all())
        self.assertTrue(torch.isinf(log_p_combined[:, 2, 1]).all() and (log_p_combined[:, 2, 1] < 0).all())

        # Assert finite values for other (in-bounds) values (e.g., SU, SN, and other SB/SL values)
        self.assertFalse(torch.isinf(log_p_combined[:, 0, :]).any())
        self.assertFalse(torch.isinf(log_p_combined[:, 3, :]).any())
        self.assertFalse(torch.isinf(log_p_combined[:, 1, 2:]).any())
        self.assertFalse(torch.isinf(log_p_combined[:, 2, 2:]).any())

    def test_combined_icdf(self):
        """Test if icdf dispatches correctly."""
        q = torch.tensor([0.2, 0.8], device=self.device)
        icdf_combined = self.combined_dist.icdf(q)

        self.assertEqual(icdf_combined.shape, (self.batch_size, self.n_vars, self.horizon, len(q)))

class TestIntegrationWithDuetProb(unittest.TestCase):
    def setUp(self):
        """Set up a minimal DUETProb model for integration testing."""
        self.seq_len = 96
        self.horizon = 24
        self.n_vars = 4
        self.channel_types = ['SU', 'SB', 'SL', 'SN']

        # Create a minimal config
        self.config = TransformerConfig(
            seq_len=self.seq_len,
            horizon=self.horizon,
            enc_in=self.n_vars,
            dec_in=self.n_vars,
            c_out=self.n_vars,
            d_model=32,
            d_ff=64,
            n_heads=2,
            e_layers=1,
            num_linear_experts=1,
            num_univariate_esn_experts=1,
            num_multivariate_esn_experts=0,
            johnson_channel_types=self.channel_types,
            channel_bounds={'channel_0': {'lower': 0, 'upper': 1}, 'channel_1': {'lower': 0, 'upper': 1}, 'channel_2': {'lower': 0, 'upper': 1}, 'channel_3': {'lower': 0, 'upper': 1}}
        )

        # Build the model
        self.model = DUETProbModel(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_forward_pass_with_johnson(self):
        """Test a full forward pass with the JohnsonOutput layer."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, self.seq_len, self.n_vars, device=self.device)

        # The model returns multiple values, we are interested in the first one (the distribution)
        denorm_distr, base_distr, _, _, _, _, _, _, _ = self.model(input_tensor)

        # Check if the output is the correct distribution type
        self.assertIsInstance(denorm_distr, DenormalizingDistribution)
        
        # Check if the parameters in the distribution have the correct shape
        self.assertEqual(denorm_distr.base_dist.gamma.shape, (batch_size, self.n_vars, self.horizon))
        self.assertEqual(denorm_distr.base_dist.delta.shape, (batch_size, self.n_vars, self.horizon))
        self.assertEqual(denorm_distr.base_dist.xi.shape, (batch_size, self.n_vars, self.horizon))
        self.assertEqual(denorm_distr.base_dist.lambda_.shape, (batch_size, self.n_vars, self.horizon))

        # Check if the masks are set up correctly inside the distribution
        self.assertTrue(torch.equal(denorm_distr.base_dist.su_mask, torch.tensor([True, False, False, False], device=self.device)))
        self.assertTrue(torch.equal(denorm_distr.base_dist.sb_mask, torch.tensor([False, True, False, False], device=self.device)))
        self.assertTrue(torch.equal(denorm_distr.base_dist.sl_mask, torch.tensor([False, False, True, False], device=self.device)))
        self.assertTrue(torch.equal(denorm_distr.base_dist.sn_mask, torch.tensor([False, False, False, True], device=self.device)))

        # Test log_prob and icdf to ensure they run without errors
        value = torch.randn(batch_size, self.horizon, self.n_vars, device=self.device)
        log_p = denorm_distr.log_prob(value)
        self.assertEqual(log_p.shape, (batch_size, self.n_vars, self.horizon))

        q = torch.tensor([0.5], device=self.device)
        quantiles = denorm_distr.icdf(q)
        self.assertEqual(quantiles.shape, (batch_size, self.n_vars, self.horizon, 1))


if __name__ == '__main__':
    unittest.main()
