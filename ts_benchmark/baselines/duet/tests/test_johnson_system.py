import unittest
import torch
import numpy as np
import scipy.stats
import pandas as pd
import math # Import math module

from ts_benchmark.baselines.duet.johnson_system import (
    JohnsonSU_torch, # Changed to _torch
    JohnsonSB_torch, # Changed to _torch
    JohnsonSL_scipy, # Still using scipy for SL icdf, but log_prob is handled by torch.distributions.LogNormal
    JohnsonSN_scipy, # Still using scipy for SN icdf, but log_prob is handled by torch.distributions.Normal
    get_best_johnson_fit,
    JohnsonOutput,
    CombinedJohnsonDistribution,
)
from ts_benchmark.baselines.duet.duet_prob import DUETProb, TransformerConfig
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel, DenormalizingDistribution

class TestJohnsonDistributions(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.horizon = 3
        self.n_vars = 4  # One for each distribution type

    def test_johnson_su(self):
        """Test the JohnsonSU_torch log_prob."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSU_torch(
            gamma=torch.tensor([gamma] * self.batch_size, device=self.device),
            delta=torch.tensor([delta] * self.batch_size, device=self.device),
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.randn(self.batch_size, device=self.device)
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.johnsonsu.logpdf(value.cpu().numpy(), a=gamma, b=delta, loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

    def test_johnson_sb(self):
        """Test the JohnsonSB_torch log_prob."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSB_torch(
            gamma=torch.tensor([gamma] * self.batch_size, device=self.device),
            delta=torch.tensor([delta] * self.batch_size, device=self.device),
            xi=torch.tensor([xi] * self.batch_size, device=self.device),
            lambda_=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.rand(self.batch_size, device=self.device) * lambda_ + xi # SB is bounded
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.johnsonsb.logpdf(value.cpu().numpy(), a=gamma, b=delta, loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))

    def test_johnson_sb_bounds(self):
        """Test JohnsonSB_torch log_prob at and outside bounds."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1.0
        dist = JohnsonSB_torch(
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
        """Test JohnsonSB_torch with very small lambda (scale) to check for numerical stability."""
        gamma, delta, xi, lambda_ = 0.5, 1.5, 0.0, 1e-7 # Very small lambda
        dist = JohnsonSB_torch(
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
        """Test the JohnsonSL (log-normal) log_prob using torch.distributions.LogNormal."""
        # Note: The JohnsonSL_scipy is only for icdf. log_prob is handled by torch.distributions.LogNormal
        # Parameters for torch.distributions.LogNormal are mean and std of the underlying normal distribution
        mean_log, std_log = 0.0, 0.9 # These map to loc and scale in scipy.stats.lognorm
        
        # Create a LogNormal distribution using PyTorch's native implementation
        dist_torch = torch.distributions.LogNormal(
            loc=torch.tensor([mean_log] * self.batch_size, device=self.device),
            scale=torch.tensor([std_log] * self.batch_size, device=self.device)
        )
        
        value = torch.rand(self.batch_size, device=self.device) + 0.1 # Must be positive
        log_p_torch = dist_torch.log_prob(value)
        
        # Compare with scipy.stats.lognorm.logpdf
        # Note: scipy.stats.lognorm uses 's' as shape, 'loc' as location, 'scale' as scale
        # For a standard LogNormal, loc=0, scale=exp(mean_log), s=std_log
        # However, the JohnsonSL_scipy uses delta, xi, lambda_ which map to s, loc, scale directly.
        # So, we use the parameters as they are in JohnsonSL_scipy for scipy comparison.
        log_p_scipy = scipy.stats.lognorm.logpdf(value.cpu().numpy(), s=std_log, loc=0.0, scale=torch.exp(torch.tensor(mean_log)).item())
        
        # Let's use the parameters as they are passed to the torch.distributions.LogNormal in CombinedJohnsonDistribution
        xi_param, lambda_param = 0.0, 1.0 # These are loc and scale for torch.distributions.LogNormal
        dist_torch_combined_style = torch.distributions.LogNormal(
            loc=torch.tensor([xi_param] * self.batch_size, device=self.device),
            scale=torch.tensor([lambda_param] * self.batch_size, device=self.device)
        )
        log_p_torch_combined_style = dist_torch_combined_style.log_prob(value)
        
        # For scipy.stats.lognorm, s is shape, loc is location, scale is scale.
        # If torch.distributions.LogNormal(loc, scale) corresponds to log(X) ~ N(loc, scale^2),
        # then X ~ LogNormal(s=scale, loc=0, scale=exp(loc)) in scipy.
        log_p_scipy_combined_style = scipy.stats.lognorm.logpdf(value.cpu().numpy(), s=lambda_param, loc=0.0, scale=np.exp(xi_param))
        
        self.assertTrue(np.allclose(log_p_torch_combined_style.cpu().numpy(), log_p_scipy_combined_style, atol=1e-5))


    def test_johnson_sl_bounds(self):
        """Test JohnsonSL log_prob at and below xi (loc)."""
        # For torch.distributions.LogNormal(loc, scale), the support is (0, inf)
        # The 'loc' parameter in torch.distributions.LogNormal is the mean of the underlying normal distribution.
        # The 'scale' parameter is the standard deviation of the underlying normal distribution.
        # The actual location parameter for the LogNormal distribution itself is usually 0.
        
        # So, values <= 0 should have -inf log_prob.
        loc_param, scale_param = 0.0, 1.0
        dist = torch.distributions.LogNormal(
            loc=torch.tensor([loc_param], device=self.device),
            scale=torch.tensor([scale_param], device=self.device)
        )

        # Value exactly at 0
        value_at_zero = torch.tensor([0.0], device=self.device)
        with self.assertRaises(ValueError):
            dist.log_prob(value_at_zero)

        # Value slightly below 0
        value_below_zero = torch.tensor([-0.001], device=self.device)
        with self.assertRaises(ValueError):
            dist.log_prob(value_below_zero)

        # Value slightly above 0 (should be finite)
        value_above_zero = torch.tensor([0.001], device=self.device)
        log_p_above_zero = dist.log_prob(value_above_zero)
        self.assertFalse(torch.isinf(log_p_above_zero).item()) # Should be finite

    def test_johnson_sl_small_lambda(self):
        """Test JohnsonSL with very small lambda (scale) to check for numerical stability."""
        # Here, lambda_ refers to the scale parameter of the underlying normal distribution.
        # A very small scale means a very sharp distribution.
        loc_param, scale_param = 0.0, 1e-7 # Very small scale
        dist = torch.distributions.LogNormal(
            loc=torch.tensor([loc_param], device=self.device),
            scale=torch.tensor([scale_param], device=self.device)
        )
        value = torch.tensor([1.0], device=self.device) # A reasonable positive value
        log_p = dist.log_prob(value)
        self.assertFalse(torch.isnan(log_p).item()) # Should not be NaN
        self.assertFalse(torch.isinf(log_p).item()) # Should not be inf

    def test_johnson_sn(self):
        """Test the JohnsonSN (normal) log_prob using torch.distributions.Normal."""
        xi, lambda_ = 0.0, 1.0 # loc, scale for Normal distribution
        dist = torch.distributions.Normal(
            loc=torch.tensor([xi] * self.batch_size, device=self.device),
            scale=torch.tensor([lambda_] * self.batch_size, device=self.device),
        )
        value = torch.randn(self.batch_size, device=self.device)
        log_p_torch = dist.log_prob(value)
        log_p_scipy = scipy.stats.norm.logpdf(value.cpu().numpy(), loc=xi, scale=lambda_)
        self.assertTrue(np.allclose(log_p_torch.cpu().numpy(), log_p_scipy, atol=1e-6))



class TestJohnsonFit(unittest.TestCase):
    def test_get_best_johnson_fit(self):
        """Test the fitting function with synthetic data."""
        np.random.seed(42)
        
        # SU data - parameters to make it clearly non-normal
        data_su = scipy.stats.johnsonsu.rvs(a=5.0, b=2.0, size=5000) 
        self.assertEqual(get_best_johnson_fit(data_su), 'SU')

        # SB data - using a wider, more stable numerical range to avoid optimizer instability
        data_sb = scipy.stats.johnsonsb.rvs(a=0.5, b=1.5, loc=10, scale=5, size=2000) 
        self.assertEqual(get_best_johnson_fit(data_sb), 'SB')

        # SL (log-normal) data
        data_sl = scipy.stats.lognorm.rvs(s=0.9, size=2000)
        self.assertIn(get_best_johnson_fit(data_sl), ['SL', 'SU'])

        # SN (normal) data
        data_sn = scipy.stats.norm.rvs(size=2000)
        self.assertEqual(get_best_johnson_fit(data_sn), 'SN')

        # Test edge case: small data
        self.assertEqual(get_best_johnson_fit(np.random.randn(15)), 'SU')
        
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
        
        # Ensure SL channel (index 2) has positive values for LogNormal
        value[:, 2, :] = torch.rand(self.batch_size, self.horizon, device=self.device) * 5 + 0.1 # Small positive values

        log_p_combined = self.combined_dist.log_prob(value)

        # Manual calculation
        log_p_manual = torch.zeros_like(log_p_combined)
        
        # SU
        su_dist = JohnsonSU_torch(self.gamma[:, 0, :], self.delta[:, 0, :], self.xi[:, 0, :], self.lambda_[:, 0, :])
        log_p_manual[:, 0, :] = su_dist.log_prob(value[:, 0, :])
        
        # SB
        sb_dist = JohnsonSB_torch(self.gamma[:, 1, :], self.delta[:, 1, :], self.xi[:, 1, :], self.lambda_[:, 1, :])
        log_p_manual[:, 1, :] = sb_dist.log_prob(value[:, 1, :])

        # SL (using torch.distributions.LogNormal)
        sl_dist = torch.distributions.LogNormal(self.xi[:, 2, :], self.lambda_[:, 2, :])
        log_p_manual[:, 2, :] = sl_dist.log_prob(value[:, 2, :])

        # SN (using torch.distributions.Normal)
        sn_dist = torch.distributions.Normal(self.xi[:, 3, :], self.lambda_[:, 3, :])
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
        value[:, 1, 2:] = self.xi[:, 1, 2:] + (torch.rand(self.batch_size, self.horizon - 2, device=self.device) * 0.8 + 0.1)

        # For SL (channel 2): Set some values <= 0 (since its support is (0, inf))
        value[:, 2, 0] = -0.1  # Less than 0
        value[:, 2, 1] = 0.0   # Equal to 0
        # Ensure remaining values are well within bounds for SL (i.e., > 0)
        value[:, 2, 2:] = torch.rand(self.batch_size, self.horizon - 2, device=self.device) * 5 + 0.1 # Small positive values

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
        
        # Ensure SL channel (index 2) has positive values for LogNormal
        value[:, :, 2] = torch.rand(batch_size, self.horizon, device=self.device) * 5 + 0.1 # Small positive values

        log_p = denorm_distr.log_prob(value)
        self.assertEqual(log_p.shape, (batch_size, self.n_vars, self.horizon))

        q = torch.tensor([0.5], device=self.device)
        quantiles = denorm_distr.icdf(q)
        self.assertEqual(quantiles.shape, (batch_size, self.n_vars, self.horizon, 1))

class TestNLLCalculation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.horizon = 10
        self.n_vars = 1 # Test with a single variable for simplicity
        torch.manual_seed(42)

    def _create_dummy_denorm_dist(self, base_dist_type, mean_val, std_val, **kwargs): # Use kwargs for flexibility
        # Create dummy stats for DenormalizingDistribution
        stats = torch.zeros(self.batch_size, self.n_vars, 2, device=self.device)
        stats[:, :, 0] = mean_val # Mean
        stats[:, :, 1] = std_val # Std

        # Default parameters for CombinedJohnsonDistribution
        gamma = torch.full((self.batch_size, self.n_vars, self.horizon), kwargs.get('gamma', 0.0), device=self.device)
        delta = torch.full((self.batch_size, self.n_vars, self.horizon), kwargs.get('delta', 1.0), device=self.device)
        # For SN and SL, xi and lambda_ should be 0.0 and 1.0 for the *normalized* distribution
        if base_dist_type == 'SN' or base_dist_type == 'SL':
            xi = torch.full((self.batch_size, self.n_vars, self.horizon), 0.0, device=self.device)
            lambda_ = torch.full((self.batch_size, self.n_vars, self.horizon), 1.0, device=self.device)
        else:
            xi = torch.full((self.batch_size, self.n_vars, self.horizon), kwargs.get('xi', 0.0), device=self.device)
            lambda_ = torch.full((self.batch_size, self.n_vars, self.horizon), kwargs.get('lambda_', 1.0), device=self.device)

        channel_types = [base_dist_type]

        base_dist = CombinedJohnsonDistribution(
            channel_types=channel_types,
            gamma=gamma, delta=delta, xi=xi, lambda_=lambda_
        )
        return DenormalizingDistribution(base_dist, stats)

    def test_nll_perfect_forecast_normal(self):
        """Test NLL for a perfect forecast with a Normal distribution."""
        mean_val = 5.0
        std_val = 1.0
        # For Normal, xi is loc, lambda_ is scale
        denorm_dist = self._create_dummy_denorm_dist('SN', mean_val, std_val)
        
        # Observation is exactly the mean
        observation = torch.full((self.batch_size, self.horizon, self.n_vars), mean_val, device=self.device)
        
        nll = -denorm_dist.log_prob(observation).mean().item()
        
        # Expected NLL for Normal(mu, sigma) at mu is -log(1/(sigma*sqrt(2*pi))) = log(sigma) + 0.5*log(2*pi)
        # NLL_denorm = NLL_base + log(std_of_denormalizing_distribution)
        # NLL_base for Normal(0,1) at 0 is 0.5 * math.log(2 * math.pi)
        expected_nll = 0.5 * math.log(2 * math.pi) + torch.log(torch.tensor(std_val)).item()

        self.assertAlmostEqual(nll, expected_nll, places=4)

    def test_nll_high_uncertainty_correct_normal(self):
        """Test NLL for high uncertainty but correct observation (Normal)."""
        mean_val = 5.0
        std_val = 10.0 # High uncertainty
        denorm_dist = self._create_dummy_denorm_dist('SN', mean_val, std_val)
        
        # Observation is still the mean
        observation = torch.full((self.batch_size, self.horizon, self.n_vars), mean_val, device=self.device)
        
        nll = -denorm_dist.log_prob(observation).mean().item()
        
        expected_nll = 0.5 * math.log(2 * math.pi) + torch.log(torch.tensor(std_val)).item()

        self.assertAlmostEqual(nll, expected_nll, places=4)
        # NLL should be higher than low uncertainty case due to log(std) term
        # (log(10) > log(1))
        self.assertGreater(nll, 0.5 * math.log(2 * math.pi) + torch.log(torch.tensor(1.0)).item() - 1e-5)

    def test_nll_low_uncertainty_incorrect_normal(self):
        """Test NLL for low uncertainty and incorrect observation (Normal)."""
        mean_val = 5.0
        std_val = 0.1 # Low uncertainty
        denorm_dist = self._create_dummy_denorm_dist('SN', mean_val, std_val)
        
        # Observation is far from the mean
        observation = torch.full((self.batch_size, self.horizon, self.n_vars), mean_val + 1.0, device=self.device)
        nll = -denorm_dist.log_prob(observation).mean().item()
        # Normalized observation: (mean_val + 1.0 - mean_val) / 0.1 = 1.0 / 0.1 = 10.0
        # NLL_denorm = 0.5 * math.log(2 * math.pi) + 0.5 * x_norm^2 + math.log(std)
        x_norm = (mean_val + 1.0 - mean_val) / std_val
        expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * x_norm**2 + torch.log(torch.tensor(std_val)).item()

        self.assertAlmostEqual(nll, expected_nll, places=4)
        # NLL should be very high due to being confident and wrong
        self.assertGreater(nll, 10.0) # Arbitrary high value, but should be significantly higher

    def test_nll_sb_out_of_bounds(self):
        """Test NLL for Johnson SB when observation is out of bounds."""
        mean_val = 0.5 # Dummy mean for normalization
        std_val = 0.1 # Dummy std for normalization
        # SB parameters: gamma, delta, xi, lambda_
        # Support is [xi, xi + lambda_]
        xi_sb = 0.0
        lambda_sb = 1.0
        denorm_dist = self._create_dummy_denorm_dist('SB', mean_val, std_val, gamma=0.0, delta=1.0, xi=xi_sb, lambda_=lambda_sb)
        
        # Observation outside the support
        observation_below = torch.full((self.batch_size, self.horizon, self.n_vars), xi_sb - 0.1, device=self.device)
        observation_above = torch.full((self.batch_size, self.horizon, self.n_vars), xi_sb + lambda_sb + 0.1, device=self.device)
        
        nll_below = -denorm_dist.log_prob(observation_below).mean().item()
        nll_above = -denorm_dist.log_prob(observation_above).mean().item()
        
        self.assertTrue(np.isinf(nll_below) and nll_below > 0) # NLL should be +inf
        self.assertTrue(np.isinf(nll_above) and nll_above > 0) # NLL should be +inf

    def test_nll_sl_out_of_bounds(self):
        """Test NLL for Johnson SL (LogNormal) when observation is out of bounds."""
        mean_val = 1.0 # Dummy mean for normalization
        std_val = 0.5 # Dummy std for normalization
        # SL parameters: xi (loc of normal), lambda_ (scale of normal)
        # Support is (0, inf)
        xi_sl = 0.0 # Mean of underlying normal
        lambda_sl = 1.0 # Std of underlying normal
        denorm_dist = self._create_dummy_denorm_dist('SL', mean_val, std_val)
        
        # Observation outside the support (<= 0)
        observation_at_zero = torch.full((self.batch_size, self.horizon, self.n_vars), 0.0, device=self.device)
        observation_below_zero = torch.full((self.batch_size, self.horizon, self.n_vars), -0.1, device=self.device)
        
        nll_at_zero = -denorm_dist.log_prob(observation_at_zero).mean().item()
        nll_below_zero = -denorm_dist.log_prob(observation_below_zero).mean().item()
        
        self.assertTrue(np.isinf(nll_at_zero) and nll_at_zero > 0) # NLL should be +inf
        self.assertTrue(np.isinf(nll_below_zero) and nll_below_zero > 0) # NLL should be +inf

    def test_nll_dry_period_scenario(self):
        """Mimic the dry period scenario with low NLL."""
        # Model predicts very low values with low uncertainty (e.g., Normal distribution around 0)
        mean_val = 0.01 # Very small predicted rain
        std_val = 0.005 # Very low uncertainty
        denorm_dist = self._create_dummy_denorm_dist('SN', mean_val, std_val)
        
        # Actual observation is 0 (no rain)
        observation = torch.full((self.batch_size, self.horizon, self.n_vars), 0.0, device=self.device)
        nll = -denorm_dist.log_prob(observation).mean().item()
        
        # Calculate expected NLL for Normal(0.01, 0.005) at 0.0
        # Normalized value: (0.0 - 0.01) / 0.005 = -2.0
        x_norm = (0.0 - mean_val) / std_val
        expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * x_norm**2 + torch.log(torch.tensor(std_val)).item()

        self.assertAlmostEqual(nll, expected_nll, places=4)
        # This NLL should be relatively low, reflecting a good forecast for a dry period.
        self.assertLess(nll, 5.0) # Arbitrary threshold, but should be low

    def test_nll_rainy_period_scenario(self):
        """Mimic the rainy period scenario with higher NLL."""
        # Model predicts higher values with higher uncertainty (e.g., Normal distribution around 3.5)
        mean_val = 3.5 # Predicted rain
        std_val = 1.0 # Higher uncertainty
        denorm_dist = self._create_dummy_denorm_dist('SN', mean_val, std_val)
        
        # Actual observation is 3.5 (perfect hit for the mean)
        observation = torch.full((self.batch_size, self.horizon, self.n_vars), 3.5, device=self.device)
        nll = -denorm_dist.log_prob(observation).mean().item()
        
        # Calculate expected NLL for Normal(3.5, 1.0) at 3.5
        # Normalized value: (3.5 - 3.5) / 1.0 = 0.0
        x_norm = (3.5 - mean_val) / std_val
        expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * x_norm**2 + torch.log(torch.tensor(std_val)).item()

        self.assertAlmostEqual(nll, expected_nll, places=4)
        # This NLL should be higher than the dry period scenario if std_val is larger.
        # log(1.0) = 0, log(0.005) = -5.29. So, the NLL for rainy period should be higher.
        self.assertGreater(nll, 0.0) # Should be positive
        self.assertGreater(nll, 0.5 * math.log(2 * math.pi) + torch.log(torch.tensor(0.005)).item() + 1e-5) # Compare to dry scenario


if __name__ == '__main__':
    unittest.main()