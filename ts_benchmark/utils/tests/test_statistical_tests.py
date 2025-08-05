
import pytest
import pandas as pd
import numpy as np
from scipy import stats

# Import the function to be tested
from ts_benchmark.utils.statistical_tests import is_significantly_zero_inflated

# --- 1. Define a diverse set of 20 base distributions and their parameters ---
# We use a variety of shapes and parameterizations to ensure robustness.
# All distributions are defined with a location > 0 to ensure the base data is positive.
DISTRIBUTION_CONFIGS = [
    # Symmetric / Normal-like
    ('norm', {'loc': 10, 'scale': 2}),
    ('t', {'df': 5, 'loc': 15, 'scale': 3}),
    ('laplace', {'loc': 20, 'scale': 5}),
    ('logistic', {'loc': 12, 'scale': 2}),

    # Skewed (Right) / Lognormal-like
    ('lognorm', {'s': 0.8, 'loc': 1, 'scale': 10}),
    ('gamma', {'a': 2.5, 'loc': 1, 'scale': 4}),
    ('expon', {'loc': 1, 'scale': 15}),
    ('chi2', {'df': 10, 'loc': 1, 'scale': 2}),
    ('weibull_min', {'c': 1.7, 'loc': 1, 'scale': 8}),
    ('f', {'dfn': 20, 'dfd': 20, 'loc': 1, 'scale': 1}),

    # Flexible / Heavy-tailed
    ('genextreme', {'c': -0.2, 'loc': 10, 'scale': 5}), # Fréchet type
    ('genextreme', {'c': 0.2, 'loc': 10, 'scale': 5}),  # Weibull type
    ('johnsonsu', {'a': 2, 'b': 2, 'loc': 15, 'scale': 10}),
    ('pareto', {'b': 2.5, 'loc': 1, 'scale': 5}),
    ('gumbel_r', {'loc': 20, 'scale': 6}),

    # Other shapes
    ('powerlaw', {'a': 2.0, 'loc': 1, 'scale': 10}),
    ('rayleigh', {'loc': 5, 'scale': 5}),
    ('beta', {'a': 2, 'b': 5, 'loc': 1, 'scale': 20}), # Scaled Beta
    ('cauchy', {'loc': 15, 'scale': 3}),
    ('invgamma', {'a': 2.0, 'loc': 1, 'scale': 5}),
]

# --- 2. Programmatically generate the 60 test cases ---
TEST_CASES = []
SAMPLE_SIZE = 2000
RNG = np.random.default_rng(42) # for reproducibility

for name, params in DISTRIBUTION_CONFIGS:
    # Get the distribution from scipy.stats
    dist = getattr(stats, name)
    
    # a) Generate the base, non-zero-inflated data
    base_data = dist.rvs(size=SAMPLE_SIZE, **params, random_state=RNG)
    series_a = pd.Series(base_data)
    TEST_CASES.append(pytest.param(series_a, False, id=f"{name}-not_inflated"))

    # b) Create a slightly zero-inflated version (15%)
    series_b = series_a.copy()
    indices_to_zero_b = RNG.choice(series_b.index, size=int(SAMPLE_SIZE * 0.15), replace=False)
    series_b[indices_to_zero_b] = 0
    TEST_CASES.append(pytest.param(series_b, True, id=f"{name}-slightly_inflated"))

    # c) Create a more zero-inflated version (40%)
    series_c = series_a.copy()
    indices_to_zero_c = RNG.choice(series_c.index, size=int(SAMPLE_SIZE * 0.40), replace=False)
    series_c[indices_to_zero_c] = 0
    TEST_CASES.append(pytest.param(series_c, True, id=f"{name}-more_inflated"))


# --- 3. Define the test function using pytest.mark.parametrize ---
@pytest.mark.parametrize("series, expected_result", TEST_CASES)
def test_is_significantly_zero_inflated_comprehensive(series, expected_result):
    """
    Tests the is_significantly_zero_inflated function across 60 diverse scenarios.
    
    - 20 different base distributions.
    - 3 levels of zero-inflation for each:
        a) None (expects False)
        b) Slight (15%) (expects True)
        c) More (40%) (expects True)
    """
    # Run the test with verbose=True to get detailed output on failure
    actual_result = is_significantly_zero_inflated(series, verbose=True)
    
    assert actual_result == expected_result
