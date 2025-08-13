import torch
import pytest
from ts_benchmark.baselines.duet.bgev_distribution import BGEVDistribution

# Test cases for BGEVDistribution

def test_bgev_distribution_init():
    q_alpha = torch.tensor(0.0)
    s_beta = torch.tensor(1.0)
    xi = torch.tensor(0.1)
    dist = BGEVDistribution(q_alpha, s_beta, xi)
    assert dist is not None

def test_bgev_distribution_cdf():
    q_alpha = torch.tensor(0.0)
    s_beta = torch.tensor(1.0)
    xi = torch.tensor(0.1)
    dist = BGEVDistribution(q_alpha, s_beta, xi)

    # Test a few values
    assert dist.cdf(torch.tensor(0.0)).item() == pytest.approx(0.5, abs=1e-2)
    assert dist.cdf(torch.tensor(1.0)).item() > dist.cdf(torch.tensor(0.0)).item()

def test_bgev_distribution_log_prob():
    q_alpha = torch.tensor(0.0)
    s_beta = torch.tensor(1.0)
    xi = torch.tensor(0.1)
    dist = BGEVDistribution(q_alpha, s_beta, xi)

    # log_prob should be negative (log of a probability)
    log_p = dist.log_prob(torch.tensor(0.0))
    assert log_p.item() < 0

def test_bgev_distribution_icdf():
    q_alpha = torch.tensor(0.0)
    s_beta = torch.tensor(1.0)
    xi = torch.tensor(0.1)
    dist = BGEVDistribution(q_alpha, s_beta, xi)

    # Test median (0.5 quantile) should be close to q_alpha for symmetric-ish distributions
    # This is a very rough check, as bGEV is not necessarily symmetric
    assert dist.icdf(torch.tensor(0.5)).item() == pytest.approx(q_alpha.item(), abs=0.1)

    # Quantiles should be increasing with probability
    assert dist.icdf(torch.tensor(0.9)).item() > dist.icdf(torch.tensor(0.1)).item()

