import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import scipy.stats as stats
import numpy as np

class BGEVDistribution(Distribution):
    arg_constraints = {
        'q_alpha': constraints.real,
        's_beta': constraints.positive,
        'xi': constraints.real,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, q_alpha, s_beta, xi, alpha=0.5, beta=0.5, p_a=0.05, p_b=0.2, c1=5, c2=5, validate_args=None):
        self.q_alpha, self.s_beta, self.xi = broadcast_all(q_alpha, s_beta, xi)
        self.alpha = torch.tensor(alpha, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        self.beta = torch.tensor(beta, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        self.p_a = torch.tensor(p_a, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        self.p_b = torch.tensor(p_b, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        self.c1 = torch.tensor(c1, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        self.c2 = torch.tensor(c2, dtype=self.q_alpha.dtype, device=self.q_alpha.device)
        super().__init__(self.q_alpha.shape, validate_args=validate_args)

    def _l_r(self, a, xi):
        return (-torch.log(a))**(-xi)

    def _l0_r(self, a):
        return torch.log(-torch.log(a))

    def _finverse_r(self, x, q_alpha, s_beta, xi, alpha, beta):
        return ((-torch.log(x))**(-xi) - self._l_r(alpha, xi)) * s_beta / (self._l_r(1 - beta / 2, xi) - self._l_r(beta / 2, xi)) + q_alpha

    def cdf(self, y):
        a = self._finverse_r(self.p_a, self.q_alpha, self.s_beta, self.xi, self.alpha, self.beta)
        b = self._finverse_r(self.p_b, self.q_alpha, self.s_beta, self.xi, self.alpha, self.beta)

        # Upper tail
        denominator_z1 = (self._l_r(1 - self.beta / 2, self.xi) - self._l_r(self.beta / 2, self.xi))
        # Add a small epsilon to prevent division by zero
        denominator_z1 = torch.where(denominator_z1 == 0, torch.tensor(1e-6, device=denominator_z1.device, dtype=denominator_z1.dtype), denominator_z1)
        z1 = (y - self.q_alpha) / (self.s_beta / denominator_z1) + self._l_r(self.alpha, self.xi)
        z1 = torch.clamp(z1, min=1e-6) # Clamp to avoid issues with z1**(-1/xi) if z1 is zero
        t1 = z1**(-1 / self.xi)

        # Weight
        # Ensure inputs to cdf are tensors and handle potential NaN/Inf from division by zero
        denominator_beta = (b - a)
        # Add a small epsilon to prevent division by zero
        denominator_beta = torch.where(denominator_beta == 0, torch.tensor(1e-6, device=denominator_beta.device, dtype=denominator_beta.dtype), denominator_beta)
        beta_cdf_input = (y - a) / denominator_beta
        beta_cdf_input = torch.nan_to_num(beta_cdf_input, nan=0.0, posinf=1.0, neginf=0.0)
        beta_cdf_input = torch.clamp(beta_cdf_input, 0.0, 1.0)

        # Use torch.special.betainc for beta CDF if available and differentiable, otherwise fallback to scipy
        # For now, sticking to scipy as torch.special.betainc might not be directly exposed or stable for all versions
        p = torch.from_numpy(stats.beta.cdf(np.atleast_1d(beta_cdf_input.cpu().detach().numpy()), self.c1.cpu().detach().numpy(), self.c2.cpu().detach().numpy()).astype(np.float32)).to(y.device)

        # Lower tail
        denominator_qa = (self._l0_r(self.p_a) - self._l0_r(self.p_b))
        # Add a small epsilon to prevent division by zero
        denominator_qa = torch.where(denominator_qa == 0, torch.tensor(1e-6, device=denominator_qa.device, dtype=denominator_qa.dtype), denominator_qa)
        q_a_tilde = a - (b - a) * (self._l0_r(self.alpha) - self._l0_r(self.p_a)) / denominator_qa
        
        denominator_sb = (self._l0_r(self.p_a) - self._l0_r(self.p_b))
        # Add a small epsilon to prevent division by zero
        denominator_sb = torch.where(denominator_sb == 0, torch.tensor(1e-6, device=denominator_sb.device, dtype=denominator_sb.dtype), denominator_sb)
        s_b_tilde = (b - a) * (self._l0_r(self.beta / 2) - self._l0_r(1 - self.beta / 2)) / denominator_sb
        
        denominator_z2 = (self._l0_r(self.beta / 2) - self._l0_r(1 - self.beta / 2))
        # Add a small epsilon to prevent division by zero
        denominator_z2 = torch.where(denominator_z2 == 0, torch.tensor(1e-6, device=denominator_z2.device, dtype=denominator_z2.dtype), denominator_z2)
        z2 = (y - q_a_tilde) / (s_b_tilde / denominator_z2) - self._l0_r(self.alpha)
        t2 = torch.exp(-z2)

        out = torch.exp(p * (-t1) + (1 - p) * (-t2))
        
        # Handle edge cases where p is exactly 0 or 1 using torch.where
        out = torch.where(p == 0, torch.exp((1 - p) * (-t2)), out)
        out = torch.where(p == 1, torch.exp(p * (-t1)), out)
        
        return out

    def log_prob(self, y):
        # Ensure cdf output is not zero before taking log
        cdf_val = self.cdf(y)
        return torch.log(torch.clamp(cdf_val, min=1e-30)) # Clamp to avoid log(0)

    def icdf(self, prob):
        # prob has shape [Num_Quantiles]
        # self.q_alpha, self.s_beta, self.xi have shape [B, N_vars, Horizon]

        # Expand prob to be broadcastable with distribution parameters
        # New shape: [1, 1, 1, Num_Quantiles]
        prob_expanded = prob.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Now, _finverse_r will receive prob_expanded with shape [1, 1, 1, Num_Quantiles]
        # and self.q_alpha, self.s_beta, self.xi with shape [B, N_vars, Horizon]
        # The result will be broadcasted to [B, N_vars, Horizon, Num_Quantiles]
        return self._finverse_r(prob_expanded, self.q_alpha, self.s_beta, self.xi, self.alpha, self.beta)