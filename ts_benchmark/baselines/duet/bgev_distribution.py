import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import scipy.stats as stats
import numpy as np
import sys
from tqdm import tqdm

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
        super().__init__(batch_shape=self.q_alpha.shape[:-1], event_shape=self.q_alpha.shape[-1:], validate_args=validate_args) # Corrected batch_shape and event_shape
        print(f"DEBUG: BGEVDistribution __init__ - q_alpha stats: mean={self.q_alpha.mean():.6f}, std={self.q_alpha.std():.6f}, min={self.q_alpha.min():.6f}, max={self.q_alpha.max():.6f}")
        print(f"DEBUG: BGEVDistribution __init__ - s_beta stats: mean={self.s_beta.mean():.6f}, std={self.s_beta.std():.6f}, min={self.s_beta.min():.6f}, max={self.s_beta.max():.6f}")
        print(f"DEBUG: BGEVDistribution __init__ - xi stats: mean={self.xi.mean():.6f}, std={self.xi.std():.6f}, min={self.xi.min():.6f}, max={self.xi.max():.6f}")
        sys.stdout.flush()

    def _l_r(self, a, xi):
        log_a = torch.log(a)
        print(f"DEBUG: _l_r - log_a stats: mean={log_a.mean():.6f}, std={log_a.std():.6f}, min={log_a.min():.6f}, max={log_a.max():.6f}")
        result = (-log_a)**(-xi)
        print(f"DEBUG: _l_r - result stats: mean={result.mean():.6f}, std={result.std():.6f}, min={result.min():.6f}, max={result.max():.6f}")
        return result

    def _l0_r(self, a):
        log_a = torch.log(a)
        print(f"DEBUG: _l0_r - log_a stats: mean={log_a.mean():.6f}, std={log_a.std():.6f}, min={log_a.min():.6f}, max={log_a.max():.6f}")
        result = torch.log(-log_a)
        print(f"DEBUG: _l0_r - result stats: mean={result.mean():.6f}, std={result.std():.6f}, min={result.min():.6f}, max={result.max():.6f}")
        return result

    def _finverse_r(self, x, q_alpha, s_beta, xi, alpha, beta):
        print(f"DEBUG: _finverse_r - Input x stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
        term1_log = -torch.log(x)
        print(f"DEBUG: _finverse_r - term1_log stats: mean={term1_log.mean():.6f}, std={term1_log.std():.6f}, min={term1_log.min():.6f}, max={term1_log.max():.6f}")
        term1 = term1_log**(-xi)
        print(f"DEBUG: _finverse_r - term1 stats: mean={term1.mean():.6f}, std={term1.std():.6f}, min={term1.min():.6f}, max={term1.max():.6f}")
        term2 = self._l_r(alpha, xi)
        print(f"DEBUG: _finverse_r - term2 stats: mean={term2.mean():.6f}, std={term2.std():.6f}, min={term2.min():.6f}, max={term2.max():.6f}")
        
        numerator = term1 - term2
        print(f"DEBUG: _finverse_r - numerator stats: mean={numerator.mean():.6f}, std={numerator.std():.6f}, min={numerator.min():.6f}, max={numerator.max():.6f}")

        denominator = (self._l_r(1 - beta / 2, xi) - self._l_r(beta / 2, xi))
        print(f"DEBUG: _finverse_r - denominator stats: mean={denominator.mean():.6f}, std={denominator.std():.6f}, min={denominator.min():.6f}, max={denominator.max():.6f}")

        # Add a small epsilon to prevent division by zero
        denominator = torch.where(denominator == 0, torch.tensor(1e-6, device=denominator.device, dtype=denominator.dtype), denominator)
        
        result = numerator * s_beta / denominator + q_alpha
        print(f"DEBUG: _finverse_r - result stats: mean={result.mean():.6f}, std={result.std():.6f}, min={result.min():.6f}, max={result.max():.6f}")
        return result

    def cdf(self, y):
        print(f"DEBUG: cdf - Input y stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")
        a = self._finverse_r(self.p_a, self.q_alpha, self.s_beta, self.xi, self.alpha, self.beta)
        print(f"DEBUG: cdf - a stats: mean={a.mean():.6f}, std={a.std():.6f}, min={a.min():.6f}, max={a.max():.6f}")
        b = self._finverse_r(self.p_b, self.q_alpha, self.s_beta, self.xi, self.alpha, self.beta)
        print(f"DEBUG: cdf - b stats: mean={b.mean():.6f}, std={b.std():.6f}, min={b.min():.6f}, max={b.max():.6f}")

        # Upper tail
        denominator_z1 = (self._l_r(1 - self.beta / 2, self.xi) - self._l_r(self.beta / 2, self.xi))
        print(f"DEBUG: cdf - denominator_z1 stats: mean={denominator_z1.mean():.6f}, std={denominator_z1.std():.6f}, min={denominator_z1.min():.6f}, max={denominator_z1.max():.6f}")
        # Add a small epsilon to prevent division by zero
        denominator_z1 = torch.where(denominator_z1 == 0, torch.tensor(1e-6, device=denominator_z1.device, dtype=denominator_z1.dtype), denominator_z1)
        z1 = (y - self.q_alpha) / (self.s_beta / denominator_z1) + self._l_r(self.alpha, self.xi)
        print(f"DEBUG: cdf - z1 stats: mean={z1.mean():.6f}, std={z1.std():.6f}, min={z1.min():.6f}, max={z1.max():.6f}")
        z1 = torch.clamp(z1, min=1e-6) # Clamp to avoid issues with z1**(-1/xi) if z1 is zero
        t1 = torch.exp(-z1)
        print(f"DEBUG: cdf - t1 stats: mean={t1.mean():.6f}, std={t1.std():.6f}, min={t1.min():.6f}, max={t1.max():.6f}")

        # Weight
        # Ensure inputs to cdf are tensors and handle potential NaN/Inf from division by zero
        denominator_beta = (b - a)
        print(f"DEBUG: cdf - denominator_beta stats: mean={denominator_beta.mean():.6f}, std={denominator_beta.std():.6f}, min={denominator_beta.min():.6f}, max={denominator_beta.max():.6f}")
        # Add a small epsilon to prevent division by zero
        denominator_beta = torch.where(denominator_beta == 0, torch.tensor(1e-6, device=denominator_beta.device, dtype=denominator_beta.dtype), denominator_beta)
        beta_cdf_input = (y - a) / denominator_beta
        print(f"DEBUG: cdf - beta_cdf_input (before clamp) stats: mean={beta_cdf_input.mean():.6f}, std={beta_cdf_input.std():.6f}, min={beta_cdf_input.min():.6f}, max={beta_cdf_input.max():.6f}")
        beta_cdf_input = torch.nan_to_num(beta_cdf_input, nan=0.0, posinf=1.0, neginf=0.0)
        beta_cdf_input = torch.clamp(beta_cdf_input, 0.0, 1.0)
        print(f"DEBUG: cdf - beta_cdf_input (after clamp) stats: mean={beta_cdf_input.mean():.6f}, std={beta_cdf_input.std():.6f}, min={beta_cdf_input.min():.6f}, max={beta_cdf_input.max():.6f}")

        # Use torch.special.betainc for beta CDF if available and differentiable, otherwise fallback to scipy
        # For now, sticking to scipy as torch.special.betainc might not be directly exposed or stable for all versions
        p = torch.from_numpy(stats.beta.cdf(np.atleast_1d(beta_cdf_input.cpu().detach().numpy()), self.c1.cpu().detach().numpy(), self.c2.cpu().detach().numpy()).astype(np.float32)).to(y.device)
        print(f"DEBUG: cdf - p stats: mean={p.mean():.6f}, std={p.std():.6f}, min={p.min():.6f}, max={p.max():.6f}")

        # Lower tail
        denominator_qa = (self._l0_r(self.p_a) - self._l0_r(self.p_b))
        print(f"DEBUG: cdf - denominator_qa stats: mean={denominator_qa.mean():.6f}, std={denominator_qa.std():.6f}, min={denominator_qa.min():.6f}, max={denominator_qa.max():.6f}")
        # Add a small epsilon to prevent division by zero
        denominator_qa = torch.where(denominator_qa == 0, torch.tensor(1e-6, device=denominator_qa.device, dtype=denominator_qa.dtype), denominator_qa)
        q_a_tilde = a - (b - a) * (self._l0_r(self.alpha) - self._l0_r(self.p_a)) / denominator_qa
        print(f"DEBUG: cdf - q_a_tilde stats: mean={q_a_tilde.mean():.6f}, std={q_a_tilde.std():.6f}, min={q_a_tilde.min():.6f}, max={q_a_tilde.max():.6f}")
        
        denominator_sb = (self._l0_r(self.p_a) - self._l0_r(self.p_b))
        print(f"DEBUG: cdf - denominator_sb stats: mean={denominator_sb.mean():.6f}, std={denominator_sb.std():.6f}, min={denominator_sb.min():.6f}, max={denominator_sb.max():.6f}")
        # Add a small epsilon to prevent division by zero
        denominator_sb = torch.where(denominator_sb == 0, torch.tensor(1e-6, device=denominator_sb.device, dtype=denominator_sb.dtype), denominator_sb)
        s_b_tilde = (b - a) * (self._l0_r(self.beta / 2) - self._l0_r(1 - self.beta / 2)) / denominator_sb
        print(f"DEBUG: cdf - s_b_tilde stats: mean={s_b_tilde.mean():.6f}, std={s_b_tilde.std():.6f}, min={s_b_tilde.min():.6f}, max={s_b_tilde.max():.6f}")
        
        denominator_z2 = (self._l0_r(self.beta / 2) - self._l0_r(1 - self.beta / 2))
        print(f"DEBUG: cdf - denominator_z2 stats: mean={denominator_z2.mean():.6f}, std={denominator_z2.std():.6f}, min={denominator_z2.min():.6f}, max={denominator_z2.max():.6f}")
        # Add a small epsilon to prevent division by zero
        denominator_z2 = torch.where(denominator_z2 == 0, torch.tensor(1e-6, device=denominator_z2.device, dtype=denominator_z2.dtype), denominator_z2)
        z2 = (y - q_a_tilde) / (s_b_tilde / denominator_z2) - self._l0_r(self.alpha)
        print(f"DEBUG: cdf - z2 stats: mean={z2.mean():.6f}, std={z2.std():.6f}, min={z2.min():.6f}, max={z2.max():.6f}")
        
        # Check for negative values in z2 before exp
        if (z2 < 0).any():
            tqdm.write(f"DEBUG: cdf - WARNING: Negative values found in z2 before exp. Min z2: {z2.min().item()}")
            # Optionally, clamp z2 to prevent inf/nan from exp(-negative_large_number)
            # z2 = torch.clamp(z2, min=-100.0) # Example clamp to a reasonable negative value

        t2 = torch.exp(-z2)
        print(f"DEBUG: cdf - t2 stats: mean={t2.mean():.6f}, std={t2.std():.6f}, min={t2.min():.6f}, max={t2.max():.6f}")

        out = torch.exp(p * (-t1) + (1 - p) * (-t2))
        print(f"DEBUG: cdf - out (before where) stats: mean={out.mean():.6f}, std={out.std():.6f}, min={out.min():.6f}, max={out.max():.6f}")
        
        # Handle edge cases where p is exactly 0 or 1 using torch.where
        out = torch.where(p == 0, torch.exp((1 - p) * (-t2)), out)
        out = torch.where(p == 1, torch.exp(p * (-t1)), out)
        print(f"DEBUG: cdf - out (after where) stats: mean={out.mean():.6f}, std={out.std():.6f}, min={out.min():.6f}, max={out.max():.6f}")
        
        return out

    def log_prob(self, y):
        print(f"DEBUG: log_prob - Input y stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")
        # Ensure cdf output is not zero before taking log
        cdf_val = self.cdf(y)
        print(f"DEBUG: log_prob - cdf_val stats: mean={cdf_val.mean():.6f}, std={cdf_val.std():.6f}, min={cdf_val.min():.6f}, max={cdf_val.max():.6f}")
        result = torch.log(torch.clamp(cdf_val, min=1e-30)) # Clamp to avoid log(0)
        print(f"DEBUG: log_prob - result stats: mean={result.mean():.6f}, std={result.std():.6f}, min={result.min():.6f}, max={result.max():.6f}")
        return result

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