import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

class ZeroInflatedExtendedGPD_M1_Continuous(Distribution):
    arg_constraints = {
        'pi': constraints.interval(0.0, 1.0), # Zero-inflation proportion
        'kappa_raw': constraints.real,       # Raw parameter for kappa (softplus applied)
        'sigma_raw': constraints.real,       # Raw parameter for sigma (softplus applied)
        'xi': constraints.real               # Shape parameter xi
    }
    support = constraints.greater_than_eq(0.0) # Continuous non-negative support

    has_rsample = False # For now, no reparameterization trick

    def __init__(self, pi, kappa_raw, sigma_raw, xi, validate_args=None):
        # Broadcast all parameters to ensure consistent shape
        self.pi, self.kappa_raw, self.sigma_raw, self.xi = broadcast_all(pi, kappa_raw, sigma_raw, xi)
        
        # Apply softplus to ensure kappa and sigma are positive
        self.kappa = F.softplus(self.kappa_raw) + 1e-4 # Add epsilon for stability
        self.sigma = F.softplus(self.sigma_raw) + 1e-4 # Add epsilon for stability

        super().__init__(self.pi.shape, validate_args=validate_args)
    
    # --- Hilfsfunktionen für GPD F(z) und W(u) ---
    def _gpd_cdf(self, z, sigma, xi):
        # Handle z < 0 or (1 + xi*z/sigma) <= 0 cases explicitly for F(z) = 0
        safe_z = torch.max(z, torch.tensor(0.0, device=z.device, dtype=z.dtype))
        
        # Calculate term (1 + xi*safe_z/sigma)
        arg_power = 1 + xi * safe_z / sigma
        
        # Clamp arg_power to be positive to avoid NaN/Inf for power/log
        # For xi < 0, arg_power could become <=0.
        arg_power = torch.clamp(arg_power, min=1e-6) 

        # xi = 0 case (Exponential)
        exp_case = 1 - torch.exp(-safe_z / sigma)
        
        # xi != 0 case (GPD)
        gpd_case = 1 - arg_power.pow(-1/xi)
        
        # Combine cases based on xi
        # Using a small epsilon for xi to treat very small xi as 0
        return torch.where(torch.abs(xi) < 1e-9, exp_case, gpd_case)

    def _gpd_pdf(self, z, sigma, xi):
        # For the derivative (pdf), also handle z < 0 or (1 + xi*z/sigma) <= 0 cases explicitly
        safe_z = torch.max(z, torch.tensor(0.0, device=z.device, dtype=z.dtype))

        arg_power = 1 + xi * safe_z / sigma
        arg_power = torch.clamp(arg_power, min=1e-6)

        # xi = 0 case (Exponential PDF)
        exp_pdf_case = (1/sigma) * torch.exp(-safe_z / sigma)

        # xi != 0 case (GPD PDF = d/dz F(z))
        gpd_pdf_case = (1/sigma) * arg_power.pow(-1/xi - 1)
        
        return torch.where(torch.abs(xi) < 1e-9, exp_pdf_case, gpd_pdf_case)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Parameters (self.pi etc.) have shape e.g. [B, N_vars, H]
        # value has shape [B, H, N_vars]. We need to align them for broadcasting.
        try:
            # Permute params from [B, N_vars, H] to [B, H, N_vars]
            pi, kappa, sigma, xi = [p.permute(0, 2, 1) for p in (self.pi, self.kappa, self.sigma, self.xi)]
        except (IndexError, RuntimeError): # Fallback if params are not 3D
            pi, kappa, sigma, xi = self.pi, self.kappa, self.sigma, self.xi

        value_safe = torch.clamp(value, min=0.0)

        # Log prob for zero values is simply log(pi)
        log_prob_at_zero = torch.log(torch.clamp(pi, min=1e-9)) # Clamp pi to prevent log(0)

        # Log prob for non-zero values is log((1-pi) * pdf)
        gpd_cdf_val = self._gpd_cdf(value_safe, sigma, xi)
        gpd_pdf_val = self._gpd_pdf(value_safe, sigma, xi)

        # Ensure gpd_cdf_val and gpd_pdf_val are positive before use
        gpd_cdf_val = torch.clamp(gpd_cdf_val, min=1e-6) # Increased clamp for stability
        gpd_pdf_val = torch.clamp(gpd_pdf_val, min=1e-6) # Increased clamp for stability

        # pdf is d/dz W(F(z)) = kappa * F(z)^(kappa-1) * dF/dz
        # Clamp gpd_cdf_val.pow(kappa - 1) to prevent inf/nan if kappa - 1 is very negative and gpd_cdf_val is very small
        pow_term = torch.clamp(gpd_cdf_val, min=1e-6).pow(kappa - 1)
        continuous_density = kappa * pow_term * gpd_pdf_val
        continuous_density = torch.clamp(continuous_density, min=1e-9) # Clamp again before log

        # Add epsilon to (1 - pi) to prevent log(0) if pi is 1
        one_minus_pi = torch.clamp(1 - pi, min=1e-9)
        log_prob_continuous = torch.log(one_minus_pi * continuous_density)

        # Combine based on whether value is zero
        is_zero = (value == 0)
        log_probs = torch.where(is_zero, log_prob_at_zero, log_prob_continuous)

        # Handle values outside support
        log_probs[value < 0] = -float('inf')

        return log_probs

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # self.pi etc: [B, N_vars, H]
        # q: [Q]
        # target output: [B, N_vars, H, Q]

        # Unsqueeze params and q for broadcasting
        pi = self.pi.unsqueeze(-1)
        kappa = self.kappa.unsqueeze(-1)
        sigma = self.sigma.unsqueeze(-1)
        xi = self.xi.unsqueeze(-1)
        q_exp = q.view(1, 1, 1, -1)

        # Adjust q for the zero-inflated part: q* = (q - pi) / (1 - pi)
        # Add a small epsilon to (1 - pi) to prevent division by zero when pi is close to 1
        one_minus_pi = torch.clamp(1 - pi, min=1e-9)
        q_star = (q_exp - pi) / one_minus_pi
        q_star = torch.clamp(q_star, 1e-9, 1 - 1e-9) # Clamp q_star to (0, 1)

        # Inverse of extension function: W_inv(q*) = (q*)^(1/kappa)
        # Add epsilon to kappa to prevent 1/kappa from being inf
        kappa_safe = torch.clamp(kappa, min=1e-9)
        w_inv_q_star = q_star.pow(1 / kappa_safe)

        # Inverse CDF of the base GPD
        term_1_minus_w_inv = 1 - w_inv_q_star
        term_1_minus_w_inv = torch.clamp(term_1_minus_w_inv, min=1e-9)

        # xi = 0 case (Exponential)
        icdf_exp_case = -sigma * torch.log(term_1_minus_w_inv)

        # xi != 0 case (GPD)
        # Add epsilon to xi to prevent division by zero
        xi_safe = torch.where(torch.abs(xi) < 1e-9, torch.tensor(1e-9, device=xi.device, dtype=xi.dtype), xi)
        icdf_gpd_case = (sigma / xi_safe) * (term_1_minus_w_inv.pow(-xi) - 1)

        quantiles_raw = torch.where(torch.abs(xi) < 1e-9, icdf_exp_case, icdf_gpd_case)

        # If q <= pi, the quantile is 0 due to the zero-inflation mass
        quantiles = torch.where(
            q_exp <= pi,
            torch.tensor(0.0, device=q.device, dtype=q.dtype),
            quantiles_raw
        )

        return torch.max(quantiles, torch.tensor(0.0, device=quantiles.device, dtype=quantiles.dtype))

    @property
    def batch_shape(self):
        return self.pi.shape
