import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

class ZeroInflatedExtendedGPD_M1_Continuous(Distribution):
    arg_constraints = {
        'pi_raw': constraints.real,          # Raw logit for pi (sigmoid applied)
        'kappa_raw': constraints.real,       # Raw parameter for kappa (softplus applied)
        'sigma_raw': constraints.real,       # Raw parameter for sigma (softplus applied)
        'xi_raw': constraints.real           # Raw shape parameter xi (tanh applied)
    }
    support = constraints.greater_than_eq(0.0) # Continuous non-negative support

    has_rsample = False # For now, no reparameterization trick

    def __init__(self, pi_raw, kappa_raw, sigma_raw, xi_raw, stats: torch.Tensor = None, validate_args=True):
        # Broadcast all parameters to ensure consistent shape
        self.pi_raw, self.kappa_raw, self.sigma_raw, self.xi_raw = broadcast_all(pi_raw, kappa_raw, sigma_raw, xi_raw)
        
        # Apply transformations to get valid distribution parameters
        self.pi = torch.clamp(torch.sigmoid(self.pi_raw), min=1e-6, max=1-1e-6)
        self.xi = 1.0 * torch.tanh(self.xi_raw) # Constrain xi to [-0.5, 0.5] for stability

        SCALE_FACTOR = 10.0 
        self.kappa = SCALE_FACTOR * torch.sigmoid(self.kappa_raw) + 1e-6
        self.sigma = SCALE_FACTOR * torch.sigmoid(self.sigma_raw) + 1e-6
        
        if stats is not None:
            # stats has the shape: [B, N_vars, 2]
            # self._mean, self._std get the shape: [B, N_vars, 1] for broadcasting
            self._mean = stats[:, :, 0].unsqueeze(-1)
            STD_FLOOR = 1e-6  # Safety floor for the standard deviation
            self._std = torch.clamp(stats[:, :, 1], min=STD_FLOOR).unsqueeze(-1)
        else:
            # Provide default mean=0 and std=1 if stats are not provided.
            # This is important for tests that instantiate the distribution directly.
            self._mean = torch.zeros_like(self.pi)
            self._std = torch.ones_like(self.pi)

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
        # value comes in as [B, H, N_vars], but params are [B, N_vars, H]. Let's align them.
        if value.dim() == 3 and self.pi.dim() == 3 and value.shape[1] == self.pi.shape[2] and value.shape[2] == self.pi.shape[1]:
            value = value.permute(0, 2, 1)

        # value has shape [B, N_vars, H]
        # Parameters have shape [B, N_vars, H]
        kappa, sigma, xi = self.kappa, self.sigma, self.xi

        # This method now receives UNNORMALIZED values.
        # The check for zeros must happen on the raw data.
        log_probs = torch.zeros_like(value, dtype=value.dtype, device=value.device)

        # Case 1: value is zero or extremely close to it (zero-inflated part)
        # Using a tolerance to handle floating point inaccuracies.
        zero_mask = torch.abs(value) < 1e-9
        if self.pi.dim() > 0:
            log_probs[zero_mask] = torch.log(self.pi[zero_mask])
        elif zero_mask.all():
            log_probs = torch.log(self.pi)

        # Case 2: value > 0 (Extended GPD part)
        # The positive mask must exclude the values already handled by the zero mask.
        positive_mask = (value > 0) & (~zero_mask)
        if torch.any(positive_mask):
            v_pos = value[positive_mask]

            # --- Internal Normalization for the GPD component ---
            # Reshape mean and std to match the shape of v_pos
            # self._mean/std have shape [B, N_vars, 1]. We need to align them with `positive_mask`.
            mean_expanded = self._mean.expand_as(value)[positive_mask]
            std_expanded = self._std.expand_as(value)[positive_mask]
            v_pos_norm = (v_pos - mean_expanded) / std_expanded
            # --- End Internal Normalization ---

            if kappa.dim() > 0:
                k_pos = kappa[positive_mask]
                s_pos = sigma[positive_mask]
                x_pos = xi[positive_mask]
                pi_pos = self.pi[positive_mask]
            else:
                k_pos = kappa
                s_pos = sigma
                x_pos = xi
                pi_pos = self.pi

            # Calculate F(z) and f(z) for the base GPD on NORMALIZED data
            gpd_cdf_val = self._gpd_cdf(v_pos_norm, s_pos, x_pos)
            gpd_pdf_val = self._gpd_pdf(v_pos_norm, s_pos, x_pos)

            # Clamp values for numerical stability before log
            gpd_cdf_val = torch.clamp(gpd_cdf_val, min=1e-9, max=1-1e-9)
            gpd_pdf_val = torch.clamp(gpd_pdf_val, min=1e-9)
            
            # Calculate the log-probability using the correct formula
            log_one_minus_pi = torch.log(1 - pi_pos)
            log_kappa = torch.log(k_pos)
            log_gpd_cdf = torch.log(gpd_cdf_val)
            log_gpd_pdf = torch.log(gpd_pdf_val)

            # Jacobian correction for the normalization transformation (log(1/std))
            log_det_jacobian = -torch.log(std_expanded)

            log_probs[positive_mask] = (
                log_one_minus_pi + 
                log_kappa + 
                (k_pos - 1) * log_gpd_cdf + 
                log_gpd_pdf + 
                log_det_jacobian
            )

        # Handle values outside support (value < 0)
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

        # This is a NORMALIZED quantile
        quantiles_norm = torch.where(torch.abs(xi) < 1e-9, icdf_exp_case, icdf_gpd_case)

        # If q <= pi, the quantile is 0 due to the zero-inflation mass
        # For the positive part, we need to denormalize the result
        # Reshape mean and std for broadcasting with quantiles_norm [B, N_vars, H, Q]
        mean_for_bcast = self._mean.unsqueeze(-1)
        std_for_bcast = self._std.unsqueeze(-1)
        quantiles_denorm = quantiles_norm * std_for_bcast + mean_for_bcast

        quantiles = torch.where(
            q_exp <= pi,
            torch.tensor(0.0, device=q.device, dtype=q.dtype),
            quantiles_denorm
        )

        return torch.max(quantiles, torch.tensor(0.0, device=quantiles.device, dtype=quantiles.dtype))

    @property
    def batch_shape(self):
        return self.pi.shape
