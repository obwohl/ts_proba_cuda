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
        self.xi = 1.0 * torch.tanh(self.xi_raw) # Constrain xi to [-1.0, 1.0] for stability

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
        # Support is z >= 0, and if xi < 0, then 1 + xi * z / sigma > 0 => z <= -sigma / xi
        support_mask = (z >= 0) & ((xi >= 0) | (z <= -sigma / xi))

        # Initialize cdf. Values outside lower bound support are 0.
        cdf = torch.zeros_like(z)

        # For values within the support, calculate the CDF
        if support_mask.any():
            z_sup, sigma_sup, xi_sup = z[support_mask], sigma[support_mask], xi[support_mask]

            arg_power = 1 + xi_sup * z_sup / sigma_sup
            
            exp_case = 1 - torch.exp(-z_sup / sigma_sup)
            gpd_case = 1 - arg_power.pow(-1 / xi_sup)
            
            cdf[support_mask] = torch.where(torch.abs(xi_sup) < 1e-9, exp_case, gpd_case)

        # For values outside the support upper bound, CDF is 1.
        cdf[(z > 0) & ~support_mask] = 1.0
        return cdf

    def _gpd_pdf(self, z, sigma, xi):
        # Support is z >= 0, and if xi < 0, then 1 + xi * z / sigma > 0 => z <= -sigma / xi
        support_mask = (z >= 0) & ((xi >= 0) | (z <= -sigma / xi))

        # Initialize pdf with 0 for values outside support
        pdf = torch.zeros_like(z)

        # For values within the support, calculate the PDF
        if support_mask.any():
            z_sup, sigma_sup, xi_sup = z[support_mask], sigma[support_mask], xi[support_mask]

            arg_power = 1 + xi_sup * z_sup / sigma_sup

            exp_pdf_case = (1 / sigma_sup) * torch.exp(-z_sup / sigma_sup)
            gpd_pdf_case = (1 / sigma_sup) * arg_power.pow(-1 / xi_sup - 1)
            
            pdf[support_mask] = torch.where(torch.abs(xi_sup) < 1e-9, exp_pdf_case, gpd_pdf_case)

        return pdf

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # value comes in as [B, H, N_vars], but params are [B, N_vars, H]. Let's align them.
        if value.dim() == 3 and self.pi.dim() == 3 and value.shape[1] == self.pi.shape[2] and value.shape[2] == self.pi.shape[1]:
            value = value.permute(0, 2, 1)

        # value has shape [B, N_vars, H]
        # Parameters have shape [B, N_vars, H]
        kappa, sigma, xi = self.kappa, self.sigma, self.xi

        # This method now receives UNNORMALIZED values.
        log_probs = torch.zeros_like(value, dtype=value.dtype, device=value.device)

        # --- MASK DEFINITIONS ---
        zero_mask = torch.abs(value) < 1e-9
        positive_mask = (value > 0) & (~zero_mask)

        # --- HANDLE ZERO VALUES ---
        if torch.any(zero_mask):
            # Default behavior for ZI channels: log(pi)
            log_probs_for_zeros = torch.log(self.pi)

            # Identify non-ZI channels (where pi was forced to be near-zero)
            non_zi_mask = self.pi < 2e-6

            # Create a combined mask for channels that are non-ZI AND have zero values
            non_zi_zero_mask = non_zi_mask & zero_mask

            if torch.any(non_zi_zero_mask):
                # For these specific points, calculate log_prob using the CDF heuristic to avoid log(0).
                
                # Denormalization stats for the target elements
                mean_expanded = self._mean.expand_as(value)[non_zi_zero_mask]
                std_expanded = self._std.expand_as(value)[non_zi_zero_mask]
                
                # Parameters for the target elements
                sigma_masked = sigma[non_zi_zero_mask]
                xi_masked = xi[non_zi_zero_mask]

                # Heuristic: P(X=0) ~= P(0 < X_norm < epsilon_norm) = CDF(epsilon_norm)
                epsilon = 1e-7
                epsilon_norm = (epsilon - mean_expanded) / std_expanded
                
                # The log_prob is log(CDF(epsilon_norm)).
                # This corresponds to the log-likelihood of the underlying continuous GPD for values near zero.
                log_prob_non_zi_zero = torch.log(torch.clamp(self._gpd_cdf(epsilon_norm, sigma_masked, xi_masked), min=1e-9))
                
                # Update the log_probs tensor only at the required locations
                log_probs_for_zeros[non_zi_zero_mask] = log_prob_non_zi_zero

            log_probs[zero_mask] = log_probs_for_zeros[zero_mask]

        # --- HANDLE POSITIVE VALUES ---
        # This part is correct for both ZI and non-ZI cases.
        # For non-ZI, pi is ~0, so log(1-pi) is ~0, and the term vanishes as required.
        if torch.any(positive_mask):
            v_pos = value[positive_mask]

            # Internal Normalization
            mean_expanded = self._mean.expand_as(value)[positive_mask]
            std_expanded = self._std.expand_as(value)[positive_mask]
            v_pos_norm = (v_pos - mean_expanded) / std_expanded

            # Parameters for the positive values
            k_pos, s_pos, x_pos, pi_pos = kappa[positive_mask], sigma[positive_mask], xi[positive_mask], self.pi[positive_mask]

            # Base GPD calculations
            gpd_cdf_val = self._gpd_cdf(v_pos_norm, s_pos, x_pos)
            gpd_pdf_val = self._gpd_pdf(v_pos_norm, s_pos, x_pos)

            # Clamp for numerical stability
            gpd_cdf_val = torch.clamp(gpd_cdf_val, min=1e-9, max=1-1e-9)
            gpd_pdf_val = torch.clamp(gpd_pdf_val, min=1e-9)
            
            # Log-probability calculation
            log_one_minus_pi = torch.log(1 - pi_pos)
            log_kappa = torch.log(k_pos)
            log_gpd_cdf = torch.log(gpd_cdf_val)
            log_gpd_pdf = torch.log(gpd_pdf_val)
            log_det_jacobian = -torch.log(std_expanded)

            log_probs[positive_mask] = (
                log_one_minus_pi + 
                log_kappa + 
                (k_pos - 1) * log_gpd_cdf + 
                log_gpd_pdf + 
                log_det_jacobian
            )

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
