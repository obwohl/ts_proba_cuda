# ts_benchmark/baselines/duet/johnson_system.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from typing import List, Dict
import math


class JohnsonSU_torch(Distribution):
    """Pure PyTorch implementation of the Johnson SU distribution's log_prob."""
    arg_constraints = {
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, gamma, delta, xi, lambda_, validate_args=None):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape, validate_args=validate_args)
        self._log_delta = self.delta.log()
        self._log_lambda = self.lambda_.log()
        self._log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        y = (value - self.xi) / self.lambda_
        z = self.gamma + self.delta * torch.asinh(y)
        log_p = self._log_delta - self._log_lambda - self._log_sqrt_2pi - 0.5 * z.pow(2) - 0.5 * (1 + y.pow(2)).log()
        return log_p

class JohnsonSB_torch(Distribution):
    """Pure PyTorch implementation of the Johnson SB distribution's log_prob."""
    arg_constraints = {
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    has_rsample = False

    def __init__(self, gamma, delta, xi, lambda_, validate_args=None):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape, validate_args=validate_args)
        self._support = constraints.interval(self.xi, self.xi + self.lambda_)
        self._log_delta = self.delta.log()
        self._log_lambda = self.lambda_.log()
        self._log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))

    @property
    def support(self):
        return self._support
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        lower, upper = self.xi, self.xi + self.lambda_
        # Clamp value to be strictly within the support to avoid log(0)
        value_safe = torch.clamp(value, lower + 1e-9, upper - 1e-9)
        
        y = (value_safe - self.xi) / self.lambda_
        
        # --- FIX: Clamp the input to log to prevent numerical instability ---
        # Add a small epsilon to prevent log(0) or log(inf)
        y_safe_for_log = torch.clamp(y, 1e-9, 1 - 1e-9)
        
        z = self.gamma + self.delta * (y_safe_for_log / (1 - y_safe_for_log)).log()
        
        log_p = self._log_delta - self._log_lambda - self._log_sqrt_2pi - 0.5 * z.pow(2) - y_safe_for_log.log() - (1 - y_safe_for_log).log()
        
        # Ensure that any original values outside the support get a large negative probability
        log_p = torch.where((value > lower) & (value < upper), log_p, torch.full_like(log_p, -1e8))
        return log_p

# --- SCIPY WRAPPERS (ONLY FOR NON-CRITICAL ICDF FUNCTION) ---

def _scipy_to_tensor(func, *args, **kwargs):
    """
    A generic wrapper that converts PyTorch tensors to NumPy arrays,
    executes a SciPy function, and converts the result back to a tensor.
    Used only for inference (icdf), not for training.
    """
    device = 'cpu'
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            device = arg.device
            break

    numpy_args = [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
    numpy_kwargs = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    
    result_np = func(*numpy_args, **numpy_kwargs)
    
    result_np = np.nan_to_num(result_np, nan=-1e9, posinf=-1e9, neginf=-1e9)

    return torch.from_numpy(np.array(result_np, dtype=np.float32)).to(device)

class JohnsonSU_scipy(Distribution):
    arg_constraints = {} # Added
    # SciPy-based version for icdf
    def __init__(self, gamma, delta, xi, lambda_):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape)
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        q_reshaped = q.view([1] * self.gamma.dim() + [-1])
        gamma, delta, xi, lambda_ = (d.unsqueeze(-1) for d in (self.gamma, self.delta, self.xi, self.lambda_))
        return _scipy_to_tensor(scipy.stats.johnsonsu.ppf, q_reshaped, a=gamma, b=delta, loc=xi, scale=lambda_)

class JohnsonSB_scipy(Distribution):
    arg_constraints = {} # Added
    # SciPy-based version for icdf
    def __init__(self, gamma, delta, xi, lambda_):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape)
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        q_reshaped = q.view([1] * self.gamma.dim() + [-1])
        gamma, delta, xi, lambda_ = (d.unsqueeze(-1) for d in (self.gamma, self.delta, self.xi, self.lambda_))
        return _scipy_to_tensor(scipy.stats.johnsonsb.ppf, q_reshaped, a=gamma, b=delta, loc=xi, scale=lambda_)

class JohnsonSL_scipy(Distribution):
    arg_constraints = {} # Added
    # SciPy-based version for icdf
    def __init__(self, delta, xi, lambda_):
        self.delta, self.xi, self.lambda_ = broadcast_all(delta, xi, lambda_)
        super().__init__(self.delta.shape)
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        q_reshaped = q.view([1] * self.delta.dim() + [-1])
        delta, xi, lambda_ = (d.unsqueeze(-1) for d in (self.delta, self.xi, self.lambda_))
        return _scipy_to_tensor(scipy.stats.lognorm.ppf, q_reshaped, s=delta, loc=xi, scale=lambda_)

class JohnsonSN_scipy(Distribution):
    arg_constraints = {} # Added
    # SciPy-based version for icdf
    def __init__(self, xi, lambda_):
        self.xi, self.lambda_ = broadcast_all(xi, lambda_)
        super().__init__(self.xi.shape)
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        q_reshaped = q.view([1] * self.xi.dim() + [-1])
        xi, lambda_ = (d.unsqueeze(-1) for d in (self.xi, self.lambda_))
        return _scipy_to_tensor(scipy.stats.norm.ppf, q_reshaped, loc=xi, scale=lambda_)


def _fit_johnson_torch(data_tensor: torch.Tensor, dist_type: str) -> float:
    """
    Performs Maximum Likelihood Estimation for a given Johnson family in pure PyTorch.
    Returns the final Negative Log-Likelihood (NLL).
    """
    # Use a shared inverse_softplus function
    def inverse_softplus(x, eps=1e-6):
        x = x.clamp(min=eps)
        return torch.log(torch.expm1(x))

    if dist_type == 'SU':
        mu, std = data_tensor.mean(), data_tensor.std()
        if std < 1e-6: return np.inf

        # Stable, simpler initialization
        gamma = nn.Parameter(torch.tensor(0.0, device=data_tensor.device))
        delta_raw = nn.Parameter(inverse_softplus(torch.tensor(1.0)))
        xi = nn.Parameter(mu.clone().detach())
        lambda_raw = nn.Parameter(inverse_softplus(std.clone().detach()))

        params = [gamma, delta_raw, xi, lambda_raw]
        optimizer = torch.optim.Adam(params, lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, min_lr=1e-6)

        # Early stopping
        patience = 250
        min_delta = 1e-5
        best_nll = float('inf')
        patience_counter = 0
        
        for i in range(5000): # Max iterations
            optimizer.zero_grad()
            delta = F.softplus(delta_raw) + 1e-6
            lambda_ = F.softplus(lambda_raw) + 1e-6
            dist = JohnsonSU_torch(gamma, delta, xi, lambda_)
            nll = -dist.log_prob(data_tensor).sum()

            if torch.isnan(nll) or torch.isinf(nll): return np.inf
            
            nll.backward()
            optimizer.step()
            
            current_nll = nll.item()
            scheduler.step(current_nll)
            
            if best_nll - current_nll > min_delta:
                best_nll = current_nll
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"SU early stopping at iteration {i}")
                break
        
        print(f"SU NLL: {best_nll}")
        return best_nll

    elif dist_type == 'SB':
        min_val, max_val = data_tensor.min(), data_tensor.max()
        if not (max_val > min_val): return np.inf
        
        gamma = nn.Parameter(torch.tensor(0.0, device=data_tensor.device))
        delta_raw = nn.Parameter(inverse_softplus(torch.tensor(1.0)))
        xi = nn.Parameter(min_val - (max_val - min_val) * 0.1)
        lambda_raw = nn.Parameter(inverse_softplus((max_val - min_val) * 1.2))

        params = [gamma, delta_raw, xi, lambda_raw]
        optimizer = torch.optim.Adam(params, lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, min_lr=1e-6)
        
        # Early stopping
        patience = 250
        min_delta = 1e-5
        best_nll = float('inf')
        patience_counter = 0
        
        for i in range(5000): # Max iterations
            optimizer.zero_grad()
            delta = F.softplus(delta_raw) + 1e-6
            lambda_ = F.softplus(lambda_raw) + 1e-6
            dist = JohnsonSB_torch(gamma, delta, xi, lambda_)
            nll = -dist.log_prob(data_tensor).sum()

            if torch.isnan(nll) or torch.isinf(nll): return np.inf
                
            nll.backward()
            optimizer.step()

            current_nll = nll.item()
            scheduler.step(current_nll)
            
            if best_nll - current_nll > min_delta:
                best_nll = current_nll
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"SB early stopping at iteration {i}")
                break
        
        print(f"SB NLL: {best_nll}")
        return best_nll

    elif dist_type == 'SL':
        if (data_tensor <= 0).any(): return np.inf
        data_safe = data_tensor[data_tensor > 0]
        if len(data_safe) < 10: return np.inf
        
        mu_log, std_log = data_safe.log().mean(), data_safe.log().std()
        if std_log < 1e-6: return np.inf

        loc = nn.Parameter(mu_log)
        scale_raw = nn.Parameter(inverse_softplus(std_log))
        optimizer = torch.optim.Adam([loc, scale_raw], lr=0.01)
        
        for _ in range(2000):
            optimizer.zero_grad()
            dist = torch.distributions.LogNormal(loc, F.softplus(scale_raw) + 1e-6)
            nll = -dist.log_prob(data_safe).sum()

            if torch.isnan(nll) or torch.isinf(nll): return np.inf
                
            nll.backward()
            optimizer.step()
            
        final_nll = -torch.distributions.LogNormal(loc.detach(), F.softplus(scale_raw.detach()) + 1e-6).log_prob(data_safe).sum()
        if torch.isnan(final_nll) or torch.isinf(final_nll): return np.inf

        print(f"SL NLL: {final_nll.item()}")
        return final_nll.item()

    elif dist_type == 'SN':
        mu, std = data_tensor.mean(), data_tensor.std()
        if std < 1e-6: return np.inf
        nll = -torch.distributions.Normal(loc=mu, scale=std).log_prob(data_tensor).sum()
        print(f"SN NLL: {nll.item()}")
        return nll.item()

    return np.inf

def get_best_johnson_fit(data: np.ndarray) -> str:
    """
    Finds the best-fitting Johnson family using a robust, pure-PyTorch optimizer.
    """
    data = data[~np.isnan(data)]
    if len(data) < 20 or len(np.unique(data)) < 10:
        return 'SU'

    data_tensor = torch.from_numpy(data.astype(np.float32))
    n = len(data_tensor)

    nlls = {}
    for dist_type in ['SU', 'SB', 'SL', 'SN']:
        try:
            nlls[dist_type] = _fit_johnson_torch(data_tensor, dist_type)
        except Exception as e:
            print(f"Error fitting {dist_type}: {e}")
            nlls[dist_type] = np.inf
    
    print(f"Calculated NLLs: {nlls}")

    if not nlls or all(v == np.inf for v in nlls.values()):
        return 'SU' 

    # Calculate BIC for each distribution
    bics = {}
    param_counts = {'SN': 2, 'SL': 2, 'SU': 4, 'SB': 4} # Corrected SL param count

    for dist_type, nll_val in nlls.items():
        if nll_val != np.inf:
            k = param_counts.get(dist_type, 4) # Default to 4 for safety
            bics[dist_type] = nll_val + k * math.log(n) / 2
        else:
            bics[dist_type] = np.inf
    
    print(f"Calculated BICs: {bics}")

    # Find best fit based on lowest BIC
    best_fit_type = min(bics, key=bics.get)
    

    return best_fit_type


class JohnsonOutput:
    args_dim: int = 4

    def __init__(self, channel_types: List[str]):
        self.channel_types = channel_types
        self.n_vars = len(channel_types)

    def distribution(self, distr_args: torch.Tensor, epoch=None, batch_idx=None) -> "CombinedJohnsonDistribution":
        gamma_raw, delta_raw, xi_raw, lambda_raw = torch.chunk(distr_args, chunks=4, dim=-1)

        gamma = gamma_raw.squeeze(-1)
        delta = F.softplus(delta_raw.squeeze(-1)) + 1e-6
        xi = xi_raw.squeeze(-1)
        lambda_ = F.softplus(lambda_raw.squeeze(-1)) + 1e-6

        # Sanitize parameters to prevent NaNs/Infs from propagating
        gamma = torch.nan_to_num(gamma, nan=0.0, posinf=1.0, neginf=-1.0)
        delta = torch.nan_to_num(delta, nan=1.0, posinf=1.0, neginf=1.0)
        xi = torch.nan_to_num(xi, nan=0.0, posinf=1.0, neginf=-1.0)
        lambda_ = torch.nan_to_num(lambda_, nan=1.0, posinf=1.0, neginf=1.0)

        return CombinedJohnsonDistribution(
            channel_types=self.channel_types, # FIX: Pass self.channel_types
            gamma=gamma, delta=delta, xi=xi, lambda_=lambda_
        )

class CombinedJohnsonDistribution(Distribution):
    arg_constraints = {} # Added

    def __init__(self, channel_types, gamma, delta, xi, lambda_, validate_args=None):
        self.channel_types = channel_types
        self.n_vars = len(channel_types)
        
        self.gamma, self.delta, self.xi, self.lambda_ = gamma, delta, xi, lambda_
        
        self.su_mask = torch.tensor([t == 'SU' for t in channel_types], device=gamma.device)
        self.sb_mask = torch.tensor([t == 'SB' for t in channel_types], device=gamma.device)
        self.sl_mask = torch.tensor([t == 'SL' for t in channel_types], device=gamma.device)
        self.sn_mask = torch.tensor([t == 'SN' for t in channel_types], device=gamma.device)

        super().__init__(gamma.shape, validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_p = torch.zeros_like(value)

        if self.su_mask.any():
            indices = self.su_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSU_torch(
                self.gamma[:, indices, :], self.delta[:, indices, :],
                self.xi[:, indices, :], self.lambda_[:, indices, :]
            )
            log_p[:, indices, :] = dist.log_prob(value[:, indices, :])

        if self.sb_mask.any():
            indices = self.sb_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSB_torch(
                self.gamma[:, indices, :], self.delta[:, indices, :],
                self.xi[:, indices, :], self.lambda_[:, indices, :]
            )
            log_p[:, indices, :] = dist.log_prob(value[:, indices, :])

        if self.sl_mask.any():
            indices = self.sl_mask.nonzero(as_tuple=True)[0]
            # For LogNormal, values must be strictly positive. Handle out-of-support values.
            sl_value = value[:, indices, :]
            # Create a mask for values that are <= 0
            out_of_support_mask = sl_value <= 0

            # Calculate log_prob only for in-support values
            # Clamp values to a small positive number to avoid log(0) or log(negative) for in-support calculation
            # This is for numerical stability during calculation, actual out-of-support values will be set to -inf
            # Removed +1e-9 from here, as it was causing issues with _validate_sample
            sl_value_clamped = torch.clamp(sl_value, min=1e-9)

            dist = torch.distributions.LogNormal(self.xi[:, indices, :], self.lambda_[:, indices, :])
            log_p_sl = dist.log_prob(sl_value_clamped)
            
            # Set log_prob to a large negative number for out-of-support values
            log_p[:, indices, :] = torch.where(out_of_support_mask, torch.full_like(log_p_sl, -1e8), log_p_sl)

        if self.sn_mask.any():
            indices = self.sn_mask.nonzero(as_tuple=True)[0]
            dist = torch.distributions.Normal(
                loc=self.xi[:, indices, :], 
                scale=self.lambda_[:, indices, :]
            )
            log_p[:, indices, :] = dist.log_prob(value[:, indices, :])
            
        return log_p

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        quantiles = torch.zeros(self.batch_shape + (len(q),), device=q.device)

        if self.su_mask.any():
            indices = self.su_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSU_scipy(
                self.gamma[:, indices, :], self.delta[:, indices, :],
                self.xi[:, indices, :], self.lambda_[:, indices, :]
            )
            quantiles[:, indices, :, :] = dist.icdf(q)

        if self.sb_mask.any():
            indices = self.sb_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSB_scipy(
                self.gamma[:, indices, :], self.delta[:, indices, :],
                self.xi[:, indices, :], self.lambda_[:, indices, :]
            )
            quantiles[:, indices, :, :] = dist.icdf(q)

        if self.sl_mask.any():
            indices = self.sl_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSL_scipy(
                self.delta[:, indices, :],
                self.xi[:, indices, :], 
                self.lambda_[:, indices, :]
            )
            quantiles[:, indices, :, :] = dist.icdf(q)

        if self.sn_mask.any():
            indices = self.sn_mask.nonzero(as_tuple=True)[0]
            dist = JohnsonSN_scipy(
                self.xi[:, indices, :], 
                self.lambda_[:, indices, :]
            )
            quantiles[:, indices, :, :] = dist.icdf(q)
            
        return quantiles