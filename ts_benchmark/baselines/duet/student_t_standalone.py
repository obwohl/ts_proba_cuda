# ts_benchmark/baselines/duet/student_t_standalone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# NEW: Import numpy and scipy for the fallback function
import numpy as np
import scipy.special
# We will inherit from the original torch.distributions.StudentT
from torch.distributions import StudentT as TorchStudentT

# --- NEW: SCIPY-BASED FALLBACK FOR BETAINCINV ---

def _scipy_betaincinv(a: torch.Tensor, b: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    A SciPy-based fallback for `torch.special.betaincinv`.

    This function mimics the behavior of `torch.special.betaincinv` by using
    `scipy.special.betaincinv`. It handles the conversion from PyTorch tensors
    to NumPy arrays (and back), including moving data between CPU and GPU if necessary.

    Args:
        a: The first shape parameter (alpha) of the beta distribution.
        b: The second shape parameter (beta) of the beta distribution.
        value: The probability value (from the incomplete beta function).

    Returns:
        A PyTorch tensor containing the result of the inverse incomplete beta function.
    """
    device = value.device
    a_np, b_np, value_np = a.detach().cpu().numpy(), b.detach().cpu().numpy(), value.detach().cpu().numpy()
    result_np = scipy.special.betaincinv(a_np, b_np, value_np)
    return torch.from_numpy(result_np).to(device)

# --- NEW CUSTOM STUDENT'S T DISTRIBUTION ---

class CustomStudentT(TorchStudentT):
    """
    A custom Student's T distribution class that inherits from torch.distributions.StudentT
    and implements the missing `icdf` (inverse cumulative distribution function) method,
    also known as the quantile function.
    """
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse cumulative distribution function (quantile function) using
        the inverse of the regularized incomplete beta function.

        Args:
            q: A tensor of probabilities (quantiles), with values in [0, 1].

        Returns:
            A tensor of the same shape as `q` containing the corresponding values
            from the distribution.
        """
        # The icdf calculation is first performed for a standard t-distribution (loc=0, scale=1).
        # We handle values for q > 0.5 and q < 0.5 symmetrically by using the tail probability.
        p = torch.where(q > 0.5, 1.0 - q, q)
        
        # The relationship between the t-distribution CDF and the incomplete beta function
        # allows us to use `betaincinv` to find the corresponding value.
        df_half = 0.5 * self.df
        one_half = 0.5 * torch.ones_like(df_half)
        
        # --- MODIFICATION FOR BROADCASTING ---
        # The distribution parameters (like df) have a shape like [Batch, Channels, Horizon].
        # The quantiles `q` (and `p`) have a shape like [NumQuantiles].
        # To make them broadcast-compatible for a [B, C, H, Q] output, we reshape them.

        # Reshape distribution params from [B, C, H] to [B, C, H, 1]
        df_half_reshaped = df_half.unsqueeze(-1)
        one_half_reshaped = one_half.unsqueeze(-1)

        # Reshape quantiles `p` to be broadcastable with the distribution params.
        # e.g., from [Q] to [1, 1, 1, Q] if df has 3 dimensions.
        # This aligns the quantile dimension with the new last dimension of the other tensors.
        p_reshaped = (2.0 * p).view([1] * self.df.dim() + [-1])

        # --- RESTORED: Environment-Aware Path Selection ---
        # Use the fast, native torch function if available (PyTorch >= 1.10).
        # Otherwise, use the slow SciPy fallback for compatibility.
        if hasattr(torch.special, 'betaincinv'):
            z = torch.special.betaincinv(df_half_reshaped, one_half_reshaped, p_reshaped)
        else:
            z = _scipy_betaincinv(df_half_reshaped, one_half_reshaped, p_reshaped)

        # Now, we solve for t: t = sqrt(df * (1/z - 1))
        # We unsqueeze self.df to match the shape of z for element-wise multiplication.
        t_squared = self.df.unsqueeze(-1) * (1.0 / (z + 1e-8) - 1.0)
        standard_t = torch.sqrt(t_squared)

        # Apply the correct sign based on the original quantile `q`.
        # `q` also needs to be reshaped to broadcast correctly for the `where` condition.
        q_reshaped = q.view([1] * self.df.dim() + [-1])
        signed_t = torch.where(q_reshaped > 0.5, standard_t, -standard_t)

        # Handle edge cases for q=0 and q=1. The reshaped q is used for the condition.
        signed_t = torch.where(q_reshaped == 0, torch.tensor(float('-inf'), device=q.device), signed_t)
        signed_t = torch.where(q_reshaped == 1, torch.tensor(float('inf'), device=q.device), signed_t)

        # Finally, transform the standard `t` value to the distribution's
        # actual `loc` and `scale`. We unsqueeze them to match the shape of signed_t.
        return self.loc.unsqueeze(-1) + self.scale.unsqueeze(-1) * signed_t

# --- HELPER CLASSES FOR PROJECTION (reused from SBP implementation) ---

class ProjectionResidualBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.norm(x + residual)
        return x

class MLPProjectionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers == 0:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.input_layer = nn.Linear(in_features, in_features)
            self.residual_blocks = nn.ModuleList(
                [ProjectionResidualBlock(d_model=in_features, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )
            self.final_layer = nn.Linear(in_features, out_features)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers > 0:
            x = self.input_layer(x)
            for block in self.residual_blocks:
                x = block(x)
            return self.final_layer(x)
        return self.projection(x)

# --- DISTRIBUTION OUTPUT CLASS (MODIFIED) ---

class StudentTOutput:
    """
    A helper class that connects the raw output of a neural network
    to a Student's T distribution from torch.distributions.
    """
    # The Student's T-distribution requires 3 parameters:
    # df (degrees of freedom), loc (mean), and scale (standard deviation).
    args_dim: int = 3

    def distribution(self, distr_args: torch.Tensor) -> CustomStudentT: # Return type changed
        """
        Takes the raw output tensor from the projection head and transforms it
        into the parameters for a CustomStudentT distribution.

        Args:
            distr_args: A tensor of shape [..., 3], where the last dimension
                        contains the raw values for (df, loc, scale).

        Returns:
            A CustomStudentT object with a working `icdf` method.
        """
        # Split the raw parameters from the last dimension
        df_raw, loc_raw, scale_raw = torch.chunk(distr_args, chunks=3, dim=-1)

        # 1. Degrees of Freedom (df): Must be positive.
        #    We use softplus to ensure positivity and add 2.0 to ensure that the
        #    distribution always has a finite variance (variance is defined for df > 2).
        #    This improves training stability.
        df = F.softplus(df_raw.squeeze(-1)) + 2.0

        # 2. Location (loc): Can be any real number, so no transformation is needed.
        loc = loc_raw.squeeze(-1)

        # 3. Scale: Must be positive.
        #    We use softplus to ensure positivity and add a small epsilon for
        #    numerical stability, preventing the scale from becoming zero.
        scale = F.softplus(scale_raw.squeeze(-1)) + 1e-6

        # Return the parameterized distribution object using the custom class.
        # This is the key change that fixes the NotImplementedError.
        return CustomStudentT(df=df, loc=loc, scale=scale)
