# ts_benchmark/baselines/duet/student_t_standalone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# NEW: Import numpy and scipy for the fallback function
import numpy as np
import scipy.special
# We will inherit from the original torch.distributions.StudentT
from torch.distributions import StudentT as TorchStudentT
# NEW: Imports for creating a custom distribution
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


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

# --- NEW: SKEWED STUDENT'S T DISTRIBUTION (FERNANDEZ-STEEL) ---

class SkewedStudentT(Distribution):
    """
    Fernandez-Steel Skewed Student's T-distribution.

    This distribution is constructed by "stretching" a standard symmetric
    Student's T-distribution on either side of the mode using a skewness
    parameter `skew` (also known as gamma or xi).

    Args:
        df (Tensor): Degrees of freedom. Must be positive.
        loc (Tensor): The mode of the distribution.
        scale (Tensor): The scale parameter. Must be positive.
        skew (Tensor): The skewness parameter. Must be positive. A value of 1.0
                       corresponds to a symmetric Student's T-distribution.
    """
    arg_constraints = {
        'df': constraints.positive,
        'loc': constraints.real,
        'scale': constraints.positive,
        'skew': constraints.positive,
    }
    support = constraints.real
    has_rsample = False # icdf is not easily reparameterizable

    def __init__(self, df, loc, scale, skew, validate_args=None):
        self.df, self.loc, self.scale, self.skew = broadcast_all(df, loc, scale, skew)
        # The base StudentT class is only used for its log_prob calculation
        # for a standard (loc=0, scale=1) distribution.
        self._standard_t = TorchStudentT(self.df, 0.0, 1.0)
        super().__init__(self._standard_t.batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SkewedStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.skew = self.skew.expand(batch_shape)
        new._standard_t = self._standard_t.expand(batch_shape)
        super(SkewedStudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of a value.
        """
        if self._validate_args:
            self._validate_sample(value)
        
        z = (value - self.loc) / self.scale
        skew_transform = torch.where(z >= 0, 1.0 / self.skew, self.skew)
        z_skewed = z * skew_transform

        log_p_sym = self._standard_t.log_prob(z_skewed)

        log_p = torch.log(torch.tensor(2.0, device=z.device)) \
                - torch.log(self.skew + 1.0 / self.skew) \
                - torch.log(self.scale) \
                + log_p_sym
        
        return log_p

    def _symmetric_student_t_icdf(self, q: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """Helper function to compute the icdf of a standard symmetric Student's T."""
        p = torch.where(q > 0.5, 1.0 - q, q)
        df_half = 0.5 * df
        one_half = 0.5 * torch.ones_like(df_half)
        
        if hasattr(torch.special, 'betaincinv'):
            z = torch.special.betaincinv(df_half, one_half, 2.0 * p)
        else:
            z = _scipy_betaincinv(df_half, one_half, 2.0 * p)

        t_squared = df * (1.0 / (z + 1e-8) - 1.0)
        standard_t = torch.sqrt(t_squared)
        signed_t = torch.where(q > 0.5, standard_t, -standard_t)
        
        signed_t = torch.where(q == 0, torch.tensor(float('-inf'), device=q.device), signed_t)
        signed_t = torch.where(q == 1, torch.tensor(float('inf'), device=q.device), signed_t)
        return signed_t

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        """Computes the inverse cumulative distribution function (quantile function)."""
        # Reshape quantile tensor to be broadcastable with distribution parameters
        q_reshaped = q.view([1] * self.df.dim() + [-1])

        # Unsqueeze distribution parameters to align for broadcasting with quantiles
        df, loc, scale, skew = (d.unsqueeze(-1) for d in (self.df, self.loc, self.scale, self.skew))

        # Probability mass to the left of the mode (loc)
        p_skew = skew.pow(2) / (1.0 + skew.pow(2))

        # Transform the quantile q to its equivalent in the standard symmetric t-distribution
        q_prime = torch.where(
            q_reshaped < p_skew,
            q_reshaped / (2.0 * p_skew),
            (q_reshaped - p_skew) / (2.0 * (1.0 - p_skew)) + 0.5,
        )
        
        # Get the quantile from the standard symmetric t-distribution
        t_val = self._symmetric_student_t_icdf(q_prime, df)

        # Apply the skewness transformation
        x = torch.where(
            q_reshaped < p_skew,
            skew * t_val,
            t_val / skew,
        )
        return loc + scale * x

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
    # A Skewed Student's T-distribution requires 4 parameters:
    # df, loc, scale, and skew.
    args_dim: int = 4

    def distribution(self, distr_args: torch.Tensor) -> "SkewedStudentT": # Return type will be the new class
        """
        Takes the raw output tensor from the projection head and transforms it
        into the parameters for a CustomStudentT distribution.

        Args:
            distr_args: A tensor of shape [..., 3], where the last dimension
                        contains the raw values for (df, loc, scale).
        
        Returns:
            A SkewedStudentT object.
        """
        # Split the raw parameters from the last dimension
        df_raw, loc_raw, scale_raw, skew_raw = torch.chunk(distr_args, chunks=4, dim=-1)

        # 1. Degrees of Freedom (df): Must be positive.
        #    We use softplus to ensure positivity and add 2.0 to ensure that the
        #    distribution always has a finite variance (variance is defined for df > 2).
        #    This improves training stability. Add a small epsilon for safety.
        df = F.softplus(df_raw.squeeze(-1)) + 2.001

        # 2. Location (loc): Can be any real number, so no transformation is needed.
        loc = loc_raw.squeeze(-1)

        # 3. Scale: Must be positive.
        #    We use softplus to ensure positivity and add a small epsilon for
        #    numerical stability, preventing the scale from becoming zero.
        scale = F.softplus(scale_raw.squeeze(-1)) + 1e-8

        # 4. Skew (gamma): Must be positive for Fernandez-Steel.
        #    A value of 1.0 means no skew. Softplus ensures positivity.
        skew = F.softplus(skew_raw.squeeze(-1)) + 1e-8

        # Return the parameterized distribution object using the custom class.
        return SkewedStudentT(df=df, loc=loc, scale=scale, skew=skew)
