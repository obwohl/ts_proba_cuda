import torch
from einops import rearrange
from typing import List

from .extended_gpd import ZeroInflatedExtendedGPD_M1_Continuous

class ExtendedGPDOutput:
    """
    A class to handle the output for the Zero-Inflated Extended GPD model.
    This class takes the raw output from a neural network and transforms it
    into the parameters required for the ZIEGPD distribution.
    """
    args_dim: int = 4  # pi, kappa, sigma, xi

    def __init__(self, channel_types: List[str]):
        """
        Initializes the ExtendedGPDOutput.

        Args:
            channel_types (List[str]): A list of strings identifying the distribution
                                       type for each channel. Currently unused but
                                       kept for API consistency.
        """
        self.channel_types = channel_types
        self.n_vars = len(channel_types)

    def distribution(
        self, distr_args: torch.Tensor, horizon: int
    ) -> ZeroInflatedExtendedGPD_M1_Continuous:
        """
        Creates a ZeroInflatedExtendedGPD_M1_Continuous distribution from the
        raw network output.

        Args:
            distr_args (torch.Tensor): The raw tensor from the projection head,
                                       with shape [B, N_vars, Horizon * args_dim].
            horizon (int): The forecast horizon.

        Returns:
            ZeroInflatedExtendedGPD_M1_Continuous: An instance of the distribution.
        """
        # distr_args is already in shape [B, N_vars, H, P]
        # Chunk parameters along the last dimension (p)
        # Each parameter will have shape [B, N_vars, Horizon]
        pi_raw, kappa_raw, sigma_raw, xi = torch.chunk(
            distr_args, chunks=self.args_dim, dim=-1
        )

        # Squeeze the last dimension which is now 1
        pi_raw = pi_raw.squeeze(-1)
        kappa_raw = kappa_raw.squeeze(-1)
        sigma_raw = sigma_raw.squeeze(-1)
        xi = xi.squeeze(-1)

        # Apply sigmoid to the raw pi parameter to constrain it to (0, 1)
        pi = torch.sigmoid(pi_raw)

        # The ZeroInflatedExtendedGPD_M1_Continuous distribution expects raw
        # (unconstrained) parameters for kappa and sigma and applies softplus
        # internally. xi is unconstrained.
        return ZeroInflatedExtendedGPD_M1_Continuous(
            pi=pi, kappa_raw=kappa_raw, sigma_raw=sigma_raw, xi=xi
        )
