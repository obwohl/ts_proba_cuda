import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    
    A layer that normalizes a time series channel-wise and can reverse the process.
    This version is simplified to subtract the last or median value, or perform no
    normalization at all, helping to stabilize training for non-stationary series.
    """
    def __init__(self, num_features: int, eps=1e-4, affine=True, norm_mode='identity'):
        """
        Args:
            num_features (int): The number of features or channels in the time series.
            eps (float): A small value added for numerical stability (to avoid division by zero).
            affine (bool): If True, this module has learnable affine parameters (gamma and beta).
            norm_mode (str): The normalization mode. One of:
                             - 'identity': No normalization is applied. The input is passed through.
                             - 'subtract_last': Subtracts the last value of the sequence from the entire sequence.
                             - 'subtract_median': Subtracts the median of the sequence from the entire sequence.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.norm_mode = norm_mode

        # Validate the normalization mode
        if self.norm_mode not in ['identity', 'subtract_last', 'subtract_median']:
            raise ValueError(f"Unknown norm_mode: '{self.norm_mode}'")

        if self.affine:
            # Learnable affine parameters (reshaped for broadcasting)
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_features))
            self.gamma.register_hook(lambda grad: print(f"[DIAGNOSTIC LOG | RevIN] Gamma Grad: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}, has_nan={torch.isnan(grad).any().item()}"))
            self.beta.register_hook(lambda grad: print(f"[DIAGNOSTIC LOG | RevIN] Beta Grad: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}, has_nan={torch.isnan(grad).any().item()}"))

    def forward(self, x: torch.Tensor, mode: str, stats_to_use=None):
        """
        Performs forward normalization or backward de-normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, NumFeatures].
            mode (str): Either 'norm' for normalization or 'denorm' for de-normalization.
            stats_to_use (torch.Tensor, optional): External stats to use for de-normalization.
                                                    Shape: [Batch, NumFeatures, 2].
                                                    If provided during 'denorm', the layer will
                                                    use these instead of its internally stored stats.
        
        Returns:
            torch.Tensor: The normalized or de-normalized tensor.
            torch.Tensor (optional): If mode is 'norm', returns the statistics used for normalization.
                                     Shape: [Batch, NumFeatures, 2]
        """
        if mode == 'norm':
            self._get_statistics(x)
            x_norm = self._normalize(x)
            # The stats tensor has shape [B, N, 2] for location and scale
            stats = torch.stack([self.location_stat.squeeze(1), self.scale_stat.squeeze(1)], dim=-1)
            return x_norm, stats
            
        elif mode == 'denorm':
            x_denorm = self._denormalize(x, stats_to_use)
            return x_denorm
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def _get_statistics(self, x):
        """Calculates and stores the statistics for normalization."""
        if self.norm_mode == 'identity':
            # For identity, location is 0 and scale is 1.
            self.location_stat = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            self.scale_stat = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            
        elif self.norm_mode == 'subtract_last':
            # Location is the last value in the sequence.
            self.location_stat = x[:, -1:, :] # CORRECTED: .detach() removed
            # Scale is the RMS of the sequence after subtracting the location.
            x_centered = x - self.location_stat
            # --- NEW DIAGNOSTIC LOGGING ---
            if self.training: # Only log during training
                print(f"\n[DIAGNOSTIC LOG | RevIN] Before scale_stat calculation (subtract_last):")
                print(f"  x_centered mean: {x_centered.mean().item():.6f}, std: {x_centered.std().item():.6f}")
                print(f"  x_centered min: {x_centered.min().item():.6f}, max: {x_centered.max().item():.6f}")
                print(f"  x_centered has nan: {torch.isnan(x_centered).any().item()}")
                mean_squared_x_centered = torch.mean(x_centered**2, dim=1, keepdim=True)
                print(f"  mean_squared_x_centered mean: {mean_squared_x_centered.mean().item():.6f}, std: {mean_squared_x_centered.std().item():.6f}")
                print(f"  mean_squared_x_centered min: {mean_squared_x_centered.min().item():.6f}, max: {mean_squared_x_centered.max().item():.6f}")
                print(f"  mean_squared_x_centered has nan: {torch.isnan(mean_squared_x_centered).any().item()}")
            # --- END NEW DIAGNOSTIC LOGGING ---
            self.scale_stat = torch.sqrt(torch.mean(x_centered**2, dim=1, keepdim=True) + self.eps) # CORRECTED: .detach() removed

        elif self.norm_mode == 'subtract_median':
            # Location is the median value of the sequence.
            self.location_stat = torch.median(x, dim=1, keepdim=True)[0] # CORRECTED: .detach() removed
            # Scale is the RMS of the sequence after subtracting the location.
            x_centered = x - self.location_stat
            # --- NEW DIAGNOSTIC LOGGING ---
            if self.training: # Only log during training
                print(f"\n[DIAGNOSTIC LOG | RevIN] Before scale_stat calculation (subtract_median):")
                print(f"  x_centered mean: {x_centered.mean().item():.6f}, std: {x_centered.std().item():.6f}")
                print(f"  x_centered min: {x_centered.min().item():.6f}, max: {x_centered.max().item():.6f}")
                print(f"  x_centered has nan: {torch.isnan(x_centered).any().item()}")
                mean_squared_x_centered = torch.mean(x_centered**2, dim=1, keepdim=True)
                print(f"  mean_squared_x_centered mean: {mean_squared_x_centered.mean().item():.6f}, std: {mean_squared_x_centered.std().item():.6f}")
                print(f"  mean_squared_x_centered min: {mean_squared_x_centered.min().item():.6f}, max: {mean_squared_x_centered.max().item():.6f}")
                print(f"  mean_squared_x_centered has nan: {torch.isnan(mean_squared_x_centered).any().item()}")
            # --- END NEW DIAGNOSTIC LOGGING ---
            self.scale_stat = torch.sqrt(torch.mean(x_centered**2, dim=1, keepdim=True) + self.eps) # CORRECTED: .detach() removed

        # --- NEW DIAGNOSTIC LOGGING ---
        if self.training: # Only log during training
            print(f"\n[DIAGNOSTIC LOG | RevIN] After _get_statistics:")
            print(f"  norm_mode: {self.norm_mode}")
            print(f"  location_stat mean: {self.location_stat.mean().item():.6f}, std: {self.location_stat.std().item():.6f}")
            print(f"  location_stat min: {self.location_stat.min().item():.6f}, max: {self.location_stat.max().item():.6f}")
            print(f"  location_stat has nan: {torch.isnan(self.location_stat).any().item()}")
            print(f"  scale_stat mean: {self.scale_stat.mean().item():.6f}, std: {self.scale_stat.std().item():.6f}")
            print(f"  scale_stat min: {self.scale_stat.min().item():.6f}, max: {self.scale_stat.max().item():.6f}")
            print(f"  scale_stat has nan: {torch.isnan(self.scale_stat).any().item()}")
        # --- END NEW DIAGNOSTIC LOGGING ---
            
    def _normalize(self, x):
        """Applies the normalization."""
        x_norm = (x - self.location_stat) / self.scale_stat

        # --- NEW DIAGNOSTIC LOGGING ---
        if self.training: # Only log during training
            print(f"\n[DIAGNOSTIC LOG | RevIN] After initial normalization (before affine):")
            print(f"  x_norm mean: {x_norm.mean().item():.6f}, std: {x_norm.std().item():.6f}")
            print(f"  x_norm min: {x_norm.min().item():.6f}, max: {x_norm.max().item():.6f}")
            print(f"  x_norm has nan: {torch.isnan(x_norm).any().item()}")
        # --- END NEW DIAGNOSTIC LOGGING ---

        if self.affine:
            # --- NEW DIAGNOSTIC LOGGING ---
            if self.training: # Only log during training
                print(f"\n[DIAGNOSTIC LOG | RevIN] Before affine transformation:")
                print(f"  gamma mean: {self.gamma.mean().item():.6f}, std: {self.gamma.std().item():.6f}")
                print(f"  gamma min: {self.gamma.min().item():.6f}, max: {self.gamma.max().item():.6f}")
                print(f"  gamma has nan: {torch.isnan(self.gamma).any().item()}")
                print(f"  beta mean: {self.beta.mean().item():.6f}, std: {self.beta.std().item():.6f}")
                print(f"  beta min: {self.beta.min().item():.6f}, max: {self.beta.max().item():.6f}")
                print(f"  beta has nan: {torch.isnan(self.beta).any().item()}")
            # --- END NEW DIAGNOSTIC LOGGING ---
            
            x_norm = x_norm * self.gamma + self.beta
            # --- NEW DIAGNOSTIC LOGGING ---
            if self.training: # Only log during training
                print(f"\n[DIAGNOSTIC LOG | RevIN] After affine transformation:")
                print(f"  x_norm (after affine) mean: {x_norm.mean().item():.6f}, std: {x_norm.std().item():.6f}")
                print(f"  x_norm (after affine) min: {x_norm.min().item():.6f}, max: {x_norm.max().item():.6f}")
                print(f"  x_norm (after affine) has nan: {torch.isnan(x_norm).any().item()}")
                if x_norm.grad is not None:
                    print(f"  x_norm grad (after affine) mean: {x_norm.grad.mean().item():.6f}, std: {x_norm.grad.std().item():.6f}")
                    print(f"  x_norm grad (after affine) has nan: {torch.isnan(x_norm.grad).any().item()}")
                else:
                    print(f"  x_norm grad (after affine): None")
            # --- END NEW DIAGNOSTIC LOGGING ---
        return x_norm

    def _denormalize(self, x, stats_to_use):
        """
        Reverses the normalization.
        Can use either internal stats or externally provided ones.
        """
        # Determine which stats to use
        if stats_to_use is not None:
            # Reshape external stats for broadcasting: [B, N, 2] -> [B, 1, N]
            location = stats_to_use[..., 0].unsqueeze(1)
            scale = stats_to_use[..., 1].unsqueeze(1)
        else:
            location = self.location_stat
            scale = self.scale_stat
            
        x_denorm = x
        if self.affine:
            # Correctly reverse the affine transformation (symmetric operation)
            x_denorm = (x_denorm - self.beta) / self.gamma
        
        x_denorm = x_denorm * scale + location
        return x_denorm