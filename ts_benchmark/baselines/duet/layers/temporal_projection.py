
import torch
import torch.nn as nn

class TemporalProjectionHead(nn.Module):
    """
    A projection head that maps a sequence of features to a sequence of parameters.
    It uses two linear layers to transform both the temporal and feature dimensions.
    
    Input:  [Batch, InputSeqLen, InputDim]
    Output: [Batch, OutputSeqLen, OutputDim]
    """
    def __init__(self, input_seq_len: int, output_seq_len: int, input_dim: int, output_dim: int):
        super().__init__()
        self.temporal_mapper = nn.Linear(input_seq_len, output_seq_len)
        self.feature_mapper = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, InputSeqLen, InputDim]
        
        # 1. Map temporal dimension
        # Transpose to [Batch, InputDim, InputSeqLen] for the linear layer
        x = x.transpose(1, 2)
        # Apply linear layer to map InputSeqLen -> OutputSeqLen
        x = self.temporal_mapper(x)
        # Transpose back to [Batch, OutputSeqLen, InputDim]
        x = x.transpose(1, 2)
        
        # 2. Map feature dimension
        # Apply linear layer to map InputDim -> OutputDim
        x = self.feature_mapper(x)
        
        return x
