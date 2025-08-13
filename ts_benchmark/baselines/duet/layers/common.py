# ts_benchmark/baselines/duet/layers/common.py

import torch.nn as nn
import torch

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
            return self.final_layer(x).to(torch.float32)
        return self.projection(x).to(torch.float32)
