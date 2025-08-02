import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, config, num_experts, input_dim):  # Akzeptiert num_experts und input_dim als Argument
        super(encoder, self).__init__()
        # Definiere encoder_hidden_size und gib ihm einen Standardwert
        encoder_hidden_size = getattr(config, 'hidden_size', 128) 

        self.distribution_fit = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_size, bias=False), nn.ReLU(),
            nn.Linear(encoder_hidden_size, 1, bias=False))  # Output a single logit per conditioned input

    def forward(self, x):
        out = self.distribution_fit(x)
        return out
