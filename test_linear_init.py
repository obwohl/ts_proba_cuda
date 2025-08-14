import torch
import torch.nn as nn

linear_layer = nn.Linear(3, 128)

print(f"Weight mean: {linear_layer.weight.mean().item()}")
print(f"Weight std: {linear_layer.weight.std().item()}")
print(f"Weight min: {linear_layer.weight.min().item()}")
print(f"Weight max: {linear_layer.weight.max().item()}")

print(f"Bias mean: {linear_layer.bias.mean().item()}")
print(f"Bias std: {linear_layer.bias.std().item()}")
print(f"Bias min: {linear_layer.bias.min().item()}")
print(f"Bias max: {linear_layer.bias.max().item()}")

# Check if all weights are zero
print(f"All weights zero: {torch.all(linear_layer.weight == 0).item()}")
print(f"All bias zero: {torch.all(linear_layer.bias == 0).item()}")
