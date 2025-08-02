import torch
import torch.nn as nn
from unittest.mock import Mock
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import numpy as np

def test_gating_diversity():
    # 1. Mock Configuration
    config = Mock()
    config.seq_len = 96
    config.d_model = 32
    config.enc_in = 1 # Assuming single channel for simplicity in this test
    config.num_linear_experts = 3
    config.num_univariate_esn_experts = 3
    config.num_multivariate_esn_experts = 3
    config.k = 3 # Number of top experts to select
    config.norm_mode = 'subtract_median' # Required by RevIN
    config.expert_embedding_dim = 16 # NEW: Add expert embedding dimension

    config.moving_avg = 25 # Add a concrete integer value for moving_avg

    # ESN specific configs (can be minimal for this test)
    config.hidden_size = 32
    config.reservoir_size_uni = 16
    config.spectral_radius_uni = 0.99
    config.sparsity_uni = 0.1
    config.leak_rate_uni = 0.5
    config.input_scaling_uni = 0.5
    config.esn_uni_weight_decay = 1e-5
    config.reservoir_size_multi = 16
    config.spectral_radius_multi = 0.99
    config.sparsity_multi = 0.1
    config.leak_rate_multi = 0.5
    config.input_scaling_multi = 0.5
    config.esn_multi_weight_decay = 1e-5
    config.projection_head_layers = 2
    config.projection_head_dim_factor = 2
    config.projection_head_dropout = 0.1

    # 2. Instantiate Model
    model = Linear_extractor_cluster(config)
    model.train() # Set to train mode to enable noisy gating

    # 3. Simulate Input
    batch_size = 128 # Increased batch size for better diversity
    num_channels = 1 # For simplicity, assuming one channel for now
    x = torch.randn(batch_size, config.seq_len, num_channels) # [B, L, N]

    # Accumulate gating weights and selection counts over multiple forward passes
    num_iterations = 10
    all_gate_weights_linear = []
    all_gate_weights_uni_esn = []
    all_gate_weights_multi_esn = []
    all_selection_counts = []

    for _ in range(num_iterations):
        y, loss_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts = model(x)
        
        if avg_gate_weights_linear.numel() > 0:
            all_gate_weights_linear.append(avg_gate_weights_linear.detach().cpu().numpy())
        if avg_gate_weights_uni_esn.numel() > 0:
            all_gate_weights_uni_esn.append(avg_gate_weights_uni_esn.detach().cpu().numpy())
        if avg_gate_weights_multi_esn.numel() > 0:
            all_gate_weights_multi_esn.append(avg_gate_weights_multi_esn.detach().cpu().numpy())
        if expert_selection_counts.numel() > 0:
            all_selection_counts.append(expert_selection_counts.detach().cpu().numpy())

    # Convert lists to numpy arrays for statistical checks
    all_gate_weights_linear = np.array(all_gate_weights_linear)
    all_gate_weights_uni_esn = np.array(all_gate_weights_uni_esn)
    all_gate_weights_multi_esn = np.array(all_gate_weights_multi_esn)
    all_selection_counts = np.array(all_selection_counts)

    # 5. Assert Diversity
    # Define a small epsilon for diversity check
    epsilon = 1e-6

    # Check if gating weights for linear experts show diversity
    if all_gate_weights_linear.shape[1] > 1: # Check if there's more than one linear expert
        # Check if the standard deviation across experts is greater than epsilon
        assert np.std(all_gate_weights_linear, axis=1).mean() > epsilon, \
            f"Linear expert gating weights lack diversity. Std Dev: {np.std(all_gate_weights_linear, axis=1).mean()}"

    # Check if gating weights for univariate ESN experts show diversity
    if all_gate_weights_uni_esn.shape[1] > 1: # Check if there's more than one univariate ESN expert
        assert np.std(all_gate_weights_uni_esn, axis=1).mean() > epsilon, \
            f"Univariate ESN expert gating weights lack diversity. Std Dev: {np.std(all_gate_weights_uni_esn, axis=1).mean()}"

    # Check if gating weights for multivariate ESN experts show diversity
    if all_gate_weights_multi_esn.shape[1] > 1: # Check if there's more than one multivariate ESN expert
        assert np.std(all_gate_weights_multi_esn, axis=1).mean() > epsilon, \
            f"Multivariate ESN expert gating weights lack diversity. Std Dev: {np.std(all_gate_weights_multi_esn, axis=1).mean()}"

    # Check if expert selection counts show diversity
    if all_selection_counts.shape[1] > 1: # Check if there's more than one expert overall
        assert np.std(all_selection_counts, axis=1).mean() > epsilon, \
            f"Expert selection counts lack diversity. Std Dev: {np.std(all_selection_counts, axis=1).mean()}"

    # Check if combined output is not all zeros or NaNs
    assert not torch.allclose(y, torch.zeros_like(y)), "Combined output is all zeros!"
    assert not torch.isnan(y).any(), "Combined output contains NaNs!"

    print("Gating diversity test passed!")

# To run this test, you would typically use pytest:
# pytest ts_benchmark/baselines/duet/tests/test_gating_diversity.py
