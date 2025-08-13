import pytest
import pandas as pd
import torch
from ts_benchmark.baselines.duet.duet_prob import DUETProb

def test_end_to_end_training_bgev():
    # Create a dummy DataFrame for testing
    # It needs at least one column and enough rows for seq_len and horizon
    data = {
        'feature1': [float(i) for i in range(1000)],
        'feature2': [float(i * 2) for i in range(1000)]
    }
    index = pd.date_range(start='2023-01-01', periods=1000, freq='h')
    train_valid_data = pd.DataFrame(data, index=index)

    # Minimal configuration for DUETProb
    config_params = {
        "seq_len": 480,
        "horizon": 96, # Assuming horizon is 96 from the original log
        "distribution_family": "bgev",
        "num_epochs": 5,
        "batch_size": 512,
        "num_workers": 0,
        "patience": 3,
        "lr": 4.677262051374299e-05,
        "d_model": 512,
        "d_ff": 512,
        "n_heads": 2,
        "e_layers": 3,
        "moving_avg": 96,
        "dropout": 0.06837556852071201,
        "fc_dropout": 0.03920422913510781,
        "use_agc": False,
        "num_linear_experts": 4,
        "num_univariate_esn_experts": 4,
        "num_multivariate_esn_experts": 2,
        "k": 3,
        "hidden_size": 128,
        "reservoir_size_uni": 128,
        "spectral_radius_uni": 1.1067583460798478,
        "sparsity_uni": 0.43462206775695794,
        "leak_rate_uni": 0.8764606215249523,
        "input_scaling_uni": 0.17490420249024774,
        "esn_uni_weight_decay": 6.674998483387268e-05,
        "reservoir_size_multi": 128,
        "spectral_radius_multi": 0.62612037278307,
        "sparsity_multi": 0.3378953000489371,
        "leak_rate_multi": 0.4308226194693835,
        "input_scaling_multi": 0.019303218665464843,
        "esn_multi_weight_decay": 1.8964547479742005e-05,
        "noise_epsilon": 0.00239014953828634,
        "projection_head_layers": 3,
        "projection_head_dim_factor": 2,
        "projection_head_dropout": 0.024511905939680223,
        "projection_head_weight_decay": 0.008512856606392125,
        "enable_diagnostic_plots": False,
        "profile_epoch": None,
        "tqdm_update_freq": 1000,
        "tqdm_min_interval": 1000,
        "loss_coef": 4.387031931822492,
        "cvar_alpha": 0.05,
        "optimization_metric": "avg_nll",
        "use_channel_adjacency_prior": False,
    }

    # Ensure the device is set to MPS if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running end-to-end test on device: {device}")

    # Instantiate the model
    model = DUETProb(**config_params)
    
    

    # Run the training process
    try:
        model.forecast_fit(train_valid_data, train_ratio_in_tv=0.8)
        assert True # If we reach here, it means no exception was raised
    except Exception as e:
        pytest.fail(f"End-to-end training failed with exception: {e}")
