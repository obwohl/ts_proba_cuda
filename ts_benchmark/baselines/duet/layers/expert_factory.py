import torch
import torch.nn as nn
from copy import deepcopy
import warnings
import time
import os

# Importiere die beiden Experten-Typen, die wir erstellen können
from .linear_pattern_extractor import Linear_extractor
from .esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert


def create_experts(config) -> nn.ModuleList:
    """
    Erstellt eine `nn.ModuleList` mit einer Mischung aus linearen und ESN-Experten.
    """
    experts = nn.ModuleList()

    # 1. Erstelle die linearen Experten
    for _ in range(getattr(config, 'num_linear_experts', 0)):
        experts.append(Linear_extractor(config))
    
    # Hole die Anzahl der neuen, spezifischen ESN-Typen
    num_uni_esn = getattr(config, 'num_univariate_esn_experts', 0)
    num_multi_esn = getattr(config, 'num_multivariate_esn_experts', 0)

    # Fallback für alte Konfigurationen, um Abwärtskompatibilität zu gewährleisten
    if num_uni_esn == 0 and num_multi_esn == 0 and hasattr(config, 'num_esn_experts'):
        num_legacy_esn = getattr(config, 'num_esn_experts', 0)
        if num_legacy_esn > 0:
            warnings.warn(
                "'num_esn_experts' is deprecated. Please use 'num_univariate_esn_experts' "
                "and 'num_multivariate_esn_experts'. Treating legacy experts as 'univariate'.",
                DeprecationWarning
            )
            num_uni_esn = num_legacy_esn

    # 2. Erstelle die univariaten ESN-Experten
    for _ in range(num_uni_esn):
        experts.append(UnivariateReservoirExpert(config))

    # 3. Erstelle die multivariaten ESN-Experten
    for _ in range(num_multi_esn):
        experts.append(MultivariateReservoirExpert(config))

    # 4. Wende die Initialisierung auf ALLE erstellten Experten an
    torch.manual_seed(int(time.time() * 1000) + os.getpid())
    
    for i, expert in enumerate(experts):
        expert.reset_parameters(expert_index=i)

    return experts