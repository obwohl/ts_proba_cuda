import torch.nn as nn
from copy import deepcopy
import warnings

# Importiere die beiden Experten-Typen, die wir erstellen können
from .linear_pattern_extractor import Linear_extractor
from .esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert


def create_experts(config) -> nn.ModuleList:
    """
    Erstellt eine `nn.ModuleList` mit einer Mischung aus linearen und ESN-Experten.

    Diese Fabrik-Funktion liest die Konfiguration, um die Anzahl und die spezifischen
    Parameter für jeden Expertentyp zu bestimmen.

    Args:
        config: Das Konfigurationsobjekt, das die folgenden Attribute enthalten muss:
                - num_linear_experts (int): Anzahl der linearen Experten.
                - num_univariate_esn_experts (int): Anzahl der univariaten ESN-Experten.
                - num_multivariate_esn_experts (int): Anzahl der multivariaten ESN-Experten.

                Veraltete Parameter (werden ignoriert, wenn neue vorhanden sind):
                - num_esn_experts (int): Wird ignoriert, wenn die neuen Parameter gesetzt sind.
                - esn_configs (list): Wird derzeit nicht für die neue hybride Architektur verwendet.

    Returns:
        nn.ModuleList: Eine Liste, die die instanziierten Experten-Module enthält.
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

    return experts