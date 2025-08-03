

import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

# --- Kopie von JohnsonSU_torch aus johnson_system.py ---
class JohnsonSU_torch(Distribution):
    arg_constraints = {
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, gamma, delta, xi, lambda_, validate_args=None):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape, validate_args=validate_args)
        self._log_delta = self.delta.log()
        self._log_lambda = self.lambda_.log()
        self._log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        y = (value - self.xi) / self.lambda_
        z = self.gamma + self.delta * torch.asinh(y)
        log_p = self._log_delta - self._log_lambda - self._log_sqrt_2pi - 0.5 * z.pow(2) - 0.5 * (1 + y.pow(2)).log()
        return log_p

# --- SciPy Wrapper für ICDF (Inverse CDF) ---
def _scipy_to_tensor(func, *args, **kwargs):
    device = 'cpu'
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            device = arg.device
            break

    numpy_args = [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
    numpy_kwargs = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    
    result_np = func(*numpy_args, **numpy_kwargs)
    result_np = np.nan_to_num(result_np, nan=-1e9, posinf=-1e9, neginf=-1e9)
    return torch.from_numpy(np.array(result_np, dtype=np.float32)).to(device)

class JohnsonSU_scipy(Distribution):
    arg_constraints = {} 
    def __init__(self, gamma, delta, xi, lambda_):
        self.gamma, self.delta, self.xi, self.lambda_ = broadcast_all(gamma, delta, xi, lambda_)
        super().__init__(self.gamma.shape)
    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        q_reshaped = q.view([1] * self.gamma.dim() + [-1])
        gamma, delta, xi, lambda_ = (d.unsqueeze(-1) for d in (self.gamma, self.delta, self.xi, self.lambda_))
        return _scipy_to_tensor(scipy.stats.johnsonsu.ppf, q_reshaped, a=gamma, b=delta, loc=xi, scale=lambda_)

# --- Zero-Inflated Johnson SU Verteilung ---
class ZeroInflatedJohnsonSU(Distribution):
    arg_constraints = {
        'p_zero_logit': constraints.real,
        'gamma': constraints.real, 'delta': constraints.positive,
        'xi': constraints.real, 'lambda_': constraints.positive,
    }
    support = constraints.greater_than_eq(0.0)
    has_rsample = False

    def __init__(self, p_zero_logit, gamma, delta, xi, lambda_, validate_args=None):
        self.p_zero_logit, self.gamma, self.delta, self.xi, self.lambda_ = \
            broadcast_all(p_zero_logit, gamma, delta, xi, lambda_)
        super().__init__(self.p_zero_logit.shape, validate_args=validate_args)

        self.p_zero = torch.sigmoid(self.p_zero_logit)
        self.johnson_su_dist_torch = JohnsonSU_torch(gamma, delta, xi, lambda_)
        self.johnson_su_dist_scipy = JohnsonSU_scipy(gamma, delta, xi, lambda_)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if (value < 0).any():
            raise ValueError("Werte für ZeroInflatedJohnsonSU müssen nicht-negativ sein.")

        log_prob_zero = torch.log(self.p_zero)

        positive_values_mask = (value > 0)
        log_prob_positive_component = torch.full_like(value, -float('inf'))

        if positive_values_mask.any():
            log_prob_positive_component[positive_values_mask] = \
                torch.log(1 - self.p_zero) + self.johnson_su_dist_torch.log_prob(value[positive_values_mask])

        result_log_prob = torch.where(value == 0, log_prob_zero, log_prob_positive_component)
        return result_log_prob

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # Sicherstellen, dass q im Bereich [0, 1] liegt
        if not (q >= 0).all() and (q <= 1).all():
            raise ValueError("Quantile q müssen im Bereich [0, 1] liegen.")

        # Wenn q <= p_zero, ist das Quantil 0
        result = torch.where(q <= self.p_zero, torch.tensor(0.0, device=q.device), torch.tensor(0.0, device=q.device))

        # Für q > p_zero, berechne das Quantil aus der Johnson SU Verteilung
        mask_positive_quantile = (q > self.p_zero)
        if mask_positive_quantile.any():
            # Skaliere das Quantil für die Johnson SU Verteilung
            q_scaled = (q[mask_positive_quantile] - self.p_zero) / (1 - self.p_zero)
            result[mask_positive_quantile] = self.johnson_su_dist_scipy.icdf(q_scaled)
        
        return result

# --- Hilfsfunktion zum Plotten ---
def plot_distribution(dist, scenario_name, data=None, ax=None, color='blue', label_prefix=''):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot PDF (approximiert)
    x_vals = torch.linspace(0, 15, 500) # Range für Regenmengen
    log_probs = dist.log_prob(x_vals)
    pdf_vals = torch.exp(log_probs)
    pdf_vals[x_vals < 0] = 0 # Keine negativen Regenmengen

    # Setze PDF bei 0 auf 0, da die Masse dort diskret ist
    pdf_vals[x_vals == 0] = 0 

    ax.plot(x_vals.numpy(), pdf_vals.numpy(), color=color, linestyle='-', label=f'{label_prefix}PDF')

    # Plot Quantile
    quantiles_to_plot = torch.tensor([0.1, 0.5, 0.9])
    quantile_values = dist.icdf(quantiles_to_plot)
    
    for i, q_val in enumerate(quantiles_to_plot):
        ax.axvline(x=quantile_values[i].item(), color=color, linestyle='--', alpha=0.7, 
                   label=f'{label_prefix}{int(q_val.item()*100)}% Quantil: {quantile_values[i].item():.2f}')

    # Plot discrete mass at zero
    p_zero_val = dist.p_zero.item()
    if p_zero_val > 0.001: # Nur plotten, wenn signifikant
        ax.bar(0, p_zero_val * 5, width=0.1, color=color, alpha=0.3, label=f'{label_prefix}P(X=0)={p_zero_val:.2f}') # Skaliert für Sichtbarkeit

    if data is not None:
        ax.hist(data.numpy(), bins=50, density=True, alpha=0.6, color='gray', label='Synthetische Daten')

    ax.set_title(f'Verteilung für {scenario_name}')
    ax.set_xlabel('Regenmenge')
    ax.set_ylabel('Wahrscheinlichkeitsdichte / Wahrscheinlichkeit')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True)

# --- Szenarien und Parameter ---

# Szenario 1: Nieselregen-Tag
# Hohe Wahrscheinlichkeit für Regen, aber geringe Intensität
# p_zero: niedrig (z.B. 0.3 -> logit ca. -0.85)
# Johnson SU: konzentriert nahe Null, leicht rechtsschief
params_drizzle = {
    'p_zero_logit': torch.tensor(-0.85),
    'gamma': torch.tensor(-0.5), 
    'delta': torch.tensor(1.0),
    'xi': torch.tensor(0.01),
    'lambda_': torch.tensor(0.1)
}

# Szenario 2: Gewitter-Tag
# Geringe Wahrscheinlichkeit für Regen, aber hohe Intensität, wenn es regnet
# p_zero: hoch (z.B. 0.9 -> logit ca. 2.2)
# Johnson SU: verschoben zu höheren Werten, stark rechtsschief
params_thunderstorm = {
    'p_zero_logit': torch.tensor(2.2),
    'gamma': torch.tensor(-2.5), 
    'delta': torch.tensor(1.0),
    'xi': torch.tensor(5.0),
    'lambda_': torch.tensor(2.0)
}

# --- Generiere synthetische Daten ---
def generate_synthetic_data(dist_params, num_samples, scenario_type):
    p_zero = torch.sigmoid(dist_params['p_zero_logit']).item()
    data = []
    for _ in range(num_samples):
        if np.random.rand() < p_zero:
            data.append(0.0)
        else:
            # Generiere aus Johnson SU für positive Werte
            # Dies ist eine Vereinfachung, da wir keine rsample für JohnsonSU_torch haben.
            # Wir nutzen hier scipy.stats.johnsonsu.rvs zur Generierung.
            positive_val = scipy.stats.johnsonsu.rvs(
                a=dist_params['gamma'].item(), 
                b=dist_params['delta'].item(), 
                loc=dist_params['xi'].item(), 
                scale=dist_params['lambda_'].item(),
                size=1
            )[0]
            data.append(max(0.0, positive_val)) # Sicherstellen, dass es nicht negativ ist
    return torch.tensor(data, dtype=torch.float32)

num_samples_per_scenario = 1000

# Szenario 2a: Gewitter-Tag, es gewittert tatsächlich
data_thunderstorm_actual = generate_synthetic_data(params_thunderstorm, num_samples_per_scenario, 'thunderstorm_actual')

# Szenario 2b: Gewitter-Tag, es gewittert aber nicht (nur Nullen)
data_thunderstorm_no_rain = torch.zeros(num_samples_per_scenario, dtype=torch.float32)

# Szenario 1: Nieselregen-Tag (es nieselt tatsächlich)
data_drizzle_actual = generate_synthetic_data(params_drizzle, num_samples_per_scenario, 'drizzle_actual')

# --- Erstelle Verteilungsinstanzen ---
zi_dist_drizzle = ZeroInflatedJohnsonSU(**params_drizzle)
zi_dist_thunderstorm = ZeroInflatedJohnsonSU(**params_thunderstorm)

# --- Plotten und NLL berechnen ---
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Zero-Inflated Johnson SU Verteilungsdemonstration', fontsize=16)

# Szenario 1: Nieselregen-Tag
print("\n--- Szenario: Nieselregen-Tag ---")
plot_distribution(zi_dist_drizzle, 'Nieselregen-Tag (ZI)', data_drizzle_actual, ax=axes[0, 0])
nll_drizzle_zi = -zi_dist_drizzle.log_prob(data_drizzle_actual).sum().item()
print(f"  NLL (ZI-Modell) für Nieselregen-Daten: {nll_drizzle_zi:.2f}")

# Vergleich mit Standard Johnson SU für Nieselregen-Tag
# Hier setzen wir p_zero_logit auf einen sehr kleinen Wert, um p_zero nahe 0 zu bekommen
params_drizzle_std = params_drizzle.copy()
params_drizzle_std['p_zero_logit'] = torch.tensor(-10.0) # p_zero ~ 0
std_dist_drizzle = ZeroInflatedJohnsonSU(**params_drizzle_std) # Nutze ZI-Klasse, aber mit p_zero=0
plot_distribution(std_dist_drizzle, 'Nieselregen-Tag (Standard SU)', data_drizzle_actual, ax=axes[0, 1], color='red', label_prefix='Std. ')
nll_drizzle_std = -std_dist_drizzle.log_prob(data_drizzle_actual).sum().item()
print(f"  NLL (Standard SU) für Nieselregen-Daten: {nll_drizzle_std:.2f}")
print(f"  Vorteil ZI vs. Standard SU: {nll_drizzle_std - nll_drizzle_zi:.2f}")

# Szenario 2a: Gewitter-Tag, es gewittert tatsächlich
print("\n--- Szenario: Gewitter-Tag (es regnet) ---")
plot_distribution(zi_dist_thunderstorm, 'Gewitter-Tag (ZI, es regnet)', data_thunderstorm_actual, ax=axes[1, 0])
nll_thunderstorm_actual_zi = -zi_dist_thunderstorm.log_prob(data_thunderstorm_actual).sum().item()
print(f"  NLL (ZI-Modell) für Gewitter-Daten (es regnet): {nll_thunderstorm_actual_zi:.2f}")

# Vergleich mit Standard Johnson SU für Gewitter-Tag (es regnet)
params_thunderstorm_std = params_thunderstorm.copy()
params_thunderstorm_std['p_zero_logit'] = torch.tensor(-10.0) # p_zero ~ 0
std_dist_thunderstorm = ZeroInflatedJohnsonSU(**params_thunderstorm_std)
plot_distribution(std_dist_thunderstorm, 'Gewitter-Tag (Standard SU, es regnet)', data_thunderstorm_actual, ax=axes[1, 1], color='red', label_prefix='Std. ')
nll_thunderstorm_actual_std = -std_dist_thunderstorm.log_prob(data_thunderstorm_actual).sum().item()
print(f"  NLL (Standard SU) für Gewitter-Daten (es regnet): {nll_thunderstorm_actual_std:.2f}")
print(f"  Vorteil ZI vs. Standard SU: {nll_thunderstorm_actual_std - nll_thunderstorm_actual_zi:.2f}")

# Szenario 2b: Gewitter-Tag, es gewittert aber nicht (nur Nullen)
print("\n--- Szenario: Gewitter-Tag (es regnet nicht) ---")
plot_distribution(zi_dist_thunderstorm, 'Gewitter-Tag (ZI, es regnet nicht)', data_thunderstorm_no_rain, ax=axes[2, 0])
nll_thunderstorm_no_rain_zi = -zi_dist_thunderstorm.log_prob(data_thunderstorm_no_rain).sum().item()
print(f"  NLL (ZI-Modell) für Gewitter-Daten (es regnet nicht): {nll_thunderstorm_no_rain_zi:.2f}")

# Vergleich mit Standard Johnson SU für Gewitter-Tag (es regnet nicht)
# Hier wird der NLL sehr hoch sein, da die Standard SU keine Masse bei 0 hat.
plot_distribution(std_dist_thunderstorm, 'Gewitter-Tag (Standard SU, es regnet nicht)', data_thunderstorm_no_rain, ax=axes[2, 1], color='red', label_prefix='Std. ')
nll_thunderstorm_no_rain_std = -std_dist_thunderstorm.log_prob(data_thunderstorm_no_rain).sum().item()
print(f"  NLL (Standard SU) für Gewitter-Daten (es regnet nicht): {nll_thunderstorm_no_rain_std:.2f}")
print(f"  Vorteil ZI vs. Standard SU: {nll_thunderstorm_no_rain_std - nll_thunderstorm_no_rain_zi:.2f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
