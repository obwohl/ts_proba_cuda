# Dateipfad: ts_benchmark/baselines/duet/utils/crps.py

import torch
from torch.distributions import Distribution

def crps_loss(
    distr: Distribution, 
    y_true: torch.Tensor, 
    num_quantiles: int = 99
) -> torch.Tensor:
    """
    Berechnet den Continuous Ranked Probability Score (CRPS) durch eine
    Approximation über den Pinball Loss.
    """
    
    # 1. Erzeuge die Quantil-Niveaus
    quantiles = torch.linspace(
        start=0.5 / num_quantiles,
        end=1.0 - 0.5 / num_quantiles,
        steps=num_quantiles,
        device=y_true.device,
    )

    # 2. Berechne die vorhergesagten Werte für jedes Quantil
    # distr.icdf gibt [B, N_Vars, H_predicted, Q] zurück
    y_pred_quantiles = distr.icdf(quantiles)

    # 3. KORREKTUR: Schneide die Vorhersage auf die korrekte Horizontlänge zu
    # y_true hat die Form [B, N_Vars, H_true]
    true_horizon = y_true.shape[2]
    # Schneide entlang der Horizont-Dimension (Index 2)
    y_pred_sliced = y_pred_quantiles[:, :, :true_horizon, :]

    # 4. Forme y_true für das Broadcasting korrekt vor
    # Wir brauchen [B, N_Vars, H_true, 1]
    y_true_compatible = y_true.unsqueeze(-1)

    # 5. Berechne den Pinball Loss (Dimensionen passen jetzt)
    # [B,N_Vars,H_true,Q] - [B,N_Vars,H_true,1] -> OK
    error = y_pred_sliced - y_true_compatible
    
    # quantiles ist 1D [Q], wir brauchen [1, 1, 1, Q] für Broadcasting
    quantiles_bcast = quantiles.view(1, 1, -1) # Form für [B, N_Vars, H, Q]

    loss_term1 = quantiles_bcast * error
    loss_term2 = (1.0 - quantiles_bcast) * error
    pinball_loss_per_quantile = torch.max(loss_term1, -loss_term2)

    # 6. Approximiere den CRPS durch den Durchschnitt
    crps_raw = 2 * pinball_loss_per_quantile.mean(dim=-1)

    # Mache die Funktion robust gegen NaN/Inf in den Zielwerten
    mask = torch.isfinite(y_true)
    crps = torch.where(mask, crps_raw, torch.tensor(0.0, device=crps_raw.device))

    # Gib den Loss in der Originalform zurück
    return crps