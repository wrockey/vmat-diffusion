"""
Multi-component loss with automatic uncertainty weighting.

Implements Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses",
CVPR 2018. Learns one scalar σ per loss component during training; weights become
1/(2σ²) automatically. Eliminates manual weight tuning for combined losses.

Formula for each term:
    weighted_i = L_i / (2 * σ_i²) + log(σ_i)
Total loss = sum(weighted_i)

Usage:
    # In LightningModule.__init__:
    self.uncertainty_loss = UncertaintyWeightedLoss(
        loss_names=["base", "gradient", "structure", "asymmetric", "dvh"],
        initial_log_sigma=0.0,  # global default
        initial_log_sigmas={"base": -1.5, "dvh": 0.3},  # per-component (from calibration)
    )

    # In training_step:
    raw_losses = {
        "base": mse_loss,
        "gradient": grad_loss,
        "structure": struct_loss,
        "asymmetric": asym_loss,
        "dvh": dvh_loss,
    }
    result = self.uncertainty_loss(raw_losses)
    total_loss = result["loss"]

    # Log sigma evolution (shows which losses the model finds most uncertain):
    for name in self.uncertainty_loss.loss_names:
        self.log(f"train/raw_{name}", result[f"raw_{name}"])
        self.log(f"train/sigma_{name}", result[f"sigma_{name}"])

Reference: https://arxiv.org/abs/1705.07115
Attribution: Initial implementation by Grok (external review, 2026-02-17),
             reviewed and integrated by Claude.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-component loss with automatic uncertainty weighting
    (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018)

    Formula for each term:
        weighted_i = L_i / (2 * σ_i²) + log(σ_i)
    Total loss = sum(weighted_i)

    Learns one σ per loss component during training. Extremely stable and
    eliminates manual weight tuning.
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_sigma: float = 0.0,   # start with equal weighting
        initial_log_sigmas: Dict[str, float] = None,  # per-component overrides
        min_sigma: float = 1e-4
    ):
        super().__init__()
        self.loss_names = loss_names
        self.min_sigma = min_sigma

        # Learnable log(σ) per loss (keeps σ > 0)
        # Per-component values override the global default
        self.log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(
                initial_log_sigmas.get(name, initial_log_sigma)
                if initial_log_sigmas else initial_log_sigma,
                dtype=torch.float32
            ))
            for name in loss_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            losses: dict of raw loss tensors, e.g.
                    {"base": L_base, "gradient": L_grad, "structure": L_struct, ...}
                    All should be scalar (already .mean() reduced)

        Returns:
            dict with:
                "loss"               -> total weighted loss (for .backward())
                "raw_{name}"         -> original loss value
                "weighted_{name}"    -> weighted contribution
                "sigma_{name}"       -> learned uncertainty (for logging)
        """
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        output = {}

        for name, loss_val in losses.items():
            if name not in self.log_sigmas:
                # Fallback for any unweighted terms
                output[name] = loss_val
                total_loss = total_loss + loss_val
                continue

            log_sigma = self.log_sigmas[name]
            sigma = torch.exp(log_sigma).clamp(min=self.min_sigma)

            # Uncertainty-weighted term (Kendall 2018: L/(2σ²) + log(σ))
            # Use clamped sigma for stability but original log_sigma for regularization
            weighted = loss_val / (2 * sigma ** 2) + log_sigma

            total_loss = total_loss + weighted

            # Store for logging & inspection
            output[f"raw_{name}"] = loss_val.detach()
            output[f"weighted_{name}"] = weighted.detach()
            output[f"sigma_{name}"] = sigma.detach()

        output["loss"] = total_loss
        return output
