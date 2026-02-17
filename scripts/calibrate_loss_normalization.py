#!/usr/bin/env python
"""
Standalone Loss Normalization Calibrator for Phase 2.

Loads a small number of .npz files from the processed dataset, simulates
realistic initial predictions (GT + Gaussian noise), computes raw average
value of each loss component, and recommends initial_log_sigma values for
UncertaintyWeightedLoss (Kendall et al. 2018).

Heuristic: initial_log_sigma = log(sqrt(avg_loss)) -> starts balanced.

Run example:
    python scripts/calibrate_loss_normalization.py \
        --data_dir /path/to/processed_npz \
        --num_samples 12

Phase 2 integration TODO:
    Replace the stub loss functions below with imports from
    train_baseline_unet.py. The stubs are placeholders to show the
    expected interface; the real losses are already implemented and
    tested in the pilot experiments.

Attribution: Initial implementation by Grok (external review, 2026-02-17),
             reviewed and saved by Claude. Stubs to be replaced with real
             loss functions during Phase 2 setup.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict


# =============================================================================
# LOSS FUNCTIONS — STUBS (replace with real implementations from training script)
#
# Phase 2 TODO: Replace these with imports from train_baseline_unet.py:
#   - base_loss -> F.mse_loss (or F.l1_loss)
#   - gradient_loss -> GradientLoss3D (lines ~420-470 in train_baseline_unet.py)
#   - structure_weighted_loss -> StructureWeightedLoss (lines ~600+ in train_baseline_unet.py)
#   - asymmetric_ptv_loss -> AsymmetricPTVLoss (lines ~730+ in train_baseline_unet.py)
#   - dvh_aware_loss -> DVHAwareLoss (lines ~500-600 in train_baseline_unet.py)
#
# These stubs produce plausible scalar values for testing the calibration
# pipeline itself, but they are NOT the real loss functions.
# =============================================================================

def base_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 or MSE — replace with your actual base loss."""
    return torch.nn.functional.mse_loss(pred, target, reduction="mean")


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3D Sobel gradient loss — STUB, replace with GradientLoss3D."""
    diff = torch.abs(torch.diff(pred, dim=-1)) - torch.abs(torch.diff(target, dim=-1))
    return torch.mean(torch.abs(diff))


def structure_weighted_loss(
    pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Structure-weighted loss — STUB, replace with StructureWeightedLoss."""
    weights = torch.mean(masks, dim=(1, 2, 3), keepdim=True) + 1e-5
    weights = weights / weights.sum()
    return torch.mean(weights * torch.abs(pred - target))


def asymmetric_ptv_loss(
    pred: torch.Tensor, target: torch.Tensor, ptv_mask: torch.Tensor
) -> torch.Tensor:
    """Asymmetric PTV loss — STUB, replace with AsymmetricPTVLoss."""
    error = pred - target
    under = torch.where(error < 0, torch.abs(error), torch.tensor(0.0, device=error.device))
    over = torch.where(error > 0, error, torch.tensor(0.0, device=error.device))
    return torch.mean(under * 2.0 + over * 0.5) * ptv_mask.float().mean()


def dvh_aware_loss(pred: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """DVH-aware loss — STUB, replace with DVHAwareLoss."""
    return torch.mean(torch.abs(pred.mean(dim=(1, 2, 3)) - constraints[:5]))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sample(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Load a single .npz file in the v2.2.0 format."""
    data = np.load(npz_path, allow_pickle=True)

    # NPZ v2.2.0 format: dose is (H, W, D), masks is (8, H, W, D)
    dose = torch.from_numpy(data['dose']).unsqueeze(0).float()       # (1, H, W, D)
    masks = torch.from_numpy(data['masks']).float()                  # (8, H, W, D)
    constraints = torch.from_numpy(data['constraints']).float()      # (13,)

    # PTV70 mask is channel 0
    ptv_mask = masks[0]

    return {
        "dose": dose,
        "masks": masks,
        "constraints": constraints,
        "ptv_mask": ptv_mask,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate loss normalization for UncertaintyWeightedLoss"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Folder containing .npz files"
    )
    parser.add_argument(
        "--num_samples", type=int, default=12,
        help="Number of samples for calibration"
    )
    parser.add_argument(
        "--noise_std", type=float, default=5.0,
        help="Noise std (Gy) for simulated initial prediction"
    )
    parser.add_argument(
        "--rx_dose_gy", type=float, default=70.0,
        help="Prescription dose for denormalization"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    npz_files = sorted(data_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    selected_files = npz_files[:args.num_samples]
    print(f"Calibrating on {len(selected_files)} samples from {data_dir}")

    # Noise std in normalized dose units (dose is normalized to Rx in NPZ)
    noise_std_normalized = args.noise_std / args.rx_dose_gy

    loss_sums: Dict[str, float] = {
        "base": 0.0, "gradient": 0.0, "structure": 0.0,
        "asymmetric": 0.0, "dvh": 0.0,
    }
    count = 0

    for f in selected_files:
        sample = load_sample(f)
        target = sample["dose"]

        # Simulate untrained model output: GT + Gaussian noise
        pred = target + torch.randn_like(target) * noise_std_normalized

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = pred.to(device)
        target = target.to(device)
        masks = sample["masks"].to(device)
        ptv_mask = sample["ptv_mask"].to(device)
        constraints = sample["constraints"].to(device)

        with torch.no_grad():
            component_losses = {
                "base":       base_loss(pred, target),
                "gradient":   gradient_loss(pred, target),
                "structure":  structure_weighted_loss(pred, target, masks),
                "asymmetric": asymmetric_ptv_loss(pred, target, ptv_mask),
                "dvh":        dvh_aware_loss(pred, constraints),
            }

        for name, val in component_losses.items():
            loss_sums[name] += val.item()
        count += 1

        print(f"  [{count}/{len(selected_files)}] {f.name}")

    # Average losses
    avg_losses = {name: loss_sums[name] / count for name in loss_sums}

    # Recommended initial_log_sigma = log(sqrt(avg))
    # This makes the weighted term start at ~0.5 + log(sqrt(avg))
    recommendations = {
        name: float(np.log(np.sqrt(avg))) if avg > 0 else 0.0
        for name, avg in avg_losses.items()
    }

    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {count}, Noise std: {args.noise_std} Gy "
          f"({noise_std_normalized:.4f} normalized)")

    print("\nAverage raw loss values:")
    for name, avg in avg_losses.items():
        print(f"  {name:12s} : {avg:.6f}")

    print("\nRecommended initial_log_sigma for UncertaintyWeightedLoss:")
    for name, sigma in recommendations.items():
        print(f"  {name:12s} : {sigma:.4f}")

    # Save to JSON
    out_file = data_dir / "loss_normalization_calib.json"
    result = {
        "avg_raw_losses": avg_losses,
        "recommended_initial_log_sigma": recommendations,
        "samples_used": count,
        "noise_std_gy": args.noise_std,
        "noise_std_normalized": noise_std_normalized,
        "rx_dose_gy": args.rx_dose_gy,
    }
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_file}")

    print("\nUsage in training script:")
    print("  from uncertainty_loss import UncertaintyWeightedLoss")
    print("  self.uncertainty_loss = UncertaintyWeightedLoss(")
    print(f"      loss_names={list(recommendations.keys())},")
    print("      initial_log_sigmas={")
    for name, sigma in recommendations.items():
        print(f"          '{name}': {sigma:.4f},")
    print("      }")
    print("  )")


if __name__ == "__main__":
    main()
