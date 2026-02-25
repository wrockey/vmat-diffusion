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

Attribution: Initial implementation by Grok (external review, 2026-02-17),
             reviewed and integrated by Claude. Stubs replaced with real
             loss imports from train_baseline_unet.py (2026-02-25).
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict

# Import real loss implementations from training script
from train_baseline_unet import (
    GradientLoss3D,
    DVHAwareLoss,
    StructureWeightedLoss,
    AsymmetricPTVLoss,
)


# =============================================================================
# LOSS FUNCTIONS — real implementations imported from train_baseline_unet.py
# Instantiated inside main() to use CLI rx_dose_gy argument.
# =============================================================================


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sample(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Load a single .npz file and build tensors matching training format.

    Returns tensors with batch dimension (B=1) for compatibility with loss functions.
    The condition tensor is (1, 9, H, W, D) = CT + 8 SDF channels.
    """
    data = np.load(npz_path, allow_pickle=True)

    ct = torch.from_numpy(data['ct']).float()                        # (H, W, D)
    dose = torch.from_numpy(data['dose']).float()                    # (H, W, D)
    masks_sdf = torch.from_numpy(data['masks_sdf']).float()          # (8, H, W, D)

    # Build condition tensor: CT (1 ch) + SDFs (8 ch) = 9 channels
    condition = torch.cat([ct.unsqueeze(0), masks_sdf], dim=0)       # (9, H, W, D)

    # Add batch dimension
    dose = dose.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W, D)
    condition = condition.unsqueeze(0)              # (1, 9, H, W, D)

    return {
        "dose": dose,
        "condition": condition,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate loss modules with CLI rx_dose_gy (Fix: was hard-coded at module level)
    gradient_loss = GradientLoss3D().to(device)
    dvh_loss = DVHAwareLoss(rx_dose_gy=args.rx_dose_gy).to(device)
    structure_weighted_loss = StructureWeightedLoss().to(device)
    asymmetric_ptv_loss = AsymmetricPTVLoss(rx_dose_gy=args.rx_dose_gy).to(device)

    for f in selected_files:
        sample = load_sample(f)
        target = sample["dose"].to(device)           # (1, 1, H, W, D)
        condition = sample["condition"].to(device)    # (1, 9, H, W, D)

        # Simulate untrained model output: GT + Gaussian noise
        pred = target + torch.randn_like(target) * noise_std_normalized

        # Extract 128^3 patch centered on PTV70 centroid (SDF channel 1, where SDF < 0 = inside)
        # This ensures DVH and asymmetric losses see actual PTV anatomy, not empty space.
        H, W, D = target.shape[2:]
        ph, pw, pd = min(128, H), min(128, W), min(128, D)

        ptv_sdf = condition[0, 1]  # PTV70 SDF: (H, W, D)
        ptv_mask = ptv_sdf < 0
        if ptv_mask.any():
            coords = torch.nonzero(ptv_mask)  # (N, 3)
            centroid = coords.float().mean(dim=0).long()
            sy = max(0, min(centroid[0].item() - ph // 2, H - ph))
            sx = max(0, min(centroid[1].item() - pw // 2, W - pw))
            sz = max(0, min(centroid[2].item() - pd // 2, D - pd))
        else:
            # Fallback to center crop if PTV not found
            sy = (H - ph) // 2
            sx = (W - pw) // 2
            sz = (D - pd) // 2
            print(f"    [WARN] No PTV70 voxels found in {f.name}, using center crop")

        pred_p = pred[:, :, sy:sy+ph, sx:sx+pw, sz:sz+pd]
        target_p = target[:, :, sy:sy+ph, sx:sx+pw, sz:sz+pd]
        condition_p = condition[:, :, sy:sy+ph, sx:sx+pw, sz:sz+pd]

        # Sanity check: count PTV voxels in patch
        ptv_in_patch = (condition_p[0, 1] < 0).sum().item()
        if ptv_in_patch < 100:
            print(f"    [WARN] Only {ptv_in_patch} PTV voxels in patch for {f.name}")

        with torch.no_grad():
            # Base loss: MSE
            base = F.mse_loss(pred_p, target_p)

            # Gradient loss
            grad = gradient_loss(pred_p, target_p)

            # Structure-weighted loss (returns loss, metrics_dict)
            struct, _ = structure_weighted_loss(pred_p, target_p, condition_p)

            # Asymmetric PTV loss (returns loss, metrics_dict)
            asym, _ = asymmetric_ptv_loss(pred_p, target_p, condition_p)

            # DVH-aware loss (returns loss, metrics_dict)
            dvh, _ = dvh_loss(pred_p, target_p, condition_p)

            component_losses = {
                "base": base,
                "gradient": grad,
                "structure": struct,
                "asymmetric": asym,
                "dvh": dvh,
            }

        for name, val in component_losses.items():
            loss_sums[name] += val.item()
        count += 1

        print(f"  [{count}/{len(selected_files)}] {f.name} (PTV voxels: {ptv_in_patch})")

    # Average losses
    avg_losses = {name: loss_sums[name] / count for name in loss_sums}

    # Recommended initial_log_sigma = log(sqrt(avg))
    # This makes the data-fit term L/(2*sigma^2) start at ~0.5 for all components.
    # The regularization term log(sigma) will vary across components.
    # Clamp avg to [1e-4, 1e4] to prevent extreme sigma values that could
    # cause gradient explosion (tiny avg → huge weight) or effective silencing.
    MIN_AVG_LOSS = 1e-4
    MAX_AVG_LOSS = 1e4
    recommendations = {
        name: float(np.log(np.sqrt(np.clip(avg, MIN_AVG_LOSS, MAX_AVG_LOSS)))) if avg > 0 else 0.0
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
