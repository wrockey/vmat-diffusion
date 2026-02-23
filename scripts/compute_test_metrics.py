"""
Compute evaluation metrics for saved predictions.

This script computes MAE, Gamma, and DVH metrics for predictions.
Run after inference to compute metrics separately.

Usage:
    python scripts/compute_test_metrics.py \
        --pred_dir predictions/dvh_aware_loss_test \
        --data_dir test_cases \
        --output_file predictions/dvh_aware_loss_test/evaluation_results.json
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

# Try to import pymedphys for gamma
try:
    from pymedphys import gamma as pymedphys_gamma
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False
    print("Warning: pymedphys not installed, gamma will be skipped")

DEFAULT_SPACING_MM = (1.0, 1.0, 2.0)


def get_spacing_from_metadata(metadata):
    """
    Extract voxel spacing from NPZ metadata with backwards-compatible fallback.

    Fallback chain:
        1. voxel_spacing_mm (v2.3+ native spacing)
        2. target_spacing_mm (v2.2 resampled spacing)
        3. DEFAULT_SPACING_MM
    """
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()

    if 'voxel_spacing_mm' in metadata:
        spacing = metadata['voxel_spacing_mm']
        return tuple(float(s) for s in spacing)

    if 'target_spacing_mm' in metadata:
        spacing = metadata['target_spacing_mm']
        return tuple(float(s) for s in spacing)

    return DEFAULT_SPACING_MM


def compute_dose_metrics(pred: np.ndarray, target: np.ndarray, rx_dose_gy: float = 70.0) -> Dict:
    """Compute dose comparison metrics."""
    pred_gy = pred * rx_dose_gy
    target_gy = target * rx_dose_gy

    diff = pred_gy - target_gy

    # Basic metrics
    mae_gy = float(np.abs(diff).mean())
    rmse_gy = float(np.sqrt((diff ** 2).mean()))
    max_error_gy = float(np.abs(diff).max())

    # Threshold-based metrics
    target_max = target_gy.max()
    mask_10pct = target_gy > (0.1 * target_max)
    mask_50pct = target_gy > (0.5 * target_max)

    metrics = {
        'mae_gy': mae_gy,
        'rmse_gy': rmse_gy,
        'max_error_gy': max_error_gy,
        'pred_max_gy': float(pred_gy.max()),
        'target_max_gy': float(target_max),
    }

    if mask_10pct.any():
        metrics['mae_gy_above_10pct'] = float(np.abs(diff[mask_10pct]).mean())
    if mask_50pct.any():
        metrics['mae_gy_above_50pct'] = float(np.abs(diff[mask_50pct]).mean())

    return metrics


def compute_gamma(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    spacing_mm: tuple = DEFAULT_SPACING_MM,
    subsample: int = 4,
    dose_threshold_pct: float = 3.0,
    distance_mm: float = 3.0,
) -> Dict:
    """Compute gamma pass rate with subsampling for speed."""
    if not HAS_PYMEDPHYS:
        return {'gamma_pass_rate': None, 'error': 'pymedphys not installed'}

    # Ensure float64 for pymedphys
    pred_gy = pred_gy.astype(np.float64)
    target_gy = target_gy.astype(np.float64)

    # Subsample for speed
    pred_sub = pred_gy[::subsample, ::subsample, ::subsample]
    target_sub = target_gy[::subsample, ::subsample, ::subsample]
    spacing_sub = tuple(s * subsample for s in spacing_mm)

    # Create coordinate axes
    axes = tuple(
        np.arange(s) * sp for s, sp in zip(pred_sub.shape, spacing_sub)
    )

    print(f"    Computing gamma on {pred_sub.shape} volume (subsample={subsample})...")

    try:
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_sub,
            axes_evaluation=axes,
            dose_evaluation=pred_sub,
            dose_percent_threshold=dose_threshold_pct,
            distance_mm_threshold=distance_mm,
            lower_percent_dose_cutoff=10.0,
        )

        valid = np.isfinite(gamma_map)
        if not valid.any():
            return {'gamma_pass_rate': 0.0, 'voxels_evaluated': 0}

        return {
            'gamma_pass_rate': float(np.mean(gamma_map[valid] <= 1.0) * 100),
            'gamma_mean': float(np.mean(gamma_map[valid])),
            'gamma_max': float(np.max(gamma_map[valid])),
            'gamma_median': float(np.median(gamma_map[valid])),
            'voxels_evaluated': int(valid.sum()),
        }
    except Exception as e:
        return {'gamma_pass_rate': None, 'error': str(e)}


def compute_dvh_metrics(pred: np.ndarray, target: np.ndarray, masks: np.ndarray,
                        structure_names: Dict[int, str], rx_dose_gy: float = 70.0,
                        spacing_mm: tuple = None) -> Dict:
    """Compute DVH metrics for each structure."""
    if spacing_mm is None:
        spacing_mm = DEFAULT_SPACING_MM
    pred_gy = pred * rx_dose_gy
    target_gy = target * rx_dose_gy

    voxel_vol_cc = float(np.prod(spacing_mm)) / 1000.0  # mm^3 to cc

    results = {}

    for idx, name in structure_names.items():
        if idx >= masks.shape[0]:
            continue

        mask = masks[idx].astype(bool)
        if not mask.any():
            results[name] = {'exists': False}
            continue

        pred_struct = pred_gy[mask]
        target_struct = target_gy[mask]

        results[name] = {
            'exists': True,
            'volume_cc': float(mask.sum() * voxel_vol_cc),
            'mae_gy': float(np.abs(pred_struct - target_struct).mean()),
            'pred_mean_gy': float(pred_struct.mean()),
            'target_mean_gy': float(target_struct.mean()),
            'pred_D95': float(np.percentile(pred_struct, 5)),  # D95 = dose to 95% of volume
            'target_D95': float(np.percentile(target_struct, 5)),
            'D95_error': float(np.percentile(pred_struct, 5) - np.percentile(target_struct, 5)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predictions')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with ground truth data')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    parser.add_argument('--gamma_subsample', type=int, default=4)

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*_pred.npz"))
    print(f"Found {len(pred_files)} prediction files")

    structure_names = {
        0: 'PTV70', 1: 'PTV56', 2: 'Prostate', 3: 'Rectum',
        4: 'Bladder', 5: 'Femur_L', 6: 'Femur_R', 7: 'Bowel'
    }

    all_results = []

    for pred_path in pred_files:
        # Find corresponding ground truth
        case_id = pred_path.stem.replace('_pred', '')
        data_path = data_dir / f"{case_id}.npz"

        if not data_path.exists():
            print(f"Warning: No ground truth found for {case_id}")
            continue

        print(f"\nProcessing: {case_id}")

        # Load data
        pred_data = np.load(pred_path)
        gt_data = np.load(data_path, allow_pickle=True)

        pred = pred_data['dose']
        target = gt_data['dose']
        masks = gt_data['masks']

        # Read spacing from metadata
        metadata = gt_data['metadata'].item() if 'metadata' in gt_data.files else {}
        spacing = get_spacing_from_metadata(metadata)

        results = {
            'case_id': case_id,
            'timestamp': datetime.now().isoformat(),
            'spacing_mm': spacing,
        }

        # Dose metrics
        print("  Computing dose metrics...")
        results['dose_metrics'] = compute_dose_metrics(pred, target, args.rx_dose_gy)

        # Gamma
        print("  Computing gamma...")
        pred_gy = pred * args.rx_dose_gy
        target_gy = target * args.rx_dose_gy
        results['gamma'] = compute_gamma(pred_gy, target_gy, spacing_mm=spacing, subsample=args.gamma_subsample)

        # DVH metrics
        print("  Computing DVH metrics...")
        results['dvh_metrics'] = compute_dvh_metrics(pred, target, masks, structure_names, args.rx_dose_gy, spacing_mm=spacing)

        all_results.append(results)

        # Print summary
        print(f"  MAE: {results['dose_metrics']['mae_gy']:.2f} Gy")
        if results['gamma'].get('gamma_pass_rate') is not None:
            print(f"  Gamma: {results['gamma']['gamma_pass_rate']:.1f}%")

    # Aggregate metrics
    mae_values = [r['dose_metrics']['mae_gy'] for r in all_results]
    gamma_values = [r['gamma']['gamma_pass_rate'] for r in all_results
                   if r['gamma'].get('gamma_pass_rate') is not None]

    summary = {
        'model': 'dvh_aware_loss',
        'n_cases': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'gamma_subsample': args.gamma_subsample,
        'aggregate_metrics': {
            'mae_gy_mean': float(np.mean(mae_values)),
            'mae_gy_std': float(np.std(mae_values)),
        },
        'per_case_results': all_results,
    }

    if gamma_values:
        summary['aggregate_metrics']['gamma_pass_rate_mean'] = float(np.mean(gamma_values))
        summary['aggregate_metrics']['gamma_pass_rate_std'] = float(np.std(gamma_values))

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print('='*60)
    print(f"Cases evaluated: {len(all_results)}")
    print(f"MAE: {summary['aggregate_metrics']['mae_gy_mean']:.2f} ± {summary['aggregate_metrics']['mae_gy_std']:.2f} Gy")
    if gamma_values:
        print(f"Gamma (3%/3mm): {summary['aggregate_metrics']['gamma_pass_rate_mean']:.1f} ± {summary['aggregate_metrics']['gamma_pass_rate_std']:.1f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
