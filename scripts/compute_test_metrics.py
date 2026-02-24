"""
Compute evaluation metrics for saved predictions.

Uses the centralized evaluation framework (eval_core, eval_metrics, eval_clinical).
Replaces the previous standalone implementations with framework imports.

Usage:
    python scripts/compute_test_metrics.py \
        --pred_dir predictions/dvh_aware_loss_test \
        --data_dir test_cases \
        --output_file predictions/dvh_aware_loss_test/evaluation_results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from eval_core import (
    STRUCTURE_CHANNELS,
    get_spacing_from_metadata,
    denormalize_dose,
    FRAMEWORK_VERSION,
)
from eval_metrics import (
    compute_dose_metrics,
    compute_gamma,
    compute_dvh_metrics,
    evaluate_case,
)
from eval_clinical import check_clinical_constraints
from eval_statistics import NumpyEncoder


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predictions')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with ground truth data')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    parser.add_argument('--gamma_subsample', type=int, default=4)
    parser.add_argument('--skip_gamma', action='store_true', help='Skip gamma computation')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*_pred.npz"))
    print(f"Found {len(pred_files)} prediction files")

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

        # Run full evaluation
        result = evaluate_case(
            pred_normalized=pred,
            target_normalized=target,
            masks=masks,
            spacing_mm=spacing,
            case_id=case_id,
            rx_dose_gy=args.rx_dose_gy,
            compute_gamma_metric=not args.skip_gamma,
            gamma_subsample=args.gamma_subsample,
        )
        result_dict = result.to_dict()
        result_dict['timestamp'] = datetime.now().isoformat()
        all_results.append(result_dict)

        # Print summary
        dm = result_dict['dose_metrics']
        print(f"  MAE: {dm['mae_gy']:.2f} Gy")
        gamma_global = result_dict.get('gamma', {}).get('global_3mm3pct', {})
        if gamma_global.get('gamma_pass_rate') is not None:
            print(f"  Gamma: {gamma_global['gamma_pass_rate']:.1f}%")
        if result_dict['clinical_constraints'].get('violations'):
            n_v = len(result_dict['clinical_constraints']['violations'])
            print(f"  Clinical constraints: {n_v} violation(s)")
        else:
            print(f"  Clinical constraints: all passed")

    # Aggregate metrics
    mae_values = [r['dose_metrics']['mae_gy'] for r in all_results]
    gamma_values = [
        r['gamma']['global_3mm3pct']['gamma_pass_rate']
        for r in all_results
        if r.get('gamma', {}).get('global_3mm3pct', {}).get('gamma_pass_rate') is not None
    ]

    summary = {
        'framework_version': FRAMEWORK_VERSION,
        'n_cases': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'gamma_subsample': args.gamma_subsample,
        'aggregate_metrics': {
            'mae_gy_mean': float(np.mean(mae_values)) if mae_values else None,
            'mae_gy_std': float(np.std(mae_values)) if mae_values else None,
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
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print('='*60)
    print(f"Cases evaluated: {len(all_results)}")
    if mae_values:
        print(f"MAE: {summary['aggregate_metrics']['mae_gy_mean']:.2f} +/- {summary['aggregate_metrics']['mae_gy_std']:.2f} Gy")
    if gamma_values:
        print(f"Gamma (3%/3mm): {summary['aggregate_metrics']['gamma_pass_rate_mean']:.1f} +/- {summary['aggregate_metrics']['gamma_pass_rate_std']:.1f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
