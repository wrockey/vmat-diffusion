"""
Inference script for trained VMAT Dose DDPM model.

Performs full-volume dose prediction using sliding window with overlap averaging.
Computes evaluation metrics: MAE, gamma pass rate, DVH comparison.
Checks predictions against QUANTEC-based clinical constraints.

Usage:
    # Single case
    python inference_dose_ddpm.py \
        --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
        --input ./processed_npz/case_0001.npz \
        --output ./predictions/case_0001_pred.npz
    
    # Batch evaluation
    python inference_dose_ddpm.py \
        --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
        --input_dir ./processed_npz \
        --output_dir ./predictions \
        --compute_metrics

Output includes:
    - dose_metrics: MAE, RMSE, max error
    - gamma: 3%/3mm pass rate (full volume)
    - dvh_metrics: D95, D50, V70, etc. for each structure
    - clinical_constraints: pass/fail against QUANTEC limits
"""

import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import model from training script
from train_dose_ddpm_v2 import DoseDDPM, VMATDoseFullVolumeDataset

# Import centralized evaluation framework
from eval_core import (
    STRUCTURE_CHANNELS,
    DEFAULT_SPACING_MM,
    get_spacing_from_metadata,
    denormalize_dose,
)
from eval_metrics import (
    compute_dose_metrics,
    compute_gamma,
    compute_dvh_metrics,
    evaluate_case,
    HAS_PYMEDPHYS,
)
from eval_clinical import check_clinical_constraints, CLINICAL_CONSTRAINTS
from eval_statistics import NumpyEncoder


# NOTE: CLINICAL_CONSTRAINTS, check_clinical_constraints, compute_dose_metrics,
# compute_gamma, compute_dvh_metrics are now imported from the centralized
# evaluation framework (eval_clinical, eval_metrics).


# =============================================================================
# Inference Functions
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> DoseDDPM:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    model = DoseDDPM.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    print(f"  Model loaded: {type(model.model).__name__}")
    print(f"  Timesteps: {model.hparams.timesteps}")
    print(f"  Rx dose: {model.hparams.rx_dose_gy} Gy")
    
    return model


def predict_single_case(
    model: DoseDDPM,
    npz_path: str,
    patch_size: int = 128,
    overlap: int = 64,
    ddim_steps: int = 50,
    device: str = 'cuda',
) -> Tuple[np.ndarray, Dict]:
    """
    Run inference on a single case.
    
    Returns:
        Tuple of (predicted_dose, metadata_dict)
    """
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    ct = torch.from_numpy(data['ct'].astype(np.float32)).unsqueeze(0)  # (1, H, W, D)
    masks_sdf = torch.from_numpy(data['masks_sdf'].astype(np.float32))  # (8, H, W, D)
    constraints = torch.from_numpy(data['constraints'].astype(np.float32))  # (13,)
    
    # Build condition tensor
    condition = torch.cat([ct, masks_sdf], dim=0).unsqueeze(0).to(device)  # (1, 9, H, W, D)
    constraints = constraints.unsqueeze(0).to(device)  # (1, 13)
    
    # Predict
    pred_dose = model.predict_full_volume(
        condition=condition,
        constraints=constraints,
        patch_size=patch_size,
        overlap=overlap,
        ddim_steps=ddim_steps,
        verbose=True,
    )
    
    # Convert to numpy
    pred_dose_np = pred_dose[0, 0].cpu().numpy()
    
    # Metadata
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    
    return pred_dose_np, metadata


def evaluate_single_case(
    pred: np.ndarray,
    npz_path: str,
    rx_dose_gy: float = 70.0,
    compute_gamma_metric: bool = True,
    gamma_subsample: int = 2,
) -> Dict:
    """
    Evaluate prediction against ground truth using centralized framework.

    Returns:
        Dict with all metrics (backward-compatible format)
    """
    # Load ground truth
    data = np.load(npz_path, allow_pickle=True)
    target = data['dose']
    masks = data['masks']

    # Read spacing from metadata
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    print("  Computing evaluation metrics...")
    result = evaluate_case(
        pred_normalized=pred,
        target_normalized=target,
        masks=masks,
        spacing_mm=spacing,
        case_id=Path(npz_path).stem,
        rx_dose_gy=rx_dose_gy,
        compute_gamma_metric=compute_gamma_metric,
        gamma_subsample=gamma_subsample,
    )

    # Convert to backward-compatible dict format
    result_dict = result.to_dict()
    result_dict['timestamp'] = datetime.now().isoformat()

    # Print summary
    cc = result_dict.get('clinical_constraints', {})
    if cc.get('violations'):
        print(f"  {len(cc['violations'])} constraint violation(s)")
    else:
        print("  All clinical constraints passed")

    return result_dict


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VMAT Dose DDPM Inference")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Input/Output
    parser.add_argument('--input', type=str, default=None,
                       help='Path to single .npz file')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Path to directory of .npz files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single prediction')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for batch predictions')
    
    # Inference settings
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='DDIM sampling steps (more = slower but better)')
    
    # Evaluation
    parser.add_argument('--compute_metrics', action='store_true',
                       help='Compute evaluation metrics')
    parser.add_argument('--gamma_subsample', type=int, default=2,
                       help='Subsample factor for gamma computation')
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Validate input
    if args.input is None and args.input_dir is None:
        raise ValueError("Must specify --input or --input_dir")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    rx_dose = args.rx_dose_gy
    
    # Get input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_files = sorted(Path(args.input_dir).glob("*.npz"))
    
    print(f"\nProcessing {len(input_files)} case(s)...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each case
    all_results = []
    
    for npz_path in tqdm(input_files, desc="Inference"):
        case_id = npz_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {case_id}")
        print('='*60)
        
        # Predict
        pred_dose, metadata = predict_single_case(
            model=model,
            npz_path=str(npz_path),
            patch_size=args.patch_size,
            overlap=args.overlap,
            ddim_steps=args.ddim_steps,
            device=args.device,
        )
        
        # Save prediction
        if args.input and args.output:
            output_path = Path(args.output)
        else:
            output_path = output_dir / f"{case_id}_pred.npz"
        
        # Load original data to include in output
        original_data = np.load(npz_path, allow_pickle=True)
        
        np.savez_compressed(
            output_path,
            pred_dose=pred_dose,
            target_dose=original_data['dose'],
            ct=original_data['ct'],
            masks=original_data['masks'],
            masks_sdf=original_data['masks_sdf'],
            constraints=original_data['constraints'],
            metadata=metadata,
            inference_settings={
                'checkpoint': args.checkpoint,
                'patch_size': args.patch_size,
                'overlap': args.overlap,
                'ddim_steps': args.ddim_steps,
                'rx_dose_gy': rx_dose,
                'timestamp': datetime.now().isoformat(),
            }
        )
        print(f"  Saved: {output_path}")
        
        # Evaluate
        if args.compute_metrics:
            results = evaluate_single_case(
                pred=pred_dose,
                npz_path=str(npz_path),
                rx_dose_gy=rx_dose,
                compute_gamma_metric=HAS_PYMEDPHYS,
                gamma_subsample=args.gamma_subsample,
            )
            all_results.append(results)
            
            # Print summary
            dm = results['dose_metrics']
            print(f"\n  Results for {case_id}:")
            print(f"    MAE: {dm['mae_gy']:.2f} Gy")
            print(f"    RMSE: {dm['rmse_gy']:.2f} Gy")
            print(f"    Max error: {dm['max_error_gy']:.2f} Gy")
            
            gamma_global = results.get('gamma', {}).get('global_3mm3pct', {})
            if gamma_global.get('gamma_pass_rate') is not None:
                print(f"    Gamma (3%/3mm): {gamma_global['gamma_pass_rate']:.1f}%")
            
            # PTV70 DVH
            if 'PTV70' in results['dvh_metrics'] and results['dvh_metrics']['PTV70']['exists']:
                ptv = results['dvh_metrics']['PTV70']
                print(f"    PTV70 D95 error: {ptv['D95_error']:.2f} Gy")
            
            # Clinical constraints summary for this case
            if 'clinical_constraints' in results:
                cc = results['clinical_constraints']
                if cc['overall_pass']:
                    print(f"    Clinical constraints: ALL PASSED")
                else:
                    print(f"    Clinical constraints: {len(cc['violations'])} violation(s)")
                    for v in cc['violations'][:3]:  # Show up to 3
                        print(f"      - {v['structure']} {v['metric']}: {v['predicted']} (limit: {v['limit']})")
    
    # Save aggregate results
    if args.compute_metrics and all_results:
        results_path = output_dir / 'evaluation_results.json'
        
        # Compute aggregate metrics
        mae_values = [r['dose_metrics']['mae_gy'] for r in all_results]
        gamma_values = [
            r['gamma']['global_3mm3pct']['gamma_pass_rate']
            for r in all_results
            if r.get('gamma', {}).get('global_3mm3pct', {}).get('gamma_pass_rate') is not None
        ]
        
        summary = {
            'n_cases': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': {
                'mae_gy_mean': float(np.mean(mae_values)),
                'mae_gy_std': float(np.std(mae_values)),
                'mae_gy_median': float(np.median(mae_values)),
                'mae_gy_max': float(np.max(mae_values)),
            },
            'per_case_results': all_results,
        }
        
        if gamma_values:
            summary['aggregate_metrics']['gamma_pass_rate_mean'] = float(np.mean(gamma_values))
            summary['aggregate_metrics']['gamma_pass_rate_std'] = float(np.std(gamma_values))
            summary['aggregate_metrics']['gamma_pass_rate_min'] = float(np.min(gamma_values))
        
        # Aggregate clinical constraint results
        cases_all_pass = sum(1 for r in all_results 
                           if r.get('clinical_constraints', {}).get('overall_pass', False))
        total_violations = sum(len(r.get('clinical_constraints', {}).get('violations', [])) 
                              for r in all_results)
        
        # Count violations by structure/metric
        violation_counts = {}
        for r in all_results:
            for v in r.get('clinical_constraints', {}).get('violations', []):
                key = f"{v['structure']}_{v['metric']}"
                violation_counts[key] = violation_counts.get(key, 0) + 1
        
        summary['clinical_constraints'] = {
            'cases_all_passed': cases_all_pass,
            'cases_with_violations': len(all_results) - cases_all_pass,
            'total_violations': total_violations,
            'violation_counts': violation_counts,
            'most_common_violations': sorted(violation_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:5],
        }
        
        # Check against goals
        goal_mae = 2.0  # Gy
        goal_gamma = 95.0  # %
        
        summary['goal_assessment'] = {
            'mae_goal_gy': goal_mae,
            'mae_goal_met': summary['aggregate_metrics']['mae_gy_mean'] < goal_mae,
            'gamma_goal_pct': goal_gamma,
            'gamma_goal_met': (summary['aggregate_metrics'].get('gamma_pass_rate_mean', 0) >= goal_gamma 
                              if gamma_values else None),
            'clinical_all_pass_pct': 100 * cases_all_pass / len(all_results) if all_results else 0,
        }
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print('='*60)
        print(f"Cases evaluated: {len(all_results)}")
        print(f"MAE: {summary['aggregate_metrics']['mae_gy_mean']:.2f} ± {summary['aggregate_metrics']['mae_gy_std']:.2f} Gy")
        print(f"     (Goal: < {goal_mae} Gy) {'✓ MET' if summary['goal_assessment']['mae_goal_met'] else '✗ NOT MET'}")
        
        if gamma_values:
            print(f"Gamma (3%/3mm): {summary['aggregate_metrics']['gamma_pass_rate_mean']:.1f} ± {summary['aggregate_metrics']['gamma_pass_rate_std']:.1f}%")
            print(f"     (Goal: >= {goal_gamma}%) {'✓ MET' if summary['goal_assessment']['gamma_goal_met'] else '✗ NOT MET'}")
        
        # Clinical constraints summary
        print(f"\n--- Clinical Constraints (QUANTEC-based) ---")
        print(f"Cases passing ALL constraints: {cases_all_pass}/{len(all_results)} ({100*cases_all_pass/len(all_results):.0f}%)")
        print(f"Total constraint violations: {total_violations}")
        
        if violation_counts:
            print("Most common violations:")
            for violation, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {violation}: {count} case(s)")
        
        print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
