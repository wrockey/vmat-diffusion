#!/usr/bin/env python3
"""
Phase 1 DDPM Optimization Experiments: Inference-Only Improvements

Experiment 1.1: Sampling Steps Ablation
- Tests: [50, 100, 250, 500, 1000] DDIM steps
- Measures: MAE, inference time, gamma pass rate

Experiment 1.2: Ensemble Averaging
- Tests: [1, 3, 5, 10] samples averaged
- Uses optimal steps from Exp 1.1
- Measures: MAE, std across samples

Usage:
    python scripts/run_phase1_experiments.py \
        --checkpoint runs/vmat_dose_ddpm/checkpoints/best-epoch=004-val/mae_gy=11.48.ckpt \
        --data_dir /mnt/i/processed_npz \
        --output_dir experiments/phase1_sampling
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from train_dose_ddpm_v2 import DoseDDPM, VMATDoseFullVolumeDataset

from eval_core import DEFAULT_SPACING_MM, get_spacing_from_metadata
from eval_metrics import compute_gamma, HAS_PYMEDPHYS


def load_model(checkpoint_path: str, device: str = 'cuda') -> DoseDDPM:
    """Load trained DDPM model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    model = DoseDDPM.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    print(f"  Timesteps: {model.hparams.timesteps}")
    print(f"  Schedule: {model.hparams.schedule}")
    print(f"  Rx dose: {model.hparams.rx_dose_gy} Gy")

    return model


def predict_single_case(
    model: DoseDDPM,
    npz_path: str,
    ddim_steps: int = 50,
    device: str = 'cuda',
    patch_size: int = 128,
    overlap: int = 32,
) -> Tuple[np.ndarray, float]:
    """
    Run inference on a single case and return prediction + inference time.
    """
    # Load data
    data = np.load(npz_path, allow_pickle=True)

    ct = torch.from_numpy(data['ct'].astype(np.float32)).unsqueeze(0)  # (1, H, W, D)
    masks_sdf = torch.from_numpy(data['masks_sdf'].astype(np.float32))  # (8, H, W, D)
    constraints = torch.from_numpy(data['constraints'].astype(np.float32))  # (13,)

    # Build condition tensor
    condition = torch.cat([ct, masks_sdf], dim=0).unsqueeze(0).to(device)  # (1, 9, H, W, D)
    constraints = constraints.unsqueeze(0).to(device)  # (1, 13)

    # Time the prediction
    torch.cuda.synchronize()
    start_time = time.time()

    pred_dose = model.predict_full_volume(
        condition=condition,
        constraints=constraints,
        patch_size=patch_size,
        overlap=overlap,
        ddim_steps=ddim_steps,
        verbose=False,
    )

    torch.cuda.synchronize()
    inference_time = time.time() - start_time

    # Convert to numpy
    pred_dose_np = pred_dose[0, 0].cpu().numpy()

    return pred_dose_np, inference_time


def evaluate_prediction(
    pred: np.ndarray,
    npz_path: str,
    rx_dose_gy: float = 70.0,
) -> Dict[str, float]:
    """Evaluate prediction against ground truth."""
    data = np.load(npz_path, allow_pickle=True)
    target = data['dose']

    # Read spacing from metadata
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    # Convert to Gy
    pred_gy = pred * rx_dose_gy
    target_gy = target * rx_dose_gy

    # Compute metrics
    metrics = {
        'mae_gy': float(np.mean(np.abs(pred_gy - target_gy))),
        'rmse_gy': float(np.sqrt(np.mean((pred_gy - target_gy)**2))),
        'max_error_gy': float(np.max(np.abs(pred_gy - target_gy))),
        'pred_mean_gy': float(pred_gy.mean()),
        'target_mean_gy': float(target_gy.mean()),
        'pred_max_gy': float(pred_gy.max()),
        'target_max_gy': float(target_gy.max()),
        'spacing_mm': spacing,
    }

    # Gamma pass rate
    gamma_result = compute_gamma(pred_gy, target_gy, spacing_mm=spacing, subsample=4)
    metrics.update({f'gamma_{k}': v for k, v in gamma_result.items()})

    return metrics


def run_sampling_ablation(
    model: DoseDDPM,
    val_files: List[str],
    steps_to_test: List[int],
    device: str = 'cuda',
    rx_dose_gy: float = 70.0,
) -> Dict:
    """
    Experiment 1.1: Sampling Steps Ablation

    Tests different numbers of DDIM sampling steps to find optimal tradeoff
    between quality and speed.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.1: SAMPLING STEPS ABLATION")
    print("="*60)

    results = {
        'experiment': 'sampling_steps_ablation',
        'timestamp': datetime.now().isoformat(),
        'steps_tested': steps_to_test,
        'val_cases': [Path(f).stem for f in val_files],
        'per_step_results': {},
        'summary': {},
    }

    for steps in steps_to_test:
        print(f"\n--- Testing {steps} DDIM steps ---")

        step_results = {
            'cases': [],
            'mae_values': [],
            'inference_times': [],
            'gamma_values': [],
        }

        for npz_path in val_files:
            case_id = Path(npz_path).stem
            print(f"  Processing {case_id}...", end=" ", flush=True)

            # Predict
            pred, inf_time = predict_single_case(
                model, str(npz_path), ddim_steps=steps, device=device
            )

            # Evaluate
            metrics = evaluate_prediction(pred, str(npz_path), rx_dose_gy)

            step_results['cases'].append({
                'case_id': case_id,
                'mae_gy': metrics['mae_gy'],
                'rmse_gy': metrics['rmse_gy'],
                'inference_time_s': inf_time,
                'gamma_pass_rate': metrics.get('gamma_gamma_pass_rate'),
            })
            step_results['mae_values'].append(metrics['mae_gy'])
            step_results['inference_times'].append(inf_time)
            if metrics.get('gamma_gamma_pass_rate') is not None:
                step_results['gamma_values'].append(metrics['gamma_gamma_pass_rate'])

            print(f"MAE: {metrics['mae_gy']:.2f} Gy, Time: {inf_time:.1f}s")

        # Aggregate for this step count
        step_results['aggregate'] = {
            'mean_mae_gy': float(np.mean(step_results['mae_values'])),
            'std_mae_gy': float(np.std(step_results['mae_values'])),
            'mean_inference_time_s': float(np.mean(step_results['inference_times'])),
            'mean_gamma_pass_rate': float(np.mean(step_results['gamma_values'])) if step_results['gamma_values'] else None,
        }

        results['per_step_results'][str(steps)] = step_results

        print(f"  Summary: MAE = {step_results['aggregate']['mean_mae_gy']:.2f} ± "
              f"{step_results['aggregate']['std_mae_gy']:.2f} Gy, "
              f"Time = {step_results['aggregate']['mean_inference_time_s']:.1f}s")

    # Find optimal steps
    best_steps = min(
        results['per_step_results'].keys(),
        key=lambda k: results['per_step_results'][k]['aggregate']['mean_mae_gy']
    )

    results['summary'] = {
        'optimal_steps': int(best_steps),
        'optimal_mae_gy': results['per_step_results'][best_steps]['aggregate']['mean_mae_gy'],
        'optimal_inference_time_s': results['per_step_results'][best_steps]['aggregate']['mean_inference_time_s'],
        'comparison': {
            str(s): {
                'mae_gy': results['per_step_results'][str(s)]['aggregate']['mean_mae_gy'],
                'time_s': results['per_step_results'][str(s)]['aggregate']['mean_inference_time_s'],
            }
            for s in steps_to_test
        }
    }

    print(f"\n*** OPTIMAL: {best_steps} steps with "
          f"{results['summary']['optimal_mae_gy']:.2f} Gy MAE ***")

    return results


def run_ensemble_averaging(
    model: DoseDDPM,
    val_files: List[str],
    n_samples_to_test: List[int],
    ddim_steps: int = 50,
    device: str = 'cuda',
    rx_dose_gy: float = 70.0,
) -> Dict:
    """
    Experiment 1.2: Ensemble Averaging

    Tests averaging multiple predictions to reduce variance.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.2: ENSEMBLE AVERAGING")
    print("="*60)
    print(f"Using {ddim_steps} DDIM steps per sample")

    results = {
        'experiment': 'ensemble_averaging',
        'timestamp': datetime.now().isoformat(),
        'ddim_steps': ddim_steps,
        'n_samples_tested': n_samples_to_test,
        'val_cases': [Path(f).stem for f in val_files],
        'per_n_results': {},
        'summary': {},
    }

    # For each case, generate max_n predictions and then test subsets
    max_n = max(n_samples_to_test)

    for npz_path in val_files:
        case_id = Path(npz_path).stem
        print(f"\n--- Processing {case_id} ---")

        # Generate max_n predictions
        predictions = []
        total_time = 0

        for i in range(max_n):
            print(f"  Sample {i+1}/{max_n}...", end=" ", flush=True)

            # Set different random seed for each sample
            torch.manual_seed(42 + i)

            pred, inf_time = predict_single_case(
                model, str(npz_path), ddim_steps=ddim_steps, device=device
            )
            predictions.append(pred)
            total_time += inf_time
            print(f"done ({inf_time:.1f}s)")

        predictions = np.stack(predictions)  # (N, H, W, D)

        # Evaluate for each n_samples
        for n in n_samples_to_test:
            key = str(n)
            if key not in results['per_n_results']:
                results['per_n_results'][key] = {
                    'cases': [],
                    'mae_values': [],
                    'std_across_samples': [],
                }

            # Average first n predictions
            ensemble_pred = np.mean(predictions[:n], axis=0)

            # Compute std across samples (variability measure)
            if n > 1:
                sample_std = np.mean(np.std(predictions[:n], axis=0))
            else:
                sample_std = 0.0

            # Evaluate
            metrics = evaluate_prediction(ensemble_pred, str(npz_path), rx_dose_gy)

            results['per_n_results'][key]['cases'].append({
                'case_id': case_id,
                'mae_gy': metrics['mae_gy'],
                'std_across_samples': float(sample_std),
                'gamma_pass_rate': metrics.get('gamma_gamma_pass_rate'),
            })
            results['per_n_results'][key]['mae_values'].append(metrics['mae_gy'])
            results['per_n_results'][key]['std_across_samples'].append(sample_std)

            print(f"  n={n}: MAE = {metrics['mae_gy']:.2f} Gy, sample_std = {sample_std:.4f}")

    # Aggregate results
    for n in n_samples_to_test:
        key = str(n)
        res = results['per_n_results'][key]
        res['aggregate'] = {
            'mean_mae_gy': float(np.mean(res['mae_values'])),
            'std_mae_gy': float(np.std(res['mae_values'])),
            'mean_sample_std': float(np.mean(res['std_across_samples'])),
        }

    # Find optimal n
    best_n = min(
        results['per_n_results'].keys(),
        key=lambda k: results['per_n_results'][k]['aggregate']['mean_mae_gy']
    )

    results['summary'] = {
        'optimal_n_samples': int(best_n),
        'optimal_mae_gy': results['per_n_results'][best_n]['aggregate']['mean_mae_gy'],
        'comparison': {
            str(n): {
                'mae_gy': results['per_n_results'][str(n)]['aggregate']['mean_mae_gy'],
                'sample_variability': results['per_n_results'][str(n)]['aggregate']['mean_sample_std'],
            }
            for n in n_samples_to_test
        }
    }

    print(f"\n*** OPTIMAL: {best_n} samples with "
          f"{results['summary']['optimal_mae_gy']:.2f} Gy MAE ***")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 DDPM Optimization Experiments")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to DDPM checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to preprocessed NPZ files')
    parser.add_argument('--output_dir', type=str, default='./experiments/phase1_sampling',
                       help='Output directory for results')
    parser.add_argument('--rx_dose_gy', type=float, default=70.0,
                       help='Prescription dose in Gy')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Experiment settings
    parser.add_argument('--steps', type=str, default='50,100,250,500,1000',
                       help='Comma-separated list of DDIM steps to test')
    parser.add_argument('--n_samples', type=str, default='1,3,5,10',
                       help='Comma-separated list of ensemble sizes to test')
    parser.add_argument('--skip_ensemble', action='store_true',
                       help='Skip ensemble experiment (faster)')

    args = parser.parse_args()

    # Parse experiment parameters
    steps_to_test = [int(s) for s in args.steps.split(',')]
    n_samples_to_test = [int(n) for n in args.n_samples.split(',')]

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get validation files (use same split as training)
    data_dir = Path(args.data_dir)
    all_files = sorted(list(data_dir.glob("*.npz")))

    # Use same seed-based split as training (10% val, 10% test)
    n_files = len(all_files)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_files)

    n_test = max(1, int(n_files * 0.1))
    n_val = max(1, int(n_files * 0.1))
    n_train = n_files - n_val - n_test

    val_indices = indices[n_train:n_train + n_val]
    val_files = [all_files[i] for i in val_indices]

    print(f"\nUsing {len(val_files)} validation cases: {[f.stem for f in val_files]}")

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Run experiments
    print("\n" + "="*60)
    print("PHASE 1: QUICK WINS (NO RETRAINING)")
    print("="*60)

    # Experiment 1.1: Sampling Steps
    sampling_results = run_sampling_ablation(
        model=model,
        val_files=val_files,
        steps_to_test=steps_to_test,
        device=args.device,
        rx_dose_gy=args.rx_dose_gy,
    )

    # Save sampling results
    sampling_path = output_dir / 'exp1_1_sampling_results.json'
    with open(sampling_path, 'w') as f:
        json.dump(sampling_results, f, indent=2)
    print(f"\nSampling results saved to: {sampling_path}")

    # Experiment 1.2: Ensemble Averaging
    if not args.skip_ensemble:
        # Use optimal steps from Exp 1.1
        optimal_steps = sampling_results['summary']['optimal_steps']

        ensemble_results = run_ensemble_averaging(
            model=model,
            val_files=val_files,
            n_samples_to_test=n_samples_to_test,
            ddim_steps=optimal_steps,
            device=args.device,
            rx_dose_gy=args.rx_dose_gy,
        )

        # Save ensemble results
        ensemble_dir = output_dir.parent / 'phase1_ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = ensemble_dir / 'exp1_2_ensemble_results.json'
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_results, f, indent=2)
        print(f"\nEnsemble results saved to: {ensemble_path}")
    else:
        ensemble_results = None

    # Combined Phase 1 summary
    phase1_summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'val_cases': [f.stem for f in val_files],
        'exp1_1_sampling': {
            'optimal_steps': sampling_results['summary']['optimal_steps'],
            'optimal_mae_gy': sampling_results['summary']['optimal_mae_gy'],
            'comparison': sampling_results['summary']['comparison'],
        },
        'exp1_2_ensemble': ensemble_results['summary'] if ensemble_results else None,
        'baseline_comparison': {
            'baseline_mae_gy': 3.73,  # From documentation
            'ddpm_v1_mae_gy': 12.19,  # From documentation
        },
        'phase1_best_mae_gy': min(
            sampling_results['summary']['optimal_mae_gy'],
            ensemble_results['summary']['optimal_mae_gy'] if ensemble_results else float('inf')
        ),
        'proceed_to_phase2': None,  # Will be determined after analysis
    }

    # Decision: Proceed to Phase 2 if best MAE > 6 Gy
    best_phase1_mae = phase1_summary['phase1_best_mae_gy']
    phase1_summary['proceed_to_phase2'] = best_phase1_mae > 6.0

    summary_path = output_dir / 'phase1_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(phase1_summary, f, indent=2)

    # Print final summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nExp 1.1 - Sampling Steps:")
    print(f"  Optimal: {sampling_results['summary']['optimal_steps']} steps")
    print(f"  MAE: {sampling_results['summary']['optimal_mae_gy']:.2f} Gy")

    if ensemble_results:
        print(f"\nExp 1.2 - Ensemble Averaging:")
        print(f"  Optimal: {ensemble_results['summary']['optimal_n_samples']} samples")
        print(f"  MAE: {ensemble_results['summary']['optimal_mae_gy']:.2f} Gy")

    print(f"\nBest Phase 1 MAE: {best_phase1_mae:.2f} Gy")
    print(f"Baseline MAE: 3.73 Gy")
    print(f"DDPM v1 MAE: 12.19 Gy")

    if phase1_summary['proceed_to_phase2']:
        print(f"\n*** RECOMMENDATION: Proceed to Phase 2 (MAE > 6 Gy threshold) ***")
    else:
        print(f"\n*** Phase 1 achieved significant improvement! Review results. ***")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
