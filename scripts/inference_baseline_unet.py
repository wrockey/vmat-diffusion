"""
Inference script for trained Baseline U-Net model.

Much faster than diffusion: single forward pass per patch (~10 sec vs ~10 min).

Usage:
    python inference_baseline_unet.py \
        --checkpoint ./runs/baseline_unet/checkpoints/best.ckpt \
        --input_dir ./test_npz \
        --output_dir ./predictions \
        --compute_metrics
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

import torch
from tqdm import tqdm

# Import baseline model
from train_baseline_unet import BaselineDosePredictor, DEFAULT_SPACING_MM, get_spacing_from_metadata

# Import evaluation functions from diffusion inference script
from inference_dose_ddpm import (
    compute_dose_metrics,
    compute_gamma,
    compute_dvh_metrics,
    check_clinical_constraints,
    HAS_PYMEDPHYS,
)


def load_model(checkpoint_path: str, device: str = 'cuda') -> BaselineDosePredictor:
    """Load trained baseline model."""
    print(f"Loading model from {checkpoint_path}")
    
    model = BaselineDosePredictor.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    print(f"  Model: BaselineUNet3D (Direct Regression)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def predict_single_case(
    model: BaselineDosePredictor,
    npz_path: str,
    patch_size: int = 128,
    overlap: int = 32,
    device: str = 'cuda',
):
    """Predict dose for a single case using sliding window."""
    import time
    start_time = time.time()
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    ct = torch.from_numpy(data['ct'].astype(np.float32)).unsqueeze(0)
    masks_sdf = torch.from_numpy(data['masks_sdf'].astype(np.float32))
    constraints = torch.from_numpy(data['constraints'].astype(np.float32))
    
    # Build condition tensor
    condition = torch.cat([ct, masks_sdf], dim=0).unsqueeze(0).to(device)
    constraints = constraints.unsqueeze(0).to(device)
    
    # Predict (single forward pass per patch - fast!)
    pred_dose = model.predict_full_volume(
        condition=condition,
        constraints=constraints,
        patch_size=patch_size,
        overlap=overlap,
        verbose=True,
    )
    
    # Convert to numpy
    pred_dose_np = pred_dose[0, 0].cpu().numpy()
    
    elapsed = time.time() - start_time
    print(f"  Inference time: {elapsed:.1f} sec")
    
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    
    return pred_dose_np, metadata


def evaluate_single_case(
    pred: np.ndarray,
    npz_path: str,
    rx_dose_gy: float = 70.0,
    compute_gamma_metric: bool = True,
    gamma_subsample: int = 2,
) -> Dict:
    """Evaluate prediction against ground truth."""
    data = np.load(npz_path, allow_pickle=True)
    target = data['dose']
    masks = data['masks']

    # Read spacing from metadata
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    structure_names = {
        0: 'PTV70', 1: 'PTV56', 2: 'Prostate', 3: 'Rectum',
        4: 'Bladder', 5: 'Femur_L', 6: 'Femur_R', 7: 'Bowel'
    }

    results = {
        'case_id': Path(npz_path).stem,
        'model_type': 'baseline_unet',
        'timestamp': datetime.now().isoformat(),
        'spacing_mm': spacing,
    }

    # Dose metrics
    print("  Computing dose metrics...")
    results['dose_metrics'] = compute_dose_metrics(pred, target, rx_dose_gy)

    # Gamma
    if compute_gamma_metric and HAS_PYMEDPHYS:
        print("  Computing gamma...")
        pred_gy = pred * rx_dose_gy
        target_gy = target * rx_dose_gy
        results['gamma'] = compute_gamma(pred_gy, target_gy, spacing_mm=spacing, subsample=gamma_subsample)

    # DVH metrics
    print("  Computing DVH metrics...")
    results['dvh_metrics'] = compute_dvh_metrics(pred, target, masks, structure_names, rx_dose_gy, spacing_mm=spacing)

    # Clinical constraints
    print("  Checking clinical constraints...")
    results['clinical_constraints'] = check_clinical_constraints(results['dvh_metrics'])

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline U-Net Inference")
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, default=None, help='Single .npz file')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory of .npz files')
    parser.add_argument('--output', type=str, default=None, help='Output .npz file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--gamma_subsample', type=int, default=2)
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)
    
    # Determine input files
    if args.input:
        input_files = [Path(args.input)]
    elif args.input_dir:
        input_files = sorted(Path(args.input_dir).glob("*.npz"))
    else:
        raise ValueError("Must specify --input or --input_dir")
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    print(f"\nProcessing {len(input_files)} case(s)...")
    
    all_results = []
    
    for npz_path in tqdm(input_files, desc="Inference"):
        case_id = npz_path.stem
        print(f"\nProcessing: {case_id}")
        
        # Predict
        pred_dose, metadata = predict_single_case(
            model, str(npz_path),
            patch_size=args.patch_size,
            overlap=args.overlap,
            device=device,
        )
        
        # Save prediction
        if output_dir:
            out_path = output_dir / f"{case_id}_pred.npz"
            np.savez_compressed(out_path, dose=pred_dose, metadata=metadata)
            print(f"  Saved: {out_path}")
        
        # Evaluate
        if args.compute_metrics:
            results = evaluate_single_case(
                pred_dose, str(npz_path),
                rx_dose_gy=args.rx_dose_gy,
                compute_gamma_metric=HAS_PYMEDPHYS,
                gamma_subsample=args.gamma_subsample,
            )
            all_results.append(results)
            
            # Print summary
            dm = results['dose_metrics']
            print(f"  MAE: {dm['mae_gy']:.2f} Gy")
            if 'gamma' in results and results['gamma'].get('gamma_pass_rate'):
                print(f"  Gamma: {results['gamma']['gamma_pass_rate']:.1f}%")
    
    # Save aggregate results
    if args.compute_metrics and all_results and output_dir:
        results_path = output_dir / 'baseline_evaluation_results.json'
        
        mae_values = [r['dose_metrics']['mae_gy'] for r in all_results]
        gamma_values = [r['gamma']['gamma_pass_rate'] for r in all_results
                       if 'gamma' in r and r['gamma'].get('gamma_pass_rate')]
        
        summary = {
            'model_type': 'baseline_unet',
            'n_cases': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': {
                'mae_gy_mean': float(np.mean(mae_values)),
                'mae_gy_std': float(np.std(mae_values)),
            },
            'per_case_results': all_results,
        }
        
        if gamma_values:
            summary['aggregate_metrics']['gamma_pass_rate_mean'] = float(np.mean(gamma_values))
            summary['aggregate_metrics']['gamma_pass_rate_std'] = float(np.std(gamma_values))
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BASELINE EVALUATION SUMMARY")
        print('='*60)
        print(f"Cases evaluated: {len(all_results)}")
        print(f"MAE: {summary['aggregate_metrics']['mae_gy_mean']:.2f} ± {summary['aggregate_metrics']['mae_gy_std']:.2f} Gy")
        if gamma_values:
            print(f"Gamma: {summary['aggregate_metrics']['gamma_pass_rate_mean']:.1f} ± {summary['aggregate_metrics']['gamma_pass_rate_std']:.1f}%")
        print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
