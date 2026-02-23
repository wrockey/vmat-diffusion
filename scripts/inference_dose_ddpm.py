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
from train_dose_ddpm_v2 import DoseDDPM, VMATDoseFullVolumeDataset, DEFAULT_SPACING_MM


def get_spacing_from_metadata(metadata):
    """
    Extract voxel spacing from NPZ metadata with backwards-compatible fallback.

    Fallback chain:
        1. voxel_spacing_mm (v2.3+ native spacing)
        2. target_spacing_mm (v2.2 resampled spacing)
        3. DEFAULT_SPACING_MM
    """
    import numpy as np
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()

    if 'voxel_spacing_mm' in metadata:
        spacing = metadata['voxel_spacing_mm']
        return tuple(float(s) for s in spacing)

    if 'target_spacing_mm' in metadata:
        spacing = metadata['target_spacing_mm']
        return tuple(float(s) for s in spacing)

    return DEFAULT_SPACING_MM

# Optional: pymedphys for gamma
try:
    from pymedphys import gamma as pymedphys_gamma
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False
    print("Warning: pymedphys not installed. Gamma evaluation disabled.")


# =============================================================================
# Clinical Constraints (QUANTEC-based for Prostate VMAT)
# =============================================================================

# These are typical clinical constraints for prostate VMAT with SIB (70/56 Gy)
# Based on QUANTEC guidelines and common clinical practice
# Users should adjust these based on their institutional protocols

CLINICAL_CONSTRAINTS = {
    # PTV constraints (targets)
    'PTV70': {
        'D95_min': 66.5,    # 95% of Rx (Gy) - minimum acceptable
        'D95_target': 70.0,  # Ideal D95
        'V95_min': 95.0,     # % volume receiving 95% of Rx
        'description': 'High-dose PTV (70 Gy target)',
    },
    'PTV56': {
        'D95_min': 53.2,    # 95% of Rx (Gy)
        'V95_min': 95.0,
        'description': 'Intermediate PTV (56 Gy target)',
    },
    
    # OAR constraints (organs at risk)
    'Rectum': {
        'V70_max': 15.0,    # % volume receiving >= 70 Gy
        'V60_max': 25.0,    # % volume receiving >= 60 Gy
        'V50_max': 50.0,    # % volume receiving >= 50 Gy
        'Dmax_max': 75.0,   # Maximum dose (Gy)
        'description': 'Rectum - critical OAR',
    },
    'Bladder': {
        'V70_max': 25.0,
        'V60_max': 35.0,
        'V50_max': 50.0,
        'Dmax_max': 75.0,
        'description': 'Bladder - critical OAR',
    },
    'Femur_L': {
        'Dmax_max': 50.0,   # Maximum dose to femoral head
        'V50_max': 5.0,     # Very little volume above 50 Gy
        'description': 'Left femoral head',
    },
    'Femur_R': {
        'Dmax_max': 50.0,
        'V50_max': 5.0,
        'description': 'Right femoral head',
    },
    'Bowel': {
        'Dmax_max': 52.0,
        'V45_max': 195.0,   # cc, but we'll check % for simplicity
        'description': 'Bowel bag',
    },
}


def check_clinical_constraints(
    dvh_metrics: Dict[str, Dict[str, float]],
    constraints: Dict = CLINICAL_CONSTRAINTS,
) -> Dict[str, Dict]:
    """
    Check if predicted DVH metrics meet clinical constraints.
    
    Args:
        dvh_metrics: DVH metrics from compute_dvh_metrics()
        constraints: Clinical constraint dictionary
    
    Returns:
        Dict with pass/fail status and details for each structure
    """
    results = {
        'overall_pass': True,
        'structures': {},
        'violations': [],
        'summary': {
            'total_constraints': 0,
            'passed': 0,
            'failed': 0,
        }
    }
    
    for structure, struct_constraints in constraints.items():
        if structure not in dvh_metrics:
            continue
        
        metrics = dvh_metrics[structure]
        if not metrics.get('exists', False):
            continue
        
        struct_results = {
            'description': struct_constraints.get('description', ''),
            'constraints_checked': [],
            'passed': True,
        }
        
        # Check each constraint for this structure
        for constraint_name, limit in struct_constraints.items():
            if constraint_name == 'description':
                continue
            
            results['summary']['total_constraints'] += 1
            
            # Parse constraint type
            if constraint_name.startswith('D') and '_min' in constraint_name:
                # Minimum Dx constraint (for targets)
                metric_name = constraint_name.replace('_min', '').replace('_target', '')
                pred_key = f'pred_{metric_name}'
                if pred_key in metrics:
                    pred_value = metrics[pred_key]
                    passed = pred_value >= limit
                    constraint_type = 'min'
                else:
                    continue
                    
            elif constraint_name.startswith('D') and '_max' in constraint_name:
                # Maximum Dx constraint
                metric_name = constraint_name.replace('_max', '')
                if metric_name == 'Dmax':
                    pred_value = metrics.get('pred_max_gy', 0)
                else:
                    pred_key = f'pred_{metric_name}'
                    pred_value = metrics.get(pred_key, 0)
                passed = pred_value <= limit
                constraint_type = 'max'
                
            elif constraint_name.startswith('V') and '_max' in constraint_name:
                # Maximum Vx constraint (for OARs)
                metric_name = constraint_name.replace('_max', '')
                pred_key = f'pred_{metric_name}'
                if pred_key in metrics:
                    pred_value = metrics[pred_key]
                    passed = pred_value <= limit
                    constraint_type = 'max'
                else:
                    continue
                    
            elif constraint_name.startswith('V') and '_min' in constraint_name:
                # Minimum Vx constraint (for targets)
                metric_name = constraint_name.replace('_min', '')
                pred_key = f'pred_{metric_name}'
                if pred_key in metrics:
                    pred_value = metrics[pred_key]
                    passed = pred_value >= limit
                    constraint_type = 'min'
                else:
                    continue
            else:
                continue
            
            # Record result
            constraint_result = {
                'constraint': constraint_name,
                'limit': limit,
                'predicted': round(pred_value, 2),
                'passed': passed,
                'type': constraint_type,
            }
            struct_results['constraints_checked'].append(constraint_result)
            
            if passed:
                results['summary']['passed'] += 1
            else:
                results['summary']['failed'] += 1
                struct_results['passed'] = False
                results['overall_pass'] = False
                results['violations'].append({
                    'structure': structure,
                    'constraint': constraint_name,
                    'limit': limit,
                    'predicted': round(pred_value, 2),
                    'type': constraint_type,
                })
        
        results['structures'][structure] = struct_results
    
    return results


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Absolute Error, optionally within a mask."""
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    return float(np.mean(np.abs(pred - target)))


def compute_dose_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    rx_dose_gy: float = 70.0,
) -> Dict[str, float]:
    """Compute various dose comparison metrics."""
    
    # Convert to Gy
    pred_gy = pred * rx_dose_gy
    target_gy = target * rx_dose_gy
    
    metrics = {
        'mae_gy': compute_mae(pred_gy, target_gy),
        'mae_normalized': compute_mae(pred, target),
        'max_error_gy': float(np.max(np.abs(pred_gy - target_gy))),
        'rmse_gy': float(np.sqrt(np.mean((pred_gy - target_gy)**2))),
        'pred_max_gy': float(pred_gy.max()),
        'target_max_gy': float(target_gy.max()),
        'pred_mean_gy': float(pred_gy.mean()),
        'target_mean_gy': float(target_gy.mean()),
    }
    
    # Dose within threshold regions
    for threshold in [0.1, 0.5, 0.9]:  # 10%, 50%, 90% of Rx
        mask = target >= threshold
        if mask.any():
            metrics[f'mae_gy_above_{int(threshold*100)}pct'] = compute_mae(
                pred_gy, target_gy, mask.astype(np.float32)
            )
    
    return metrics


def compute_gamma(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_mm: Tuple[float, ...] = DEFAULT_SPACING_MM,
    dose_threshold_pct: float = 3.0,
    distance_mm: float = 3.0,
    lower_dose_cutoff_pct: float = 10.0,
    subsample: int = 1,
) -> Dict[str, float]:
    """
    Compute gamma pass rate.
    
    Args:
        pred: Predicted dose in Gy
        target: Ground truth dose in Gy
        spacing_mm: Voxel spacing
        dose_threshold_pct: Dose difference threshold (%)
        distance_mm: Distance-to-agreement threshold (mm)
        lower_dose_cutoff_pct: Ignore voxels below this % of max dose
        subsample: Subsample factor for speed (1 = full resolution)
    
    Returns:
        Dict with gamma pass rate and statistics
    """
    if not HAS_PYMEDPHYS:
        return {'gamma_pass_rate': None, 'error': 'pymedphys not installed'}
    
    # Subsample if requested
    if subsample > 1:
        pred = pred[::subsample, ::subsample, ::subsample]
        target = target[::subsample, ::subsample, ::subsample]
        spacing_mm = tuple(s * subsample for s in spacing_mm)
    
    # Create coordinate axes
    axes = tuple(
        np.arange(s) * sp for s, sp in zip(pred.shape, spacing_mm)
    )
    
    try:
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target,
            axes_evaluation=axes,
            dose_evaluation=pred,
            dose_percent_threshold=dose_threshold_pct,
            distance_mm_threshold=distance_mm,
            lower_percent_dose_cutoff=lower_dose_cutoff_pct,
        )
        
        valid = np.isfinite(gamma_map)
        if not valid.any():
            return {'gamma_pass_rate': 0.0, 'gamma_mean': None, 'gamma_max': None}
        
        pass_rate = float(np.mean(gamma_map[valid] <= 1.0) * 100)
        
        return {
            'gamma_pass_rate': pass_rate,
            'gamma_mean': float(np.mean(gamma_map[valid])),
            'gamma_max': float(np.max(gamma_map[valid])),
            'gamma_median': float(np.median(gamma_map[valid])),
            'voxels_evaluated': int(valid.sum()),
        }
    
    except Exception as e:
        return {'gamma_pass_rate': None, 'error': str(e)}


def compute_dvh_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    masks: np.ndarray,
    structure_names: Dict[int, str],
    rx_dose_gy: float = 70.0,
    spacing_mm: Tuple[float, ...] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute DVH-based metrics for each structure.

    Args:
        pred: Predicted dose (normalized)
        target: Ground truth dose (normalized)
        masks: Binary masks (C, H, W, D)
        structure_names: Channel to structure name mapping
        rx_dose_gy: Prescription dose for denormalization
        spacing_mm: Voxel spacing in mm (default: DEFAULT_SPACING_MM)

    Returns:
        Dict of metrics per structure
    """
    if spacing_mm is None:
        spacing_mm = DEFAULT_SPACING_MM
    pred_gy = pred * rx_dose_gy
    target_gy = target * rx_dose_gy
    
    results = {}
    
    for ch, name in structure_names.items():
        if ch >= masks.shape[0]:
            continue
        
        mask = masks[ch] > 0
        if not mask.any():
            results[name] = {'exists': False}
            continue
        
        pred_struct = pred_gy[mask]
        target_struct = target_gy[mask]
        
        # Basic stats
        metrics = {
            'exists': True,
            'volume_cc': float(mask.sum() * np.prod(spacing_mm) / 1000),
            'mae_gy': float(np.mean(np.abs(pred_struct - target_struct))),
            'pred_mean_gy': float(pred_struct.mean()),
            'target_mean_gy': float(target_struct.mean()),
            'pred_max_gy': float(pred_struct.max()),
            'target_max_gy': float(target_struct.max()),
            'pred_min_gy': float(pred_struct.min()),
            'target_min_gy': float(target_struct.min()),
        }
        
        # DVH points (Dx = dose to x% of volume)
        for pct in [95, 50, 5, 2]:
            metrics[f'pred_D{pct}'] = float(np.percentile(pred_struct, 100 - pct))
            metrics[f'target_D{pct}'] = float(np.percentile(target_struct, 100 - pct))
            metrics[f'D{pct}_error'] = metrics[f'pred_D{pct}'] - metrics[f'target_D{pct}']
        
        # Volume receiving dose (Vx = volume receiving >= x Gy)
        for dose_level in [70, 60, 50, 40]:
            pred_vx = float((pred_struct >= dose_level).sum() / len(pred_struct) * 100)
            target_vx = float((target_struct >= dose_level).sum() / len(target_struct) * 100)
            metrics[f'pred_V{dose_level}'] = pred_vx
            metrics[f'target_V{dose_level}'] = target_vx
            metrics[f'V{dose_level}_error'] = pred_vx - target_vx
        
        results[name] = metrics
    
    return results


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
    overlap: int = 32,
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
    Evaluate prediction against ground truth.
    
    Returns:
        Dict with all metrics
    """
    # Load ground truth
    data = np.load(npz_path, allow_pickle=True)
    target = data['dose']
    masks = data['masks']

    # Read spacing from metadata
    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    # Structure names
    structure_names = {
        0: 'PTV70', 1: 'PTV56', 2: 'Prostate', 3: 'Rectum',
        4: 'Bladder', 5: 'Femur_L', 6: 'Femur_R', 7: 'Bowel'
    }

    results = {
        'case_id': Path(npz_path).stem,
        'timestamp': datetime.now().isoformat(),
        'spacing_mm': spacing,
    }

    # Dose metrics
    print("  Computing dose metrics...")
    results['dose_metrics'] = compute_dose_metrics(pred, target, rx_dose_gy)

    # Gamma
    if compute_gamma_metric and HAS_PYMEDPHYS:
        print("  Computing gamma (this may take a moment)...")
        pred_gy = pred * rx_dose_gy
        target_gy = target * rx_dose_gy
        results['gamma'] = compute_gamma(
            pred_gy, target_gy,
            spacing_mm=spacing,
            subsample=gamma_subsample,
        )

    # DVH metrics
    print("  Computing DVH metrics...")
    results['dvh_metrics'] = compute_dvh_metrics(
        pred, target, masks, structure_names, rx_dose_gy, spacing_mm=spacing
    )
    
    # Clinical constraint check
    print("  Checking clinical constraints...")
    results['clinical_constraints'] = check_clinical_constraints(results['dvh_metrics'])
    
    if results['clinical_constraints']['violations']:
        print(f"  ⚠️  {len(results['clinical_constraints']['violations'])} constraint violation(s)")
    else:
        print("  ✓ All clinical constraints passed")
    
    return results


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
    parser.add_argument('--overlap', type=int, default=32)
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
            
            if 'gamma' in results and results['gamma'].get('gamma_pass_rate') is not None:
                print(f"    Gamma (3%/3mm): {results['gamma']['gamma_pass_rate']:.1f}%")
            
            # PTV70 DVH
            if 'PTV70' in results['dvh_metrics'] and results['dvh_metrics']['PTV70']['exists']:
                ptv = results['dvh_metrics']['PTV70']
                print(f"    PTV70 D95 error: {ptv['D95_error']:.2f} Gy")
            
            # Clinical constraints summary for this case
            if 'clinical_constraints' in results:
                cc = results['clinical_constraints']
                if cc['overall_pass']:
                    print(f"    Clinical constraints: ✓ ALL PASSED")
                else:
                    print(f"    Clinical constraints: ⚠️ {len(cc['violations'])} violation(s)")
                    for v in cc['violations'][:3]:  # Show up to 3
                        print(f"      - {v['structure']} {v['constraint']}: {v['predicted']} (limit: {v['limit']})")
    
    # Save aggregate results
    if args.compute_metrics and all_results:
        results_path = output_dir / 'evaluation_results.json'
        
        # Compute aggregate metrics
        mae_values = [r['dose_metrics']['mae_gy'] for r in all_results]
        gamma_values = [r['gamma']['gamma_pass_rate'] for r in all_results 
                       if 'gamma' in r and r['gamma'].get('gamma_pass_rate') is not None]
        
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
        
        # Count violations by structure/constraint
        violation_counts = {}
        for r in all_results:
            for v in r.get('clinical_constraints', {}).get('violations', []):
                key = f"{v['structure']}_{v['constraint']}"
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
            json.dump(summary, f, indent=2)
        
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
