"""
Centralized metric computations for VMAT dose prediction evaluation.

Computes dose error metrics, gamma analysis, DVH metrics, and region-specific
gamma. All metric functions return plain dicts for JSON serialization.

Fixes vs. prior scattered implementations:
    - D95 via np.percentile(dose, 5, method='lower') (nearest-rank, matches TPS)
    - All-NaN gamma returns None (not 0.0)
    - max_gamma=2.0 default for speed
    - float64 for gamma computation
    - V45_cc for Bowel (absolute volume, not %)
    - dvh_reliable flag for <100 voxels
    - Input validation (NaN, Inf, negative dose)

Version: 1.0.0
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from eval_core import (
    STRUCTURE_CHANNELS,
    STRUCTURE_INDEX,
    PTV_STRUCTURES,
    DEFAULT_SPACING_MM,
    PRIMARY_PRESCRIPTION_GY,
    SECONDARY_PRESCRIPTION_GY,
    MIN_VOXELS_RELIABLE,
    EvaluationResult,
    denormalize_dose,
    get_structure_mask,
    validate_dose_array,
    sanitize_dose,
)
from eval_clinical import check_clinical_constraints

# Optional pymedphys
try:
    from pymedphys import gamma as pymedphys_gamma
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False


# =============================================================================
# Dose Error Metrics
# =============================================================================

def compute_dose_metrics(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
) -> Dict[str, float]:
    """
    Compute dose comparison metrics between prediction and target in Gy.

    Args:
        pred_gy: Predicted dose in Gy
        target_gy: Ground truth dose in Gy

    Returns:
        Dict with mae_gy, rmse_gy, max_error_gy, pred_max_gy, target_max_gy,
        pred_mean_gy, target_mean_gy, and threshold-based MAE.
    """
    diff = pred_gy - target_gy
    abs_diff = np.abs(diff)

    metrics = {
        'mae_gy': float(abs_diff.mean()),
        'rmse_gy': float(np.sqrt((diff ** 2).mean())),
        'max_error_gy': float(abs_diff.max()),
        'pred_max_gy': float(pred_gy.max()),
        'target_max_gy': float(target_gy.max()),
        'pred_mean_gy': float(pred_gy.mean()),
        'target_mean_gy': float(target_gy.mean()),
    }

    # Threshold-based MAE (% of prescription)
    for threshold_pct in [10, 50, 90]:
        threshold_frac = threshold_pct / 100.0
        mask = target_gy >= (threshold_frac * target_gy.max())
        if mask.any():
            metrics[f'mae_gy_above_{threshold_pct}pct'] = float(
                abs_diff[mask].mean()
            )

    return metrics


def compute_per_structure_mae(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE within each anatomical structure.

    Args:
        pred_gy: Predicted dose in Gy
        target_gy: Ground truth dose in Gy
        masks: (8, Y, X, Z) binary masks

    Returns:
        Dict mapping structure name to MAE in Gy
    """
    result = {}
    for ch, name in STRUCTURE_CHANNELS.items():
        if ch >= masks.shape[0]:
            continue
        mask = masks[ch] > 0
        if mask.any():
            result[name] = float(np.mean(np.abs(pred_gy[mask] - target_gy[mask])))
    return result


# =============================================================================
# Gamma Analysis
# =============================================================================

def compute_gamma(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    spacing_mm: Tuple[float, ...] = DEFAULT_SPACING_MM,
    dose_threshold_pct: float = 3.0,
    distance_mm: float = 3.0,
    lower_dose_cutoff_pct: float = 10.0,
    max_gamma: float = 2.0,
    subsample: int = 1,
) -> Dict:
    """
    Compute gamma pass rate (global normalization, AAPM TG-218 standard).

    Args:
        pred_gy: Predicted dose in Gy (will be cast to float64)
        target_gy: Ground truth dose in Gy (will be cast to float64)
        spacing_mm: Voxel spacing (y, x, z) in mm
        dose_threshold_pct: Dose difference threshold (%)
        distance_mm: Distance-to-agreement threshold (mm)
        lower_dose_cutoff_pct: Ignore voxels below this % of max dose
        max_gamma: Cap gamma search distance (2.0 = no effect on pass rates, 3-5x speedup)
        subsample: Subsample factor for speed (1 = full resolution)

    Returns:
        Dict with gamma_pass_rate (%), gamma_mean, gamma_median, gamma_max,
        voxels_evaluated. Returns gamma_pass_rate=None if pymedphys unavailable
        or computation fails.
    """
    if not HAS_PYMEDPHYS:
        return {'gamma_pass_rate': None, 'error': 'pymedphys not installed'}

    # Cast to float64 for numerical stability
    pred_f64 = pred_gy.astype(np.float64)
    target_f64 = target_gy.astype(np.float64)

    # Subsample if requested
    if subsample > 1:
        pred_f64 = pred_f64[::subsample, ::subsample, ::subsample]
        target_f64 = target_f64[::subsample, ::subsample, ::subsample]
        spacing_mm = tuple(s * subsample for s in spacing_mm)

    # Create coordinate axes
    axes = tuple(
        np.arange(s) * sp for s, sp in zip(pred_f64.shape, spacing_mm)
    )

    try:
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_f64,
            axes_evaluation=axes,
            dose_evaluation=pred_f64,
            dose_percent_threshold=dose_threshold_pct,
            distance_mm_threshold=distance_mm,
            lower_percent_dose_cutoff=lower_dose_cutoff_pct,
            max_gamma=max_gamma,
        )

        valid = np.isfinite(gamma_map)
        if not valid.any():
            # No voxels above dose cutoff — return None, NOT 0.0
            return {
                'gamma_pass_rate': None,
                'gamma_mean': None,
                'gamma_median': None,
                'gamma_max': None,
                'voxels_evaluated': 0,
                'note': 'No voxels above lower dose cutoff',
            }

        gamma_valid = gamma_map[valid]
        return {
            'gamma_pass_rate': float(np.mean(gamma_valid <= 1.0) * 100),
            'gamma_mean': float(np.mean(gamma_valid)),
            'gamma_median': float(np.median(gamma_valid)),
            'gamma_max': float(np.max(gamma_valid)),
            'voxels_evaluated': int(valid.sum()),
        }

    except Exception as e:
        return {'gamma_pass_rate': None, 'error': str(e)}


def compute_region_gamma(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    region_mask: np.ndarray,
    spacing_mm: Tuple[float, ...] = DEFAULT_SPACING_MM,
    dose_threshold_pct: float = 3.0,
    distance_mm: float = 3.0,
    lower_dose_cutoff_pct: float = 10.0,
    max_gamma: float = 2.0,
    subsample: int = 2,
) -> Dict:
    """
    Compute gamma within a specific region (compute-full-then-mask approach).

    The gamma map is computed on the full volume first, then masked to the
    region of interest. This ensures DTA search crosses mask boundaries
    correctly (masking first creates artificial dose cliffs).

    Args:
        pred_gy: Full predicted dose in Gy
        target_gy: Full ground truth dose in Gy
        region_mask: Boolean mask defining the region of interest
        spacing_mm: Voxel spacing in mm
        dose_threshold_pct: Dose difference threshold (%)
        distance_mm: Distance-to-agreement threshold (mm)
        lower_dose_cutoff_pct: Lower dose cutoff (% of max)
        max_gamma: Cap gamma search
        subsample: Subsample factor

    Returns:
        Dict with gamma_pass_rate within region, or None if no valid voxels
    """
    if not HAS_PYMEDPHYS:
        return {'gamma_pass_rate': None, 'error': 'pymedphys not installed'}

    if not region_mask.any():
        return {'gamma_pass_rate': None, 'error': 'empty region mask',
                'voxels_evaluated': 0}

    # Cast to float64
    pred_f64 = pred_gy.astype(np.float64)
    target_f64 = target_gy.astype(np.float64)

    # Subsample everything together
    if subsample > 1:
        pred_f64 = pred_f64[::subsample, ::subsample, ::subsample]
        target_f64 = target_f64[::subsample, ::subsample, ::subsample]
        region_mask_sub = region_mask[::subsample, ::subsample, ::subsample]
        spacing_mm = tuple(s * subsample for s in spacing_mm)
    else:
        region_mask_sub = region_mask

    # Compute full gamma map
    axes = tuple(
        np.arange(s) * sp for s, sp in zip(pred_f64.shape, spacing_mm)
    )

    try:
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_f64,
            axes_evaluation=axes,
            dose_evaluation=pred_f64,
            dose_percent_threshold=dose_threshold_pct,
            distance_mm_threshold=distance_mm,
            lower_percent_dose_cutoff=lower_dose_cutoff_pct,
            max_gamma=max_gamma,
        )

        # Extract gamma values within mask where gamma is valid
        valid_in_region = region_mask_sub & np.isfinite(gamma_map)
        if not valid_in_region.any():
            return {
                'gamma_pass_rate': None,
                'gamma_mean': None,
                'gamma_median': None,
                'gamma_max': None,
                'voxels_evaluated': 0,
                'note': 'No valid gamma voxels in region',
            }

        gamma_region = gamma_map[valid_in_region]
        return {
            'gamma_pass_rate': float(np.mean(gamma_region <= 1.0) * 100),
            'gamma_mean': float(np.mean(gamma_region)),
            'gamma_median': float(np.median(gamma_region)),
            'gamma_max': float(np.max(gamma_region)),
            'voxels_evaluated': int(valid_in_region.sum()),
        }

    except Exception as e:
        return {'gamma_pass_rate': None, 'error': str(e)}


def compute_ptv_region_gamma(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    masks: np.ndarray,
    spacing_mm: Tuple[float, ...] = DEFAULT_SPACING_MM,
    margin_mm: float = 5.0,
    subsample: int = 2,
    **kwargs,
) -> Dict:
    """
    Compute gamma within PTV + margin region.

    Dilates the combined PTV mask by margin_mm using scipy binary_dilation,
    then computes gamma within that expanded region.

    Args:
        pred_gy: Predicted dose in Gy
        target_gy: Ground truth dose in Gy
        masks: (8, Y, X, Z) binary masks
        spacing_mm: Voxel spacing in mm
        margin_mm: Expansion margin in mm
        subsample: Subsample factor
        **kwargs: Additional args passed to compute_region_gamma

    Returns:
        Dict with PTV-region gamma metrics
    """
    from scipy.ndimage import binary_dilation

    # Combine PTV masks
    ptv70_idx = STRUCTURE_INDEX['PTV70']
    ptv56_idx = STRUCTURE_INDEX['PTV56']

    ptv_mask = np.zeros(masks.shape[1:], dtype=bool)
    if ptv70_idx < masks.shape[0]:
        ptv_mask |= (masks[ptv70_idx] > 0)
    if ptv56_idx < masks.shape[0]:
        ptv_mask |= (masks[ptv56_idx] > 0)

    if not ptv_mask.any():
        return {'gamma_pass_rate': None, 'error': 'no PTV voxels',
                'voxels_evaluated': 0}

    # Compute dilation iterations per axis based on spacing
    iterations_per_axis = [
        max(1, int(round(margin_mm / sp))) for sp in spacing_mm
    ]
    # Use the max to create a roughly spherical expansion
    # (binary_dilation with a ball structure element would be more precise,
    # but this is a reasonable approximation)
    struct = np.ones((
        2 * iterations_per_axis[0] + 1,
        2 * iterations_per_axis[1] + 1,
        2 * iterations_per_axis[2] + 1,
    ), dtype=bool)
    ptv_expanded = binary_dilation(ptv_mask, structure=struct, iterations=1)

    return compute_region_gamma(
        pred_gy, target_gy, ptv_expanded,
        spacing_mm=spacing_mm, subsample=subsample, **kwargs
    )


# =============================================================================
# DVH Metrics
# =============================================================================

def compute_dvh_metrics(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    masks: np.ndarray,
    spacing_mm: Tuple[float, ...] = DEFAULT_SPACING_MM,
    rx_dose_gy: float = PRIMARY_PRESCRIPTION_GY,
) -> Dict[str, Dict]:
    """
    Compute comprehensive DVH metrics for each structure.

    For each structure, computes:
        - Dx: D98, D95, D50, D5, D2 using np.percentile(method='lower')
        - Statistics: Dmean, Dmax, Dmin
        - Vx: V70, V60, V50, V45, V40 (as % of structure volume)
        - V45_cc for Bowel (absolute volume in cc)
        - V95 for PTVs (% volume receiving >= 95% of structure's Rx)
        - Error terms: pred - GT for each metric
        - dvh_reliable flag: False if < MIN_VOXELS_RELIABLE voxels

    Args:
        pred_gy: Predicted dose in Gy
        target_gy: Ground truth dose in Gy
        masks: (8, Y, X, Z) binary masks
        spacing_mm: Voxel spacing in mm
        rx_dose_gy: Primary prescription dose for normalization context

    Returns:
        Dict mapping structure name to metrics dict
    """
    voxel_vol_cc = float(np.prod(spacing_mm)) / 1000.0  # mm^3 to cc

    results = {}

    for ch, name in STRUCTURE_CHANNELS.items():
        if ch >= masks.shape[0]:
            continue

        mask = masks[ch] > 0
        if not mask.any():
            results[name] = {'exists': False}
            continue

        pred_struct = pred_gy[mask]
        target_struct = target_gy[mask]
        n_voxels = len(pred_struct)
        dvh_reliable = n_voxels >= MIN_VOXELS_RELIABLE

        metrics = {
            'exists': True,
            'n_voxels': n_voxels,
            'dvh_reliable': dvh_reliable,
            'volume_cc': float(n_voxels * voxel_vol_cc),
        }

        if not dvh_reliable:
            warnings.warn(
                f"Structure {name} has only {n_voxels} voxels "
                f"(< {MIN_VOXELS_RELIABLE}); DVH metrics may be unreliable"
            )

        # --- Basic statistics ---
        metrics['pred_mean_gy'] = float(pred_struct.mean())
        metrics['target_mean_gy'] = float(target_struct.mean())
        metrics['pred_max_gy'] = float(pred_struct.max())
        metrics['target_max_gy'] = float(target_struct.max())
        metrics['pred_min_gy'] = float(pred_struct.min())
        metrics['target_min_gy'] = float(target_struct.min())

        # --- Dx metrics (dose to x% of volume) ---
        # D95 = dose received by at least 95% of volume = 5th percentile
        # Using method='lower' (nearest-rank) to match TPS convention
        for pct in [98, 95, 50, 5, 2]:
            percentile_value = 100 - pct  # D95 -> 5th percentile
            pred_dx = float(np.percentile(pred_struct, percentile_value, method='lower'))
            target_dx = float(np.percentile(target_struct, percentile_value, method='lower'))
            metrics[f'pred_D{pct}'] = pred_dx
            metrics[f'target_D{pct}'] = target_dx
            metrics[f'D{pct}_error'] = round(pred_dx - target_dx, 4)

        # --- MAE within structure ---
        metrics['mae_gy'] = float(np.mean(np.abs(pred_struct - target_struct)))

        # --- Vx metrics (volume receiving >= x Gy, as % of structure) ---
        for dose_level in [70, 60, 50, 45, 40]:
            pred_vx = float((pred_struct >= dose_level).sum() / n_voxels * 100)
            target_vx = float((target_struct >= dose_level).sum() / n_voxels * 100)
            metrics[f'pred_V{dose_level}'] = pred_vx
            metrics[f'target_V{dose_level}'] = target_vx
            metrics[f'V{dose_level}_error'] = round(pred_vx - target_vx, 4)

        # --- V45_cc for Bowel (absolute volume in cc) ---
        if name == 'Bowel':
            pred_v45_cc = float((pred_struct >= 45.0).sum() * voxel_vol_cc)
            target_v45_cc = float((target_struct >= 45.0).sum() * voxel_vol_cc)
            metrics['pred_V45_cc'] = pred_v45_cc
            metrics['target_V45_cc'] = target_v45_cc
            metrics['V45_cc_error'] = round(pred_v45_cc - target_v45_cc, 4)

        # --- V95 for PTVs (% volume receiving >= 95% of structure's Rx) ---
        if name in PTV_STRUCTURES:
            struct_rx = (PRIMARY_PRESCRIPTION_GY if name == 'PTV70'
                         else SECONDARY_PRESCRIPTION_GY)
            threshold_95 = 0.95 * struct_rx
            pred_v95 = float((pred_struct >= threshold_95).sum() / n_voxels * 100)
            target_v95 = float((target_struct >= threshold_95).sum() / n_voxels * 100)
            metrics['pred_V95'] = pred_v95
            metrics['target_V95'] = target_v95
            metrics['V95_error'] = round(pred_v95 - target_v95, 4)

        results[name] = metrics

    return results


# =============================================================================
# Primary Entry Point
# =============================================================================

def evaluate_case(
    pred_normalized: np.ndarray,
    target_normalized: np.ndarray,
    masks: np.ndarray,
    spacing_mm: Tuple[float, ...],
    case_id: str = "unknown",
    rx_dose_gy: float = PRIMARY_PRESCRIPTION_GY,
    compute_gamma_metric: bool = True,
    gamma_subsample: int = 2,
    compute_ptv_gamma: bool = True,
) -> EvaluationResult:
    """
    Primary evaluation entry point — runs all metrics for a single case.

    Args:
        pred_normalized: Predicted dose, normalized [0, 1]
        target_normalized: Ground truth dose, normalized [0, 1]
        masks: (8, Y, X, Z) binary masks
        spacing_mm: Voxel spacing (y, x, z) in mm
        case_id: Case identifier
        rx_dose_gy: Prescription dose in Gy
        compute_gamma_metric: Whether to compute gamma (requires pymedphys)
        gamma_subsample: Subsample factor for gamma (2 for publication, 4 for training)
        compute_ptv_gamma: Whether to compute PTV-region gamma

    Returns:
        EvaluationResult with all metrics populated
    """
    result = EvaluationResult(
        case_id=case_id,
        spacing_mm=tuple(spacing_mm),
    )

    # Denormalize to Gy
    pred_gy = denormalize_dose(pred_normalized, rx_dose_gy)
    target_gy = denormalize_dose(target_normalized, rx_dose_gy)

    # Validate inputs
    result.validation_warnings.extend(validate_dose_array(pred_gy, "predicted"))
    result.validation_warnings.extend(validate_dose_array(target_gy, "target"))

    # Sanitize (clamp negatives, replace NaN/Inf)
    pred_gy = sanitize_dose(pred_gy)
    target_gy = sanitize_dose(target_gy)

    # 1. Dose metrics
    result.dose_metrics = compute_dose_metrics(pred_gy, target_gy)
    result.dose_metrics['per_structure_mae'] = compute_per_structure_mae(
        pred_gy, target_gy, masks
    )

    # 2. Global gamma
    if compute_gamma_metric:
        result.gamma['global_3mm3pct'] = compute_gamma(
            pred_gy, target_gy,
            spacing_mm=spacing_mm,
            subsample=gamma_subsample,
        )

        # 3. PTV-region gamma
        if compute_ptv_gamma:
            result.gamma['ptv_region_3mm3pct'] = compute_ptv_region_gamma(
                pred_gy, target_gy, masks,
                spacing_mm=spacing_mm,
                subsample=gamma_subsample,
            )

    # 4. DVH metrics
    result.dvh_metrics = compute_dvh_metrics(
        pred_gy, target_gy, masks,
        spacing_mm=spacing_mm,
        rx_dose_gy=rx_dose_gy,
    )

    # 5. Clinical constraints
    result.clinical_constraints = check_clinical_constraints(result.dvh_metrics)

    return result
