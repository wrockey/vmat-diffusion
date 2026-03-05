"""
Anatomical variability analysis: characterize dataset features and correlate
with prediction error to identify drivers of outlier cases.

This is an analysis experiment (no GPU needed). It extracts ~38 anatomical
features per case from NPZ files and DICOM RTSTRUCT, loads evaluation results
from baseline and combined_loss_2.5to1 experiments (3 seeds each), computes
Spearman correlations, cross-seed ICC, and train-vs-test distribution tests.

Usage:
    python scripts/analyze_anatomical_variability.py \
        --data_dir /home/wrockey/data/processed_npz \
        --dicom_dir /home/wrockey/data/anonymized_dicom

Output:
    runs/anatomical_variability/features_all_74_cases.csv
    runs/anatomical_variability/analysis_results.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_core import (
    STRUCTURE_CHANNELS,
    STRUCTURE_INDEX,
    OAR_STRUCTURES,
    PTV_STRUCTURES,
    ALL_STRUCTURES,
    PRIMARY_PRESCRIPTION_GY,
    DEFAULT_SPACING_MM,
    get_spacing_from_metadata,
)
from eval_statistics import NumpyEncoder

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SDF clip distance used during preprocessing
SDF_CLIP_MM = 50.0

# Test case IDs (consistent across all experiments)
TEST_CASE_IDS = [
    'prostate70gy_0005', 'prostate70gy_0018', 'prostate70gy_0024',
    'prostate70gy_0027', 'prostate70gy_0056', 'prostate70gy_0065',
    'prostate70gy_0079',
]

# Experiments to load for error correlation
EXPERIMENTS = {
    'baseline_v23': {
        'seeds': [42, 123, 456],
        'pred_dir_template': 'predictions/baseline_v23_seed{seed}_test',
        'eval_file': 'baseline_evaluation_results.json',
    },
    'combined_loss_2.5to1': {
        'seeds': [42, 123, 456],
        'pred_dir_template': 'predictions/combined_loss_2.5to1_seed{seed}_test',
        'eval_file': 'baseline_evaluation_results.json',
    },
}


# =============================================================================
# PTV target detection from DICOM RTSTRUCT
# =============================================================================

# Regex patterns for PTV dose levels
PTV_PATTERNS = {
    'ptv7000': re.compile(r'(?i)\bptv[_\s]?70'),
    'ptv5600': re.compile(r'(?i)\bptv[_\s]?56'),
    'ptv5040': re.compile(r'(?i)\bptv[_\s]?50[_\s.]?4'),
    'ptv6160': re.compile(r'(?i)\bptv[_\s]?6160'),
}

# Exclusion patterns for non-target structures
EXCLUDE_PATTERNS = re.compile(
    r'(?i)(^ring|^anti|^xxx|\+\d+mm|_export$|^ptvs\+)'
)


def detect_ptv_targets(rtstruct_path: str) -> Dict[str, Any]:
    """Scan RTSTRUCT for PTV dose level targets.

    Returns dict with ptv7000_present, ptv5600_present, ptv5040_present,
    ptv6160_present (booleans) and matched_roi_names for debugging.
    """
    import pydicom

    result = {
        'ptv7000_present': False,
        'ptv5600_present': False,
        'ptv5040_present': False,
        'ptv6160_present': False,
        'matched_roi_names': {},
    }

    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        roi_names = [roi.ROIName for roi in ds.StructureSetROISequence]
    except Exception as e:
        result['dicom_error'] = str(e)
        return result

    for roi_name in roi_names:
        # Skip non-target structures
        if EXCLUDE_PATTERNS.search(roi_name):
            continue

        for ptv_key, pattern in PTV_PATTERNS.items():
            if pattern.search(roi_name):
                result[f'{ptv_key}_present'] = True
                if ptv_key not in result['matched_roi_names']:
                    result['matched_roi_names'][ptv_key] = []
                result['matched_roi_names'][ptv_key].append(roi_name)

    return result


def scan_all_dicom_ptv_targets(dicom_dir: str) -> Dict[str, Dict]:
    """Scan all DICOM directories for PTV targets.

    Returns {case_id: ptv_result_dict}.
    """
    dicom_path = Path(dicom_dir)
    results = {}

    for case_dir in sorted(dicom_path.iterdir()):
        if not case_dir.is_dir():
            continue
        rtstruct = case_dir / 'RTSTRUCT.dcm'
        if not rtstruct.exists():
            continue

        case_id = case_dir.name  # e.g., prostate70gy_XXXX
        results[case_id] = detect_ptv_targets(str(rtstruct))

    return results


# =============================================================================
# Feature extraction from NPZ files
# =============================================================================

def extract_features_single_case(npz_path: str) -> Dict[str, Any]:
    """Extract anatomical features from a single NPZ file.

    Loads one case at a time to avoid OOM.
    """
    data = np.load(npz_path, allow_pickle=True)
    case_id = Path(npz_path).stem  # e.g., prostate70gy_XXXX

    # Extract arrays
    ct = data['ct']                    # (Y, X, Z)
    dose = data['dose']                # (Y, X, Z) normalized [0,1]
    masks = data['masks']              # (8, Y, X, Z) uint8
    masks_sdf = data['masks_sdf']      # (8, Y, X, Z) float32

    # Metadata for spacing
    metadata = data['metadata'].item() if 'metadata' in data else {}
    spacing_mm = get_spacing_from_metadata(metadata)
    voxel_vol_cc = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0

    features: Dict[str, Any] = {'case_id': case_id}

    # --- Structure volumes and existence ---
    for ch_idx, struct_name in STRUCTURE_CHANNELS.items():
        mask = masks[ch_idx]
        n_voxels = int(mask.sum())
        features[f'{struct_name}_exists'] = n_voxels > 0
        features[f'{struct_name}_volume_cc'] = n_voxels * voxel_vol_cc

    # --- Spatial extent ---
    volume_shape = ct.shape
    features['volume_z_slices'] = volume_shape[2]
    features['z_extent_mm'] = volume_shape[2] * spacing_mm[2]

    # PTV70 spatial characteristics
    ptv70_mask = masks[STRUCTURE_INDEX['PTV70']]
    if ptv70_mask.sum() > 0:
        ptv70_coords = np.argwhere(ptv70_mask > 0)
        ptv70_min = ptv70_coords.min(axis=0)
        ptv70_max = ptv70_coords.max(axis=0)
        ptv70_extent = (ptv70_max - ptv70_min + 1).astype(float)
        # Convert to mm
        ptv70_extent_mm = ptv70_extent * np.array(spacing_mm)
        features['ptv70_y_extent_mm'] = ptv70_extent_mm[0]
        features['ptv70_x_extent_mm'] = ptv70_extent_mm[1]
        features['ptv70_z_extent_mm'] = ptv70_extent_mm[2]
        # Aspect ratio: lateral extent / AP extent
        features['ptv70_aspect_ratio'] = ptv70_extent_mm[1] / max(ptv70_extent_mm[0], 1e-6)
    else:
        features['ptv70_y_extent_mm'] = 0.0
        features['ptv70_x_extent_mm'] = 0.0
        features['ptv70_z_extent_mm'] = 0.0
        features['ptv70_aspect_ratio'] = 0.0

    # --- PTV-OAR proximity (from SDF) ---
    ptv70_sdf = masks_sdf[STRUCTURE_INDEX['PTV70']]
    ptv70_inside = ptv70_sdf < 0  # Inside PTV70

    for oar_name in ['Rectum', 'Bladder', 'Bowel']:
        oar_idx = STRUCTURE_INDEX[oar_name]
        oar_sdf = masks_sdf[oar_idx]

        if ptv70_inside.sum() > 0 and masks[oar_idx].sum() > 0:
            # SDF values at PTV70 voxels — negative means inside OAR
            oar_sdf_at_ptv = oar_sdf[ptv70_inside]
            # Convert SDF from [-1, 1] to mm using clip distance
            oar_dist_mm_at_ptv = oar_sdf_at_ptv * SDF_CLIP_MM
            features[f'ptv70_to_{oar_name}_min_dist_mm'] = float(oar_dist_mm_at_ptv.min())
            features[f'ptv70_to_{oar_name}_mean_dist_mm'] = float(oar_dist_mm_at_ptv.mean())

            # Overlap: fraction of PTV70 voxels inside OAR (SDF < 0)
            overlap_count = (oar_sdf_at_ptv < 0).sum()
            features[f'ptv70_{oar_name}_overlap_pct'] = float(
                100.0 * overlap_count / ptv70_inside.sum()
            )
        else:
            features[f'ptv70_to_{oar_name}_min_dist_mm'] = np.nan
            features[f'ptv70_to_{oar_name}_mean_dist_mm'] = np.nan
            features[f'ptv70_{oar_name}_overlap_pct'] = 0.0

    # --- Dose complexity (ground truth) ---
    dose_gy = dose * PRIMARY_PRESCRIPTION_GY
    features['dose_mean_gy'] = float(dose_gy.mean())
    features['dose_std_gy'] = float(dose_gy.std())
    features['dose_max_gy'] = float(dose_gy.max())

    # High dose volume: fraction of voxels > 50% of Rx
    features['dose_high_volume_pct'] = float(
        100.0 * (dose_gy > 0.5 * PRIMARY_PRESCRIPTION_GY).sum() / dose_gy.size
    )

    # Conformity index: V_Rx / V_PTV70 (loose approximation)
    rx_volume = (dose_gy >= 0.95 * PRIMARY_PRESCRIPTION_GY).sum()
    ptv70_volume = ptv70_mask.sum()
    if ptv70_volume > 0:
        features['conformity_index'] = float(rx_volume / ptv70_volume)
    else:
        features['conformity_index'] = np.nan

    # Dose gradient steepness: mean gradient magnitude in PTV boundary region
    # Use PTV70 SDF between -0.1 and 0.1 (boundary region)
    boundary_mask = (np.abs(ptv70_sdf) < 0.1)
    if boundary_mask.sum() > 100:
        # Compute gradient magnitude of dose
        grad = np.gradient(dose_gy, *spacing_mm)
        grad_mag = np.sqrt(sum(g**2 for g in grad))
        features['ptv70_boundary_grad_gy_mm'] = float(grad_mag[boundary_mask].mean())
    else:
        features['ptv70_boundary_grad_gy_mm'] = np.nan

    # --- Voxel spacing ---
    features['spacing_y_mm'] = spacing_mm[0]
    features['spacing_x_mm'] = spacing_mm[1]
    features['spacing_z_mm'] = spacing_mm[2]

    return features


def extract_all_features(data_dir: str, dicom_dir: Optional[str] = None) -> pd.DataFrame:
    """Extract features from all NPZ files and merge DICOM PTV targets."""
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob('prostate70gy_*.npz'))

    print(f"Found {len(npz_files)} NPZ files in {data_dir}")

    # Extract features from each NPZ
    all_features = []
    for i, npz_path in enumerate(npz_files):
        print(f"  [{i+1}/{len(npz_files)}] {npz_path.stem}...", end='', flush=True)
        feats = extract_features_single_case(str(npz_path))
        all_features.append(feats)
        print(" done")

    df = pd.DataFrame(all_features)

    # Merge DICOM PTV targets if available
    if dicom_dir and Path(dicom_dir).exists():
        print(f"\nScanning DICOM RTSTRUCT files in {dicom_dir}...")
        ptv_results = scan_all_dicom_ptv_targets(dicom_dir)

        # Merge by case_id
        ptv_rows = []
        for case_id, ptv_info in ptv_results.items():
            ptv_rows.append({
                'case_id': case_id,
                'ptv7000_present': ptv_info['ptv7000_present'],
                'ptv5600_present': ptv_info['ptv5600_present'],
                'ptv5040_present': ptv_info['ptv5040_present'],
                'ptv6160_present': ptv_info['ptv6160_present'],
            })
        ptv_df = pd.DataFrame(ptv_rows)

        # Merge — only cases that exist in NPZ
        df = df.merge(ptv_df, on='case_id', how='left')
        print(f"  Merged PTV targets for {ptv_df.shape[0]} DICOM cases "
              f"({df['ptv5040_present'].sum()} have PTV5040)")
    else:
        print("No DICOM directory provided — skipping PTV target detection")

    # Add train/test split indicator
    df['is_test'] = df['case_id'].isin(TEST_CASE_IDS)

    return df


# =============================================================================
# Load evaluation results
# =============================================================================

def load_evaluation_results(project_root: Path) -> Dict[str, Dict]:
    """Load per-case evaluation results from baseline and combined loss experiments.

    Returns nested dict: {experiment: {seed: {case_id: metrics_dict}}}.
    """
    results = {}

    for exp_name, exp_config in EXPERIMENTS.items():
        results[exp_name] = {}
        for seed in exp_config['seeds']:
            pred_dir = project_root / exp_config['pred_dir_template'].format(seed=seed)
            eval_path = pred_dir / exp_config['eval_file']

            if not eval_path.exists():
                print(f"  WARNING: {eval_path} not found, skipping")
                continue

            with open(eval_path) as f:
                eval_data = json.load(f)

            seed_results = {}
            for case_result in eval_data['per_case_results']:
                case_id = case_result['case_id']
                seed_results[case_id] = {
                    'mae_gy': case_result['dose_metrics']['mae_gy'],
                    'rmse_gy': case_result['dose_metrics']['rmse_gy'],
                    'max_error_gy': case_result['dose_metrics']['max_error_gy'],
                    'global_gamma': case_result['gamma']['global_3mm3pct']['gamma_pass_rate'],
                    'ptv_gamma': case_result['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'],
                }
                # DVH metrics (may not exist for all structures)
                for struct in ['PTV70', 'PTV56', 'Rectum', 'Bladder']:
                    if struct in case_result['dvh_metrics']:
                        dvh = case_result['dvh_metrics'][struct]
                        if dvh.get('exists', False) and dvh.get('dvh_reliable', False):
                            seed_results[case_id][f'{struct}_D95_error'] = dvh.get('D95_error', np.nan)
                            seed_results[case_id][f'{struct}_Dmean_error'] = (
                                dvh.get('pred_mean_gy', 0) - dvh.get('target_mean_gy', 0)
                            )

            results[exp_name][seed] = seed_results

    return results


def build_error_dataframe(eval_results: Dict) -> pd.DataFrame:
    """Build a tidy dataframe of error metrics: one row per (case, experiment, seed)."""
    rows = []
    for exp_name, seeds_data in eval_results.items():
        for seed, cases_data in seeds_data.items():
            for case_id, metrics in cases_data.items():
                row = {'case_id': case_id, 'experiment': exp_name, 'seed': seed}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Correlation analysis
# =============================================================================

def compute_spearman_correlations(
    features_df: pd.DataFrame,
    error_df: pd.DataFrame,
    experiment: str = 'combined_loss_2.5to1',
) -> Dict[str, Dict]:
    """Compute Spearman rank correlations between features and seed-averaged errors.

    Uses test-set cases only (n=7). Reports rho and p-value.
    """
    # Seed-averaged errors per case
    exp_errors = error_df[error_df['experiment'] == experiment]
    avg_errors = exp_errors.groupby('case_id').agg({
        'mae_gy': 'mean',
        'ptv_gamma': 'mean',
        'global_gamma': 'mean',
        'PTV70_D95_error': 'mean',
    }).reset_index()

    # Merge with features (test cases only)
    test_features = features_df[features_df['is_test']].copy()
    merged = test_features.merge(avg_errors, on='case_id', suffixes=('_feat', '_err'))

    # Select numeric feature columns
    error_cols = ['mae_gy', 'ptv_gamma', 'global_gamma', 'PTV70_D95_error']
    exclude = {'case_id', 'is_test'} | set(error_cols)
    feature_cols = [c for c in merged.columns
                    if c not in exclude
                    and merged[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')]

    correlations = {}
    for feat in feature_cols:
        correlations[feat] = {}
        for err_metric in error_cols:
            x = merged[feat].values.astype(float)
            y = merged[err_metric].values.astype(float)

            # Drop NaN pairs
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 4:
                correlations[feat][err_metric] = {'rho': np.nan, 'p_value': np.nan, 'n': int(valid.sum())}
                continue

            rho, p = stats.spearmanr(x[valid], y[valid])
            correlations[feat][err_metric] = {
                'rho': float(rho),
                'p_value': float(p),
                'n': int(valid.sum()),
            }

    return correlations


def compute_cross_seed_icc(error_df: pd.DataFrame, experiment: str) -> Dict[str, Dict]:
    """Compute ICC(3,1) across seeds for each error metric.

    ICC quantifies how much variance is due to case (anatomy) vs seed (randomness).
    """
    exp_data = error_df[error_df['experiment'] == experiment].copy()
    metrics = ['mae_gy', 'ptv_gamma', 'global_gamma']

    icc_results = {}
    for metric in metrics:
        # Pivot: rows=cases, cols=seeds
        pivot = exp_data.pivot(index='case_id', columns='seed', values=metric).dropna()
        if pivot.shape[1] < 2:
            icc_results[metric] = {'icc': np.nan, 'n_cases': 0, 'n_seeds': 0}
            continue

        n = pivot.shape[0]  # cases
        k = pivot.shape[1]  # seeds
        data = pivot.values

        # Two-way ANOVA components for ICC(3,1)
        grand_mean = data.mean()
        row_means = data.mean(axis=1)
        col_means = data.mean(axis=0)

        ss_row = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
        ss_col = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_error = ss_total - ss_row - ss_col

        ms_row = ss_row / max(n - 1, 1)
        ms_error = ss_error / max((n - 1) * (k - 1), 1)

        # ICC(3,1): (MS_row - MS_error) / (MS_row + (k-1)*MS_error)
        denom = ms_row + (k - 1) * ms_error
        icc = (ms_row - ms_error) / denom if denom > 0 else np.nan

        icc_results[metric] = {
            'icc': float(icc),
            'n_cases': int(n),
            'n_seeds': int(k),
            'case_variance_pct': float(100.0 * ss_row / max(ss_total, 1e-10)),
            'seed_variance_pct': float(100.0 * ss_col / max(ss_total, 1e-10)),
        }

    return icc_results


def compute_train_test_distribution_tests(features_df: pd.DataFrame) -> Dict[str, Dict]:
    """KS test comparing train vs test distributions for key features."""
    train = features_df[~features_df['is_test']]
    test = features_df[features_df['is_test']]

    numeric_cols = [c for c in features_df.columns
                    if c not in ('case_id', 'is_test')
                    and features_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    results = {}
    for col in numeric_cols:
        train_vals = train[col].dropna().values.astype(float)
        test_vals = test[col].dropna().values.astype(float)

        if len(train_vals) < 3 or len(test_vals) < 3:
            continue

        ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
        results[col] = {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'train_mean': float(train_vals.mean()),
            'train_std': float(train_vals.std()),
            'test_mean': float(test_vals.mean()),
            'test_std': float(test_vals.std()),
            'n_train': len(train_vals),
            'n_test': len(test_vals),
        }

    # Categorical: PTV target presence (Fisher's exact)
    for ptv_col in ['ptv5040_present', 'ptv5600_present', 'ptv6160_present']:
        if ptv_col not in features_df.columns:
            continue
        train_pos = train[ptv_col].sum()
        train_neg = len(train) - train_pos
        test_pos = test[ptv_col].sum()
        test_neg = len(test) - test_pos

        table = np.array([[train_pos, train_neg], [test_pos, test_neg]])
        try:
            _, fisher_p = stats.fisher_exact(table)
        except ValueError:
            fisher_p = np.nan

        results[ptv_col] = {
            'test_type': 'fisher_exact',
            'fisher_p_value': float(fisher_p),
            'train_prevalence': float(train_pos / max(len(train), 1)),
            'test_prevalence': float(test_pos / max(len(test), 1)),
            'n_train': int(len(train)),
            'n_test': int(len(test)),
        }

    return results


def compute_ptv_target_stratification(
    features_df: pd.DataFrame,
    error_df: pd.DataFrame,
    experiment: str = 'combined_loss_2.5to1',
) -> Dict[str, Dict]:
    """Compare error metrics grouped by PTV target presence (test set only)."""
    # Seed-averaged errors
    exp_errors = error_df[error_df['experiment'] == experiment]
    avg_errors = exp_errors.groupby('case_id').agg({
        'mae_gy': 'mean',
        'ptv_gamma': 'mean',
    }).reset_index()

    test_features = features_df[features_df['is_test']].copy()
    merged = test_features.merge(avg_errors, on='case_id')

    results = {}
    for ptv_col in ['ptv5040_present', 'ptv5600_present', 'ptv6160_present']:
        if ptv_col not in merged.columns:
            continue

        group_yes = merged[merged[ptv_col] == True]
        group_no = merged[merged[ptv_col] == False]

        strat = {
            'n_yes': len(group_yes),
            'n_no': len(group_no),
            'cases_yes': group_yes['case_id'].tolist(),
            'cases_no': group_no['case_id'].tolist(),
        }

        for metric in ['mae_gy', 'ptv_gamma']:
            vals_yes = group_yes[metric].values
            vals_no = group_no[metric].values

            strat[f'{metric}_yes_mean'] = float(vals_yes.mean()) if len(vals_yes) > 0 else np.nan
            strat[f'{metric}_no_mean'] = float(vals_no.mean()) if len(vals_no) > 0 else np.nan

            # Mann-Whitney U if both groups have >= 2 observations
            if len(vals_yes) >= 2 and len(vals_no) >= 2:
                try:
                    u_stat, u_p = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
                    strat[f'{metric}_mannwhitney_U'] = float(u_stat)
                    strat[f'{metric}_mannwhitney_p'] = float(u_p)
                except ValueError:
                    strat[f'{metric}_mannwhitney_U'] = np.nan
                    strat[f'{metric}_mannwhitney_p'] = np.nan
            else:
                strat[f'{metric}_mannwhitney_U'] = np.nan
                strat[f'{metric}_mannwhitney_p'] = np.nan

        results[ptv_col] = strat

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Anatomical variability analysis')
    parser.add_argument('--data_dir', type=str,
                        default='/home/wrockey/data/processed_npz',
                        help='Path to processed NPZ directory')
    parser.add_argument('--dicom_dir', type=str,
                        default='/home/wrockey/data/anonymized_dicom',
                        help='Path to anonymized DICOM directory')
    parser.add_argument('--output_dir', type=str,
                        default='runs/anatomical_variability',
                        help='Output directory for results')
    args = parser.parse_args()

    # Resolve output dir relative to project root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANATOMICAL VARIABILITY ANALYSIS")
    print("=" * 70)
    print(f"Data dir:   {args.data_dir}")
    print(f"DICOM dir:  {args.dicom_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print()

    # Step 1: Extract features
    print("Step 1: Extracting anatomical features...")
    features_df = extract_all_features(args.data_dir, args.dicom_dir)
    print(f"  → {features_df.shape[0]} cases, {features_df.shape[1]} features")

    # Save features CSV
    csv_path = output_dir / f'features_all_{features_df.shape[0]}_cases.csv'
    features_df.to_csv(csv_path, index=False)
    print(f"  → Saved to {csv_path}")

    # Step 2: Load evaluation results
    print("\nStep 2: Loading evaluation results...")
    eval_results = load_evaluation_results(_PROJECT_ROOT)
    error_df = build_error_dataframe(eval_results)
    n_obs = len(error_df)
    print(f"  → {n_obs} observations (case × experiment × seed)")

    # Step 3: Spearman correlations (test set)
    print("\nStep 3: Computing Spearman correlations (test set, n=7)...")
    correlations = {}
    for exp_name in EXPERIMENTS:
        correlations[exp_name] = compute_spearman_correlations(
            features_df, error_df, experiment=exp_name
        )

    # Find top correlations
    top_corrs = []
    for feat, metrics in correlations['combined_loss_2.5to1'].items():
        for metric, vals in metrics.items():
            if not np.isnan(vals['rho']):
                top_corrs.append({
                    'feature': feat,
                    'metric': metric,
                    'rho': vals['rho'],
                    'p_value': vals['p_value'],
                    'abs_rho': abs(vals['rho']),
                })
    top_corrs.sort(key=lambda x: x['abs_rho'], reverse=True)

    print("  Top 10 correlations (combined_loss_2.5to1):")
    for i, tc in enumerate(top_corrs[:10]):
        sig = '*' if tc['p_value'] < 0.05 else ''
        print(f"    {i+1}. {tc['feature']} vs {tc['metric']}: "
              f"rho={tc['rho']:.3f} (p={tc['p_value']:.3f}){sig}")

    # Step 4: Cross-seed ICC
    print("\nStep 4: Computing cross-seed ICC...")
    icc_results = {}
    for exp_name in EXPERIMENTS:
        icc_results[exp_name] = compute_cross_seed_icc(error_df, exp_name)
        for metric, vals in icc_results[exp_name].items():
            print(f"  {exp_name} {metric}: ICC={vals['icc']:.3f}, "
                  f"case variance={vals.get('case_variance_pct', 0):.1f}%")

    # Step 5: Train vs test distribution comparison
    print("\nStep 5: Train vs test distribution comparison...")
    dist_tests = compute_train_test_distribution_tests(features_df)
    significant = {k: v for k, v in dist_tests.items()
                   if v.get('ks_p_value', v.get('fisher_p_value', 1.0)) < 0.05}
    print(f"  → {len(significant)} features with p < 0.05")
    for feat, vals in significant.items():
        p = vals.get('ks_p_value', vals.get('fisher_p_value', np.nan))
        print(f"    {feat}: p={p:.4f}")

    # Step 6: PTV target stratification
    print("\nStep 6: PTV target stratification (test set)...")
    ptv_strat = compute_ptv_target_stratification(features_df, error_df)
    for ptv_col, strat in ptv_strat.items():
        print(f"  {ptv_col}: n_yes={strat['n_yes']}, n_no={strat['n_no']}")
        if strat['n_yes'] > 0 and strat['n_no'] > 0:
            print(f"    MAE: yes={strat.get('mae_gy_yes_mean', 'N/A'):.2f}, "
                  f"no={strat.get('mae_gy_no_mean', 'N/A'):.2f}")

    # Step 7: Build per-case error profile for test cases
    print("\nStep 7: Building per-case outlier profiles...")
    test_features = features_df[features_df['is_test']].copy()
    outlier_profiles = {}
    for _, row in test_features.iterrows():
        case_id = row['case_id']
        # Get seed-averaged errors
        case_errors = error_df[
            (error_df['case_id'] == case_id)
            & (error_df['experiment'] == 'combined_loss_2.5to1')
        ]
        if len(case_errors) > 0:
            outlier_profiles[case_id] = {
                'mae_gy_mean': float(case_errors['mae_gy'].mean()),
                'mae_gy_std': float(case_errors['mae_gy'].std()),
                'ptv_gamma_mean': float(case_errors['ptv_gamma'].mean()),
                'ptv_gamma_std': float(case_errors['ptv_gamma'].std()),
                'global_gamma_mean': float(case_errors['global_gamma'].mean()),
                'global_gamma_std': float(case_errors['global_gamma'].std()),
            }
            # Add key anatomical features
            for feat in ['PTV70_volume_cc', 'Rectum_volume_cc', 'Bladder_volume_cc',
                         'ptv70_to_Rectum_min_dist_mm', 'ptv70_Rectum_overlap_pct',
                         'dose_high_volume_pct', 'conformity_index', 'z_extent_mm']:
                outlier_profiles[case_id][feat] = float(row.get(feat, np.nan))

    print("  Per-case profiles (combined_loss_2.5to1, seed-averaged):")
    for case_id in sorted(outlier_profiles, key=lambda x: outlier_profiles[x]['mae_gy_mean'], reverse=True):
        p = outlier_profiles[case_id]
        print(f"    {case_id}: MAE={p['mae_gy_mean']:.2f}±{p['mae_gy_std']:.2f}, "
              f"PTV_gamma={p['ptv_gamma_mean']:.1f}±{p['ptv_gamma_std']:.1f}%")

    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_cases_total': int(features_df.shape[0]),
        'n_cases_test': int(features_df['is_test'].sum()),
        'n_features': int(features_df.shape[1]),
        'test_case_ids': TEST_CASE_IDS,
        'correlations': correlations,
        'top_correlations_combined_loss': top_corrs[:20],
        'icc_results': icc_results,
        'train_test_distribution_tests': dist_tests,
        'ptv_target_stratification': ptv_strat,
        'outlier_profiles': outlier_profiles,
        'feature_summary': {
            col: {
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std()),
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max()),
            }
            for col in features_df.select_dtypes(include=[np.number]).columns
        },
    }

    json_path = output_dir / 'analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n→ Results saved to {json_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
