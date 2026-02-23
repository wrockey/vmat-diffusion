"""
Analyze Gamma Metric Hypothesis: Is overall Gamma the right metric?

This script tests the hypothesis that overall Gamma may not be the appropriate
metric for evaluating dose prediction quality, because:
1. Multiple dose distributions could be clinically valid
2. The model may predict an "average" of valid solutions
3. Clinical acceptability (DVH constraints) may be a better metric

Tests:
1. DVH Clinical Acceptability - Do predictions pass clinical constraints?
2. PTV-Only Gamma - Is model accurate where it matters most?
3. Region-Specific Gamma - Where do failures concentrate?

Created: 2026-01-23
Experiment: gamma_metric_analysis
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Optional: pymedphys for gamma
try:
    from pymedphys import gamma as pymedphys_gamma
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False
    print("Warning: pymedphys not installed. Region-specific Gamma disabled.")

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-friendly colors
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'cyan': '#17becf',
    'pass': '#2ca02c',
    'fail': '#d62728',
}

# Clinical constraints for prostate VMAT with SIB
CLINICAL_CONSTRAINTS = {
    'PTV70': {
        'D95_min_gy': 66.5,  # 95% of 70 Gy
        'D95_min_pct': 0.95,
        'description': 'PTV70 D95 >= 95% Rx (66.5 Gy)',
    },
    'PTV56': {
        'D95_min_gy': 53.2,  # 95% of 56 Gy
        'D95_min_pct': 0.95,
        'description': 'PTV56 D95 >= 95% Rx (53.2 Gy)',
    },
    'Rectum': {
        'V70_max_pct': 0.15,  # V70 < 15%
        'V65_max_pct': 0.25,  # V65 < 25%
        'V50_max_pct': 0.50,  # V50 < 50%
        'description': 'Rectum V70 < 15%, V65 < 25%, V50 < 50%',
    },
    'Bladder': {
        'V70_max_pct': 0.25,  # V70 < 25%
        'V65_max_pct': 0.50,  # V65 < 50%
        'description': 'Bladder V70 < 25%, V65 < 50%',
    },
}

RX_DOSE_GY = 70.0
DEFAULT_SPACING_MM = (1.0, 1.0, 2.0)


def get_spacing_from_metadata(metadata):
    """
    Extract voxel spacing from NPZ metadata with backwards-compatible fallback.
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


def load_prediction_and_target(pred_path: Path, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Load prediction, target, masks, and spacing."""
    pred_data = np.load(pred_path)
    pred_dose_raw = pred_data['dose']  # Normalized 0-1 scale
    pred_dose = pred_dose_raw * RX_DOSE_GY  # Convert to Gy

    case_id = pred_path.stem.replace('_pred', '')
    target_path = data_dir / f'{case_id}.npz'
    target_data = np.load(target_path, allow_pickle=True)

    target_dose = target_data['dose'] * RX_DOSE_GY  # Convert to Gy (normalized 0-1)
    masks_sdf = target_data['masks_sdf']

    metadata = target_data['metadata'].item() if 'metadata' in target_data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    return pred_dose, target_dose, masks_sdf, spacing


def extract_structure_masks(masks_sdf: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract boolean masks from SDF array."""
    structure_names = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder',
                       'Femur_L', 'Femur_R', 'Bowel']
    masks = {}
    for idx, name in enumerate(structure_names):
        masks[name] = masks_sdf[idx] < 0  # Inside where SDF < 0
    return masks


def compute_dvh_metrics(dose: np.ndarray, mask: np.ndarray) -> Dict:
    """Compute DVH metrics for a structure."""
    if not mask.any():
        return {'exists': False}

    dose_in_structure = dose[mask]
    n_voxels = len(dose_in_structure)

    # Sort for DVH computation
    sorted_dose = np.sort(dose_in_structure)[::-1]  # Descending

    # Dx metrics (dose received by x% of volume)
    d95_idx = int(0.95 * n_voxels)
    d50_idx = int(0.50 * n_voxels)
    d5_idx = int(0.05 * n_voxels)

    d95 = sorted_dose[min(d95_idx, n_voxels-1)]
    d50 = sorted_dose[min(d50_idx, n_voxels-1)]
    d5 = sorted_dose[min(d5_idx, n_voxels-1)]

    # Vx metrics (volume receiving at least x Gy)
    v70 = np.sum(dose_in_structure >= 70.0) / n_voxels
    v65 = np.sum(dose_in_structure >= 65.0) / n_voxels
    v50 = np.sum(dose_in_structure >= 50.0) / n_voxels
    v40 = np.sum(dose_in_structure >= 40.0) / n_voxels

    return {
        'exists': True,
        'n_voxels': n_voxels,
        'mean_gy': float(np.mean(dose_in_structure)),
        'max_gy': float(np.max(dose_in_structure)),
        'min_gy': float(np.min(dose_in_structure)),
        'd95_gy': float(d95),
        'd50_gy': float(d50),
        'd5_gy': float(d5),
        'v70': float(v70),
        'v65': float(v65),
        'v50': float(v50),
        'v40': float(v40),
    }


def check_clinical_constraints(pred_dvh: Dict, target_dvh: Dict, structure: str) -> Dict:
    """Check if predictions meet clinical constraints."""
    results = {'structure': structure, 'constraints': []}

    if structure not in CLINICAL_CONSTRAINTS:
        return results

    constraints = CLINICAL_CONSTRAINTS[structure]

    # PTV D95 constraint
    if 'D95_min_gy' in constraints:
        pred_d95 = pred_dvh.get('d95_gy', 0)
        target_d95 = target_dvh.get('d95_gy', 0)
        threshold = constraints['D95_min_gy']

        pred_pass = pred_d95 >= threshold
        target_pass = target_d95 >= threshold

        results['constraints'].append({
            'name': f'{structure} D95 >= {threshold:.1f} Gy',
            'pred_value': pred_d95,
            'target_value': target_d95,
            'threshold': threshold,
            'pred_pass': pred_pass,
            'target_pass': target_pass,
            'type': 'D95',
        })

    # OAR Vx constraints
    for vx_key in ['V70_max_pct', 'V65_max_pct', 'V50_max_pct']:
        if vx_key in constraints:
            dose_level = int(vx_key[1:3])
            vx_name = f'v{dose_level}'
            pred_vx = pred_dvh.get(vx_name, 0)
            target_vx = target_dvh.get(vx_name, 0)
            threshold = constraints[vx_key]

            pred_pass = pred_vx <= threshold
            target_pass = target_vx <= threshold

            results['constraints'].append({
                'name': f'{structure} V{dose_level} <= {threshold*100:.0f}%',
                'pred_value': pred_vx,
                'target_value': target_vx,
                'threshold': threshold,
                'pred_pass': pred_pass,
                'target_pass': target_pass,
                'type': f'V{dose_level}',
            })

    return results


def compute_region_gamma(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple = DEFAULT_SPACING_MM,
    subsample: int = 2,
) -> Dict:
    """Compute Gamma only within a specific region (masked)."""
    if not HAS_PYMEDPHYS or not mask.any():
        return {'computed': False, 'reason': 'No voxels or pymedphys unavailable'}

    # Subsample for speed
    pred_sub = pred_gy[::subsample, ::subsample, ::subsample].astype(np.float64)
    target_sub = target_gy[::subsample, ::subsample, ::subsample].astype(np.float64)
    mask_sub = mask[::subsample, ::subsample, ::subsample]

    spacing_sub = tuple(s * subsample for s in spacing_mm)

    # Create axes
    axes = tuple(
        np.arange(s) * sp for s, sp in zip(pred_sub.shape, spacing_sub)
    )

    try:
        # Compute full gamma map
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_sub,
            axes_evaluation=axes,
            dose_evaluation=pred_sub,
            dose_percent_threshold=3.0,
            distance_mm_threshold=3.0,
            lower_percent_dose_cutoff=10.0,
            max_gamma=2.0,
        )

        # Extract gamma values within mask
        valid_gamma = gamma_map[mask_sub & ~np.isnan(gamma_map)]

        if len(valid_gamma) == 0:
            return {'computed': False, 'reason': 'No valid gamma voxels in region'}

        pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100

        return {
            'computed': True,
            'gamma_pass_rate': float(pass_rate),
            'gamma_mean': float(np.mean(valid_gamma)),
            'gamma_median': float(np.median(valid_gamma)),
            'gamma_max': float(np.max(valid_gamma)),
            'n_voxels': int(len(valid_gamma)),
        }
    except Exception as e:
        return {'computed': False, 'reason': str(e)}


def analyze_single_case(
    pred_path: Path,
    data_dir: Path,
) -> Dict:
    """Run full analysis on a single case."""
    case_id = pred_path.stem.replace('_pred', '')
    print(f"\nAnalyzing: {case_id}")

    pred_dose, target_dose, masks_sdf, spacing = load_prediction_and_target(pred_path, data_dir)
    masks = extract_structure_masks(masks_sdf)

    results = {
        'case_id': case_id,
        'timestamp': datetime.now().isoformat(),
        'spacing_mm': spacing,
        'dvh_metrics': {},
        'constraint_results': [],
        'region_gamma': {},
    }

    # Compute DVH metrics for each structure
    print("  Computing DVH metrics...")
    for structure in ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder', 'Bowel']:
        if structure in masks:
            pred_dvh = compute_dvh_metrics(pred_dose, masks[structure])
            target_dvh = compute_dvh_metrics(target_dose, masks[structure])

            results['dvh_metrics'][structure] = {
                'pred': pred_dvh,
                'target': target_dvh,
            }

            # Check clinical constraints
            constraint_check = check_clinical_constraints(pred_dvh, target_dvh, structure)
            if constraint_check['constraints']:
                results['constraint_results'].append(constraint_check)

    # Compute region-specific Gamma
    print("  Computing region-specific Gamma...")

    # PTV-only Gamma (combined PTV70 + PTV56)
    ptv_mask = masks['PTV70'] | masks['PTV56']
    results['region_gamma']['PTV_combined'] = compute_region_gamma(
        pred_dose, target_dose, ptv_mask, spacing_mm=spacing, subsample=2
    )
    print(f"    PTV Gamma: {results['region_gamma']['PTV_combined'].get('gamma_pass_rate', 'N/A'):.1f}%")

    # OAR Gamma (Rectum + Bladder)
    oar_mask = masks['Rectum'] | masks['Bladder']
    results['region_gamma']['OAR_combined'] = compute_region_gamma(
        pred_dose, target_dose, oar_mask, spacing_mm=spacing, subsample=2
    )
    print(f"    OAR Gamma: {results['region_gamma']['OAR_combined'].get('gamma_pass_rate', 'N/A'):.1f}%")

    # High-dose region (>50% Rx)
    high_dose_mask = target_dose > (0.5 * RX_DOSE_GY)
    results['region_gamma']['high_dose'] = compute_region_gamma(
        pred_dose, target_dose, high_dose_mask, spacing_mm=spacing, subsample=2
    )
    print(f"    High-dose Gamma: {results['region_gamma']['high_dose'].get('gamma_pass_rate', 'N/A'):.1f}%")

    # Low-dose region (10-50% Rx) - the "no man's land"
    low_dose_mask = (target_dose > (0.1 * RX_DOSE_GY)) & (target_dose <= (0.5 * RX_DOSE_GY))
    results['region_gamma']['low_dose'] = compute_region_gamma(
        pred_dose, target_dose, low_dose_mask, spacing_mm=spacing, subsample=2
    )
    print(f"    Low-dose Gamma: {results['region_gamma']['low_dose'].get('gamma_pass_rate', 'N/A'):.1f}%")

    return results


def summarize_results(all_results: List[Dict]) -> Dict:
    """Summarize results across all cases."""
    summary = {
        'n_cases': len(all_results),
        'constraint_summary': {},
        'gamma_summary': {},
    }

    # Aggregate constraint pass rates
    all_constraints = []
    for r in all_results:
        for struct_result in r['constraint_results']:
            for c in struct_result['constraints']:
                all_constraints.append(c)

    # Group by constraint name
    constraint_groups = {}
    for c in all_constraints:
        name = c['name']
        if name not in constraint_groups:
            constraint_groups[name] = {'pred_pass': [], 'target_pass': []}
        constraint_groups[name]['pred_pass'].append(c['pred_pass'])
        constraint_groups[name]['target_pass'].append(c['target_pass'])

    for name, data in constraint_groups.items():
        summary['constraint_summary'][name] = {
            'pred_pass_rate': sum(data['pred_pass']) / len(data['pred_pass']) * 100,
            'target_pass_rate': sum(data['target_pass']) / len(data['target_pass']) * 100,
            'n_cases': len(data['pred_pass']),
        }

    # Aggregate Gamma by region
    for region in ['PTV_combined', 'OAR_combined', 'high_dose', 'low_dose']:
        gamma_values = [
            r['region_gamma'][region].get('gamma_pass_rate')
            for r in all_results
            if r['region_gamma'][region].get('computed', False)
        ]
        if gamma_values:
            summary['gamma_summary'][region] = {
                'mean': np.mean(gamma_values),
                'std': np.std(gamma_values),
                'values': gamma_values,
            }

    return summary


def generate_figures(all_results: List[Dict], summary: Dict, output_dir: Path):
    """Generate publication-ready figures."""
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Clinical Constraint Pass Rates
    fig1_constraint_comparison(summary, output_dir)

    # Figure 2: Region-Specific Gamma Comparison
    fig2_region_gamma(summary, output_dir)

    # Figure 3: DVH Comparison (overlay pred vs target)
    fig3_dvh_comparison(all_results, output_dir)

    # Figure 4: Key Finding Summary
    fig4_key_finding(summary, output_dir)


def fig1_constraint_comparison(summary: Dict, output_dir: Path):
    """Figure 1: Clinical constraint pass rates - prediction vs target."""
    constraints = summary['constraint_summary']

    names = list(constraints.keys())
    pred_rates = [constraints[n]['pred_pass_rate'] for n in names]
    target_rates = [constraints[n]['target_pass_rate'] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, target_rates, width, label='Target (Ground Truth)',
                   color=COLORS['blue'], edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_rates, width, label='Prediction',
                   color=COLORS['orange'], edgecolor='black')

    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Clinical Constraint Pass Rates: Prediction vs Ground Truth')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, val in zip(bars1, target_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, pred_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_constraint_pass_rates.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_constraint_pass_rates.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig1_constraint_pass_rates.png'}")
    plt.close()


def fig2_region_gamma(summary: Dict, output_dir: Path):
    """Figure 2: Gamma pass rate by region."""
    gamma_data = summary['gamma_summary']

    regions = ['PTV_combined', 'OAR_combined', 'high_dose', 'low_dose']
    region_labels = ['PTV\n(Critical)', 'OAR\n(Constraints)', 'High Dose\n(>50% Rx)', 'Low Dose\n(No-Man\'s Land)']

    means = [gamma_data.get(r, {}).get('mean', 0) for r in regions]
    stds = [gamma_data.get(r, {}).get('std', 0) for r in regions]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors_region = [COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['green']]
    x = np.arange(len(regions))

    bars = ax.bar(x, means, yerr=stds, color=colors_region, edgecolor='black',
                  capsize=5, linewidth=1.5)

    ax.set_ylabel('Gamma Pass Rate (3%/3mm) %')
    ax.set_title('Region-Specific Gamma: Where Does Accuracy Matter?')
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylim(0, 100)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Clinical Target (95%)')

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight key insight
    if means[0] > means[3]:  # PTV better than low-dose
        ax.annotate('Model more accurate\nwhere it matters!',
                    xy=(0, means[0]), xytext=(1.5, means[0] + 15),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_region_gamma.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_region_gamma.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig2_region_gamma.png'}")
    plt.close()


def fig3_dvh_comparison(all_results: List[Dict], output_dir: Path):
    """Figure 3: DVH comparison - overlay pred vs target."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: PTV DVH
    ax = axes[0]
    for i, r in enumerate(all_results):
        case = r['case_id']
        if 'PTV70' in r['dvh_metrics'] and r['dvh_metrics']['PTV70']['pred']['exists']:
            pred_d95 = r['dvh_metrics']['PTV70']['pred']['d95_gy']
            target_d95 = r['dvh_metrics']['PTV70']['target']['d95_gy']
            pred_mean = r['dvh_metrics']['PTV70']['pred']['mean_gy']
            target_mean = r['dvh_metrics']['PTV70']['target']['mean_gy']

            ax.scatter(target_d95, pred_d95, s=100, color=COLORS['blue'], marker='o',
                      label=f'{case} D95' if i == 0 else '', zorder=5)
            ax.scatter(target_mean, pred_mean, s=100, color=COLORS['orange'], marker='s',
                      label=f'{case} Dmean' if i == 0 else '', zorder=5)

    # Perfect agreement line
    ax.plot([50, 80], [50, 80], 'k--', alpha=0.5, label='Perfect agreement')
    ax.axhline(y=66.5, color=COLORS['red'], linestyle=':', alpha=0.7, label='D95 threshold (66.5 Gy)')
    ax.axvline(x=66.5, color=COLORS['red'], linestyle=':', alpha=0.7)

    ax.set_xlabel('Target Dose (Gy)')
    ax.set_ylabel('Predicted Dose (Gy)')
    ax.set_title('(A) PTV70 DVH Metrics')
    ax.legend(loc='lower right')
    ax.set_xlim(50, 80)
    ax.set_ylim(50, 80)

    # Panel B: OAR DVH (Rectum V70)
    ax = axes[1]
    for i, r in enumerate(all_results):
        case = r['case_id']
        if 'Rectum' in r['dvh_metrics'] and r['dvh_metrics']['Rectum']['pred']['exists']:
            pred_v70 = r['dvh_metrics']['Rectum']['pred']['v70'] * 100
            target_v70 = r['dvh_metrics']['Rectum']['target']['v70'] * 100

            color = COLORS['pass'] if pred_v70 <= 15 else COLORS['fail']
            ax.scatter(target_v70, pred_v70, s=100, color=color, marker='o',
                      label=f'{case}' if i == 0 else '', zorder=5)

    ax.plot([0, 30], [0, 30], 'k--', alpha=0.5, label='Perfect agreement')
    ax.axhline(y=15, color=COLORS['red'], linestyle=':', alpha=0.7, label='V70 limit (15%)')
    ax.axvline(x=15, color=COLORS['red'], linestyle=':', alpha=0.7)

    ax.set_xlabel('Target V70 (%)')
    ax.set_ylabel('Predicted V70 (%)')
    ax.set_title('(B) Rectum V70 Constraint')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)

    plt.suptitle('DVH Metric Comparison: Prediction vs Ground Truth', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_dvh_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_dvh_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig3_dvh_comparison.png'}")
    plt.close()


def fig4_key_finding(summary: Dict, output_dir: Path):
    """Figure 4: Key finding - the metric problem."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    metrics = ['Overall\nGamma', 'PTV\nGamma', 'OAR\nGamma', 'DVH\nConstraints']

    # Get values (use placeholder for overall gamma from previous results)
    overall_gamma = 31.2  # From structure-weighted experiment
    ptv_gamma = summary['gamma_summary'].get('PTV_combined', {}).get('mean', 0)
    oar_gamma = summary['gamma_summary'].get('OAR_combined', {}).get('mean', 0)

    # DVH constraint pass rate (average across all constraints)
    constraint_rates = [v['pred_pass_rate'] for v in summary['constraint_summary'].values()]
    dvh_pass = np.mean(constraint_rates) if constraint_rates else 0

    values = [overall_gamma, ptv_gamma, oar_gamma, dvh_pass]
    colors_bar = [COLORS['red'], COLORS['blue'], COLORS['orange'], COLORS['green']]

    bars = ax.bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Pass Rate / Score (%)')
    ax.set_title('Key Finding: Is Overall Gamma the Right Metric?', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Clinical Target (95%)')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add interpretation box
    textstr = 'If PTV Gamma ~ DVH Pass Rate >> Overall Gamma:\n-> Model accurate where it matters\n-> Overall Gamma penalizes valid low-dose variation'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_key_finding.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_key_finding.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig4_key_finding.png'}")
    plt.close()


def main():
    print("="*70)
    print("GAMMA METRIC HYPOTHESIS ANALYSIS")
    print("Testing: Is overall Gamma the right metric for dose prediction?")
    print("="*70)

    # Paths - analyze all available test predictions
    pred_dirs = [
        Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\structure_weighted_test'),
        Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\dvh_aware_loss_test'),
    ]
    data_dir = Path(r'I:\processed_npz')
    output_dir = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\gamma_metric_analysis')
    output_dir.mkdir(exist_ok=True)

    # Find prediction files
    pred_files = []
    for pred_dir in pred_dirs:
        if pred_dir.exists():
            pred_files.extend(list(pred_dir.glob('*_pred.npz')))
            break  # Use first available

    if not pred_files:
        print("ERROR: No prediction files found!")
        return

    print(f"\nFound {len(pred_files)} prediction files")
    print(f"Output directory: {output_dir}")

    # Analyze each case
    all_results = []
    for pred_path in pred_files:
        result = analyze_single_case(pred_path, data_dir)
        all_results.append(result)

    # Summarize
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary = summarize_results(all_results)

    print("\nClinical Constraint Pass Rates:")
    for name, data in summary['constraint_summary'].items():
        status = "PASS" if data['pred_pass_rate'] >= 100 else "FAIL"
        print(f"  [{status}] {name}: Pred={data['pred_pass_rate']:.0f}%, Target={data['target_pass_rate']:.0f}%")

    print("\nRegion-Specific Gamma:")
    for region, data in summary['gamma_summary'].items():
        print(f"  {region}: {data['mean']:.1f}% ± {data['std']:.1f}%")

    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    figures_dir = output_dir / 'figures'
    generate_figures(all_results, summary, figures_dir)

    # Save results
    results_file = output_dir / 'analysis_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            return obj

        json.dump({
            'all_results': all_results,
            'summary': summary,
        }, f, indent=2, default=convert)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
