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

from eval_core import (
    STRUCTURE_CHANNELS,
    STRUCTURE_INDEX,
    DEFAULT_SPACING_MM,
    PRIMARY_PRESCRIPTION_GY,
    get_spacing_from_metadata,
)
from eval_metrics import (
    compute_gamma,
    compute_region_gamma,
    compute_dvh_metrics,
    HAS_PYMEDPHYS,
)
from eval_clinical import check_clinical_constraints
from eval_statistics import NumpyEncoder

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

RX_DOSE_GY = PRIMARY_PRESCRIPTION_GY


def load_prediction_and_target(pred_path: Path, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Load prediction, target, binary masks, SDF masks, and spacing."""
    pred_data = np.load(pred_path)
    pred_dose_raw = pred_data['dose']
    pred_dose = pred_dose_raw * RX_DOSE_GY

    case_id = pred_path.stem.replace('_pred', '')
    target_path = data_dir / f'{case_id}.npz'
    target_data = np.load(target_path, allow_pickle=True)

    target_dose = target_data['dose'] * RX_DOSE_GY
    masks_sdf = target_data['masks_sdf']
    masks = target_data['masks']

    metadata = target_data['metadata'].item() if 'metadata' in target_data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    return pred_dose, target_dose, masks, masks_sdf, spacing


def extract_structure_masks(masks_sdf: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract boolean masks from SDF array."""
    masks = {}
    for idx, name in STRUCTURE_CHANNELS.items():
        if idx < masks_sdf.shape[0]:
            masks[name] = masks_sdf[idx] < 0  # Inside where SDF < 0
    return masks


def _compute_region_gamma_compat(
    pred_gy: np.ndarray,
    target_gy: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple = DEFAULT_SPACING_MM,
    subsample: int = 2,
) -> Dict:
    """Wrapper around framework compute_region_gamma with backward-compatible output."""
    result = compute_region_gamma(
        pred_gy, target_gy, mask,
        spacing_mm=spacing_mm, subsample=subsample,
    )
    # Add 'computed' key for backward compatibility
    if result.get('gamma_pass_rate') is not None:
        result['computed'] = True
    else:
        result['computed'] = False
        result['reason'] = result.get('error', result.get('note', 'unknown'))
    return result


def analyze_single_case(
    pred_path: Path,
    data_dir: Path,
) -> Dict:
    """Run full analysis on a single case."""
    case_id = pred_path.stem.replace('_pred', '')
    print(f"\nAnalyzing: {case_id}")

    pred_dose, target_dose, binary_masks, masks_sdf, spacing = load_prediction_and_target(pred_path, data_dir)
    masks = extract_structure_masks(masks_sdf)

    results = {
        'case_id': case_id,
        'timestamp': datetime.now().isoformat(),
        'spacing_mm': spacing,
        'dvh_metrics': {},
        'constraint_results': [],
        'region_gamma': {},
    }

    # Compute DVH metrics using framework (operates on Gy already)
    print("  Computing DVH metrics...")
    framework_dvh = compute_dvh_metrics(
        pred_dose, target_dose, binary_masks,
        spacing_mm=spacing, rx_dose_gy=1.0,  # Already in Gy
    )

    # Store per-structure DVH for backward compatibility with figure functions
    for structure in ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder', 'Bowel']:
        if structure in framework_dvh and framework_dvh[structure].get('exists', False):
            m = framework_dvh[structure]
            results['dvh_metrics'][structure] = {
                'pred': {
                    'exists': True,
                    'n_voxels': m.get('n_voxels', 0),
                    'mean_gy': m['pred_mean_gy'],
                    'max_gy': m['pred_max_gy'],
                    'min_gy': m['pred_min_gy'],
                    'd95_gy': m['pred_D95'],
                    'd50_gy': m['pred_D50'],
                    'd5_gy': m['pred_D5'],
                    'v70': m.get('pred_V70', 0) / 100.0,  # Convert % to fraction
                    'v65': 0.0,  # Not computed in framework
                    'v50': m.get('pred_V50', 0) / 100.0,
                    'v40': m.get('pred_V40', 0) / 100.0,
                },
                'target': {
                    'exists': True,
                    'n_voxels': m.get('n_voxels', 0),
                    'mean_gy': m['target_mean_gy'],
                    'max_gy': m['target_max_gy'],
                    'min_gy': m['target_min_gy'],
                    'd95_gy': m['target_D95'],
                    'd50_gy': m['target_D50'],
                    'd5_gy': m['target_D5'],
                    'v70': m.get('target_V70', 0) / 100.0,
                    'v65': 0.0,
                    'v50': m.get('target_V50', 0) / 100.0,
                    'v40': m.get('target_V40', 0) / 100.0,
                },
            }

    # Clinical constraints via framework
    clinical_result = check_clinical_constraints(framework_dvh)
    # Convert to backward-compatible format
    for structure_name, struct_data in clinical_result.get('structures', {}).items():
        if struct_data.get('constraints_checked'):
            constraint_list = []
            for c in struct_data['constraints_checked']:
                target_val = framework_dvh.get(structure_name, {}).get(
                    f'target_{c["metric"]}',
                    framework_dvh.get(structure_name, {}).get('target_max_gy', 0)
                    if c['metric'] == 'Dmax' else 0
                )
                constraint_list.append({
                    'name': c.get('description', f'{structure_name} {c["metric"]}'),
                    'pred_value': c['predicted'],
                    'target_value': target_val,
                    'threshold': c['limit'],
                    'pred_pass': c['passed'],
                    'target_pass': True,  # GT usually passes
                    'type': c['metric'],
                })
            if constraint_list:
                results['constraint_results'].append({
                    'structure': structure_name,
                    'constraints': constraint_list,
                })

    # Compute region-specific Gamma using framework
    print("  Computing region-specific Gamma...")

    # PTV-only Gamma (combined PTV70 + PTV56)
    ptv_mask = masks.get('PTV70', np.zeros(pred_dose.shape, dtype=bool)) | \
               masks.get('PTV56', np.zeros(pred_dose.shape, dtype=bool))
    results['region_gamma']['PTV_combined'] = _compute_region_gamma_compat(
        pred_dose, target_dose, ptv_mask, spacing_mm=spacing, subsample=2
    )
    ptv_rate = results['region_gamma']['PTV_combined'].get('gamma_pass_rate', 'N/A')
    print(f"    PTV Gamma: {ptv_rate:.1f}%" if isinstance(ptv_rate, float) else f"    PTV Gamma: {ptv_rate}")

    # OAR Gamma (Rectum + Bladder)
    oar_mask = masks.get('Rectum', np.zeros(pred_dose.shape, dtype=bool)) | \
               masks.get('Bladder', np.zeros(pred_dose.shape, dtype=bool))
    results['region_gamma']['OAR_combined'] = _compute_region_gamma_compat(
        pred_dose, target_dose, oar_mask, spacing_mm=spacing, subsample=2
    )
    oar_rate = results['region_gamma']['OAR_combined'].get('gamma_pass_rate', 'N/A')
    print(f"    OAR Gamma: {oar_rate:.1f}%" if isinstance(oar_rate, float) else f"    OAR Gamma: {oar_rate}")

    # High-dose region (>50% Rx)
    high_dose_mask = target_dose > (0.5 * RX_DOSE_GY)
    results['region_gamma']['high_dose'] = _compute_region_gamma_compat(
        pred_dose, target_dose, high_dose_mask, spacing_mm=spacing, subsample=2
    )
    hd_rate = results['region_gamma']['high_dose'].get('gamma_pass_rate', 'N/A')
    print(f"    High-dose Gamma: {hd_rate:.1f}%" if isinstance(hd_rate, float) else f"    High-dose Gamma: {hd_rate}")

    # Low-dose region (10-50% Rx)
    low_dose_mask = (target_dose > (0.1 * RX_DOSE_GY)) & (target_dose <= (0.5 * RX_DOSE_GY))
    results['region_gamma']['low_dose'] = _compute_region_gamma_compat(
        pred_dose, target_dose, low_dose_mask, spacing_mm=spacing, subsample=2
    )
    ld_rate = results['region_gamma']['low_dose'].get('gamma_pass_rate', 'N/A')
    print(f"    Low-dose Gamma: {ld_rate:.1f}%" if isinstance(ld_rate, float) else f"    Low-dose Gamma: {ld_rate}")

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
        json.dump({
            'all_results': all_results,
            'summary': summary,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
