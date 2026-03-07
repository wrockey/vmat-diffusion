"""
Generate cross-condition comparison figures for ablation scout experiments.

Compares all 10 conditions (C1-C10) on key metrics:
1. Cross-condition bar chart (MAE, PTV Gamma, D95 Gap)
2. Addition arm comparison (C1 vs C2-C5)
3. Leave-one-out arm comparison (C6 vs C7-C10)
4. Per-case heatmap (conditions × cases × PTV Gamma)
5. DVH comparison for representative case across key conditions
6. Ablation contribution chart (delta from baseline/full)

All figures are single-seed (42), preliminary, for pipeline validation only.

Created: 2026-03-08
Experiment: ablation_scouts (C2-C5, C7-C10 + C1, C6 references)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Publication plot configuration (CLAUDE.md standard) ───────────────────────
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Colorblind-friendly palette (Wong 2011, Nature Methods)
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
}

RX_DOSE_GY = 70.0
STRUCTURE_NAMES = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder',
                   'Femur_L', 'Femur_R', 'Bowel']
DATA_DIR = Path('/home/wrockey/data/processed_npz')

# ── Condition definitions ─────────────────────────────────────────────────────

CONDITIONS = [
    ('C1',  'Baseline\n(MSE)',      'baseline_v23'),
    ('C2',  '+Gradient',            'C2_gradient_only'),
    ('C3',  '+DVH',                 'C3_dvh_only'),
    ('C4',  '+Structure',           'C4_structure_only'),
    ('C5',  '+AsymPTV',             'C5_asymptv_only'),
    ('C6',  'Full\ncombined',       'combined_loss_2.5to1'),
    ('C7',  'Full\n-Gradient',      'C7_full_no_gradient'),
    ('C8',  'Full\n-DVH',           'C8_full_no_dvh'),
    ('C9',  'Full\n-Structure',     'C9_full_no_structure'),
    ('C10', 'Full\n-AsymPTV',       'C10_full_no_asymptv'),
]

# Color assignments: baseline=black, additions=blues, full=red, ablations=oranges
CONDITION_COLORS = {
    'C1':  COLORS['black'],
    'C2':  COLORS['cyan'],
    'C3':  COLORS['blue'],
    'C4':  COLORS['green'],
    'C5':  COLORS['purple'],
    'C6':  COLORS['red'],
    'C7':  COLORS['orange'],
    'C8':  COLORS['yellow'],
    'C9':  '#8B6914',  # dark yellow/brown for visibility
    'C10': '#CC6677',  # muted rose
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_condition_data(seed: int = 42) -> dict:
    """Load evaluation results for all conditions that have data."""
    results = {}
    for cid, label, exp_name in CONDITIONS:
        eval_file = (PROJECT_ROOT / 'predictions' /
                     f'{exp_name}_seed{seed}_test' /
                     'baseline_evaluation_results.json')
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            results[cid] = {
                'label': label,
                'exp_name': exp_name,
                'raw': data,
                'cases': extract_per_case_metrics(data),
            }
    return results


def extract_per_case_metrics(data: dict) -> list:
    """Extract standardized per-case metrics from evaluation results."""
    cases = []
    for c in data['per_case_results']:
        ptv70 = c['dvh_metrics'].get('PTV70', {})
        pred_d95 = ptv70.get('pred_D95')
        targ_d95 = ptv70.get('target_D95')
        d95_gap = (pred_d95 - targ_d95) if (pred_d95 is not None and targ_d95 is not None) else None

        cases.append({
            'case_id': c['case_id'],
            'mae_gy': c['dose_metrics']['mae_gy'],
            'global_gamma': c['gamma']['global_3mm3pct']['gamma_pass_rate'],
            'ptv_gamma': c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'],
            'd95_gap': d95_gap,
            'per_structure_mae': c['dose_metrics'].get('per_structure_mae', {}),
        })
    return cases


def get_metric_arrays(results: dict, metric: str) -> dict:
    """Get arrays of a given metric for each condition."""
    arrays = {}
    for cid, data in results.items():
        vals = [c[metric] for c in data['cases'] if c[metric] is not None]
        if vals:
            arrays[cid] = np.array(vals)
    return arrays


# ═══════════════════════════════════════════════════════════════════════════════
# Figure functions
# ═══════════════════════════════════════════════════════════════════════════════

def save_figure(fig, name: str, fig_dir: Path):
    """Save as PNG (300 DPI) and PDF."""
    fig.savefig(fig_dir / f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(fig_dir / f'{name}.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved {name}.png + .pdf')


def fig1_cross_condition_bars(results: dict, fig_dir: Path):
    """Bar chart comparing all conditions on MAE, PTV Gamma, D95 Gap."""
    metrics = {
        'MAE (Gy)': ('mae_gy', False),           # lower is better
        'PTV Gamma 3%/3mm (%)': ('ptv_gamma', True),  # higher is better
        'PTV70 D95 Gap (Gy)': ('d95_gap', None),      # closer to 0 is better
    }

    cids = [cid for cid, _, _ in CONDITIONS if cid in results]
    labels = [results[cid]['label'] for cid in cids]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (metric_label, (metric_key, _)) in zip(axes, metrics.items()):
        arrays = get_metric_arrays(results, metric_key)
        means = [np.mean(arrays.get(cid, [np.nan])) for cid in cids]
        stds = [np.std(arrays.get(cid, [np.nan])) for cid in cids]
        colors = [CONDITION_COLORS.get(cid, COLORS['black']) for cid in cids]

        bars = ax.bar(range(len(cids)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_ylabel(metric_label)
        ax.set_xticks(range(len(cids)))

        # Add reference lines
        if metric_key == 'ptv_gamma':
            ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% target')
            ax.legend(loc='lower left')
        elif metric_key == 'd95_gap':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[-1].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)

    # Separator between addition and ablation arms
    for ax in axes:
        ax.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Ablation Scout: All Conditions (seed 42, 70 cases, preliminary)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig1_cross_condition_bars', fig_dir)


def fig2_addition_arm(results: dict, fig_dir: Path):
    """C1 vs C2-C5: which individual component helps most?"""
    addition_cids = ['C1', 'C2', 'C3', 'C4', 'C5']
    available = [c for c in addition_cids if c in results]
    if len(available) < 2:
        print('  [SKIP] Not enough addition arm data')
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (metric_key, ylabel) in zip(axes, [
        ('mae_gy', 'MAE (Gy)'),
        ('ptv_gamma', 'PTV Gamma 3%/3mm (%)'),
        ('d95_gap', 'PTV70 D95 Gap (Gy)'),
    ]):
        arrays = get_metric_arrays(results, metric_key)
        positions = []
        data = []
        colors = []
        labels = []

        for i, cid in enumerate(available):
            if cid in arrays:
                positions.append(i)
                data.append(arrays[cid])
                colors.append(CONDITION_COLORS[cid])
                labels.append(results[cid]['label'])

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)

        if metric_key == 'ptv_gamma':
            ax.axhline(y=95, color='red', linestyle='--', alpha=0.5)
        elif metric_key == 'd95_gap':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('Addition Arm: MSE + Single Component (seed 42, preliminary)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig2_addition_arm', fig_dir)


def fig3_leaveoneout_arm(results: dict, fig_dir: Path):
    """C6 vs C7-C10: which component is most essential?"""
    loo_cids = ['C6', 'C7', 'C8', 'C9', 'C10']
    available = [c for c in loo_cids if c in results]
    if len(available) < 2:
        print('  [SKIP] Not enough leave-one-out arm data')
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (metric_key, ylabel) in zip(axes, [
        ('mae_gy', 'MAE (Gy)'),
        ('ptv_gamma', 'PTV Gamma 3%/3mm (%)'),
        ('d95_gap', 'PTV70 D95 Gap (Gy)'),
    ]):
        arrays = get_metric_arrays(results, metric_key)
        positions = []
        data = []
        colors = []
        labels = []

        for i, cid in enumerate(available):
            if cid in arrays:
                positions.append(i)
                data.append(arrays[cid])
                colors.append(CONDITION_COLORS[cid])
                labels.append(results[cid]['label'])

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)

        if metric_key == 'ptv_gamma':
            ax.axhline(y=95, color='red', linestyle='--', alpha=0.5)
        elif metric_key == 'd95_gap':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('Leave-One-Out Arm: Full Combined Minus One (seed 42, preliminary)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig3_leaveoneout_arm', fig_dir)


def fig4_per_case_heatmap(results: dict, fig_dir: Path):
    """Heatmap: conditions × cases for PTV Gamma."""
    cids = [cid for cid, _, _ in CONDITIONS if cid in results]
    if not cids:
        return

    # Get case IDs from first condition
    case_ids = [c['case_id'] for c in results[cids[0]]['cases']]

    # Build matrix
    matrix = np.full((len(cids), len(case_ids)), np.nan)
    for i, cid in enumerate(cids):
        for j, case in enumerate(results[cid]['cases']):
            matrix[i, j] = case['ptv_gamma']

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')

    # Add value annotations
    for i in range(len(cids)):
        for j in range(len(case_ids)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 70 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=9, color=color)

    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels([c.replace('prostate70gy_', '') for c in case_ids],
                       rotation=45, ha='right')
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels([f'{cid} {results[cid]["label"]}' for cid in cids], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, label='PTV Gamma 3%/3mm (%)')
    ax.set_xlabel('Test Case')
    ax.set_title('PTV Gamma Pass Rate by Condition and Case (seed 42, preliminary)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig4_per_case_heatmap', fig_dir)


def fig5_delta_chart(results: dict, fig_dir: Path):
    """Delta from baseline (additions) and from full (ablations)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Addition arm: delta from C1
    ax = axes[0]
    ax.set_title('Addition Arm: Change from Baseline (C1)', fontweight='bold')
    addition_cids = ['C2', 'C3', 'C4', 'C5']
    available_add = [c for c in addition_cids if c in results and 'C1' in results]

    if available_add and 'C1' in results:
        for metric_key, offset, marker, label in [
            ('mae_gy', -0.15, 'o', 'MAE (Gy)'),
            ('ptv_gamma', 0.0, 's', 'PTV Gamma (%)'),
            ('d95_gap', 0.15, '^', 'D95 Gap (Gy)'),
        ]:
            baseline_vals = get_metric_arrays(results, metric_key).get('C1', np.array([0]))
            baseline_mean = np.mean(baseline_vals)

            deltas = []
            for i, cid in enumerate(available_add):
                arr = get_metric_arrays(results, metric_key).get(cid, np.array([np.nan]))
                deltas.append(np.mean(arr) - baseline_mean)

            x = np.arange(len(available_add)) + offset
            ax.scatter(x, deltas, marker=marker, s=80, label=label, zorder=5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(available_add)))
        ax.set_xticklabels([results[c]['label'] for c in available_add], fontsize=10)
        ax.set_ylabel('Delta from C1 Baseline')
        ax.legend(fontsize=9)

    # Ablation arm: delta from C6
    ax = axes[1]
    ax.set_title('Ablation Arm: Change from Full (C6)', fontweight='bold')
    loo_cids = ['C7', 'C8', 'C9', 'C10']
    available_loo = [c for c in loo_cids if c in results and 'C6' in results]

    if available_loo and 'C6' in results:
        for metric_key, offset, marker, label in [
            ('mae_gy', -0.15, 'o', 'MAE (Gy)'),
            ('ptv_gamma', 0.0, 's', 'PTV Gamma (%)'),
            ('d95_gap', 0.15, '^', 'D95 Gap (Gy)'),
        ]:
            full_vals = get_metric_arrays(results, metric_key).get('C6', np.array([0]))
            full_mean = np.mean(full_vals)

            deltas = []
            for i, cid in enumerate(available_loo):
                arr = get_metric_arrays(results, metric_key).get(cid, np.array([np.nan]))
                deltas.append(np.mean(arr) - full_mean)

            x = np.arange(len(available_loo)) + offset
            ax.scatter(x, deltas, marker=marker, s=80, label=label, zorder=5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(available_loo)))
        ax.set_xticklabels([results[c]['label'] for c in available_loo], fontsize=10)
        ax.set_ylabel('Delta from C6 Full Combined')
        ax.legend(fontsize=9)

    fig.suptitle('Component Contribution Analysis (seed 42, preliminary)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig5_delta_chart', fig_dir)


def fig6_per_structure_mae(results: dict, fig_dir: Path):
    """Per-structure MAE comparison across key conditions."""
    key_cids = ['C1', 'C5', 'C6', 'C10']  # baseline, best single, full, full-asym
    available = [c for c in key_cids if c in results]
    if len(available) < 2:
        print('  [SKIP] Not enough data for per-structure MAE')
        return

    structures = ['PTV70', 'Rectum', 'Bladder', 'Femur_L', 'Femur_R', 'Bowel']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(structures))
    width = 0.8 / len(available)

    for i, cid in enumerate(available):
        struct_maes = []
        for struct in structures:
            vals = [c['per_structure_mae'].get(struct, np.nan)
                    for c in results[cid]['cases']
                    if struct in c.get('per_structure_mae', {})]
            struct_maes.append(np.mean(vals) if vals else np.nan)

        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, struct_maes, width, label=f'{cid} {results[cid]["label"]}',
               color=CONDITION_COLORS[cid], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=30, ha='right')
    ax.set_ylabel('MAE (Gy)')
    ax.legend(fontsize=9)
    ax.set_title('Per-Structure MAE: Key Conditions (seed 42, preliminary)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig6_per_structure_mae', fig_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation scout comparison figures')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for all conditions (default: 42)')
    args = parser.parse_args()

    plt.rcParams.update(PLOT_CONFIG)

    fig_dir = PROJECT_ROOT / 'runs' / 'ablation_scouts' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading evaluation data (seed {args.seed})...')
    results = load_condition_data(args.seed)
    print(f'  Found data for {len(results)} conditions: {", ".join(sorted(results.keys()))}')

    if len(results) < 2:
        print('[ERROR] Need at least 2 conditions with evaluation data')
        return

    missing = [cid for cid, _, _ in CONDITIONS if cid not in results]
    if missing:
        print(f'  Missing: {", ".join(missing)}')

    # Generate figures
    figure_funcs = [
        ('1/6', 'Cross-condition bars',   lambda: fig1_cross_condition_bars(results, fig_dir)),
        ('2/6', 'Addition arm box plots', lambda: fig2_addition_arm(results, fig_dir)),
        ('3/6', 'Leave-one-out box plots', lambda: fig3_leaveoneout_arm(results, fig_dir)),
        ('4/6', 'Per-case heatmap',       lambda: fig4_per_case_heatmap(results, fig_dir)),
        ('5/6', 'Delta chart',            lambda: fig5_delta_chart(results, fig_dir)),
        ('6/6', 'Per-structure MAE',      lambda: fig6_per_structure_mae(results, fig_dir)),
    ]

    for num, name, func in figure_funcs:
        print(f'[{num}] {name}...')
        try:
            func()
        except Exception as e:
            print(f'  [ERROR] {e}')
            import traceback
            traceback.print_exc()

    generated = list(fig_dir.glob('*.png'))
    print(f'\nGenerated {len(generated)} PNG figures + matching PDFs in {fig_dir}')


if __name__ == '__main__':
    main()
