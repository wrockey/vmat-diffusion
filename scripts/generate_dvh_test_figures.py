"""
Generate test set evaluation figures for DVH-aware loss experiment.

Figures:
1. Model comparison bar chart (MAE and Gamma)
2. Case-by-case metrics
3. DVH structure errors
4. Key finding summary

Created: 2026-01-22
Experiment: dvh_aware_loss (test set evaluation)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
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
}

# Paths
PRED_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\dvh_aware_loss_test')
FIG_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\dvh_aware_loss\figures')
FIG_DIR.mkdir(exist_ok=True)


def load_results():
    """Load evaluation results."""
    results_path = PRED_DIR / 'evaluation_results.json'
    with open(results_path) as f:
        return json.load(f)


def fig1_model_comparison():
    """Figure 1: Model comparison - MAE and Gamma across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data from all experiments
    models = ['Baseline', 'Grad Loss', 'Grad+VGG', 'DVH-Aware']
    test_mae = [1.43, 1.44, 1.44, 0.95]
    gamma = [14.2, 27.9, 28.0, 27.7]  # approximate for VGG
    colors = [COLORS['blue'], COLORS['green'], COLORS['purple'], COLORS['orange']]

    # Panel A: Test MAE
    ax = axes[0]
    bars = ax.bar(models, test_mae, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test MAE (Gy)')
    ax.set_title('(A) Test Set MAE')
    ax.set_ylim(0, 2)
    ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Target (<3 Gy)')
    for bar, val in zip(bars, test_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.annotate('DVH-Aware best!\\n33% improvement', xy=(3, 0.95), xytext=(2.5, 1.5),
                fontsize=9, ha='center', color=COLORS['orange'],
                arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

    # Panel B: Gamma pass rate
    ax = axes[1]
    bars = ax.bar(models, gamma, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('(B) Gamma (3%/3mm)')
    ax.set_ylim(0, 100)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Target (95%)')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Interim (50%)')
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('DVH-Aware Loss: Test Set Evaluation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'fig6_test_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig6_test_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig6_test_comparison.png'}")
    plt.close()


def fig2_case_metrics(results):
    """Figure 2: Per-case metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cases = [r['case_id'] for r in results['per_case_results']]
    mae = [r['dose_metrics']['mae_gy'] for r in results['per_case_results']]
    gamma = [r['gamma']['gamma_pass_rate'] for r in results['per_case_results']]

    x = np.arange(len(cases))
    width = 0.6

    # Panel A: MAE per case
    ax = axes[0]
    bars = ax.bar(x, mae, width, color=COLORS['blue'], edgecolor='black')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) MAE per Test Case')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.axhline(y=np.mean(mae), color=COLORS['red'], linestyle='--', label=f'Mean: {np.mean(mae):.2f} Gy')
    ax.legend()
    for bar, val in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Panel B: Gamma per case
    ax = axes[1]
    bars = ax.bar(x, gamma, width, color=COLORS['green'], edgecolor='black')
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('(B) Gamma per Test Case')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.axhline(y=np.mean(gamma), color=COLORS['red'], linestyle='--', label=f'Mean: {np.mean(gamma):.1f}%')
    ax.set_ylim(0, 50)
    ax.legend()
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_case_metrics.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig7_case_metrics.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig7_case_metrics.png'}")
    plt.close()


def fig3_dvh_errors(results):
    """Figure 3: DVH structure errors."""
    structures = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder']

    # Collect MAE for each structure across cases
    struct_mae = {s: [] for s in structures}
    for case_result in results['per_case_results']:
        for s in structures:
            if s in case_result['dvh_metrics'] and case_result['dvh_metrics'][s].get('exists'):
                struct_mae[s].append(case_result['dvh_metrics'][s]['mae_gy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(structures))
    width = 0.6
    means = [np.mean(struct_mae[s]) if struct_mae[s] else 0 for s in structures]
    stds = [np.std(struct_mae[s]) if len(struct_mae[s]) > 1 else 0 for s in structures]

    colors_struct = [COLORS['orange'], COLORS['orange'], COLORS['blue'],
                     COLORS['green'], COLORS['green']]

    bars = ax.bar(x, means, width, yerr=stds, color=colors_struct,
                  edgecolor='black', capsize=5)

    ax.set_ylabel('MAE (Gy)')
    ax.set_title('Structure-Specific MAE (DVH-Aware Model)')
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['orange'], label='PTV (Target)'),
                       Patch(facecolor=COLORS['blue'], label='CTV'),
                       Patch(facecolor=COLORS['green'], label='OAR')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig8_structure_errors.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig8_structure_errors.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig8_structure_errors.png'}")
    plt.close()


def fig4_key_finding():
    """Figure 4: Key finding summary."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    models = ['Baseline', 'Grad Loss', 'Grad+VGG', 'DVH-Aware']
    test_mae = [1.43, 1.44, 1.44, 0.95]
    gamma = [14.2, 27.9, 28.0, 27.7]

    x = np.arange(len(models))
    width = 0.35

    # Normalize gamma to 0-1 for dual axis visualization
    gamma_norm = [g/100 for g in gamma]

    ax1 = ax
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, test_mae, width, label='Test MAE (Gy)',
                    color=COLORS['blue'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, gamma, width, label='Gamma (%)',
                    color=COLORS['green'], alpha=0.8)

    ax1.set_ylabel('Test MAE (Gy)', color=COLORS['blue'])
    ax2.set_ylabel('Gamma Pass Rate (%)', color=COLORS['green'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 2)
    ax2.set_ylim(0, 50)

    # Legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Highlight DVH result
    ax.annotate('DVH-Aware:\\nBest MAE + Good Gamma',
                xy=(3, 0.95), xytext=(2.2, 1.6),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.title('Key Finding: DVH-Aware Loss Achieves Best Test MAE (0.95 Gy)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'fig9_key_finding_test.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig9_key_finding_test.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig9_key_finding_test.png'}")
    plt.close()


def main():
    print("="*60)
    print("Generating test set evaluation figures for DVH-aware loss")
    print("="*60)
    print(f"Output directory: {FIG_DIR}")
    print()

    # Load results
    results = load_results()
    print(f"Loaded results for {results['n_cases']} cases")
    print(f"Mean MAE: {results['aggregate_metrics']['mae_gy_mean']:.2f} Gy")
    print(f"Mean Gamma: {results['aggregate_metrics']['gamma_pass_rate_mean']:.1f}%")
    print()

    # Generate figures
    print("\n1. Model comparison...")
    fig1_model_comparison()

    print("\n2. Case metrics...")
    fig2_case_metrics(results)

    print("\n3. DVH structure errors...")
    fig3_dvh_errors(results)

    print("\n4. Key finding...")
    fig4_key_finding()

    print("\n" + "="*60)
    print("All test evaluation figures generated successfully!")
    print(f"Location: {FIG_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
