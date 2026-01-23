"""
Generate publication-ready figures for structure-weighted loss experiment.

Figures:
1. Model comparison bar chart (MAE and Gamma across all models)
2. Training curves (loss, val MAE over epochs)
3. Per-case test metrics
4. Key finding summary

Created: 2026-01-22
Experiment: structure_weighted_loss
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import csv

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
RUN_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\structure_weighted_loss')
PRED_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\structure_weighted_test')
FIG_DIR = RUN_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_results():
    """Load evaluation results."""
    results_path = PRED_DIR / 'evaluation_results.json'
    with open(results_path) as f:
        return json.load(f)


def load_metrics_csv():
    """Load training metrics CSV."""
    csv_path = RUN_DIR / 'version_1' / 'metrics.csv'
    epochs = []
    val_mae = []
    train_loss = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('val/mae_gy'):
                epochs.append(int(row['epoch']))
                val_mae.append(float(row['val/mae_gy']))
            if row.get('train/loss_epoch'):
                train_loss.append(float(row['train/loss_epoch']))

    return epochs, val_mae, train_loss


def fig1_model_comparison():
    """Figure 1: Model comparison - MAE and Gamma across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data from all experiments
    models = ['Baseline', 'Grad Loss', 'DVH-Aware', 'Struct-Weight']
    val_mae = [3.73, 3.67, 3.61, 2.91]
    test_mae = [1.43, 1.44, 0.95, 1.40]
    gamma = [14.2, 27.9, 27.7, 31.2]
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
    ax.annotate('New best!', xy=(3, 31.2), xytext=(2.5, 50),
                fontsize=10, ha='center', color=COLORS['orange'],
                arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

    plt.suptitle('Structure-Weighted Loss: Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'fig1_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_model_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig1_model_comparison.png'}")
    plt.close()


def fig2_training_curves():
    """Figure 2: Training curves."""
    epochs, val_mae, train_loss = load_metrics_csv()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Validation MAE
    ax = axes[0]
    ax.plot(epochs, val_mae, color=COLORS['blue'], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE (Gy)')
    ax.set_title('(A) Validation MAE Over Training')
    best_idx = np.argmin(val_mae)
    ax.axhline(y=val_mae[best_idx], color=COLORS['red'], linestyle='--', alpha=0.7,
               label=f'Best: {val_mae[best_idx]:.2f} Gy (epoch {epochs[best_idx]})')
    ax.scatter([epochs[best_idx]], [val_mae[best_idx]], color=COLORS['red'], s=100, zorder=5)
    ax.legend()

    # Panel B: Training loss
    ax = axes[1]
    loss_epochs = list(range(len(train_loss)))
    ax.plot(loss_epochs, train_loss, color=COLORS['green'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(B) Training Loss')
    ax.set_yscale('log')

    plt.suptitle('Structure-Weighted Loss: Training Progress', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'fig2_training_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_training_curves.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig2_training_curves.png'}")
    plt.close()


def fig3_case_metrics(results):
    """Figure 3: Per-case test metrics."""
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
    ax.axhline(y=np.mean(mae), color=COLORS['red'], linestyle='--',
               label=f'Mean: {np.mean(mae):.2f} Gy')
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
    ax.axhline(y=np.mean(gamma), color=COLORS['red'], linestyle='--',
               label=f'Mean: {np.mean(gamma):.1f}%')
    ax.set_ylim(0, 50)
    ax.legend()
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.suptitle('Structure-Weighted Loss: Per-Case Test Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_case_metrics.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_case_metrics.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig3_case_metrics.png'}")
    plt.close()


def fig4_key_finding():
    """Figure 4: Key finding - Gamma improvement timeline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Timeline of experiments
    experiments = ['Baseline', 'Grad Loss', 'DVH-Aware', 'Struct-Weight']
    gamma_values = [14.2, 27.9, 27.7, 31.2]
    x = np.arange(len(experiments))

    colors_exp = [COLORS['blue'], COLORS['green'], COLORS['purple'], COLORS['orange']]
    bars = ax.bar(x, gamma_values, color=colors_exp, edgecolor='black', linewidth=1.5)

    # Highlight improvement
    ax.annotate('', xy=(3, 31.2), xytext=(1, 27.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('+3.3%', xy=(2, 32), fontsize=12, ha='center', fontweight='bold')

    ax.set_ylabel('Gamma Pass Rate (3%/3mm) %')
    ax.set_xlabel('Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylim(0, 100)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Clinical Target (95%)')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Interim Target (50%)')

    for bar, val in zip(bars, gamma_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend(loc='upper right')
    ax.set_title('Key Finding: Structure-Weighted Loss Achieves Best Gamma (31.2%)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_key_finding.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_key_finding.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig4_key_finding.png'}")
    plt.close()


def main():
    print("="*60)
    print("Generating figures for structure-weighted loss experiment")
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

    print("\n2. Training curves...")
    fig2_training_curves()

    print("\n3. Case metrics...")
    fig3_case_metrics(results)

    print("\n4. Key finding...")
    fig4_key_finding()

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Location: {FIG_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
