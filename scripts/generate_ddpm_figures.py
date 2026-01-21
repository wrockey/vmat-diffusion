"""
Generate publication-ready figures for DDPM experiments.

Figures:
1. Training curves (loss and volatile MAE)
2. Sampling steps ablation (50-1000 steps)
3. Ensemble averaging results
4. Model comparison (DDPM vs baseline vs gradient loss)
5. "More steps = worse" demonstration

Created: 2026-01-20
Experiments: ddpm_dose_v1, phase1_sampling, phase1_ensemble
Git commit: 3efbea0 (DDPM training)
"""

import json
import numpy as np
import pandas as pd
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
}

# Paths
RUN_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\vmat_dose_ddpm')
EXP_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\experiments')
FIG_DIR = RUN_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_training_metrics():
    """Load DDPM training metrics."""
    metrics_path = RUN_DIR / 'epoch_metrics.csv'
    return pd.read_csv(metrics_path)


def load_sampling_results():
    """Load Phase 1 sampling ablation results."""
    with open(EXP_DIR / 'phase1_sampling' / 'exp1_1_sampling_results.json') as f:
        return json.load(f)


def load_ensemble_results():
    """Load Phase 1 ensemble results."""
    with open(EXP_DIR / 'phase1_ensemble' / 'exp1_2_ensemble_results.json') as f:
        return json.load(f)


def fig1_ddpm_training_curves(metrics):
    """Figure 1: DDPM training curves showing volatile MAE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Loss curves
    ax = axes[0]
    ax.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', color=COLORS['blue'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', color=COLORS['orange'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Noise Prediction MSE)')
    ax.set_title('(A) DDPM Training Loss')
    ax.legend()
    ax.set_xlim(0, metrics['epoch'].max())
    ax.set_yscale('log')

    # Panel B: Validation MAE (volatile!)
    ax = axes[1]
    ax.plot(metrics['epoch'], metrics['val_mae_gy'], label='Val MAE', color=COLORS['red'], linewidth=2, alpha=0.8)
    ax.axhline(y=3.73, color=COLORS['green'], linestyle='--', label='Baseline (3.73 Gy)', linewidth=1.5)
    ax.fill_between(metrics['epoch'], 0, metrics['val_mae_gy'], alpha=0.2, color=COLORS['red'])

    # Mark best and worst
    best_idx = metrics['val_mae_gy'].idxmin()
    worst_idx = metrics['val_mae_gy'].idxmax()
    ax.scatter([metrics.loc[best_idx, 'epoch']], [metrics.loc[best_idx, 'val_mae_gy']],
               color=COLORS['green'], s=100, zorder=5, marker='v', label=f'Best: {metrics.loc[best_idx, "val_mae_gy"]:.1f} Gy')
    ax.scatter([metrics.loc[worst_idx, 'epoch']], [metrics.loc[worst_idx, 'val_mae_gy']],
               color=COLORS['red'], s=100, zorder=5, marker='^', label=f'Worst: {metrics.loc[worst_idx, "val_mae_gy"]:.1f} Gy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(B) Validation MAE - Extreme Volatility')
    ax.legend(loc='upper right')
    ax.set_xlim(0, metrics['epoch'].max())
    ax.set_ylim(0, 70)

    # Add annotation about volatility
    ax.annotate('Range: 12-64 Gy\n(Very unstable!)', xy=(20, 50), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig1_ddpm_training_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_ddpm_training_curves.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig1_ddpm_training_curves.png'}")
    plt.close()


def fig2_sampling_steps_ablation(sampling_results):
    """Figure 2: More steps = worse MAE (key finding)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = [50, 100, 250, 500, 1000]
    maes = [sampling_results['per_step_results'][str(s)]['aggregate']['mean_mae_gy'] for s in steps]
    times = [sampling_results['per_step_results'][str(s)]['aggregate']['mean_inference_time_s'] / 60 for s in steps]  # Convert to minutes

    # Panel A: MAE vs Steps
    ax = axes[0]
    bars = ax.bar([str(s) for s in steps], maes, color=COLORS['blue'], edgecolor='black', linewidth=1.5)
    ax.axhline(y=3.73, color=COLORS['green'], linestyle='--', label='Baseline (3.73 Gy)', linewidth=2)

    # Color best bar differently
    bars[0].set_color(COLORS['green'])

    ax.set_xlabel('DDIM Sampling Steps')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) More Steps = Worse MAE')
    ax.legend()

    # Add values on bars
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{mae:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add trend arrow
    ax.annotate('', xy=(4, 6.5), xytext=(0, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2, 7.0, 'Worse with more steps!', fontsize=11, color='red', ha='center')

    # Panel B: Inference Time vs Steps
    ax = axes[1]
    ax.plot(steps, times, 'o-', color=COLORS['orange'], linewidth=2, markersize=10)
    ax.set_xlabel('DDIM Sampling Steps')
    ax.set_ylabel('Inference Time (minutes)')
    ax.set_title('(B) Inference Time Scales Linearly')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add values
    for s, t in zip(steps, times):
        ax.annotate(f'{t:.0f} min', xy=(s, t), xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig2_sampling_steps_ablation.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_sampling_steps_ablation.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig2_sampling_steps_ablation.png'}")
    plt.close()


def fig3_ensemble_averaging(ensemble_results):
    """Figure 3: Ensemble averaging provides no benefit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_samples = [1, 3, 5, 10]
    maes = [ensemble_results['per_n_results'][str(n)]['aggregate']['mean_mae_gy'] for n in n_samples]
    variability = [ensemble_results['per_n_results'][str(n)]['aggregate']['mean_sample_std'] for n in n_samples]

    # Panel A: MAE vs ensemble size
    ax = axes[0]
    bars = ax.bar([str(n) for n in n_samples], maes, color=COLORS['blue'], edgecolor='black', linewidth=1.5)
    bars[0].set_color(COLORS['green'])  # Best is n=1

    ax.axhline(y=3.73, color=COLORS['orange'], linestyle='--', label='Baseline (3.73 Gy)', linewidth=2)
    ax.set_xlabel('Ensemble Size (n)')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) Ensemble Size vs MAE')
    ax.legend()
    ax.set_ylim(3.5, 4.0)

    # Add values
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{mae:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel B: Sample variability
    ax = axes[1]
    ax.bar([str(n) for n in n_samples], variability, color=COLORS['purple'], edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Ensemble Size (n)')
    ax.set_ylabel('Sample Std Dev (MAE Gy)')
    ax.set_title('(B) Near-Zero Sample Variability')

    # Add annotation
    ax.annotate('Very low variability\n≈ deterministic output', xy=(2, 0.02), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig3_ensemble_averaging.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_ensemble_averaging.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig3_ensemble_averaging.png'}")
    plt.close()


def fig4_model_comparison():
    """Figure 4: Full model comparison including gradient loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Baseline\nU-Net', 'DDPM\n(optimized)', 'Gradient\nLoss']
    val_mae = [3.73, 3.78, 3.67]
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]

    # Panel A: Validation MAE comparison
    ax = axes[0]
    bars = ax.bar(models, val_mae, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Validation MAE (Gy)')
    ax.set_title('(A) Validation MAE Comparison')
    ax.set_ylim(3.5, 4.0)

    for bar, mae in zip(bars, val_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{mae:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel B: Summary table as text
    ax = axes[1]
    ax.axis('off')

    table_text = """
    Model Comparison Summary
    ========================

    | Model          | Val MAE | Complexity | Recommended |
    |----------------|---------|------------|-------------|
    | Baseline U-Net | 3.73 Gy | Low        | Yes         |
    | DDPM optimized | 3.78 Gy | High       | No          |
    | Gradient Loss  | 3.67 Gy | Low        | Yes (best)  |

    Key Finding: DDPM provides no benefit over simple baseline.
    The added complexity of diffusion (1000 timesteps, iterative
    sampling) yields equivalent or worse results.

    Recommendation: Use Gradient Loss model for best accuracy
    and improved Gamma pass rate (27.9% vs 14.2%).
    """
    ax.text(0.1, 0.9, table_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig4_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_model_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig4_model_comparison.png'}")
    plt.close()


def fig5_key_finding():
    """Figure 5: Single summary figure of key DDPM finding."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    steps = [50, 100, 250, 500, 1000]
    maes = [3.80, 4.89, 5.24, 5.93, 6.73]

    # Plot
    ax.plot(steps, maes, 'o-', color=COLORS['red'], linewidth=3, markersize=12, label='DDPM MAE')
    ax.axhline(y=3.73, color=COLORS['green'], linestyle='--', linewidth=3, label='Baseline U-Net (3.73 Gy)')
    ax.axhline(y=3.67, color=COLORS['blue'], linestyle=':', linewidth=3, label='Gradient Loss (3.67 Gy)')

    ax.fill_between(steps, 3.67, maes, alpha=0.2, color=COLORS['red'])

    ax.set_xlabel('DDIM Sampling Steps', fontsize=14)
    ax.set_ylabel('MAE (Gy)', fontsize=14)
    ax.set_title('Key Finding: DDPM "More Steps = Worse Results"', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_ylim(3, 8)

    # Annotations
    ax.annotate('Optimal: 50 steps\n(still worse than baseline)',
                xy=(50, 3.80), xytext=(100, 4.5),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='black'))

    ax.annotate('1000 steps:\n77% worse than baseline!',
                xy=(1000, 6.73), xytext=(500, 7.2),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig5_key_finding.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_key_finding.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig5_key_finding.png'}")
    plt.close()


def main():
    print("Generating DDPM experiment figures...")
    print(f"Output directory: {FIG_DIR}")
    print()

    # Load data
    metrics = load_training_metrics()
    sampling_results = load_sampling_results()
    ensemble_results = load_ensemble_results()

    print(f"Loaded training metrics: {len(metrics)} epochs")
    print(f"Loaded sampling results: {len(sampling_results['steps_tested'])} step counts tested")
    print(f"Loaded ensemble results: {len(ensemble_results['n_samples_tested'])} ensemble sizes tested")

    # Generate figures
    print("\n1. DDPM training curves...")
    fig1_ddpm_training_curves(metrics)

    print("\n2. Sampling steps ablation...")
    fig2_sampling_steps_ablation(sampling_results)

    print("\n3. Ensemble averaging...")
    fig3_ensemble_averaging(ensemble_results)

    print("\n4. Model comparison...")
    fig4_model_comparison()

    print("\n5. Key finding summary...")
    fig5_key_finding()

    print("\n" + "="*50)
    print("All DDPM figures generated successfully!")
    print(f"Location: {FIG_DIR}")


if __name__ == '__main__':
    main()
