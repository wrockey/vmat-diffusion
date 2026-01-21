"""
Generate publication-ready figures for grad_vgg_combined experiment.

Figures:
1. Training curves (loss and MAE vs epoch)
2. Model comparison bar chart (baseline vs gradient loss vs grad+VGG)
3. Dose slice comparison (prediction vs ground truth)
4. Loss components breakdown (MSE, Gradient, VGG)

Created: 2026-01-21
Experiment: grad_vgg_combined
Git commit: To be recorded after commit
"""

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
    'cyan': '#17becf',
}

# Paths
RUN_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\grad_vgg_combined')
PRED_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\grad_vgg_combined_test')
DATA_DIR = Path(r'I:\processed_npz')
FIG_DIR = RUN_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_metrics():
    """Load training metrics from CSV."""
    metrics_path = RUN_DIR / 'version_1' / 'metrics.csv'
    df = pd.read_csv(metrics_path)

    # Extract epoch-level validation metrics
    val_data = df[df['val/mae_gy'].notna()][['epoch', 'val/loss', 'val/mae_gy']].copy()
    val_data = val_data.rename(columns={'val/loss': 'val_loss', 'val/mae_gy': 'val_mae_gy'})

    # Extract epoch-level training metrics
    train_cols = ['epoch', 'train/loss_epoch', 'train/grad_loss', 'train/mse_loss', 'train/vgg_loss']
    train_data = df[df['train/loss_epoch'].notna()][train_cols].copy()
    train_data = train_data.rename(columns={
        'train/loss_epoch': 'train_loss',
        'train/grad_loss': 'grad_loss',
        'train/mse_loss': 'mse_loss',
        'train/vgg_loss': 'vgg_loss'
    })

    # Merge
    metrics = pd.merge(val_data, train_data, on='epoch', how='outer').sort_values('epoch')
    return metrics


def fig1_training_curves(metrics):
    """Figure 1: Training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Loss curves
    ax = axes[0]
    ax.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', color=COLORS['blue'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', color=COLORS['orange'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(A) Training and Validation Loss')
    ax.legend()
    ax.set_xlim(0, metrics['epoch'].max())

    # Panel B: Validation MAE
    ax = axes[1]
    ax.plot(metrics['epoch'], metrics['val_mae_gy'], label='Val MAE', color=COLORS['green'], linewidth=2)
    ax.axhline(y=3.73, color=COLORS['red'], linestyle='--', label='Baseline (3.73 Gy)', linewidth=1.5)
    ax.axhline(y=3.67, color=COLORS['blue'], linestyle=':', label='Grad Loss (3.67 Gy)', linewidth=1.5)
    ax.axhline(y=2.27, color=COLORS['purple'], linestyle='-', label='Best (2.27 Gy)', linewidth=1.5, alpha=0.7)

    # Mark best epoch
    best_idx = metrics['val_mae_gy'].idxmin()
    best_epoch = metrics.loc[best_idx, 'epoch']
    best_mae = metrics.loc[best_idx, 'val_mae_gy']
    ax.scatter([best_epoch], [best_mae], color=COLORS['purple'], s=100, zorder=5, marker='*')
    ax.annotate(f'Best: {best_mae:.2f} Gy\n(Epoch {int(best_epoch)})',
                xy=(best_epoch, best_mae), xytext=(best_epoch + 5, best_mae + 1.5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(B) Validation MAE')
    ax.legend(loc='upper right')
    ax.set_xlim(0, metrics['epoch'].max())
    ax.set_ylim(0, 15)

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig1_training_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_training_curves.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig1_training_curves.png'}")
    plt.close()


def fig2_model_comparison():
    """Figure 2: Comparison bar chart of metrics across all models."""
    # Data from experiments
    models = ['Baseline', 'Grad Loss', 'Grad+VGG']
    val_mae = [3.73, 3.67, 2.27]  # Validation MAE
    test_mae = [1.43, 1.44, 1.44]  # Test MAE
    gamma = [14.2, 27.9, 27.85]  # Gamma pass rate

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]

    # Panel A: Validation MAE
    ax = axes[0]
    bars = ax.bar(models, val_mae, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) Validation MAE')
    ax.set_ylim(0, 5)
    for bar, val in zip(bars, val_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    # Improvement annotation
    ax.annotate(f'-38%\nvs Baseline',
                xy=(2, val_mae[2]), xytext=(2.3, 3.5),
                fontsize=9, ha='left', color=COLORS['purple'],
                arrowprops=dict(arrowstyle='->', color=COLORS['purple']))

    # Panel B: Test MAE
    ax = axes[1]
    bars = ax.bar(models, test_mae, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(B) Test MAE')
    ax.set_ylim(0, 2.5)
    for bar, val in zip(bars, test_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel C: Gamma Pass Rate
    ax = axes[2]
    bars = ax.bar(models, gamma, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=50, color=COLORS['red'], linestyle='--', label='Target (50%)', linewidth=1.5)
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('(C) Gamma (3%/3mm)')
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right')
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Note about Gamma
    ax.annotate('VGG does not\nimprove Gamma',
                xy=(2, gamma[2]), xytext=(1.8, 42),
                fontsize=9, ha='center', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig2_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_model_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig2_model_comparison.png'}")
    plt.close()


def fig3_dose_slices():
    """Figure 3: Dose slice comparison."""
    # Load data
    case_id = 'case_0007'
    pred_data = np.load(PRED_DIR / f'{case_id}_pred.npz')
    gt_data = np.load(DATA_DIR / f'{case_id}.npz')

    pred_dose = pred_data['dose'] * 70.0  # Convert to Gy
    gt_dose = gt_data['dose'] * 70.0

    # Get central slice
    z_mid = pred_dose.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin, vmax = 0, 75

    # Ground truth
    ax = axes[0]
    im = ax.imshow(gt_dose[z_mid], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('(A) Ground Truth')
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.axis('off')

    # Prediction
    ax = axes[1]
    ax.imshow(pred_dose[z_mid], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('(B) Prediction (Grad+VGG)')
    ax.axis('off')

    # Difference
    ax = axes[2]
    diff = pred_dose[z_mid] - gt_dose[z_mid]
    im_diff = ax.imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
    ax.set_title('(C) Difference (Pred - GT)')
    ax.axis('off')

    # Colorbars
    cbar1 = fig.colorbar(im, ax=axes[:2], shrink=0.8, label='Dose (Gy)')
    cbar2 = fig.colorbar(im_diff, ax=axes[2], shrink=0.8, label='Difference (Gy)')

    plt.suptitle(f'Dose Distribution - {case_id} (Axial Slice z={z_mid})', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig3_dose_slices.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_dose_slices.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig3_dose_slices.png'}")
    plt.close()


def fig4_loss_components(metrics):
    """Figure 4: Loss component breakdown (MSE, Gradient, VGG)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics['epoch'], metrics['mse_loss'], label='MSE Loss', color=COLORS['blue'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['grad_loss'], label='Gradient Loss (x0.1)', color=COLORS['orange'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['vgg_loss'], label='VGG Loss (x0.001)', color=COLORS['cyan'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['train_loss'], label='Total Loss', color=COLORS['green'], linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components (Grad+VGG)')
    ax.legend()
    ax.set_xlim(0, metrics['epoch'].max())
    ax.set_ylim(0, 0.05)

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig4_loss_components.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_loss_components.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig4_loss_components.png'}")
    plt.close()


def fig5_key_finding():
    """Figure 5: Key finding - VGG helps MAE but not Gamma."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: MAE Improvement
    ax = axes[0]
    models = ['Baseline', 'Grad', 'Grad+VGG']
    val_mae = [3.73, 3.67, 2.27]
    colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]
    bars = ax.bar(models, val_mae, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Validation MAE (Gy)')
    ax.set_title('(A) MAE: VGG Helps (-38%)')
    ax.set_ylim(0, 5)
    for bar, val in zip(bars, val_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Arrow showing improvement
    ax.annotate('', xy=(2, 2.27), xytext=(0, 3.73),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Panel B: Gamma unchanged
    ax = axes[1]
    gamma = [14.2, 27.9, 27.85]
    bars = ax.bar(models, gamma, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=95, color=COLORS['red'], linestyle='--', label='Target (95%)', linewidth=1.5)
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('(B) Gamma: VGG Does Not Help')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Annotation
    ax.annotate('~28% plateau',
                xy=(2, 28), xytext=(2.2, 50),
                fontsize=10, ha='left', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.suptitle('Key Finding: VGG Loss Improves MAE but Not Gamma', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig5_key_finding.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_key_finding.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig5_key_finding.png'}")
    plt.close()


def main():
    print("="*60)
    print("Generating publication-ready figures for grad_vgg_combined")
    print("="*60)
    print(f"Output directory: {FIG_DIR}")
    print()

    # Load metrics
    metrics = load_metrics()
    print(f"Loaded metrics: {len(metrics)} epochs")

    # Generate figures
    print("\n1. Training curves...")
    fig1_training_curves(metrics)

    print("\n2. Model comparison...")
    fig2_model_comparison()

    print("\n3. Dose slices...")
    fig3_dose_slices()

    print("\n4. Loss components...")
    fig4_loss_components(metrics)

    print("\n5. Key finding...")
    fig5_key_finding()

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Location: {FIG_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
