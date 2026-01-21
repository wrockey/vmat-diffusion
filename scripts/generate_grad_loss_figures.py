"""
Generate publication-ready figures for grad_loss_0.1 experiment.

Figures:
1. Training curves (loss and MAE vs epoch)
2. Model comparison bar chart (baseline vs gradient loss)
3. Dose slice comparison (prediction vs ground truth)
4. Gamma analysis visualization

Created: 2026-01-20
Experiment: grad_loss_0.1
Git commit: 5d111a0
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
}

# Paths
RUN_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\runs\grad_loss_0.1')
PRED_DIR = Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\grad_loss_0.1_test')
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
    train_data = df[df['train/loss_epoch'].notna()][['epoch', 'train/loss_epoch', 'train/grad_loss', 'train/mse_loss']].copy()
    train_data = train_data.rename(columns={
        'train/loss_epoch': 'train_loss',
        'train/grad_loss': 'grad_loss',
        'train/mse_loss': 'mse_loss'
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
    ax.axhline(y=3.67, color=COLORS['purple'], linestyle=':', label='Best (3.67 Gy)', linewidth=1.5)

    # Mark best epoch
    best_idx = metrics['val_mae_gy'].idxmin()
    best_epoch = metrics.loc[best_idx, 'epoch']
    best_mae = metrics.loc[best_idx, 'val_mae_gy']
    ax.scatter([best_epoch], [best_mae], color=COLORS['purple'], s=100, zorder=5, marker='*')
    ax.annotate(f'Best: {best_mae:.2f} Gy\n(Epoch {int(best_epoch)})',
                xy=(best_epoch, best_mae), xytext=(best_epoch + 5, best_mae + 1),
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
    """Figure 2: Comparison bar chart of metrics."""
    # Data
    models = ['Baseline U-Net', 'Gradient Loss (0.1)']
    val_mae = [3.73, 3.67]
    test_mae = [1.43, 1.44]
    gamma = [14.2, 27.9]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Validation MAE
    ax = axes[0]
    bars = ax.bar(models, val_mae, color=[COLORS['blue'], COLORS['green']], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) Validation MAE')
    ax.set_ylim(0, 5)
    for bar, val in zip(bars, val_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel B: Test MAE
    ax = axes[1]
    bars = ax.bar(models, test_mae, color=[COLORS['blue'], COLORS['green']], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(B) Test MAE')
    ax.set_ylim(0, 2.5)
    for bar, val in zip(bars, test_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel C: Gamma Pass Rate
    ax = axes[2]
    bars = ax.bar(models, gamma, color=[COLORS['blue'], COLORS['green']], edgecolor='black', linewidth=1.5)
    ax.axhline(y=50, color=COLORS['red'], linestyle='--', label='Target (50%)', linewidth=1.5)
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('(C) Gamma (3%/3mm)')
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right')
    for bar, val in zip(bars, gamma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement annotation
    improvement = gamma[1] - gamma[0]
    ax.annotate(f'+{improvement:.1f}%\nimprovement',
                xy=(1, gamma[1]), xytext=(1.3, 35),
                fontsize=10, ha='left', color=COLORS['green'],
                arrowprops=dict(arrowstyle='->', color=COLORS['green']))

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
    ax.set_title('(B) Prediction (Gradient Loss)')
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
    """Figure 4: Loss component breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics['epoch'], metrics['mse_loss'], label='MSE Loss', color=COLORS['blue'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['grad_loss'], label='Gradient Loss', color=COLORS['orange'], linewidth=2)
    ax.plot(metrics['epoch'], metrics['train_loss'], label='Total Loss', color=COLORS['green'], linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.set_xlim(0, metrics['epoch'].max())
    ax.set_ylim(0, 0.03)

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'fig4_loss_components.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_loss_components.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig4_loss_components.png'}")
    plt.close()


def main():
    print("Generating publication-ready figures...")
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

    print("\n" + "="*50)
    print("All figures generated successfully!")
    print(f"Location: {FIG_DIR}")


if __name__ == '__main__':
    main()
