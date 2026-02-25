"""
Generate publication-ready figures for baseline_v23 experiment.

Pipeline validation run on v2.3 preprocessed data.
Generates the standard Medical Physics Figure Set per CLAUDE.md.
Supports --seed flag to generate figures for any seed (42, 123, 456).

Figures:
1. Training curves (loss, MAE, Gamma vs epoch)
2. Dose colorwash (pred vs GT, 3 views)
3. Dose difference map (pred - GT, 3 views)
4. DVH comparison (pred vs GT per structure)
5. Gamma analysis (PTV-region vs global bar chart)
6. Per-case box/strip plots (MAE, Gamma, D95 error)
7. QUANTEC compliance heatmap
8. Femur L/R asymmetry paired bar chart

Created: 2026-02-24
Updated: 2026-02-25 — made seed-configurable
Experiment: baseline_v23
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
    # bbox_inches and pad_inches are passed directly to savefig() calls
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
COLOR_ORDER = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']

# ── Structure definitions ─────────────────────────────────────────────────────
STRUCTURE_NAMES = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder',
                   'Femur_L', 'Femur_R', 'Bowel']
STRUCTURE_CHANNELS = {name: i for i, name in enumerate(STRUCTURE_NAMES)}

# Structures to plot in DVH (skip PTV56 — typically not present in data)
DVH_STRUCTURES = ['PTV70', 'Rectum', 'Bladder', 'Femur_L', 'Femur_R', 'Bowel']
DVH_COLORS = {
    'PTV70':   COLORS['red'],
    'Rectum':  COLORS['orange'],
    'Bladder': COLORS['blue'],
    'Femur_L': COLORS['green'],
    'Femur_R': COLORS['cyan'],
    'Bowel':   COLORS['purple'],
}

# ── Data paths — set dynamically by configure_paths() ─────────────────────────
RUN_DIR = None
METRICS_CSV = None
EVAL_JSON = None
PRED_DIR = None
DATA_DIR = Path('/home/wrockey/data/processed_npz')
FIG_DIR = None
SEED_LABEL = ''

RX_DOSE_GY = 70.0


def configure_paths(seed: int):
    """Set module-level path variables for the given seed.

    Handles the path inconsistency where seed42 artifacts are under PROJECT_ROOT
    while seed123+ artifacts are under PROJECT_ROOT/scripts/ (launched from scripts/ dir).
    """
    global RUN_DIR, METRICS_CSV, EVAL_JSON, PRED_DIR, FIG_DIR, SEED_LABEL
    SEED_LABEL = f'seed {seed}'

    exp_name = f'baseline_v23_seed{seed}'

    # Find run dir — check both project root and scripts/ subdir
    for base in [PROJECT_ROOT, PROJECT_ROOT / 'scripts']:
        candidate = base / 'runs' / exp_name
        if candidate.exists():
            RUN_DIR = candidate
            break
    else:
        RUN_DIR = PROJECT_ROOT / 'runs' / exp_name  # fallback

    # Find metrics CSV — check version_0 and version_1
    for ver in ['version_1', 'version_0']:
        candidate = RUN_DIR / ver / 'metrics.csv'
        if candidate.exists():
            METRICS_CSV = candidate
            break
    else:
        METRICS_CSV = RUN_DIR / 'version_0' / 'metrics.csv'  # fallback

    # Find eval JSON — check both locations
    for base in [PROJECT_ROOT, PROJECT_ROOT / 'scripts']:
        candidate = base / 'predictions' / f'{exp_name}_test' / 'baseline_evaluation_results.json'
        if candidate.exists():
            EVAL_JSON = candidate
            PRED_DIR = candidate.parent
            break
    else:
        PRED_DIR = PROJECT_ROOT / 'predictions' / f'{exp_name}_test'
        EVAL_JSON = PRED_DIR / 'baseline_evaluation_results.json'

    FIG_DIR = PROJECT_ROOT / 'runs' / 'baseline_v23' / f'figures_seed{seed}'


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_training_metrics() -> pd.DataFrame:
    """Load and parse the PyTorch Lightning metrics CSV.

    Returns a DataFrame with one row per epoch, columns:
        epoch, train_loss, mse_loss, neg_penalty, val_loss, val_mae_gy, val_gamma
    """
    if not METRICS_CSV.exists():
        print(f"  [WARN] Metrics CSV not found: {METRICS_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(METRICS_CSV)

    # Validation metrics: rows where val/mae_gy is not NaN
    val_cols = ['epoch', 'val/loss', 'val/mae_gy', 'val/gamma_3mm3pct']
    val_present = [c for c in val_cols if c in df.columns]
    val_data = df[df['val/mae_gy'].notna()][val_present].copy()
    rename_val = {
        'val/loss': 'val_loss',
        'val/mae_gy': 'val_mae_gy',
        'val/gamma_3mm3pct': 'val_gamma',
    }
    val_data = val_data.rename(columns={k: v for k, v in rename_val.items() if k in val_data.columns})

    # Training epoch-level metrics: rows where train/loss_epoch is not NaN
    train_cols = ['epoch', 'train/loss_epoch', 'train/mse_loss', 'train/neg_penalty']
    train_present = [c for c in train_cols if c in df.columns]
    train_data = df[df['train/loss_epoch'].notna()][train_present].copy()
    rename_train = {
        'train/loss_epoch': 'train_loss',
        'train/mse_loss': 'mse_loss',
        'train/neg_penalty': 'neg_penalty',
    }
    train_data = train_data.rename(columns={k: v for k, v in rename_train.items() if k in train_data.columns})

    # Merge on epoch
    metrics = pd.merge(val_data, train_data, on='epoch', how='outer').sort_values('epoch')
    metrics = metrics.reset_index(drop=True)

    return metrics


def load_evaluation_results() -> dict:
    """Load the evaluation results JSON."""
    if not EVAL_JSON.exists():
        print(f"  [WARN] Evaluation results not found: {EVAL_JSON}")
        return {}
    with open(EVAL_JSON) as f:
        return json.load(f)


def select_representative_case(eval_data: dict, override_case: str = None) -> str:
    """Select the median-MAE case from test results (or use override).

    Returns the case_id string.
    """
    if override_case:
        # Validate that the case exists in eval data
        case_ids = [c['case_id'] for c in eval_data.get('per_case_results', [])]
        if override_case in case_ids:
            return override_case
        print(f"  [WARN] Requested case '{override_case}' not found in results. "
              f"Available: {case_ids}")
        print(f"  Falling back to median-MAE case.")

    cases = eval_data.get('per_case_results', [])
    if not cases:
        return ''

    # Sort by MAE, pick case just below median (avoids showing worst-case artifacts)
    sorted_cases = sorted(cases, key=lambda c: c['dose_metrics']['mae_gy'])
    median_idx = max(0, len(sorted_cases) // 2 - 1)
    return sorted_cases[median_idx]['case_id']


def get_case_data(eval_data: dict, case_id: str) -> dict:
    """Get per-case result dict for a given case_id."""
    for case in eval_data.get('per_case_results', []):
        if case['case_id'] == case_id:
            return case
    return {}


def load_npz_volumes(case_id: str):
    """Load GT and prediction NPZ volumes for a case.

    Returns (ct, gt_dose_gy, pred_dose_gy, masks) or None if files missing.
    ct: (Y, X, Z) float32 [0,1]
    gt_dose_gy: (Y, X, Z) float32 in Gy
    pred_dose_gy: (Y, X, Z) float32 in Gy
    masks: (8, Y, X, Z) uint8
    """
    gt_path = DATA_DIR / f'{case_id}.npz'
    pred_path = PRED_DIR / f'{case_id}_pred.npz'

    if not gt_path.exists():
        print(f"  [WARN] GT NPZ not found: {gt_path}")
        return None
    if not pred_path.exists():
        print(f"  [WARN] Pred NPZ not found: {pred_path}")
        return None

    gt_npz = np.load(gt_path, allow_pickle=True)
    pred_npz = np.load(pred_path, allow_pickle=True)

    ct = gt_npz['ct']                      # (Y, X, Z), [0,1]
    gt_dose = gt_npz['dose'] * RX_DOSE_GY  # normalized -> Gy
    pred_dose = pred_npz['dose'] * RX_DOSE_GY
    masks = gt_npz['masks']                 # (8, Y, X, Z)

    # Handle shape mismatch between pred and GT (pred may differ if sliding
    # window output was slightly different)
    if pred_dose.shape != gt_dose.shape:
        print(f"  [WARN] Shape mismatch: pred={pred_dose.shape}, gt={gt_dose.shape}")
        # Crop to minimum common shape
        min_shape = tuple(min(p, g) for p, g in zip(pred_dose.shape, gt_dose.shape))
        pred_dose = pred_dose[:min_shape[0], :min_shape[1], :min_shape[2]]
        gt_dose = gt_dose[:min_shape[0], :min_shape[1], :min_shape[2]]
        ct = ct[:min_shape[0], :min_shape[1], :min_shape[2]]
        masks = masks[:, :min_shape[0], :min_shape[1], :min_shape[2]]

    return ct, gt_dose, pred_dose, masks


def find_ptv70_centroid(masks: np.ndarray) -> tuple:
    """Find the centroid (y, x, z) of PTV70 mask."""
    ptv70 = masks[STRUCTURE_CHANNELS['PTV70']]
    if ptv70.sum() == 0:
        # Fallback to volume center
        return tuple(s // 2 for s in ptv70.shape)
    coords = np.argwhere(ptv70 > 0)
    centroid = coords.mean(axis=0).astype(int)
    return tuple(centroid)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure functions
# ═══════════════════════════════════════════════════════════════════════════════

def save_figure(fig, name: str):
    """Save figure as both PNG (300 DPI) and PDF to FIG_DIR."""
    png_path = FIG_DIR / f'{name}.png'
    pdf_path = FIG_DIR / f'{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


def fig1_training_curves(metrics: pd.DataFrame):
    """Figure 1: Training curves.

    (A) Train loss and val loss vs epoch.
    (B) Val MAE (Gy) and val Gamma (%) vs epoch.
    Best checkpoint epoch marked with vertical dashed line.
    Overfitting ratio annotated.
    """
    if metrics.empty:
        print("  [SKIP] No training metrics available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Loss curves ---
    ax = axes[0]
    train_loss = metrics.dropna(subset=['train_loss'])
    val_loss = metrics.dropna(subset=['val_loss'])

    ax.plot(train_loss['epoch'], train_loss['train_loss'],
            label='Train Loss', color=COLORS['blue'], linewidth=2)
    ax.plot(val_loss['epoch'], val_loss['val_loss'],
            label='Val Loss', color=COLORS['orange'], linewidth=2)

    # Find best val MAE epoch for vertical line
    val_mae_data = metrics.dropna(subset=['val_mae_gy'])
    if not val_mae_data.empty:
        best_idx = val_mae_data['val_mae_gy'].idxmin()
        best_epoch = val_mae_data.loc[best_idx, 'epoch']
        ax.axvline(x=best_epoch, color=COLORS['black'], linestyle='--',
                   alpha=0.5, linewidth=1, label=f'Best ckpt (ep {int(best_epoch)})')

    # Overfitting ratio at final epoch
    final_train = train_loss['train_loss'].iloc[-1] if not train_loss.empty else None
    final_val = val_loss['val_loss'].iloc[-1] if not val_loss.empty else None
    if final_train is not None and final_val is not None and final_train > 0:
        overfit_ratio = final_val / final_train
        ax.annotate(f'Overfit ratio: {overfit_ratio:.1f}x',
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(A) Training and Validation Loss')
    ax.legend(loc='upper right')
    ax.set_xlim(0, metrics['epoch'].max())

    # --- Panel B: Val MAE and Val Gamma ---
    ax = axes[1]
    ax_gamma = ax.twinx()

    if not val_mae_data.empty:
        line1, = ax.plot(val_mae_data['epoch'], val_mae_data['val_mae_gy'],
                         label='Val MAE (Gy)', color=COLORS['blue'], linewidth=2)

        # Mark best MAE
        best_mae = val_mae_data.loc[best_idx, 'val_mae_gy']
        ax.scatter([best_epoch], [best_mae], color=COLORS['blue'], s=100,
                   zorder=5, marker='*')
        ax.annotate(f'Best: {best_mae:.2f} Gy\n(Epoch {int(best_epoch)})',
                    xy=(best_epoch, best_mae),
                    xytext=(best_epoch + metrics['epoch'].max() * 0.05, best_mae + 1),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    val_gamma_data = metrics.dropna(subset=['val_gamma'])
    if not val_gamma_data.empty:
        line2, = ax_gamma.plot(val_gamma_data['epoch'], val_gamma_data['val_gamma'],
                               label='Val Gamma 3%/3mm (%)', color=COLORS['green'],
                               linewidth=2, linestyle='--')
        ax_gamma.set_ylabel('Gamma Pass Rate (%)', color=COLORS['green'])
        ax_gamma.tick_params(axis='y', labelcolor=COLORS['green'])
        ax_gamma.spines['right'].set_visible(True)

    # Best checkpoint vertical line
    if not val_mae_data.empty:
        ax.axvline(x=best_epoch, color=COLORS['black'], linestyle='--',
                   alpha=0.5, linewidth=1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (Gy)', color=COLORS['blue'])
    ax.tick_params(axis='y', labelcolor=COLORS['blue'])
    ax.set_title('(B) Validation MAE and Gamma Pass Rate')
    ax.set_xlim(0, metrics['epoch'].max())

    # Combined legend
    lines = []
    labels = []
    if not val_mae_data.empty:
        lines.append(line1)
        labels.append('Val MAE (Gy)')
    if not val_gamma_data.empty:
        lines.append(line2)
        labels.append('Val Gamma 3%/3mm (%)')
    ax.legend(lines, labels, loc='upper right')

    fig.tight_layout()
    save_figure(fig, 'fig1_training_curves')


def fig2_dose_colorwash(case_id: str, volumes):
    """Figure 2: Dose colorwash — pred vs GT, axial/coronal/sagittal through PTV70 centroid.

    2x3 grid: top row = prediction, bottom row = ground truth.
    Columns: axial, coronal, sagittal.
    CT as grayscale background, dose overlay with transparency.
    """
    if volumes is None:
        print("  [SKIP] Volume data not available.")
        return

    ct, gt_dose, pred_dose, masks = volumes
    cy, cx, cz = find_ptv70_centroid(masks)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    dose_vmin, dose_vmax = 0, RX_DOSE_GY * 1.1  # 0 to 77 Gy
    dose_threshold = 5.0  # Only show dose > 5 Gy

    titles_col = ['Axial (z={})'.format(cz),
                  'Coronal (y={})'.format(cy),
                  'Sagittal (x={})'.format(cx)]
    titles_row = ['Predicted', 'Ground Truth']

    for row_idx, (dose, row_label) in enumerate([(pred_dose, 'Predicted'),
                                                  (gt_dose, 'Ground Truth')]):
        # Extract slices: axes are (Y, X, Z)
        slices = [
            (ct[:, :, cz], dose[:, :, cz]),       # Axial: Y x X
            (ct[cy, :, :], dose[cy, :, :]),        # Coronal: X x Z
            (ct[:, cx, :], dose[:, cx, :]),         # Sagittal: Y x Z
        ]

        for col_idx, (ct_slice, dose_slice) in enumerate(slices):
            ax = axes[row_idx, col_idx]

            # CT background (grayscale)
            ax.imshow(ct_slice, cmap='gray', aspect='auto',
                      vmin=0, vmax=1, origin='lower')

            # Dose overlay with transparency (mask low dose)
            dose_masked = np.ma.masked_where(dose_slice < dose_threshold, dose_slice)
            im = ax.imshow(dose_masked, cmap='jet', aspect='auto',
                           vmin=dose_vmin, vmax=dose_vmax, alpha=0.6,
                           origin='lower')

            ax.set_title(f'{row_label} - {titles_col[col_idx]}', fontsize=12)
            ax.axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Dose (Gy)', fontsize=12)

    fig.suptitle(f'Dose Colorwash: {case_id}', fontsize=16, y=0.98)
    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.89)
    save_figure(fig, 'fig2_dose_colorwash')


def fig3_dose_difference(case_id: str, volumes):
    """Figure 3: Dose difference (pred - GT) overlaid on CT, 3 views.

    RdBu_r diverging colormap centered at 0, colorbar in Gy.
    """
    if volumes is None:
        print("  [SKIP] Volume data not available.")
        return

    ct, gt_dose, pred_dose, masks = volumes
    cy, cx, cz = find_ptv70_centroid(masks)
    diff = pred_dose - gt_dose

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Determine symmetric color range
    diff_max = min(np.percentile(np.abs(diff), 99), 30.0)
    norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)

    view_labels = ['Axial (z={})'.format(cz),
                   'Coronal (y={})'.format(cy),
                   'Sagittal (x={})'.format(cx)]

    ct_slices = [ct[:, :, cz], ct[cy, :, :], ct[:, cx, :]]
    diff_slices = [diff[:, :, cz], diff[cy, :, :], diff[:, cx, :]]

    for idx, (ct_slice, diff_slice, label) in enumerate(
            zip(ct_slices, diff_slices, view_labels)):
        ax = axes[idx]

        # CT background
        ax.imshow(ct_slice, cmap='gray', aspect='auto', vmin=0, vmax=1,
                  origin='lower')

        # Difference overlay
        im = ax.imshow(diff_slice, cmap='RdBu_r', aspect='auto',
                       norm=norm, alpha=0.7, origin='lower')

        ax.set_title(label, fontsize=12)
        ax.axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Dose Difference (Gy): Pred - GT', fontsize=12)

    fig.suptitle(f'Dose Difference Map: {case_id}', fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig3_dose_difference')


def fig4_dvh_comparison(case_id: str, volumes):
    """Figure 4: DVH curves — GT (solid) vs predicted (dashed) per structure.

    Computes cumulative DVH from dose arrays and structure masks.
    """
    if volumes is None:
        print("  [SKIP] Volume data not available.")
        return

    ct, gt_dose, pred_dose, masks = volumes

    fig, ax = plt.subplots(figsize=(10, 7))

    dose_bins = np.linspace(0, RX_DOSE_GY * 1.15, 500)

    for struct_name in DVH_STRUCTURES:
        ch = STRUCTURE_CHANNELS[struct_name]
        mask = masks[ch]

        if mask.sum() == 0:
            print(f"  [INFO] Structure '{struct_name}' has empty mask — skipping DVH.")
            continue

        color = DVH_COLORS[struct_name]

        # Compute cumulative DVH: fraction of structure volume receiving >= dose
        gt_voxels = gt_dose[mask > 0]
        pred_voxels = pred_dose[mask > 0]

        gt_dvh = np.array([100.0 * np.mean(gt_voxels >= d) for d in dose_bins])
        pred_dvh = np.array([100.0 * np.mean(pred_voxels >= d) for d in dose_bins])

        ax.plot(dose_bins, gt_dvh, color=color, linewidth=2, linestyle='-',
                label=f'{struct_name} (GT)')
        ax.plot(dose_bins, pred_dvh, color=color, linewidth=2, linestyle='--',
                label=f'{struct_name} (Pred)')

    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume (%)')
    ax.set_title(f'DVH Comparison: {case_id}')
    ax.set_xlim(0, RX_DOSE_GY * 1.15)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'fig4_dvh_comparison')


def fig5_gamma_bar_chart(eval_data: dict):
    """Figure 5: PTV-region vs global gamma bar chart for all test cases.

    Per-voxel gamma maps are not stored in predictions, so we create a grouped
    bar chart comparing global and PTV-region gamma pass rates.
    """
    cases = eval_data.get('per_case_results', [])
    if not cases:
        print("  [SKIP] No evaluation results available.")
        return

    case_ids = [c['case_id'].replace('prostate70gy_', 'P') for c in cases]
    global_gamma = [c['gamma']['global_3mm3pct']['gamma_pass_rate'] for c in cases]
    ptv_gamma = [c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'] for c in cases]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(case_ids))
    width = 0.35

    bars1 = ax.bar(x - width/2, global_gamma, width, label='Global Gamma 3%/3mm',
                   color=COLORS['blue'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ptv_gamma, width, label='PTV-region Gamma 3%/3mm',
                   color=COLORS['orange'], edgecolor='black', linewidth=0.5)

    # 95% target line for PTV gamma
    ax.axhline(y=95, color=COLORS['red'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='Clinical target (95%)')

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Case')
    ax.set_ylabel('Gamma Pass Rate (%)')
    ax.set_title('Gamma Pass Rate: Global vs PTV-region (3%/3mm)')
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'fig5_gamma_bar_chart')


def fig6_per_case_boxplots(eval_data: dict):
    """Figure 6: Per-case strip/box plots for MAE, Global Gamma, PTV Gamma, PTV70 D95 error.

    One point per case (n=7), with individual points visible.
    """
    cases = eval_data.get('per_case_results', [])
    if not cases:
        print("  [SKIP] No evaluation results available.")
        return

    mae_vals = [c['dose_metrics']['mae_gy'] for c in cases]
    global_gamma = [c['gamma']['global_3mm3pct']['gamma_pass_rate'] for c in cases]
    ptv_gamma = [c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'] for c in cases]

    # PTV70 D95 error
    d95_errors = []
    for c in cases:
        ptv70_dvh = c.get('dvh_metrics', {}).get('PTV70', {})
        if 'D95_error' in ptv70_dvh:
            d95_errors.append(ptv70_dvh['D95_error'])
        else:
            d95_errors.append(np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    panels = [
        (axes[0], mae_vals, 'MAE (Gy)', '(A) Mean Absolute Error', COLORS['blue']),
        (axes[1], global_gamma, 'Gamma Pass Rate (%)', '(B) Global Gamma 3%/3mm', COLORS['green']),
        (axes[2], ptv_gamma, 'Gamma Pass Rate (%)', '(C) PTV-region Gamma 3%/3mm', COLORS['orange']),
        (axes[3], d95_errors, 'D95 Error (Gy)', '(D) PTV70 D95 Error', COLORS['red']),
    ]

    for ax, values, ylabel, title, color in panels:
        values_clean = [v for v in values if not np.isnan(v)]
        if not values_clean:
            ax.set_title(title + ' (no data)')
            continue

        bp = ax.boxplot(values_clean, patch_artist=True, widths=0.5,
                        boxprops=dict(facecolor=color, alpha=0.3),
                        medianprops=dict(color=COLORS['black'], linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Strip plot (individual points with jitter)
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(values_clean))
        ax.scatter(np.ones(len(values_clean)) + jitter, values_clean,
                   color=color, s=60, zorder=5, edgecolors='black', linewidths=0.5)

        # Annotate mean
        mean_val = np.mean(values_clean)
        std_val = np.std(values_clean)
        ax.axhline(y=mean_val, color=color, linestyle=':', alpha=0.5, linewidth=1)
        ax.text(1.35, mean_val, f'{mean_val:.2f} +/- {std_val:.2f}',
                fontsize=9, va='center', color=color)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([1])
        ax.set_xticklabels([f'n={len(values_clean)}'])

    fig.suptitle(f'Per-case Test Metrics (baseline_v23, {SEED_LABEL})', fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig6_per_case_boxplots')


def fig7_quantec_compliance(eval_data: dict):
    """Figure 7: QUANTEC compliance heatmap.

    Rows = cases, columns = structure/constraint combos.
    Green = pass, red = fail, gray = not evaluated.
    """
    cases = eval_data.get('per_case_results', [])
    if not cases:
        print("  [SKIP] No evaluation results available.")
        return

    # Collect all unique constraint labels across all cases
    constraint_labels = []
    for c in cases:
        structs = c.get('clinical_constraints', {}).get('structures', {})
        for struct_name, struct_data in structs.items():
            for chk in struct_data.get('constraints_checked', []):
                label = f"{struct_name}\n{chk['metric']}"
                if label not in constraint_labels:
                    constraint_labels.append(label)

    if not constraint_labels:
        print("  [SKIP] No clinical constraints data found.")
        return

    case_ids = [c['case_id'].replace('prostate70gy_', 'P') for c in cases]
    n_cases = len(case_ids)
    n_constraints = len(constraint_labels)

    # Build matrix: 1=pass, 0=fail, -1=not evaluated
    matrix = np.full((n_cases, n_constraints), -1, dtype=float)

    for i, c in enumerate(cases):
        structs = c.get('clinical_constraints', {}).get('structures', {})
        for struct_name, struct_data in structs.items():
            for chk in struct_data.get('constraints_checked', []):
                label = f"{struct_name}\n{chk['metric']}"
                if label in constraint_labels:
                    j = constraint_labels.index(label)
                    matrix[i, j] = 1.0 if chk['passed'] else 0.0

    # Custom colormap: gray=-1, red=0, green=1
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#CCCCCC', '#D55E00', '#009E73'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(12, n_constraints * 0.8), max(4, n_cases * 0.6)))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(n_constraints))
    ax.set_xticklabels(constraint_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_cases))
    ax.set_yticklabels(case_ids, fontsize=10)
    ax.set_title('QUANTEC Constraint Compliance per Case', fontsize=14)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#009E73', label='Pass'),
        mpatches.Patch(color='#D55E00', label='Fail'),
        mpatches.Patch(color='#CCCCCC', label='Not evaluated'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=10)

    # Add text annotations
    for i in range(n_cases):
        for j in range(n_constraints):
            val = matrix[i, j]
            if val == 1:
                ax.text(j, i, 'P', ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
            elif val == 0:
                ax.text(j, i, 'F', ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')

    fig.tight_layout()
    save_figure(fig, 'fig7_quantec_compliance')


def fig8_femur_asymmetry(eval_data: dict):
    """Figure 8: Femur L vs R MAE per case — highlighting systematic L/R asymmetry.

    Paired bar chart with individual case pairs connected by lines.
    """
    cases = eval_data.get('per_case_results', [])
    if not cases:
        print("  [SKIP] No evaluation results available.")
        return

    case_ids = []
    femur_l_mae = []
    femur_r_mae = []

    for c in cases:
        struct_mae = c.get('dose_metrics', {}).get('per_structure_mae', {})
        fl = struct_mae.get('Femur_L')
        fr = struct_mae.get('Femur_R')
        if fl is not None and fr is not None:
            case_ids.append(c['case_id'].replace('prostate70gy_', 'P'))
            femur_l_mae.append(fl)
            femur_r_mae.append(fr)

    if not case_ids:
        print("  [SKIP] No Femur MAE data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Paired bar chart ---
    ax = axes[0]
    x = np.arange(len(case_ids))
    width = 0.35

    bars_l = ax.bar(x - width/2, femur_l_mae, width, label='Femur_L',
                    color=COLORS['green'], edgecolor='black', linewidth=0.5)
    bars_r = ax.bar(x + width/2, femur_r_mae, width, label='Femur_R',
                    color=COLORS['cyan'], edgecolor='black', linewidth=0.5)

    # Connect paired bars with lines
    for i in range(len(case_ids)):
        ax.plot([x[i] - width/2, x[i] + width/2],
                [femur_l_mae[i], femur_r_mae[i]],
                color=COLORS['black'], linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Case')
    ax.set_ylabel('MAE (Gy)')
    ax.set_title('(A) Femur L vs R MAE per Case')
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.legend()

    # --- Panel B: Difference (L - R) ---
    ax = axes[1]
    diffs = [l - r for l, r in zip(femur_l_mae, femur_r_mae)]
    colors_bar = [COLORS['red'] if d > 0 else COLORS['blue'] for d in diffs]

    ax.bar(x, diffs, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color=COLORS['black'], linewidth=1)

    mean_diff = np.mean(diffs)
    ax.axhline(y=mean_diff, color=COLORS['orange'], linestyle='--', linewidth=1.5,
               label=f'Mean diff: {mean_diff:.2f} Gy')

    ax.set_xlabel('Case')
    ax.set_ylabel('MAE Difference: Femur_L - Femur_R (Gy)')
    ax.set_title('(B) Femur Asymmetry (L - R)')
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.legend()

    # Annotate which side is systematically worse
    n_l_worse = sum(1 for d in diffs if d > 0)
    n_r_worse = sum(1 for d in diffs if d < 0)
    summary_text = f'Femur_L worse: {n_l_worse}/{len(diffs)} cases\nFemur_R worse: {n_r_worse}/{len(diffs)} cases'
    ax.annotate(summary_text, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Femur L/R Asymmetry Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig8_femur_asymmetry')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures for baseline_v23 experiment.')
    parser.add_argument('--case', type=str, default=None,
                        help='Case ID for single-case figures (default: median MAE case)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to generate figures for (default: 42)')
    args = parser.parse_args()

    # Configure paths for requested seed
    configure_paths(args.seed)

    # Apply plot config
    plt.rcParams.update(PLOT_CONFIG)

    print('=' * 70)
    print(f'  baseline_v23 Figure Generation ({SEED_LABEL})')
    print('=' * 70)
    print()

    # Create output directory
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {FIG_DIR}')
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print('Loading data...')
    metrics = load_training_metrics()
    if not metrics.empty:
        print(f'  Training metrics: {len(metrics)} epoch records')
        val_mae_data = metrics.dropna(subset=['val_mae_gy'])
        if not val_mae_data.empty:
            best_idx = val_mae_data['val_mae_gy'].idxmin()
            best_mae = val_mae_data.loc[best_idx, 'val_mae_gy']
            best_epoch = val_mae_data.loc[best_idx, 'epoch']
            print(f'  Best val MAE: {best_mae:.2f} Gy at epoch {int(best_epoch)}')

    eval_data = load_evaluation_results()
    n_cases = len(eval_data.get('per_case_results', []))
    print(f'  Evaluation results: {n_cases} test cases')

    if eval_data.get('aggregate_metrics'):
        agg = eval_data['aggregate_metrics']
        print(f'  Aggregate MAE: {agg["mae_gy_mean"]:.2f} +/- {agg["mae_gy_std"]:.2f} Gy')
        print(f'  Aggregate Gamma: {agg["gamma_pass_rate_mean"]:.1f} +/- {agg["gamma_pass_rate_std"]:.1f}%')

    # Select representative case
    rep_case = select_representative_case(eval_data, args.case)
    if rep_case:
        rep_case_data = get_case_data(eval_data, rep_case)
        rep_mae = rep_case_data.get('dose_metrics', {}).get('mae_gy', 0)
        print(f'  Representative case: {rep_case} (MAE={rep_mae:.2f} Gy)')
    else:
        print('  [WARN] No representative case selected.')
    print()

    # Load volumes for representative case
    volumes = None
    if rep_case:
        print(f'Loading volumes for {rep_case}...')
        volumes = load_npz_volumes(rep_case)
        if volumes is not None:
            ct, gt_dose, pred_dose, masks = volumes
            print(f'  CT shape: {ct.shape}')
            print(f'  Dose shape (GT): {gt_dose.shape}, range [{gt_dose.min():.1f}, {gt_dose.max():.1f}] Gy')
            print(f'  Dose shape (pred): {pred_dose.shape}, range [{pred_dose.min():.1f}, {pred_dose.max():.1f}] Gy')
            print(f'  Masks shape: {masks.shape}')
            centroid = find_ptv70_centroid(masks)
            print(f'  PTV70 centroid (y,x,z): {centroid}')
        print()

    # ── Generate figures ──────────────────────────────────────────────────
    print('Generating figures...')
    print()

    print('[1/8] Training curves...')
    try:
        fig1_training_curves(metrics)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[2/8] Dose colorwash...')
    try:
        fig2_dose_colorwash(rep_case, volumes)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[3/8] Dose difference map...')
    try:
        fig3_dose_difference(rep_case, volumes)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[4/8] DVH comparison...')
    try:
        fig4_dvh_comparison(rep_case, volumes)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[5/8] Gamma bar chart...')
    try:
        fig5_gamma_bar_chart(eval_data)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[6/8] Per-case box plots...')
    try:
        fig6_per_case_boxplots(eval_data)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[7/8] QUANTEC compliance heatmap...')
    try:
        fig7_quantec_compliance(eval_data)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('[8/8] Femur asymmetry...')
    try:
        fig8_femur_asymmetry(eval_data)
    except Exception as e:
        print(f'  [ERROR] {e}')

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    generated = list(FIG_DIR.glob('*.png'))
    print(f'  Generated {len(generated)} PNG figures + matching PDFs')
    print(f'  Location: {FIG_DIR}')
    for f in sorted(generated):
        print(f'    {f.name}')
    print('=' * 70)


if __name__ == '__main__':
    main()
