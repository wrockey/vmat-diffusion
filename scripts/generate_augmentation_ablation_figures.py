"""
Generate publication-ready figures for augmentation_ablation experiment.

Compares no-augmentation (seed42) vs baseline (seed42 WITH augmentation)
on the same 7 test cases. Generates the standard Medical Physics Figure Set
per CLAUDE.md, reusing figure functions from generate_baseline_v23_figures.py.

Figures:
1. Training curves (loss, MAE, Gamma vs epoch)
2. Dose colorwash (pred vs GT, 3 views)
3. Dose difference map (pred - GT, 3 views)
4. DVH comparison (pred vs GT per structure)
5. Gamma analysis (PTV-region vs global bar chart)
6. Per-case box/strip plots (MAE, Gamma, D95 error)
7. QUANTEC compliance heatmap
8. Femur L/R asymmetry paired bar chart

Created: 2026-02-27
Experiment: augmentation_ablation (seed42, --no_augmentation)
"""

import argparse
import sys
from pathlib import Path

# Import everything from the baseline figure script
# (same figure functions, same plot config, same data loaders)
import generate_baseline_v23_figures as base_figs

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure_augmentation_ablation_paths():
    """Set module-level path variables for the augmentation ablation experiment."""
    base_figs.SEED_LABEL = 'augmentation ablation (no aug, seed 42)'

    exp_name = 'augmentation_ablation_seed42'

    # Run dir — this experiment was started before the path fix,
    # so it lives in scripts/runs/
    for base_dir in [PROJECT_ROOT, PROJECT_ROOT / 'scripts']:
        candidate = base_dir / 'runs' / exp_name
        if candidate.exists():
            base_figs.RUN_DIR = candidate
            break
    else:
        base_figs.RUN_DIR = PROJECT_ROOT / 'scripts' / 'runs' / exp_name

    # Metrics CSV — find latest version
    for ver in ['version_3', 'version_2', 'version_1', 'version_0']:
        candidate = base_figs.RUN_DIR / ver / 'metrics.csv'
        if candidate.exists():
            base_figs.METRICS_CSV = candidate
            break
    else:
        base_figs.METRICS_CSV = base_figs.RUN_DIR / 'version_0' / 'metrics.csv'

    # Eval JSON — predictions are in PROJECT_ROOT/predictions/ (consolidated)
    base_figs.PRED_DIR = PROJECT_ROOT / 'predictions' / f'{exp_name}_test'
    base_figs.EVAL_JSON = base_figs.PRED_DIR / 'baseline_evaluation_results.json'

    # Figure output directory
    base_figs.FIG_DIR = PROJECT_ROOT / 'runs' / 'augmentation_ablation' / 'figures'


def main():
    parser = argparse.ArgumentParser(
        description='Generate figures for augmentation ablation experiment.')
    parser.add_argument('--case', type=str, default=None,
                        help='Case ID for single-case figures (default: below-median MAE case)')
    args = parser.parse_args()

    # Configure paths for augmentation ablation
    configure_augmentation_ablation_paths()

    # Apply plot config
    import matplotlib.pyplot as plt
    plt.rcParams.update(base_figs.PLOT_CONFIG)

    print('=' * 70)
    print(f'  Augmentation Ablation Figure Generation')
    print('=' * 70)
    print()
    print(f'  Run dir:  {base_figs.RUN_DIR}')
    print(f'  Metrics:  {base_figs.METRICS_CSV}')
    print(f'  Eval:     {base_figs.EVAL_JSON}')
    print(f'  Pred dir: {base_figs.PRED_DIR}')
    print(f'  Fig dir:  {base_figs.FIG_DIR}')
    print()

    # Create output directory
    base_figs.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print('Loading data...')
    metrics = base_figs.load_training_metrics()
    if not metrics.empty:
        print(f'  Training metrics: {len(metrics)} epoch records')
        val_mae_data = metrics.dropna(subset=['val_mae_gy'])
        if not val_mae_data.empty:
            best_idx = val_mae_data['val_mae_gy'].idxmin()
            best_mae = val_mae_data.loc[best_idx, 'val_mae_gy']
            best_epoch = val_mae_data.loc[best_idx, 'epoch']
            print(f'  Best val MAE: {best_mae:.2f} Gy at epoch {int(best_epoch)}')

    eval_data = base_figs.load_evaluation_results()
    n_cases = len(eval_data.get('per_case_results', []))
    print(f'  Evaluation results: {n_cases} test cases')

    if eval_data.get('aggregate_metrics'):
        agg = eval_data['aggregate_metrics']
        print(f'  Aggregate MAE: {agg["mae_gy_mean"]:.2f} +/- {agg["mae_gy_std"]:.2f} Gy')
        print(f'  Aggregate Gamma: {agg["gamma_pass_rate_mean"]:.1f} +/- {agg["gamma_pass_rate_std"]:.1f}%')

    # Select representative case (below-median MAE)
    rep_case = base_figs.select_representative_case(eval_data, args.case)
    if rep_case:
        rep_case_data = base_figs.get_case_data(eval_data, rep_case)
        rep_mae = rep_case_data.get('dose_metrics', {}).get('mae_gy', 0)
        print(f'  Representative case: {rep_case} (MAE={rep_mae:.2f} Gy)')
    print()

    # Load volumes for representative case
    volumes = None
    if rep_case:
        print(f'Loading volumes for {rep_case}...')
        volumes = base_figs.load_npz_volumes(rep_case)
        if volumes is not None:
            ct, gt_dose, pred_dose, masks = volumes
            print(f'  CT shape: {ct.shape}')
            print(f'  Dose range (GT): [{gt_dose.min():.1f}, {gt_dose.max():.1f}] Gy')
            print(f'  Dose range (pred): [{pred_dose.min():.1f}, {pred_dose.max():.1f}] Gy')
            centroid = base_figs.find_ptv70_centroid(masks)
            print(f'  PTV70 centroid: {centroid}')
        print()

    # ── Generate figures ──────────────────────────────────────────────────
    print('Generating figures...')
    print()

    figure_funcs = [
        ('1/8', 'Training curves', lambda: base_figs.fig1_training_curves(metrics)),
        ('2/8', 'Dose colorwash', lambda: base_figs.fig2_dose_colorwash(rep_case, volumes)),
        ('3/8', 'Dose difference map', lambda: base_figs.fig3_dose_difference(rep_case, volumes)),
        ('4/8', 'DVH comparison', lambda: base_figs.fig4_dvh_comparison(rep_case, volumes)),
        ('5/8', 'Gamma bar chart', lambda: base_figs.fig5_gamma_bar_chart(eval_data)),
        ('6/8', 'Per-case box plots', lambda: base_figs.fig6_per_case_boxplots(eval_data)),
        ('7/8', 'QUANTEC compliance', lambda: base_figs.fig7_quantec_compliance(eval_data)),
        ('8/8', 'Femur asymmetry', lambda: base_figs.fig8_femur_asymmetry(eval_data)),
    ]

    for num, name, func in figure_funcs:
        print(f'[{num}] {name}...')
        try:
            func()
        except Exception as e:
            print(f'  [ERROR] {e}')

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    generated = list(base_figs.FIG_DIR.glob('*.png'))
    print(f'  Generated {len(generated)} PNG figures + matching PDFs')
    print(f'  Location: {base_figs.FIG_DIR}')
    for f in sorted(generated):
        print(f'    {f.name}')
    print('=' * 70)


if __name__ == '__main__':
    main()
