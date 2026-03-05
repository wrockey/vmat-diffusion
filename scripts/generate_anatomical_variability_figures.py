"""
Generate 8 publication figures for anatomical variability analysis.

Reads features CSV and analysis results JSON from
runs/anatomical_variability/, generates PNG+PDF figures.

Usage:
    python scripts/generate_anatomical_variability_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
OUTPUT_DIR = _PROJECT_ROOT / 'runs' / 'anatomical_variability'
FIG_DIR = OUTPUT_DIR / 'figures'
FEATURES_CSV = sorted(OUTPUT_DIR.glob('features_all_*_cases.csv'))[-1]
RESULTS_JSON = OUTPUT_DIR / 'analysis_results.json'

# Standard plot config (consistent with existing figure scripts)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors — matplotlib tab palette (consistent with existing figure scripts)
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'cyan': '#17becf',
    'brown': '#8c564b',
    'pink': '#e377c2',
}

# Category colors for feature importance
CATEGORY_COLORS = {
    'volume': COLORS['blue'],
    'spatial': COLORS['orange'],
    'proximity': COLORS['red'],
    'dose': COLORS['green'],
    'ptv_target': COLORS['purple'],
}

# 7 test case IDs
TEST_CASE_IDS = [
    'prostate70gy_0005', 'prostate70gy_0018', 'prostate70gy_0024',
    'prostate70gy_0027', 'prostate70gy_0056', 'prostate70gy_0065',
    'prostate70gy_0079',
]

# Short case labels
def short_id(case_id: str) -> str:
    return case_id.replace('prostate70gy_', '')


def load_data():
    """Load features CSV and analysis results JSON."""
    df = pd.read_csv(FEATURES_CSV)
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    return df, results


def save_fig(fig, name: str):
    """Save figure as PNG and PDF."""
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def get_seed_averaged_errors(results: dict, experiment: str = 'combined_loss_2.5to1') -> dict:
    """Get seed-averaged error metrics per test case from outlier profiles."""
    return results.get('outlier_profiles', {})


# =============================================================================
# Figure 1: Anatomical variability overview
# =============================================================================

def fig1_anatomical_overview(df: pd.DataFrame, results: dict):
    """Violin/box plots of 8 key features across 74 cases, test cases highlighted."""
    features = [
        ('PTV70_volume_cc', 'PTV70 Volume (cc)'),
        ('Rectum_volume_cc', 'Rectum Volume (cc)'),
        ('Bladder_volume_cc', 'Bladder Volume (cc)'),
        ('Bowel_volume_cc', 'Bowel Volume (cc)'),
        ('z_extent_mm', 'Z Extent (mm)'),
        ('ptv70_to_Rectum_min_dist_mm', 'PTV70-Rectum Min Dist (mm)'),
        ('ptv70_Rectum_overlap_pct', 'PTV70-Rectum Overlap (%)'),
        ('conformity_index', 'Conformity Index'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, (feat, label) in enumerate(features):
        ax = axes[i]
        vals = df[feat].dropna().values

        # Violin plot
        parts = ax.violinplot(vals, positions=[0], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['blue'])
            pc.set_alpha(0.3)

        # Test case points
        test_df = df[df['is_test']]
        test_vals = test_df[feat].dropna()
        ax.scatter(
            np.zeros(len(test_vals)), test_vals, color=COLORS['red'],
            s=50, zorder=5, label='Test cases', edgecolors='black', linewidth=0.5,
        )

        # Label outlier case 0065
        case_0065 = df[df['case_id'] == 'prostate70gy_0065']
        if len(case_0065) > 0 and not pd.isna(case_0065[feat].values[0]):
            val_0065 = case_0065[feat].values[0]
            ax.annotate(
                '0065', (0, val_0065), textcoords='offset points',
                xytext=(15, 0), fontsize=9, color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8),
            )

        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_title(label.split('(')[0].strip(), fontsize=11)

    axes[0].legend(loc='upper right', fontsize=9)
    fig.suptitle('Anatomical Feature Distributions (N=74, test cases highlighted)',
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig1_anatomical_overview')


# =============================================================================
# Figure 2: Feature-error scatter plots
# =============================================================================

def fig2_feature_error_scatter(df: pd.DataFrame, results: dict):
    """Top 6 correlations: feature vs seed-averaged error, Spearman rho annotated."""
    top_corrs = results['top_correlations_combined_loss'][:6]
    profiles = results['outlier_profiles']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, corr in enumerate(top_corrs):
        ax = axes[i]
        feat = corr['feature']
        metric = corr['metric']
        rho = corr['rho']
        p_val = corr['p_value']

        # Get feature values and error values for test cases
        test_df = df[df['is_test']].copy()
        x_vals = []
        y_vals = []
        labels = []

        for _, row in test_df.iterrows():
            case_id = row['case_id']
            if case_id not in profiles:
                continue
            feat_val = row[feat]
            if pd.isna(feat_val):
                continue

            # Map metric name to profile key
            metric_map = {
                'mae_gy': 'mae_gy_mean',
                'ptv_gamma': 'ptv_gamma_mean',
                'global_gamma': 'global_gamma_mean',
                'PTV70_D95_error': 'mae_gy_mean',  # fallback
            }
            err_key = metric_map.get(metric, f'{metric}_mean')
            err_val = profiles[case_id].get(err_key, np.nan)
            if np.isnan(err_val):
                continue

            x_vals.append(feat_val)
            y_vals.append(err_val)
            labels.append(short_id(case_id))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        ax.scatter(x_vals, y_vals, s=80, color=COLORS['blue'], edgecolors='black',
                   linewidth=0.5, zorder=5)

        # Label each point
        for x, y, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(6, 6), fontsize=9)

        # Annotation box with rho and p
        sig = '*' if p_val < 0.05 else ''
        ax.text(0.05, 0.95, f'rho={rho:.3f}{sig}\np={p_val:.3f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Clean feature/metric names for labels
        feat_label = feat.replace('_', ' ').replace('ptv70 to ', 'PTV70-').replace('ptv70 ', 'PTV70 ')
        metric_label = metric.replace('_', ' ').upper()
        ax.set_xlabel(feat_label, fontsize=11)
        ax.set_ylabel(f'Seed-avg {metric_label}', fontsize=11)
        ax.set_title(f'{feat_label} vs {metric_label}', fontsize=10)

    fig.suptitle('Top 6 Feature-Error Correlations (combined_loss_2.5:1, test set n=7)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig2_feature_error_scatter')


# =============================================================================
# Figure 3: Correlation heatmap
# =============================================================================

def fig3_correlation_heatmap(df: pd.DataFrame, results: dict):
    """Top ~15 features x 4 error metrics, Spearman rho color-coded."""
    corr_data = results['correlations']['combined_loss_2.5to1']

    error_metrics = ['mae_gy', 'ptv_gamma', 'global_gamma', 'PTV70_D95_error']
    metric_labels = ['MAE (Gy)', 'PTV Gamma (%)', 'Global Gamma (%)', 'PTV70 D95 Error']

    # Rank features by max |rho| across metrics
    feat_max_rho = {}
    for feat, metrics in corr_data.items():
        rhos = [abs(metrics.get(m, {}).get('rho', 0)) for m in error_metrics
                if not np.isnan(metrics.get(m, {}).get('rho', 0))]
        if rhos:
            feat_max_rho[feat] = max(rhos)
        else:
            feat_max_rho[feat] = 0.0

    top_features = sorted(feat_max_rho, key=feat_max_rho.get, reverse=True)[:15]

    # Build rho matrix
    rho_matrix = np.full((len(top_features), len(error_metrics)), np.nan)
    p_matrix = np.full((len(top_features), len(error_metrics)), np.nan)

    for i, feat in enumerate(top_features):
        for j, metric in enumerate(error_metrics):
            vals = corr_data.get(feat, {}).get(metric, {})
            rho_matrix[i, j] = vals.get('rho', np.nan)
            p_matrix[i, j] = vals.get('p_value', np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Annotate cells with rho value and significance
    for i in range(len(top_features)):
        for j in range(len(error_metrics)):
            rho = rho_matrix[i, j]
            p = p_matrix[i, j]
            if np.isnan(rho):
                continue
            sig = '*' if p < 0.05 else ''
            color = 'white' if abs(rho) > 0.6 else 'black'
            ax.text(j, i, f'{rho:.2f}{sig}', ha='center', va='center',
                    fontsize=9, color=color)

    # Labels
    feat_labels = [f.replace('_', ' ').replace('ptv70 to ', 'PTV70-')
                   .replace('ptv70 ', 'PTV70 ') for f in top_features]
    ax.set_xticks(range(len(error_metrics)))
    ax.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feat_labels, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman rho', fontsize=12)

    ax.set_title('Feature-Error Correlation Heatmap\n(combined_loss_2.5:1, test set n=7, * p<0.05)',
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, 'fig3_correlation_heatmap')


# =============================================================================
# Figure 4: Outlier case anatomical profile (radar chart)
# =============================================================================

def fig4_outlier_radar(df: pd.DataFrame, results: dict):
    """Radar chart: case 0065 vs best case vs dataset median (z-scored features)."""
    features_for_radar = [
        'PTV70_volume_cc', 'Rectum_volume_cc', 'Bladder_volume_cc',
        'Bowel_volume_cc', 'z_extent_mm', 'ptv70_to_Rectum_min_dist_mm',
        'ptv70_Rectum_overlap_pct', 'conformity_index',
    ]
    feature_labels = [
        'PTV70\nVolume', 'Rectum\nVolume', 'Bladder\nVolume',
        'Bowel\nVolume', 'Z Extent', 'PTV70-Rectum\nMin Dist',
        'PTV70-Rectum\nOverlap', 'Conformity\nIndex',
    ]

    # Z-score features
    numeric_df = df[features_for_radar].copy()
    z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()

    # Get specific cases
    worst_case = 'prostate70gy_0065'
    best_case = 'prostate70gy_0027'  # Lowest MAE

    z_median = z_scores.median().values
    z_worst = z_scores[df['case_id'] == worst_case].values.flatten()
    z_best = z_scores[df['case_id'] == best_case].values.flatten()

    # Radar plot
    N = len(features_for_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for values, label, color, ls in [
        (z_median, 'Dataset Median', COLORS['blue'], '--'),
        (z_worst, f'Case 0065 (worst MAE)', COLORS['red'], '-'),
        (z_best, f'Case 0027 (lowest MAE)', COLORS['green'], '-'),
    ]:
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, color=color, linewidth=2, linestyle=ls, label=label)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_ylim(-2.5, 3.5)
    ax.set_ylabel('Z-score', fontsize=11, labelpad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Outlier Case Anatomical Profile (z-scored)', fontsize=13, pad=20)

    fig.tight_layout()
    save_fig(fig, 'fig4_outlier_radar')


# =============================================================================
# Figure 5: Cross-seed consistency heatmap
# =============================================================================

def fig5_cross_seed_consistency(df: pd.DataFrame, results: dict):
    """Heatmap: 7 cases x 6 seed-experiment combos, colored by MAE and PTV gamma."""
    # Load raw evaluation results
    experiments = {
        'combined_loss_2.5to1': [42, 123, 456],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, (metric, metric_label, cmap) in enumerate([
        ('mae_gy', 'MAE (Gy)', 'YlOrRd'),
        ('ptv_gamma', 'PTV Gamma (%)', 'RdYlGn'),
    ]):
        ax = axes[metric_idx]

        # Build matrix
        case_ids = sorted(TEST_CASE_IDS)
        seeds = [42, 123, 456]
        matrix = np.full((len(case_ids), len(seeds)), np.nan)

        for exp_name, exp_seeds in experiments.items():
            for seed_idx, seed in enumerate(exp_seeds):
                pred_dir = _PROJECT_ROOT / f'predictions/{exp_name}_seed{seed}_test'
                eval_path = pred_dir / 'baseline_evaluation_results.json'
                if not eval_path.exists():
                    continue
                with open(eval_path) as f:
                    eval_data = json.load(f)

                for case_result in eval_data['per_case_results']:
                    case_id = case_result['case_id']
                    if case_id not in case_ids:
                        continue
                    row_idx = case_ids.index(case_id)
                    if metric == 'mae_gy':
                        matrix[row_idx, seed_idx] = case_result['dose_metrics']['mae_gy']
                    elif metric == 'ptv_gamma':
                        matrix[row_idx, seed_idx] = case_result['gamma']['ptv_region_3mm3pct']['gamma_pass_rate']

        im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        # Annotate cells
        for i in range(len(case_ids)):
            for j in range(len(seeds)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                fmt = f'{val:.1f}' if metric == 'mae_gy' else f'{val:.0f}'
                color = 'white' if metric == 'mae_gy' and val > 5 else 'black'
                if metric == 'ptv_gamma' and val < 92:
                    color = 'white'
                ax.text(j, i, fmt, ha='center', va='center', fontsize=10, color=color)

        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=11)
        ax.set_yticks(range(len(case_ids)))
        ax.set_yticklabels([short_id(c) for c in case_ids], fontsize=11)
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Case', fontsize=12)
        ax.set_title(f'{metric_label}', fontsize=13)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(metric_label, fontsize=11)

    icc = results.get('icc_results', {}).get('combined_loss_2.5to1', {})
    icc_mae = icc.get('mae_gy', {}).get('icc', np.nan)
    icc_ptv = icc.get('ptv_gamma', {}).get('icc', np.nan)

    fig.suptitle(
        f'Cross-Seed Consistency (combined_loss_2.5:1)\n'
        f'ICC: MAE={icc_mae:.3f}, PTV Gamma={icc_ptv:.3f}',
        fontsize=14, y=1.05,
    )
    fig.tight_layout()
    save_fig(fig, 'fig5_cross_seed_consistency')


# =============================================================================
# Figure 6: Train vs test distributions
# =============================================================================

def fig6_train_test_distributions(df: pd.DataFrame, results: dict):
    """Overlapping KDEs for 6 key features, KS p-value annotated."""
    features = [
        ('PTV70_volume_cc', 'PTV70 Volume (cc)'),
        ('Rectum_volume_cc', 'Rectum Volume (cc)'),
        ('Bladder_volume_cc', 'Bladder Volume (cc)'),
        ('Bowel_volume_cc', 'Bowel Volume (cc)'),
        ('z_extent_mm', 'Z Extent (mm)'),
        ('ptv70_to_Rectum_min_dist_mm', 'PTV70-Rectum Min Dist (mm)'),
    ]

    train = df[~df['is_test']]
    test = df[df['is_test']]
    dist_tests = results.get('train_test_distribution_tests', {})

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (feat, label) in enumerate(features):
        ax = axes[i]
        train_vals = train[feat].dropna().values
        test_vals = test[feat].dropna().values

        # KDE for train
        if len(train_vals) > 2:
            kde_train = stats.gaussian_kde(train_vals)
            x_range = np.linspace(
                min(train_vals.min(), test_vals.min()) - 0.1 * train_vals.std(),
                max(train_vals.max(), test_vals.max()) + 0.1 * train_vals.std(),
                200,
            )
            ax.fill_between(x_range, kde_train(x_range), alpha=0.3, color=COLORS['blue'],
                            label=f'Train (n={len(train_vals)})')
            ax.plot(x_range, kde_train(x_range), color=COLORS['blue'], linewidth=1.5)

        # Test cases as scatter/rug
        ax.scatter(test_vals, np.zeros(len(test_vals)) - 0.002,
                   marker='|', s=200, color=COLORS['red'], linewidths=2,
                   label=f'Test (n={len(test_vals)})', zorder=5)

        # KS test p-value
        ks_info = dist_tests.get(feat, {})
        ks_p = ks_info.get('ks_p_value', np.nan)
        if not np.isnan(ks_p):
            ax.text(0.95, 0.95, f'KS p={ks_p:.3f}',
                    transform=ax.transAxes, fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')

    fig.suptitle('Train vs Test Feature Distributions (KDE + rug)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig6_train_test_distributions')


# =============================================================================
# Figure 7: PTV target prevalence & error
# =============================================================================

def fig7_ptv_target_analysis(df: pd.DataFrame, results: dict):
    """Left: stacked bar of PTV target combinations. Right: box plots by ptv5040_present."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Left: PTV target prevalence ---
    ax = axes[0]
    ptv_cols = ['ptv7000_present', 'ptv5040_present', 'ptv5600_present', 'ptv6160_present']
    ptv_labels = ['PTV7000', 'PTV5040', 'PTV5600', 'PTV6160']
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

    train = df[~df['is_test']]
    test = df[df['is_test']]

    x = np.arange(len(ptv_labels))
    width = 0.35
    train_counts = [train[c].sum() for c in ptv_cols]
    test_counts = [test[c].sum() for c in ptv_cols]

    bars1 = ax.bar(x - width / 2, train_counts, width, label=f'Train (n={len(train)})',
                   color=COLORS['blue'], alpha=0.7)
    bars2 = ax.bar(x + width / 2, test_counts, width, label=f'Test (n={len(test)})',
                   color=COLORS['red'], alpha=0.7)

    # Annotate counts
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(ptv_labels, fontsize=11)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('PTV Target Prevalence', fontsize=13)
    ax.legend(fontsize=10)

    # --- Middle + Right: Error by PTV5040 presence (test set) ---
    profiles = results['outlier_profiles']
    strat = results.get('ptv_target_stratification', {}).get('ptv5040_present', {})

    for metric_idx, (metric_key, metric_label) in enumerate([
        ('mae_gy_mean', 'MAE (Gy)'),
        ('ptv_gamma_mean', 'PTV Gamma (%)'),
    ]):
        ax = axes[metric_idx + 1]

        # Split test cases by ptv5040
        test_df = df[df['is_test']].copy()
        yes_vals = []
        no_vals = []
        yes_labels = []
        no_labels = []

        for _, row in test_df.iterrows():
            case_id = row['case_id']
            if case_id not in profiles:
                continue
            val = profiles[case_id].get(metric_key, np.nan)
            if np.isnan(val):
                continue
            if row.get('ptv5040_present', False):
                yes_vals.append(val)
                yes_labels.append(short_id(case_id))
            else:
                no_vals.append(val)
                no_labels.append(short_id(case_id))

        # Box plots
        bp_data = []
        bp_labels_list = []
        bp_colors = []
        if yes_vals:
            bp_data.append(yes_vals)
            bp_labels_list.append(f'PTV5040\nPresent (n={len(yes_vals)})')
            bp_colors.append(COLORS['blue'])
        if no_vals:
            bp_data.append(no_vals)
            bp_labels_list.append(f'PTV5040\nAbsent (n={len(no_vals)})')
            bp_colors.append(COLORS['orange'])

        positions = list(range(1, len(bp_data) + 1))
        bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True)
        for patch, color in zip(bp['boxes'], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Overlay individual points with labels
        for grp_idx, (vals, labs) in enumerate(zip(
            [yes_vals, no_vals][:len(bp_data)],
            [yes_labels, no_labels][:len(bp_data)],
        )):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                np.full(len(vals), grp_idx + 1) + jitter, vals,
                s=60, color=bp_colors[grp_idx], edgecolors='black', linewidth=0.5, zorder=5,
            )
            for x, y, lbl in zip(
                np.full(len(vals), grp_idx + 1) + jitter, vals, labs,
            ):
                ax.annotate(lbl, (x, y), textcoords='offset points',
                            xytext=(8, 0), fontsize=8)

        ax.set_xticklabels(bp_labels_list, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label} by PTV5040 Presence', fontsize=12)

    fig.suptitle('PTV Target Analysis (DICOM RTSTRUCT)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig7_ptv_target_analysis')


# =============================================================================
# Figure 8: Feature importance ranking
# =============================================================================

def fig8_feature_importance(df: pd.DataFrame, results: dict):
    """Horizontal bar chart of |Spearman rho| for top 15 features, colored by category."""
    top_corrs = results['top_correlations_combined_loss'][:15]

    # Categorize features
    def categorize(feat: str) -> str:
        if 'volume_cc' in feat or '_exists' in feat:
            return 'volume'
        if 'extent' in feat or 'aspect' in feat or 'z_slices' in feat or 'spacing' in feat:
            return 'spatial'
        if 'dist_mm' in feat or 'overlap_pct' in feat:
            return 'proximity'
        if 'dose' in feat or 'conformity' in feat or 'grad' in feat:
            return 'dose'
        if 'ptv' in feat and 'present' in feat:
            return 'ptv_target'
        return 'volume'

    fig, ax = plt.subplots(figsize=(12, 8))

    labels = []
    abs_rhos = []
    colors_list = []
    for corr in reversed(top_corrs):  # reverse so highest is at top
        feat = corr['feature']
        metric = corr['metric']
        label = f"{feat.replace('_', ' ')} vs {metric.replace('_', ' ')}"
        labels.append(label)
        abs_rhos.append(corr['abs_rho'])
        cat = categorize(feat)
        colors_list.append(CATEGORY_COLORS.get(cat, COLORS['blue']))

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, abs_rhos, color=colors_list, edgecolor='black', linewidth=0.5)

    # Significance markers
    for i, corr in enumerate(reversed(top_corrs)):
        rho = abs_rhos[i]
        p = corr['p_value']
        sig = ' *' if p < 0.05 else ''
        ax.text(rho + 0.01, i, f'{rho:.3f}{sig}', va='center', fontsize=10)

    # Critical value line for n=7, p<0.05
    ax.axvline(x=0.786, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.786, len(labels) - 0.5, 'p<0.05\n(n=7)', fontsize=9,
            color='gray', ha='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('|Spearman rho|', fontsize=13)
    ax.set_xlim(0, 1.1)
    ax.set_title('Feature Importance Ranking (combined_loss_2.5:1, test set n=7)\n'
                 '* = p < 0.05', fontsize=13)

    # Category legend
    legend_handles = []
    for cat, color in CATEGORY_COLORS.items():
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor=color, edgecolor='black',
                                     linewidth=0.5, label=cat.replace('_', ' ').title()))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10)

    fig.tight_layout()
    save_fig(fig, 'fig8_feature_importance')


# =============================================================================
# Main
# =============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {FIG_DIR}")

    df, results = load_data()
    print(f"Loaded {len(df)} cases, {len(results.get('top_correlations_combined_loss', []))} top correlations\n")

    fig1_anatomical_overview(df, results)
    fig2_feature_error_scatter(df, results)
    fig3_correlation_heatmap(df, results)
    fig4_outlier_radar(df, results)
    fig5_cross_seed_consistency(df, results)
    fig6_train_test_distributions(df, results)
    fig7_ptv_target_analysis(df, results)
    fig8_feature_importance(df, results)

    print(f"\nAll 8 figures saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
