"""
Multi-seed aggregation and statistical testing for VMAT dose prediction.

Handles:
    - Aggregating results across seeds (per-case means, then global)
    - Bootstrap confidence intervals (case-level resampling)
    - Wilcoxon signed-rank tests (on per-case means, NOT pooled seed×case)
    - Holm-Bonferroni multiple comparison correction
    - Publication-ready table formatting
    - Standardized JSON output with numpy serialization

Statistical methodology notes:
    - Seeds within the same case are NOT independent observations.
      We average across seeds per-case first, then treat cases as the
      independent unit (n = number of test cases).
    - Holm-Bonferroni is uniformly more powerful than Bonferroni with
      the same family-wise error rate guarantee.
    - Bootstrap CIs resample at the case level (after seed-averaging)
      to generalize to new patients.

Version: 1.0.0
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval_core import FRAMEWORK_VERSION


# =============================================================================
# Multi-Seed Aggregation
# =============================================================================

def aggregate_across_seeds(
    seed_results: List[Dict],
    seed_ids: List[int],
    metric_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Aggregate evaluation results across multiple seeds.

    For each metric, computes per-case mean across seeds, then global
    mean ± std across cases with bootstrap CIs.

    Args:
        seed_results: List of per-seed result dicts. Each dict maps
            case_id to an EvaluationResult.to_dict(). Structure:
            [
                {case_id: result_dict, ...},  # seed 0
                {case_id: result_dict, ...},  # seed 1
                ...
            ]
        seed_ids: List of seed values (e.g. [42, 123, 456])
        metric_keys: Metrics to aggregate. If None, uses default set.

    Returns:
        Dict with:
            - framework_version: str
            - n_seeds: int
            - n_cases: int
            - seed_ids: list
            - aggregate: dict of metric -> {mean, std, ci_lower, ci_upper}
            - per_case: dict of case_id -> metric -> {mean, std, values_by_seed}
            - per_seed: list of {seed, metrics: {metric -> mean across cases}}
    """
    if metric_keys is None:
        metric_keys = _default_metric_keys()

    n_seeds = len(seed_results)
    assert n_seeds == len(seed_ids), \
        f"seed_results ({n_seeds}) and seed_ids ({len(seed_ids)}) length mismatch"

    # Collect all case IDs (union across seeds)
    all_case_ids = set()
    for sr in seed_results:
        all_case_ids.update(sr.keys())
    all_case_ids = sorted(all_case_ids)
    n_cases = len(all_case_ids)

    # Build per_case: for each case and metric, collect values across seeds
    per_case = {}
    for case_id in all_case_ids:
        per_case[case_id] = {}
        for mk in metric_keys:
            values = []
            values_by_seed = {}
            for seed_idx, sr in enumerate(seed_results):
                if case_id in sr:
                    val = _extract_metric(sr[case_id], mk)
                    if val is not None:
                        values.append(val)
                        values_by_seed[str(seed_ids[seed_idx])] = val
            if values:
                per_case[case_id][mk] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'values_by_seed': values_by_seed,
                }

    # Build aggregate: per-case means -> global mean ± std with CI
    aggregate = {}
    for mk in metric_keys:
        case_means = []
        for case_id in all_case_ids:
            if mk in per_case.get(case_id, {}):
                case_means.append(per_case[case_id][mk]['mean'])
        if case_means:
            arr = np.array(case_means)
            ci = bootstrap_ci(arr)
            aggregate[mk] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'n_cases_with_data': len(case_means),
            }

    # Build per_seed summary
    per_seed = []
    for seed_idx, sr in enumerate(seed_results):
        seed_summary = {'seed': seed_ids[seed_idx], 'metrics': {}}
        for mk in metric_keys:
            values = []
            for case_id in sr:
                val = _extract_metric(sr[case_id], mk)
                if val is not None:
                    values.append(val)
            if values:
                seed_summary['metrics'][mk] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'n_cases': len(values),
                }
        per_seed.append(seed_summary)

    return {
        'framework_version': FRAMEWORK_VERSION,
        'n_seeds': n_seeds,
        'n_cases': n_cases,
        'seed_ids': seed_ids,
        'aggregate': aggregate,
        'per_case': per_case,
        'per_seed': per_seed,
    }


def _default_metric_keys() -> List[str]:
    """Default metrics to aggregate."""
    return [
        'dose_metrics.mae_gy',
        'dose_metrics.rmse_gy',
        'dose_metrics.max_error_gy',
        'gamma.global_3mm3pct.gamma_pass_rate',
        'gamma.ptv_region_3mm3pct.gamma_pass_rate',
        'clinical_constraints.compliance_rate',
        'dvh_metrics.PTV70.D95_error',
        'dvh_metrics.PTV70.pred_D95',
        'dvh_metrics.PTV56.D95_error',
        'dvh_metrics.PTV56.pred_D95',
        'dvh_metrics.Rectum.pred_V70',
        'dvh_metrics.Bladder.pred_V70',
    ]


def _extract_metric(result_dict: Dict, metric_key: str) -> Optional[float]:
    """
    Extract a metric value from a nested result dict using dot notation.

    E.g. 'dose_metrics.mae_gy' -> result_dict['dose_metrics']['mae_gy']
    """
    parts = metric_key.split('.')
    current = result_dict
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval using percentile method.

    Resamples at the case level (each element is one case's mean).

    Args:
        values: Array of per-case values (after seed-averaging)
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        (ci_lower, ci_upper) tuple
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < 2:
        return (float(values[0]) if n == 1 else float('nan'),
                float(values[0]) if n == 1 else float('nan'))

    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        bootstrap_means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return (lower, upper)


# =============================================================================
# Statistical Comparison (Wilcoxon)
# =============================================================================

def wilcoxon_comparison(
    results_a: Dict,
    results_b: Dict,
    metric_key: str,
    n_comparisons: int = 1,
    alpha: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test comparing two experiments on a given metric.

    IMPORTANT: Uses per-case means (after averaging across seeds) as the
    unit of observation. This correctly treats seed×case pairs as
    non-independent. The sample size is n = number of test cases.

    Args:
        results_a: Aggregate results from experiment A (from aggregate_across_seeds)
        results_b: Aggregate results from experiment B
        metric_key: Dot-notation metric to compare (e.g. 'dose_metrics.mae_gy')
        n_comparisons: Total number of comparisons for Holm-Bonferroni
        alpha: Significance level

    Returns:
        Dict with statistic, p_value, p_adjusted, significant, effect_size,
        direction, n_pairs, etc.
    """
    from scipy.stats import wilcoxon as scipy_wilcoxon

    # Extract per-case means
    per_case_a = results_a.get('per_case', {})
    per_case_b = results_b.get('per_case', {})

    # Find common cases
    common_cases = sorted(set(per_case_a.keys()) & set(per_case_b.keys()))

    values_a = []
    values_b = []
    for case_id in common_cases:
        va = per_case_a.get(case_id, {}).get(metric_key, {}).get('mean')
        vb = per_case_b.get(case_id, {}).get(metric_key, {}).get('mean')
        if va is not None and vb is not None:
            values_a.append(va)
            values_b.append(vb)

    n_pairs = len(values_a)

    if n_pairs < 6:
        return {
            'metric': metric_key,
            'n_pairs': n_pairs,
            'error': f'Insufficient pairs ({n_pairs}) for Wilcoxon test (need >= 6)',
            'significant': False,
        }

    arr_a = np.array(values_a)
    arr_b = np.array(values_b)
    diffs = arr_a - arr_b

    # Wilcoxon signed-rank
    try:
        stat, p_value = scipy_wilcoxon(diffs, alternative='two-sided')
    except ValueError as e:
        return {
            'metric': metric_key,
            'n_pairs': n_pairs,
            'error': str(e),
            'significant': False,
        }

    # Cohen's d (paired)
    diff_mean = float(np.mean(diffs))
    diff_std = float(np.std(diffs, ddof=1))
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0

    # Direction
    if diff_mean < 0:
        direction = 'B > A'
    elif diff_mean > 0:
        direction = 'A > B'
    else:
        direction = 'equal'

    return {
        'metric': metric_key,
        'n_pairs': n_pairs,
        'statistic': float(stat),
        'p_value': float(p_value),
        'p_adjusted': None,  # Filled in by holm_bonferroni
        'significant': False,  # Updated by holm_bonferroni
        'alpha': alpha,
        'cohens_d': round(cohens_d, 4),
        'mean_diff': round(diff_mean, 4),
        'std_diff': round(diff_std, 4),
        'direction': direction,
        'mean_a': round(float(arr_a.mean()), 4),
        'mean_b': round(float(arr_b.mean()), 4),
    }


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def holm_bonferroni(
    test_results: List[Dict],
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Apply Holm-Bonferroni step-down correction to a list of test results.

    Uniformly more powerful than Bonferroni with the same family-wise
    error rate (FWER) guarantee.

    Args:
        test_results: List of dicts, each with 'p_value' key
        alpha: Family-wise significance level

    Returns:
        Same list with 'p_adjusted' and 'significant' fields updated
    """
    # Filter to tests that have valid p_values
    valid_indices = [
        i for i, r in enumerate(test_results)
        if r.get('p_value') is not None
    ]

    if not valid_indices:
        return test_results

    # Sort by p-value
    sorted_indices = sorted(valid_indices, key=lambda i: test_results[i]['p_value'])
    m = len(sorted_indices)

    for rank, idx in enumerate(sorted_indices):
        p = test_results[idx]['p_value']
        adjusted_alpha = alpha / (m - rank)
        p_adjusted = min(p * (m - rank), 1.0)

        test_results[idx]['p_adjusted'] = round(p_adjusted, 6)
        test_results[idx]['significant'] = p <= adjusted_alpha

    return test_results


def compare_experiments(
    results_a: Dict,
    results_b: Dict,
    metric_keys: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Run Wilcoxon comparison on multiple metrics with Holm-Bonferroni correction.

    Args:
        results_a: Aggregate results from experiment A
        results_b: Aggregate results from experiment B
        metric_keys: Metrics to compare (default: common publication metrics)
        alpha: Family-wise significance level

    Returns:
        List of comparison result dicts, corrected for multiple comparisons
    """
    if metric_keys is None:
        metric_keys = [
            'dose_metrics.mae_gy',
            'gamma.global_3mm3pct.gamma_pass_rate',
            'gamma.ptv_region_3mm3pct.gamma_pass_rate',
            'clinical_constraints.compliance_rate',
            'dvh_metrics.PTV70.D95_error',
        ]

    n_comparisons = len(metric_keys)
    test_results = []

    for mk in metric_keys:
        result = wilcoxon_comparison(
            results_a, results_b, mk,
            n_comparisons=n_comparisons, alpha=alpha
        )
        test_results.append(result)

    # Apply Holm-Bonferroni correction
    test_results = holm_bonferroni(test_results, alpha=alpha)

    return test_results


# =============================================================================
# Output Formatting
# =============================================================================

def format_publication_table(
    experiments: Dict[str, Dict],
    metric_keys: Optional[List[str]] = None,
) -> str:
    """
    Format a markdown table comparing experiments on standardized metrics.

    Args:
        experiments: Dict mapping experiment_name to aggregate results
        metric_keys: Metrics to include as columns

    Returns:
        Markdown table string
    """
    if metric_keys is None:
        metric_keys = [
            'dose_metrics.mae_gy',
            'gamma.global_3mm3pct.gamma_pass_rate',
            'gamma.ptv_region_3mm3pct.gamma_pass_rate',
            'clinical_constraints.compliance_rate',
            'dvh_metrics.PTV70.D95_error',
        ]

    # Column headers
    short_names = {
        'dose_metrics.mae_gy': 'MAE (Gy)',
        'dose_metrics.rmse_gy': 'RMSE (Gy)',
        'gamma.global_3mm3pct.gamma_pass_rate': 'Gamma 3%/3mm (%)',
        'gamma.ptv_region_3mm3pct.gamma_pass_rate': 'PTV Gamma (%)',
        'clinical_constraints.compliance_rate': 'QUANTEC (%)',
        'dvh_metrics.PTV70.D95_error': 'PTV70 D95 Gap (Gy)',
        'dvh_metrics.PTV56.D95_error': 'PTV56 D95 Gap (Gy)',
    }

    headers = ['Experiment'] + [short_names.get(mk, mk) for mk in metric_keys]
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'

    rows = [header_line, separator]
    for exp_name, agg in experiments.items():
        values = [exp_name]
        for mk in metric_keys:
            if mk in agg.get('aggregate', {}):
                m = agg['aggregate'][mk]
                mean = m['mean']
                std = m.get('std', 0)
                values.append(f'{mean:.2f} +/- {std:.2f}')
            else:
                values.append('--')
        rows.append('| ' + ' | '.join(values) + ' |')

    return '\n'.join(rows)


# =============================================================================
# JSON Serialization
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_standardized_results(
    aggregate: Dict,
    output_path: str,
    experiment_name: str = "",
) -> None:
    """
    Save aggregated results to JSON with numpy serialization.

    Args:
        aggregate: Output from aggregate_across_seeds()
        output_path: Path to write JSON file
        experiment_name: Name to include in output
    """
    from pathlib import Path
    from datetime import datetime

    output = {
        'framework_version': FRAMEWORK_VERSION,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        **aggregate,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
