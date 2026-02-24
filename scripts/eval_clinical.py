"""
Clinical constraint checking for VMAT dose predictions.

QUANTEC-based constraints for prostate VMAT with SIB (70/56 Gy in 28 fractions).
Single source of truth for constraint definitions and pass/fail evaluation.

No external dependencies beyond eval_core.

Version: 1.0.0
"""

from typing import Dict, List, Optional

from eval_core import PRIMARY_PRESCRIPTION_GY, SECONDARY_PRESCRIPTION_GY


# =============================================================================
# Clinical Constraints — THE ONE DEFINITION
# =============================================================================

# Each constraint is structured as:
#   metric: the DVH metric name (must match keys in dvh_metrics output)
#   type: 'min' (value must be >= limit) or 'max' (value must be <= limit)
#   value: the threshold
#   unit: 'Gy', '%', or 'cc'

CLINICAL_CONSTRAINTS: Dict[str, Dict] = {
    'PTV70': {
        'description': 'High-dose PTV (70 Gy target)',
        'rx_gy': PRIMARY_PRESCRIPTION_GY,
        'constraints': [
            {'metric': 'D95', 'type': 'min', 'value': 66.5, 'unit': 'Gy',
             'description': 'D95 >= 95% of Rx'},
            {'metric': 'V95', 'type': 'min', 'value': 95.0, 'unit': '%',
             'description': 'V95 >= 95% (volume receiving >= 95% of Rx)'},
        ],
    },
    'PTV56': {
        'description': 'Intermediate PTV (56 Gy target)',
        'rx_gy': SECONDARY_PRESCRIPTION_GY,
        'constraints': [
            {'metric': 'D95', 'type': 'min', 'value': 53.2, 'unit': 'Gy',
             'description': 'D95 >= 95% of Rx'},
            {'metric': 'V95', 'type': 'min', 'value': 95.0, 'unit': '%',
             'description': 'V95 >= 95% (volume receiving >= 95% of Rx)'},
        ],
    },
    'Rectum': {
        'description': 'Rectum - critical OAR',
        'constraints': [
            {'metric': 'V70', 'type': 'max', 'value': 15.0, 'unit': '%',
             'description': 'V70 <= 15%'},
            {'metric': 'V60', 'type': 'max', 'value': 25.0, 'unit': '%',
             'description': 'V60 <= 25%'},
            {'metric': 'V50', 'type': 'max', 'value': 50.0, 'unit': '%',
             'description': 'V50 <= 50%'},
            {'metric': 'Dmax', 'type': 'max', 'value': 75.0, 'unit': 'Gy',
             'description': 'Dmax <= 75 Gy'},
        ],
    },
    'Bladder': {
        'description': 'Bladder - critical OAR',
        'constraints': [
            {'metric': 'V70', 'type': 'max', 'value': 25.0, 'unit': '%',
             'description': 'V70 <= 25%'},
            {'metric': 'V60', 'type': 'max', 'value': 35.0, 'unit': '%',
             'description': 'V60 <= 35%'},
            {'metric': 'V50', 'type': 'max', 'value': 50.0, 'unit': '%',
             'description': 'V50 <= 50%'},
            {'metric': 'Dmax', 'type': 'max', 'value': 75.0, 'unit': 'Gy',
             'description': 'Dmax <= 75 Gy'},
        ],
    },
    'Femur_L': {
        'description': 'Left femoral head',
        'constraints': [
            {'metric': 'Dmax', 'type': 'max', 'value': 50.0, 'unit': 'Gy',
             'description': 'Dmax <= 50 Gy'},
            {'metric': 'V50', 'type': 'max', 'value': 5.0, 'unit': '%',
             'description': 'V50 <= 5%'},
        ],
    },
    'Femur_R': {
        'description': 'Right femoral head',
        'constraints': [
            {'metric': 'Dmax', 'type': 'max', 'value': 50.0, 'unit': 'Gy',
             'description': 'Dmax <= 50 Gy'},
            {'metric': 'V50', 'type': 'max', 'value': 5.0, 'unit': '%',
             'description': 'V50 <= 5%'},
        ],
    },
    'Bowel': {
        'description': 'Bowel bag',
        'constraints': [
            {'metric': 'Dmax', 'type': 'max', 'value': 52.0, 'unit': 'Gy',
             'description': 'Dmax <= 52 Gy'},
            {'metric': 'V45_cc', 'type': 'max', 'value': 195.0, 'unit': 'cc',
             'description': 'V45 <= 195 cc (absolute volume)'},
        ],
    },
}


def check_clinical_constraints(
    dvh_metrics: Dict[str, Dict],
    constraints: Dict[str, Dict] = None,
) -> Dict:
    """
    Check predicted DVH metrics against clinical constraints.

    Args:
        dvh_metrics: Per-structure DVH metrics from eval_metrics.compute_dvh_metrics().
            Expected keys per structure: pred_D95, pred_Dmax, pred_V70, pred_V95,
            pred_V45_cc (for Bowel), etc.
        constraints: Clinical constraint definitions (default: CLINICAL_CONSTRAINTS)

    Returns:
        Dict with:
            - overall_pass: bool
            - compliance_rate: float (% of constraints passed)
            - structures: per-structure results
            - violations: list of violation dicts
            - missing: list of metrics that couldn't be checked
            - summary: counts
    """
    if constraints is None:
        constraints = CLINICAL_CONSTRAINTS

    results = {
        'overall_pass': True,
        'compliance_rate': 0.0,
        'structures': {},
        'violations': [],
        'missing': [],
        'summary': {
            'total_constraints': 0,
            'passed': 0,
            'failed': 0,
            'not_evaluated': 0,
        },
    }

    for structure, struct_def in constraints.items():
        if structure not in dvh_metrics:
            # Structure not in prediction (may not be contoured)
            for c in struct_def.get('constraints', []):
                results['summary']['not_evaluated'] += 1
                results['missing'].append({
                    'structure': structure,
                    'metric': c['metric'],
                    'reason': 'structure not in dvh_metrics',
                })
            continue

        metrics = dvh_metrics[structure]
        if not metrics.get('exists', False):
            for c in struct_def.get('constraints', []):
                results['summary']['not_evaluated'] += 1
                results['missing'].append({
                    'structure': structure,
                    'metric': c['metric'],
                    'reason': 'structure mask is empty',
                })
            continue

        struct_results = {
            'description': struct_def.get('description', ''),
            'constraints_checked': [],
            'passed': True,
        }

        for c in struct_def.get('constraints', []):
            metric_name = c['metric']
            limit = c['value']
            constraint_type = c['type']

            # Map metric name to DVH output key
            pred_value = _get_metric_value(metrics, metric_name, prefix='pred')

            if pred_value is None:
                results['summary']['not_evaluated'] += 1
                results['missing'].append({
                    'structure': structure,
                    'metric': metric_name,
                    'reason': f'metric key not found in dvh_metrics',
                })
                continue

            results['summary']['total_constraints'] += 1

            if constraint_type == 'min':
                passed = pred_value >= limit
            else:  # 'max'
                passed = pred_value <= limit

            constraint_result = {
                'metric': metric_name,
                'constraint_type': constraint_type,
                'limit': limit,
                'unit': c['unit'],
                'predicted': round(pred_value, 2),
                'passed': passed,
                'description': c.get('description', ''),
            }
            struct_results['constraints_checked'].append(constraint_result)

            if passed:
                results['summary']['passed'] += 1
            else:
                results['summary']['failed'] += 1
                struct_results['passed'] = False
                results['overall_pass'] = False
                results['violations'].append({
                    'structure': structure,
                    'metric': metric_name,
                    'constraint_type': constraint_type,
                    'limit': limit,
                    'unit': c['unit'],
                    'predicted': round(pred_value, 2),
                    'description': c.get('description', ''),
                })

        results['structures'][structure] = struct_results

    total = results['summary']['total_constraints']
    if total > 0:
        results['compliance_rate'] = round(
            100.0 * results['summary']['passed'] / total, 1
        )

    return results


def _get_metric_value(
    metrics: Dict,
    metric_name: str,
    prefix: str = 'pred',
) -> Optional[float]:
    """
    Look up a metric value from the DVH metrics dict.

    Handles the mapping from constraint metric names to DVH output keys:
        D95 -> pred_D95
        Dmax -> pred_max_gy
        V70 -> pred_V70
        V95 -> pred_V95
        V45_cc -> pred_V45_cc

    Args:
        metrics: Structure DVH metrics dict
        metric_name: Constraint metric name (e.g. 'D95', 'Dmax', 'V70', 'V45_cc')
        prefix: 'pred' or 'target'

    Returns:
        Metric value as float, or None if not found
    """
    if metric_name == 'Dmax':
        key = f'{prefix}_max_gy'
    elif metric_name == 'V45_cc':
        key = f'{prefix}_V45_cc'
    else:
        key = f'{prefix}_{metric_name}'

    return metrics.get(key)
