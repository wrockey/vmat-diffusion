#!/bin/bash
# =============================================================================
# ablation_config.sh — Maps SGE array task ID to experiment condition + seed
#
# 13 conditions x 3 seeds = 39 tasks (task IDs 1-39)
#
# Conditions:
#   C1-C10:  Loss ablation (pre-registered)
#   C11-C13: Architecture comparison (reduced scope per PROJECT_ASSESSMENT.md)
#
# Seeds: 42, 123, 456 (standard 3-seed protocol)
#
# Usage: source this file, then use TASK_ID to look up EXP_NAME, SEED, CLI_FLAGS
# =============================================================================

# ---------- Paths (update these before first run) ----------
# Data directory on /nfsscratch (staging source)
SCRATCH_BASE="/nfsscratch/${USER}/vmat-diffusion"
DATA_DIR="${SCRATCH_BASE}/data"
CALIBRATION_JSON="${SCRATCH_BASE}/data/loss_normalization_calib.json"

# Project root (cloned repo on scratch or home)
PROJECT_ROOT="${SCRATCH_BASE}/repo"

# Results destination on /nfsscratch
RESULTS_DIR="${SCRATCH_BASE}/results"

# ---------- Seed mapping ----------
SEEDS=(42 123 456)

# ---------- Condition definitions ----------
# Each condition has a name and CLI flags (beyond the common flags)
# Common flags applied to all: --epochs 200 --batch_size 2 --lr 0.0001 --num_workers 4

NUM_CONDITIONS=13

# Condition names
CONDITION_NAMES=(
    "C01_baseline_mse"
    "C02_mse_gradient"
    "C03_mse_dvh"
    "C04_mse_structure"
    "C05_mse_asymptv"
    "C06_full_combined"
    "C07_full_no_gradient"
    "C08_full_no_dvh"
    "C09_full_no_structure"
    "C10_full_no_asymptv"
    "C11_attention_mse"
    "C12_attention_full"
    "C13_bottleneck_mse"
)

# --- Loss flags for each condition ---
# C1: Baseline (MSE only) — no extra flags
CONDITION_FLAGS_1=""

# C2: MSE + Gradient
CONDITION_FLAGS_2="--use_gradient_loss --gradient_loss_weight 0.1"

# C3: MSE + DVH
CONDITION_FLAGS_3="--use_dvh_loss --dvh_loss_weight 0.5"

# C4: MSE + Structure-weighted
CONDITION_FLAGS_4="--use_structure_weighted --structure_weighted_weight 1.0"

# C5: MSE + Asymmetric PTV
CONDITION_FLAGS_5="--use_asymmetric_ptv --asymmetric_ptv_weight 1.0"

# C6: Full combined (all 5 + uncertainty weighting)
CONDITION_FLAGS_6="\
--use_gradient_loss --gradient_loss_weight 0.1 \
--use_dvh_loss --dvh_loss_weight 0.5 \
--use_structure_weighted --structure_weighted_weight 1.0 \
--use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C7: Full - Gradient (ablation: remove gradient from full)
CONDITION_FLAGS_7="\
--use_dvh_loss --dvh_loss_weight 0.5 \
--use_structure_weighted --structure_weighted_weight 1.0 \
--use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C8: Full - DVH (ablation: remove DVH from full)
CONDITION_FLAGS_8="\
--use_gradient_loss --gradient_loss_weight 0.1 \
--use_structure_weighted --structure_weighted_weight 1.0 \
--use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C9: Full - Structure (ablation: remove structure-weighted from full)
CONDITION_FLAGS_9="\
--use_gradient_loss --gradient_loss_weight 0.1 \
--use_dvh_loss --dvh_loss_weight 0.5 \
--use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C10: Full - AsymPTV (ablation: remove asymmetric PTV from full)
CONDITION_FLAGS_10="\
--use_gradient_loss --gradient_loss_weight 0.1 \
--use_dvh_loss --dvh_loss_weight 0.5 \
--use_structure_weighted --structure_weighted_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C11: AttentionUNet + MSE (architecture control)
CONDITION_FLAGS_11="--architecture attention_unet"

# C12: AttentionUNet + Full combined (architecture + best loss)
CONDITION_FLAGS_12="\
--architecture attention_unet \
--use_gradient_loss --gradient_loss_weight 0.1 \
--use_dvh_loss --dvh_loss_weight 0.5 \
--use_structure_weighted --structure_weighted_weight 1.0 \
--use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
--use_uncertainty_weighting --calibration_json ${CALIBRATION_JSON}"

# C13: BottleneckAttn + MSE (architecture control)
CONDITION_FLAGS_13="--architecture bottleneck_attn"


# =============================================================================
# get_task_config() — Resolve SGE_TASK_ID to experiment parameters
#
# Sets: EXP_NAME, SEED, CLI_FLAGS, CONDITION_IDX
# =============================================================================
get_task_config() {
    local task_id=$1

    if [[ -z "$task_id" ]]; then
        echo "ERROR: get_task_config requires a task ID argument" >&2
        return 1
    fi

    if [[ "$task_id" -lt 1 || "$task_id" -gt $((NUM_CONDITIONS * 3)) ]]; then
        echo "ERROR: Task ID $task_id out of range [1, $((NUM_CONDITIONS * 3))]" >&2
        return 1
    fi

    # Task ID layout: conditions cycle through seeds
    # Task 1 = C1 seed42, Task 2 = C1 seed123, Task 3 = C1 seed456
    # Task 4 = C2 seed42, Task 5 = C2 seed123, Task 6 = C2 seed456
    # ...
    CONDITION_IDX=$(( (task_id - 1) / 3 + 1 ))
    local seed_idx=$(( (task_id - 1) % 3 ))
    SEED=${SEEDS[$seed_idx]}

    # Get condition name (0-indexed array)
    EXP_NAME="${CONDITION_NAMES[$((CONDITION_IDX - 1))]}"

    # Get CLI flags for this condition
    local flags_var="CONDITION_FLAGS_${CONDITION_IDX}"
    CLI_FLAGS="${!flags_var}"

    # Export for use in job scripts
    export EXP_NAME SEED CLI_FLAGS CONDITION_IDX
}


# =============================================================================
# print_all_tasks() — Print the full task mapping (for verification)
# =============================================================================
print_all_tasks() {
    echo "=============================================="
    echo "VMAT Ablation Study — Task Mapping"
    echo "13 conditions x 3 seeds = 39 tasks"
    echo "=============================================="
    echo ""
    printf "%-6s %-4s %-30s %-6s %s\n" "TASK" "C#" "EXP_NAME" "SEED" "CLI_FLAGS"
    printf "%-6s %-4s %-30s %-6s %s\n" "----" "--" "--------" "----" "---------"

    for task_id in $(seq 1 $((NUM_CONDITIONS * 3))); do
        get_task_config "$task_id"
        printf "%-6d %-4d %-30s %-6d %s\n" \
            "$task_id" "$CONDITION_IDX" "$EXP_NAME" "$SEED" "$CLI_FLAGS"
    done
}


# If run directly (not sourced), print the task mapping
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_all_tasks
fi
