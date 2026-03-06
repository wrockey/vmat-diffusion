#!/bin/bash
# =============================================================================
# submit_ablation.sh — Master submission script for VMAT ablation study
#
# Submits training and inference array jobs with dependency chaining.
#
# Usage:
#   ./submit_ablation.sh                          # Submit all 39 tasks
#   ./submit_ablation.sh --conditions 1-9         # C1-C3 only (tasks 1-9)
#   ./submit_ablation.sh --conditions 1-30        # C1-C10 (loss ablation only)
#   ./submit_ablation.sh --conditions 31-39       # C11-C13 (architecture only)
#   ./submit_ablation.sh --dry-run                # Print commands without submitting
#   ./submit_ablation.sh --train-only             # Skip inference (submit later)
#   ./submit_ablation.sh --infer-only JOB_ID      # Submit inference for existing training job
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config for path definitions and task mapping
source "${SCRIPT_DIR}/ablation_config.sh"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
DRY_RUN=false
TRAIN_ONLY=false
INFER_ONLY=false
INFER_HOLD_JID=""
TASK_RANGE="1-39"
EXTRA_QSUB_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --conditions)
            TASK_RANGE="$2"
            shift 2
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --infer-only)
            INFER_ONLY=true
            INFER_HOLD_JID="$2"
            shift 2
            ;;
        --extra)
            EXTRA_QSUB_ARGS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run              Print commands without submitting"
            echo "  --conditions RANGE     Task ID range (default: 1-39)"
            echo "                         1-30  = C1-C10 (loss ablation)"
            echo "                         31-39 = C11-C13 (architecture)"
            echo "                         1-9   = C1-C3 (quick test)"
            echo "  --train-only           Submit training only (no inference)"
            echo "  --infer-only JOB_ID    Submit inference chained to existing training job"
            echo "  --extra 'ARGS'         Extra qsub arguments (quoted)"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Condition Mapping:"
            echo "  Tasks 1-3:   C01 Baseline MSE (seeds 42, 123, 456)"
            echo "  Tasks 4-6:   C02 MSE+Gradient"
            echo "  Tasks 7-9:   C03 MSE+DVH"
            echo "  Tasks 10-12: C04 MSE+Structure"
            echo "  Tasks 13-15: C05 MSE+AsymPTV"
            echo "  Tasks 16-18: C06 Full Combined"
            echo "  Tasks 19-21: C07 Full-Gradient"
            echo "  Tasks 22-24: C08 Full-DVH"
            echo "  Tasks 25-27: C09 Full-Structure"
            echo "  Tasks 28-30: C10 Full-AsymPTV"
            echo "  Tasks 31-33: C11 AttentionUNet+MSE"
            echo "  Tasks 34-36: C12 AttentionUNet+Full"
            echo "  Tasks 37-39: C13 BottleneckAttn+MSE"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================
echo "=============================================="
echo "VMAT Ablation Study — Submission"
echo "=============================================="
echo ""

# Parse task range to count tasks
if [[ "${TASK_RANGE}" == *-* ]]; then
    RANGE_START=$(echo "${TASK_RANGE}" | cut -d- -f1)
    RANGE_END=$(echo "${TASK_RANGE}" | cut -d- -f2)
    NUM_TASKS=$(( RANGE_END - RANGE_START + 1 ))
else
    NUM_TASKS=1
    RANGE_START="${TASK_RANGE}"
    RANGE_END="${TASK_RANGE}"
fi

echo "Task range:       ${TASK_RANGE} (${NUM_TASKS} tasks)"
echo "Dry run:          ${DRY_RUN}"
echo "Train only:       ${TRAIN_ONLY}"
echo "Infer only:       ${INFER_ONLY}"
echo ""

# Show conditions in this range
echo "Conditions included:"
for task_id in $(seq "${RANGE_START}" "${RANGE_END}"); do
    get_task_config "${task_id}"
    printf "  Task %2d: C%02d %-30s seed=%d\n" "${task_id}" "${CONDITION_IDX}" "${EXP_NAME}" "${SEED}"
done
echo ""

# Check prerequisites (skip in dry-run)
if [[ "${DRY_RUN}" != "true" ]]; then
    # Check data exists
    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "ERROR: Data directory not found: ${DATA_DIR}"
        echo ""
        echo "Run setup_env.sh first to:"
        echo "  1. Create conda environment"
        echo "  2. Stage data to /nfsscratch"
        echo "  3. Set up directory structure"
        exit 1
    fi

    FILE_COUNT=$(find "${DATA_DIR}" -name "*.npz" | wc -l)
    echo "Data directory:   ${DATA_DIR}"
    echo "NPZ files:        ${FILE_COUNT}"

    if [[ "${FILE_COUNT}" -lt 10 ]]; then
        echo "WARNING: Only ${FILE_COUNT} NPZ files. Expected ~200."
        read -p "Continue anyway? [y/N] " CONFIRM
        if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
            echo "Aborted."
            exit 1
        fi
    fi

    # Check project repo
    if [[ ! -f "${PROJECT_ROOT}/scripts/train_baseline_unet.py" ]]; then
        echo "ERROR: Project not found at ${PROJECT_ROOT}"
        echo "Clone the repo first: git clone <url> ${PROJECT_ROOT}"
        exit 1
    fi

    # Check calibration JSON for uncertainty-weighted conditions
    if [[ "${RANGE_START}" -le 36 ]]; then  # C6-C10, C12 need calibration
        if [[ ! -f "${CALIBRATION_JSON}" ]]; then
            echo "WARNING: Calibration JSON not found: ${CALIBRATION_JSON}"
            echo "Conditions C6-C10 and C12 require this file."
            echo "Run: python scripts/calibrate_loss_normalization.py --data_dir ${DATA_DIR}"
        fi
    fi

    # Check conda env
    if ! conda info --envs 2>/dev/null | grep -q "vmat-diffusion"; then
        echo "WARNING: conda environment 'vmat-diffusion' not found."
        echo "Run setup_env.sh first."
    fi

    # Check qsub is available
    if ! command -v qsub &>/dev/null; then
        echo "ERROR: qsub not found. Are you on an Argon login node?"
        exit 1
    fi

    echo ""
fi

# Create output directories
if [[ "${DRY_RUN}" != "true" ]]; then
    mkdir -p "${RESULTS_DIR}/runs"
    mkdir -p "${RESULTS_DIR}/predictions"
    mkdir -p "${RESULTS_DIR}/logs"
    echo "Output directories created under ${RESULTS_DIR}/"
    echo ""
fi

# =============================================================================
# RECORD GIT STATE
# =============================================================================
if [[ "${DRY_RUN}" != "true" && -d "${PROJECT_ROOT}/.git" ]]; then
    GIT_HASH=$(cd "${PROJECT_ROOT}" && git rev-parse HEAD)
    GIT_STATUS=$(cd "${PROJECT_ROOT}" && git status --short)
    echo "Git commit: ${GIT_HASH}"
    if [[ -n "${GIT_STATUS}" ]]; then
        echo "WARNING: Uncommitted changes in ${PROJECT_ROOT}:"
        echo "${GIT_STATUS}"
        echo ""
        read -p "Submit with uncommitted changes? [y/N] " CONFIRM
        if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
            echo "Commit changes first, then re-run."
            exit 1
        fi
    else
        echo "  Working tree clean."
    fi
    echo ""
fi

# =============================================================================
# SUBMIT JOBS
# =============================================================================

run_qsub() {
    local cmd="$1"
    local description="$2"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY RUN] ${description}"
        echo "  ${cmd}"
        echo ""
        echo "DRY_RUN_JOB_ID"  # Placeholder for chaining
    else
        echo "Submitting: ${description}"
        echo "  ${cmd}"
        JOB_OUTPUT=$(eval "${cmd}")
        JOB_ID_FULL=$(echo "${JOB_OUTPUT}" | grep -oP '\d+' | head -1)
        echo "  Job ID: ${JOB_ID_FULL}"
        echo ""
        echo "${JOB_ID_FULL}"
    fi
}

# --- Training ---
if [[ "${INFER_ONLY}" != "true" ]]; then
    TRAIN_CMD="qsub -terse -t ${TASK_RANGE} ${EXTRA_QSUB_ARGS} ${SCRIPT_DIR}/train_ablation.job"
    TRAIN_JOB_ID=$(run_qsub "${TRAIN_CMD}" "Training array (tasks ${TASK_RANGE})")
else
    TRAIN_JOB_ID="${INFER_HOLD_JID}"
    echo "Using existing training job: ${TRAIN_JOB_ID}"
    echo ""
fi

# --- Inference ---
if [[ "${TRAIN_ONLY}" != "true" ]]; then
    # Use -hold_jid_ad for array dependency: each inference task waits for
    # its corresponding training task (same task ID) to complete.
    INFER_CMD="qsub -terse -t ${TASK_RANGE} -hold_jid_ad ${TRAIN_JOB_ID} ${EXTRA_QSUB_ARGS} ${SCRIPT_DIR}/infer_ablation.job"
    INFER_JOB_ID=$(run_qsub "${INFER_CMD}" "Inference array (tasks ${TASK_RANGE}, depends on ${TRAIN_JOB_ID})")
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "SUBMISSION SUMMARY"
echo "=============================================="
echo ""
echo "Tasks submitted:  ${NUM_TASKS}"
echo "Task range:       ${TASK_RANGE}"

if [[ "${INFER_ONLY}" != "true" ]]; then
    echo "Training job:     ${TRAIN_JOB_ID}"
fi
if [[ "${TRAIN_ONLY}" != "true" ]]; then
    echo "Inference job:    ${INFER_JOB_ID:-N/A}"
fi

echo ""
echo "--- GPU-Hour Estimate ---"
echo "Training:   ${NUM_TASKS} tasks x ~4 hrs = ~$(( NUM_TASKS * 4 )) GPU-hours"
echo "Inference:  ${NUM_TASKS} tasks x ~0.5 hrs = ~$(( NUM_TASKS / 2 )) GPU-hours"
echo "Total:      ~$(( NUM_TASKS * 4 + NUM_TASKS / 2 )) GPU-hours"
echo ""
echo "--- Wall-Clock Estimate ---"
echo "(Assumes Argon schedules all tasks concurrently)"
echo "Training:   ~4-6 hours"
echo "Inference:  ~30-60 minutes"
echo "Total:      ~5-7 hours wall-clock"
echo ""
echo "--- Monitoring ---"
echo "  qstat -u \$USER                    # All jobs"
echo "  qstat -j ${TRAIN_JOB_ID:-JOB_ID}  # Training details"
echo "  qstat -t -j ${TRAIN_JOB_ID:-JOB_ID}  # Per-task status"
echo "  ls ${RESULTS_DIR}/logs/            # Completed logs"
echo ""
echo "--- After Completion ---"
echo "  ls ${RESULTS_DIR}/runs/            # Training outputs"
echo "  ls ${RESULTS_DIR}/predictions/     # Inference results"
echo ""
echo "  # Aggregate results:"
echo "  python scripts/argon/collect_results.py ${RESULTS_DIR}  # (create this script)"
echo ""
echo "=============================================="
