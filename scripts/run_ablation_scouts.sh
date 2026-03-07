#!/bin/bash
# =============================================================================
# Ablation Scout Runs: C2-C5 (addition) and C7-C10 (leave-one-out)
# Single-seed (42), preliminary (70 cases), sequential on RTX 3090
#
# GitHub tracking: issue #65
# Expected runtime: ~40 hours total (~5 hours per condition)
#
# Usage:
#   bash scripts/run_ablation_scouts.sh           # Run all 8
#   bash scripts/run_ablation_scouts.sh C3 C5     # Run specific conditions
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/home/wrockey/miniforge3/envs/vmat-diffusion/bin/python"
DATA_DIR="/home/wrockey/data/processed_npz"
TEST_DIR="/home/wrockey/data/processed_npz_test"
CALIB_JSON="/home/wrockey/data/processed_npz/loss_normalization_calib.json"
SEED=42
EPOCHS=200
GH_ISSUE=65

# Common training args
COMMON_TRAIN="--data_dir $DATA_DIR --epochs $EPOCHS --seed $SEED --num_workers 2"

# Common inference args
COMMON_INFER="--input_dir $TEST_DIR --compute_metrics --overlap 64 --gamma_subsample 4"

# Track overall progress
STARTED_AT=$(date '+%Y-%m-%d %H:%M')
COMPLETED=()
FAILED=()

# ---- Helper functions ----

log() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
}

gh_comment() {
    gh issue comment "$GH_ISSUE" --body "$1" 2>/dev/null || echo "[WARN] Failed to post GH comment"
}

find_best_checkpoint() {
    local run_dir="$1"
    find "$run_dir/checkpoints" -name "best-*.ckpt" -not -name "last*" | head -1
}

run_inference() {
    local condition="$1"
    local exp_name="$2"
    local best_ckpt

    best_ckpt=$(find_best_checkpoint "runs/${exp_name}_seed${SEED}")
    if [ -z "$best_ckpt" ]; then
        echo "[ERROR] No best checkpoint found for $exp_name"
        return 1
    fi

    local pred_dir="predictions/${exp_name}_seed${SEED}_test"

    log "INFERENCE: $condition ($exp_name)"
    $PYTHON scripts/inference_baseline_unet.py \
        --checkpoint "$best_ckpt" \
        $COMMON_INFER \
        --output_dir "$pred_dir"

    # Extract key metrics from evaluation results
    local eval_file="$pred_dir/baseline_evaluation_results.json"
    if [ -f "$eval_file" ]; then
        $PYTHON -c "
import json
r = json.load(open('$eval_file'))
agg = r.get('aggregate', {})
mae = agg.get('mae_gy', {})
gam = agg.get('gamma_3pct_3mm', {})
ptv = agg.get('ptv_gamma_3pct_3mm', {})
d95 = agg.get('ptv70_d95_gap_gy', {})
print(f'MAE: {mae.get(\"mean\",\"?\"):.2f} ± {mae.get(\"std\",\"?\"):.2f} Gy')
print(f'Global Gamma: {gam.get(\"mean\",\"?\"):.1f} ± {gam.get(\"std\",\"?\"):.1f}%')
print(f'PTV Gamma: {ptv.get(\"mean\",\"?\"):.1f} ± {ptv.get(\"std\",\"?\"):.1f}%')
print(f'PTV70 D95 Gap: {d95.get(\"mean\",\"?\"):+.2f} ± {d95.get(\"std\",\"?\"):.2f} Gy')
"
    fi
}

post_results() {
    local condition="$1"
    local exp_name="$2"
    local eval_file="predictions/${exp_name}_seed${SEED}_test/baseline_evaluation_results.json"

    if [ ! -f "$eval_file" ]; then
        gh_comment "### $condition — ⚠️ No evaluation results found"
        return
    fi

    local metrics
    metrics=$($PYTHON -c "
import json
r = json.load(open('$eval_file'))
agg = r.get('aggregate', {})
mae = agg.get('mae_gy', {})
gam = agg.get('gamma_3pct_3mm', {})
ptv = agg.get('ptv_gamma_3pct_3mm', {})
d95 = agg.get('ptv70_d95_gap_gy', {})
print(f'| {mae.get(\"mean\",0):.2f} ± {mae.get(\"std\",0):.2f} | {gam.get(\"mean\",0):.1f} ± {gam.get(\"std\",0):.1f}% | {ptv.get(\"mean\",0):.1f} ± {ptv.get(\"std\",0):.1f}% | {d95.get(\"mean\",0):+.2f} ± {d95.get(\"std\",0):.2f} |')
")

    gh_comment "### $condition complete ✅

| MAE (Gy) | Global Gamma | PTV Gamma | D95 Gap (Gy) |
|----------|--------------|-----------|--------------|
$metrics

Run dir: \`runs/${exp_name}_seed${SEED}/\`"
}

run_condition() {
    local condition="$1"
    local exp_name="$2"
    shift 2
    local train_flags="$*"

    log "TRAINING: $condition ($exp_name)"
    gh_comment "### $condition — Training started 🚀
\`\`\`
$PYTHON scripts/train_baseline_unet.py $COMMON_TRAIN --exp_name $exp_name $train_flags
\`\`\`
Started: $(date '+%Y-%m-%d %H:%M')"

    # Train
    if $PYTHON scripts/train_baseline_unet.py $COMMON_TRAIN --exp_name "$exp_name" $train_flags; then
        # Environment snapshot
        conda list --export -n vmat-diffusion > "runs/${exp_name}_seed${SEED}/environment_snapshot.txt" 2>/dev/null || true

        # Inference + eval
        if run_inference "$condition" "$exp_name"; then
            post_results "$condition" "$exp_name"
            COMPLETED+=("$condition")
        else
            gh_comment "### $condition — ⚠️ Inference failed"
            FAILED+=("$condition (inference)")
        fi
    else
        gh_comment "### $condition — ❌ Training failed"
        FAILED+=("$condition (training)")
    fi
}

# ---- Define conditions ----

run_C2() {
    run_condition "C2 (MSE+Gradient)" "C2_gradient_only" \
        --use_gradient_loss --gradient_loss_weight 0.1
}

run_C3() {
    run_condition "C3 (MSE+DVH)" "C3_dvh_only" \
        --use_dvh_loss --dvh_loss_weight 0.5
}

run_C4() {
    run_condition "C4 (MSE+Structure)" "C4_structure_only" \
        --use_structure_weighted --structure_weighted_weight 1.0
}

run_C5() {
    run_condition "C5 (MSE+AsymPTV)" "C5_asymptv_only" \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0
}

run_C7() {
    run_condition "C7 (Full−Gradient)" "C7_full_no_gradient" \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C8() {
    run_condition "C8 (Full−DVH)" "C8_full_no_dvh" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C9() {
    run_condition "C9 (Full−Structure)" "C9_full_no_structure" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C10() {
    run_condition "C10 (Full−AsymPTV)" "C10_full_no_asymptv" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

# ---- Main ----

# Determine which conditions to run
if [ $# -gt 0 ]; then
    CONDITIONS=("$@")
else
    CONDITIONS=(C2 C3 C4 C5 C7 C8 C9 C10)
fi

# Verify git is clean
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "[ERROR] Git working tree is not clean. Commit changes first."
    git status --short
    exit 1
fi

GIT_HASH=$(git rev-parse --short HEAD)

log "ABLATION SCOUT BATCH — Starting"
echo "Conditions: ${CONDITIONS[*]}"
echo "Git hash:   $GIT_HASH"
echo "Seed:       $SEED"
echo "Data dir:   $DATA_DIR"
echo "Test dir:   $TEST_DIR"
echo ""

gh_comment "## Ablation Scout Batch Started

**Conditions:** ${CONDITIONS[*]}
**Git hash:** \`$GIT_HASH\`
**Seed:** $SEED
**Dataset:** 70 cases (v2.3 preliminary)
**Started:** $STARTED_AT

Each condition will post results as it completes."

for cond in "${CONDITIONS[@]}"; do
    case "$cond" in
        C2)  run_C2 ;;
        C3)  run_C3 ;;
        C4)  run_C4 ;;
        C5)  run_C5 ;;
        C7)  run_C7 ;;
        C8)  run_C8 ;;
        C9)  run_C9 ;;
        C10) run_C10 ;;
        *)   echo "[WARN] Unknown condition: $cond — skipping" ;;
    esac
done

# ---- Summary ----

FINISHED_AT=$(date '+%Y-%m-%d %H:%M')

log "ABLATION SCOUT BATCH — Complete"
echo "Started:   $STARTED_AT"
echo "Finished:  $FINISHED_AT"
echo "Completed: ${COMPLETED[*]:-none}"
echo "Failed:    ${FAILED[*]:-none}"

SUMMARY="## Ablation Scout Batch Complete

**Started:** $STARTED_AT
**Finished:** $FINISHED_AT
**Completed:** ${COMPLETED[*]:-none}
**Failed:** ${FAILED[*]:-none}

See individual comments above for per-condition metrics."

gh_comment "$SUMMARY"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
