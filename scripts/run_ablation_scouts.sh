#!/bin/bash
# =============================================================================
# Ablation Scout Runs: C2-C5 (addition) and C7-C10 (leave-one-out)
# Single-seed (42), preliminary (70 cases), sequential on RTX 3090
#
# GitHub tracking: issue #65
# Expected runtime: ~40 hours total (~5 hours per condition)
#
# Usage:
#   tmux new -s ablation
#   bash scripts/run_ablation_scouts.sh           # Run all 8
#   bash scripts/run_ablation_scouts.sh C3 C5     # Run specific conditions
#
# Resume after partial failure:
#   bash scripts/run_ablation_scouts.sh C5 C7 C8 C9 C10
#   (post-processing always covers ALL conditions with eval data, not just
#    the current invocation, so resume is safe)
# =============================================================================

set -uo pipefail
# NOTE: -e intentionally omitted. We handle errors per-condition so one
# failure doesn't kill the entire 40-hour batch. Critical failures (git
# dirty, missing data) exit explicitly.

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

# ---- Logging ----

LOG_DIR="$PROJECT_ROOT/runs/ablation_scouts"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch_$(date '+%Y%m%d_%H%M%S').log"

# Tee all output to log file AND console
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

# Track overall progress
STARTED_AT=$(date '+%Y-%m-%d %H:%M')
COMPLETED=()
FAILED=()

# Condition metadata: maps condition ID to exp_name for post-processing
declare -A CONDITION_NAMES
CONDITION_NAMES=(
    [C2]="C2_gradient_only"
    [C3]="C3_dvh_only"
    [C4]="C4_structure_only"
    [C5]="C5_asymptv_only"
    [C7]="C7_full_no_gradient"
    [C8]="C8_full_no_dvh"
    [C9]="C9_full_no_structure"
    [C10]="C10_full_no_asymptv"
)

declare -A CONDITION_LABELS
CONDITION_LABELS=(
    [C2]="MSE+Gradient"
    [C3]="MSE+DVH"
    [C4]="MSE+Structure"
    [C5]="MSE+AsymPTV"
    [C7]="Full-Gradient"
    [C8]="Full-DVH"
    [C9]="Full-Structure"
    [C10]="Full-AsymPTV"
)

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
    find "$run_dir/checkpoints" -name "best-*.ckpt" -not -name "last*" 2>/dev/null | head -1
}

# Extract metrics from evaluation results JSON (computes from per-case data)
extract_metrics() {
    local eval_file="$1"
    local format="${2:-table}"  # "table" for GH markdown, "summary" for console

    $PYTHON -c "
import json, numpy as np

r = json.load(open('$eval_file'))
cases = r['per_case_results']

maes = [c['dose_metrics']['mae_gy'] for c in cases]
globals_g = [c['gamma']['global_3mm3pct']['gamma_pass_rate'] for c in cases]
ptv_g = [c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'] for c in cases]
d95_gaps = []
for c in cases:
    ptv70 = c['dvh_metrics'].get('PTV70', {})
    pred = ptv70.get('pred_D95')
    targ = ptv70.get('target_D95')
    if pred is not None and targ is not None:
        d95_gaps.append(pred - targ)

mae_m, mae_s = np.mean(maes), np.std(maes)
gg_m, gg_s = np.mean(globals_g), np.std(globals_g)
pg_m, pg_s = np.mean(ptv_g), np.std(ptv_g)
d95_m = np.mean(d95_gaps) if d95_gaps else float('nan')
d95_s = np.std(d95_gaps) if d95_gaps else float('nan')

if '$format' == 'table':
    print(f'| {mae_m:.2f} +/- {mae_s:.2f} | {gg_m:.1f} +/- {gg_s:.1f}% | {pg_m:.1f} +/- {pg_s:.1f}% | {d95_m:+.2f} +/- {d95_s:.2f} |')
else:
    print(f'MAE: {mae_m:.2f} +/- {mae_s:.2f} Gy')
    print(f'Global Gamma: {gg_m:.1f} +/- {gg_s:.1f}%')
    print(f'PTV Gamma: {pg_m:.1f} +/- {pg_s:.1f}%')
    print(f'PTV70 D95 Gap: {d95_m:+.2f} +/- {d95_s:.2f} Gy')
" 2>/dev/null || echo "| [metric extraction failed] | — | — | — |"
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

    # Print metrics to console (non-fatal if extraction fails)
    local eval_file="$pred_dir/baseline_evaluation_results.json"
    if [ -f "$eval_file" ]; then
        extract_metrics "$eval_file" "summary" || true
    fi
}

post_results() {
    local condition="$1"
    local exp_name="$2"
    local eval_file="predictions/${exp_name}_seed${SEED}_test/baseline_evaluation_results.json"

    if [ ! -f "$eval_file" ]; then
        gh_comment "### $condition — No evaluation results found" || true
        return
    fi

    local metrics
    metrics=$(extract_metrics "$eval_file" "table") || metrics="| [extraction failed] | — | — | — |"

    gh_comment "### $condition complete

| MAE (Gy) | Global Gamma | PTV Gamma | D95 Gap (Gy) |
|----------|--------------|-----------|--------------|
$metrics

Run dir: \`runs/${exp_name}_seed${SEED}/\`" || true
}

run_condition() {
    local condition="$1"
    local exp_name="$2"
    shift 2
    local train_flags="$*"

    log "TRAINING: $condition ($exp_name)"
    gh_comment "### $condition — Training started
\`\`\`
$PYTHON scripts/train_baseline_unet.py $COMMON_TRAIN --exp_name $exp_name $train_flags
\`\`\`
Started: $(date '+%Y-%m-%d %H:%M')" || true

    # Train
    if $PYTHON scripts/train_baseline_unet.py $COMMON_TRAIN --exp_name "$exp_name" $train_flags; then
        # Environment snapshot
        conda list --export -n vmat-diffusion > "runs/${exp_name}_seed${SEED}/environment_snapshot.txt" 2>/dev/null || true

        # Inference + eval
        if run_inference "$condition" "$exp_name"; then
            post_results "$condition" "$exp_name"
            COMPLETED+=("$condition")
        else
            gh_comment "### $condition — Inference failed" || true
            FAILED+=("$condition (inference)")
        fi
    else
        gh_comment "### $condition — Training failed" || true
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
    run_condition "C7 (Full-Gradient)" "C7_full_no_gradient" \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C8() {
    run_condition "C8 (Full-DVH)" "C8_full_no_dvh" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C9() {
    run_condition "C9 (Full-Structure)" "C9_full_no_structure" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_asymmetric_ptv --asymmetric_ptv_weight 1.0 \
        --asymmetric_underdose_weight 2.5 --asymmetric_overdose_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

run_C10() {
    run_condition "C10 (Full-AsymPTV)" "C10_full_no_asymptv" \
        --use_gradient_loss --gradient_loss_weight 0.1 \
        --use_dvh_loss --dvh_loss_weight 0.5 \
        --use_structure_weighted --structure_weighted_weight 1.0 \
        --use_uncertainty_weighting --calibration_json "$CALIB_JSON"
}

# ---- Post-processing ----
# These functions scan for ALL available evaluation data on disk,
# not just the current invocation's COMPLETED list. This means
# resume runs correctly produce full comparison tables/figures.

update_experiments_index() {
    log "POST-PROCESSING: Updating EXPERIMENTS_INDEX.md"
    local index_file="notebooks/EXPERIMENTS_INDEX.md"
    local today
    today=$(date '+%Y-%m-%d')

    for cond_id in C2 C3 C4 C5 C7 C8 C9 C10; do
        local exp_name="${CONDITION_NAMES[$cond_id]}"
        local label="${CONDITION_LABELS[$cond_id]}"
        local eval_file="predictions/${exp_name}_seed${SEED}_test/baseline_evaluation_results.json"

        if [ ! -f "$eval_file" ]; then
            continue
        fi

        # Check if entry already exists (avoid duplicates)
        if grep -q "$exp_name" "$index_file" 2>/dev/null; then
            echo "[SKIP] $cond_id already in EXPERIMENTS_INDEX.md"
            continue
        fi

        # Extract metrics for index row
        local row
        row=$($PYTHON -c "
import json, numpy as np

r = json.load(open('$eval_file'))
cases = r['per_case_results']

maes = [c['dose_metrics']['mae_gy'] for c in cases]
globals_g = [c['gamma']['global_3mm3pct']['gamma_pass_rate'] for c in cases]
ptv_g = [c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'] for c in cases]
d95_gaps = []
for c in cases:
    ptv70 = c['dvh_metrics'].get('PTV70', {})
    pred = ptv70.get('pred_D95')
    targ = ptv70.get('target_D95')
    if pred is not None and targ is not None:
        d95_gaps.append(pred - targ)

mae_m, mae_s = np.mean(maes), np.std(maes)
gg_m, gg_s = np.mean(globals_g), np.std(globals_g)
pg_m, pg_s = np.mean(ptv_g), np.std(ptv_g)
d95_m = np.mean(d95_gaps) if d95_gaps else float('nan')
d95_s = np.std(d95_gaps) if d95_gaps else float('nan')

print(f'| $today | $cond_id $label (seed42) | \`$GIT_HASH\` | — | BaselineUNet3D | {mae_m:.2f} +/- {mae_s:.2f} (test, n={len(cases)}) | {gg_m:.1f} +/- {gg_s:.1f}% (global), {pg_m:.1f} +/- {pg_s:.1f}% (PTV) | {d95_m:+.2f} +/- {d95_s:.2f} Gy | Preliminary |')
" 2>/dev/null) || {
            echo "[WARN] Failed to extract metrics for $cond_id"
            continue
        }

        # Insert before the "In Progress" section marker
        sed -i "/^### v2.3 Experiments — In Progress/i\\
$row" "$index_file"
        echo "[OK] Added $cond_id to EXPERIMENTS_INDEX.md"
    done
}

generate_figures() {
    log "POST-PROCESSING: Generating ablation scout figures"

    if [ -f "scripts/generate_ablation_scouts_figures.py" ]; then
        $PYTHON scripts/generate_ablation_scouts_figures.py || {
            echo "[WARN] Figure generation failed — run manually later"
        }
    else
        echo "[WARN] Figure script not found — skipping"
    fi
}

post_summary_table() {
    log "POST-PROCESSING: Posting summary table to GH"

    # Build a comprehensive comparison table including C1 and C6
    # Scans ALL available eval data on disk (not just current invocation)
    local table
    table=$($PYTHON -c "
import json, numpy as np
from pathlib import Path

conditions = [
    ('C1', 'Baseline (MSE)', 'baseline_v23'),
    ('C2', 'MSE+Gradient', 'C2_gradient_only'),
    ('C3', 'MSE+DVH', 'C3_dvh_only'),
    ('C4', 'MSE+Structure', 'C4_structure_only'),
    ('C5', 'MSE+AsymPTV', 'C5_asymptv_only'),
    ('C6', 'Full combined', 'combined_loss_2.5to1'),
    ('C7', 'Full-Gradient', 'C7_full_no_gradient'),
    ('C8', 'Full-DVH', 'C8_full_no_dvh'),
    ('C9', 'Full-Structure', 'C9_full_no_structure'),
    ('C10', 'Full-AsymPTV', 'C10_full_no_asymptv'),
]

rows = []
for cid, label, exp in conditions:
    eval_file = Path('predictions') / f'{exp}_seed42_test' / 'baseline_evaluation_results.json'
    if not eval_file.exists():
        rows.append(f'| {cid} | {label} | — | — | — | — |')
        continue

    r = json.load(open(eval_file))
    cases = r['per_case_results']

    maes = [c['dose_metrics']['mae_gy'] for c in cases]
    gg = [c['gamma']['global_3mm3pct']['gamma_pass_rate'] for c in cases]
    pg = [c['gamma']['ptv_region_3mm3pct']['gamma_pass_rate'] for c in cases]
    d95 = []
    for c in cases:
        ptv70 = c['dvh_metrics'].get('PTV70', {})
        p, t = ptv70.get('pred_D95'), ptv70.get('target_D95')
        if p is not None and t is not None:
            d95.append(p - t)

    d95_str = f'{np.mean(d95):+.2f} +/- {np.std(d95):.2f}' if d95 else 'N/A'
    rows.append(f'| {cid} | {label} | {np.mean(maes):.2f} +/- {np.std(maes):.2f} | {np.mean(gg):.1f} +/- {np.std(gg):.1f}% | {np.mean(pg):.1f} +/- {np.std(pg):.1f}% | {d95_str} Gy |')

print('| ID | Condition | MAE (Gy) | Global Gamma | PTV Gamma | D95 Gap |')
print('|----|-----------|----------|--------------|-----------|---------|')
for row in rows:
    print(row)
" 2>/dev/null) || {
        echo "[WARN] Failed to generate summary table"
        return
    }

    gh_comment "## Full C1-C10 Scout Comparison (seed42, 70 cases)

$table

**Note:** C1 and C6 are existing 3-seed experiments shown here at seed42 only for comparison. All results are preliminary (single-seed, 70-case dataset).

**Log file:** \`$LOG_FILE\`" || true
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

# Check disk space (need ~2 GB per condition for checkpoints + predictions)
AVAIL_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
NEEDED_GB=$((${#CONDITIONS[@]} * 2))
if [ "$AVAIL_GB" -lt "$NEEDED_GB" ]; then
    echo "[WARN] Only ${AVAIL_GB}G available, need ~${NEEDED_GB}G for ${#CONDITIONS[@]} conditions"
    echo "       (2 GB per condition: checkpoints + predictions)"
fi

# Warn if not in tmux/screen
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  WARNING: Not running inside tmux or screen!"
    echo "  This batch takes ~40 hours. If your terminal closes, it dies."
    echo "  Recommended: tmux new -s ablation"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to abort..."
    read -r
fi

log "ABLATION SCOUT BATCH — Starting"
echo "Conditions: ${CONDITIONS[*]}"
echo "Git hash:   $GIT_HASH"
echo "Seed:       $SEED"
echo "Data dir:   $DATA_DIR"
echo "Test dir:   $TEST_DIR"
echo "Log file:   $LOG_FILE"
echo "Disk avail: ${AVAIL_GB}G"
echo ""

gh_comment "## Ablation Scout Batch Started

**Conditions:** ${CONDITIONS[*]}
**Git hash:** \`$GIT_HASH\`
**Seed:** $SEED
**Dataset:** 70 cases (v2.3 preliminary)
**Started:** $STARTED_AT
**Log file:** \`$LOG_FILE\`

Each condition will post results as it completes." || true

# ---- Run all conditions ----

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

# ---- Post-processing ----

FINISHED_AT=$(date '+%Y-%m-%d %H:%M')

log "ABLATION SCOUT BATCH — All training/inference complete"
echo "Started:   $STARTED_AT"
echo "Finished:  $FINISHED_AT"
echo "Completed: ${COMPLETED[*]:-none}"
echo "Failed:    ${FAILED[*]:-none}"

# Post-processing scans ALL available data on disk, so it works
# correctly even on resume runs
update_experiments_index || true
generate_figures || true
post_summary_table || true

# ---- Final summary ----

SUMMARY="## Ablation Scout Batch Complete

**Started:** $STARTED_AT
**Finished:** $FINISHED_AT
**Completed:** ${COMPLETED[*]:-none}
**Failed:** ${FAILED[*]:-none}

### Post-processing status
- EXPERIMENTS_INDEX.md: updated
- Figures: generated to \`runs/ablation_scouts/figures/\`
- Full comparison table: posted above
- Log file: \`$LOG_FILE\`

### Remaining documentation (manual)
- [ ] Create notebook: \`notebooks/$(date '+%Y-%m-%d')_ablation_scouts_preliminary.ipynb\`
- [ ] Review figures and add captions
- [ ] Commit documentation artifacts
- [ ] Update pinned issue #63 if results change the narrative"

gh_comment "$SUMMARY" || true

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILED CONDITIONS: ${FAILED[*]}"
    exit 1
fi
