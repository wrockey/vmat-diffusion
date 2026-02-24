#!/bin/bash
# =============================================================================
# End-to-End Baseline U-Net Experiment on v2.3 Data
# Issue #37: Pipeline validation + baseline metrics establishment
#
# Runs: 3 seeds (42, 123, 456) sequentially
# For each seed: train → find best checkpoint → inference on test set → evaluation
# All output logged to runs/baseline_v23_run.log
#
# Usage:
#   bash scripts/run_baseline_v23.sh
#   # Or in background:
#   nohup bash scripts/run_baseline_v23.sh > /dev/null 2>&1 &
# =============================================================================

set -euo pipefail

# ---- Configuration ----
DATA_DIR="/home/wrockey/data/processed_npz"
EXP_BASE="baseline_v23"
LOG_DIR="runs"
PRED_DIR="predictions"
SEEDS=(42 123 456)
EPOCHS=200
BATCH_SIZE=2
PATCH_SIZE=128
NUM_WORKERS=2
GAMMA_SUBSAMPLE=2
RX_DOSE_GY=70.0

# ---- Setup ----
cd /home/wrockey/projects/vmat-diffusion

# Activate conda
source /home/wrockey/miniforge3/etc/profile.d/conda.sh
conda activate vmat-diffusion

# Create output directories
mkdir -p "$LOG_DIR" "$PRED_DIR"

# Main log file
RUN_LOG="$LOG_DIR/${EXP_BASE}_run.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"
}

# ---- Record experiment metadata ----
GIT_HASH=$(git rev-parse HEAD)
PYTHON_VERSION=$(python --version 2>&1)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
N_FILES=$(ls -1 "$DATA_DIR"/*.npz 2>/dev/null | wc -l)

log "=========================================================="
log "BASELINE U-NET v2.3 EXPERIMENT — Issue #37"
log "=========================================================="
log "Git hash: $GIT_HASH"
log "Python: $PYTHON_VERSION"
log "PyTorch: $PYTORCH_VERSION"
log "CUDA: $CUDA_VERSION"
log "GPU: $GPU_NAME"
log "Data dir: $DATA_DIR ($N_FILES NPZ files)"
log "Seeds: ${SEEDS[*]}"
log "Epochs: $EPOCHS (early stopping patience=50)"
log "Batch size: $BATCH_SIZE"
log "Patch size: $PATCH_SIZE"
log ""

# Save conda environment snapshot
CONDA_SNAPSHOT="$LOG_DIR/${EXP_BASE}_environment_snapshot.txt"
conda list --export > "$CONDA_SNAPSHOT"
log "Conda snapshot saved to: $CONDA_SNAPSHOT"

# ---- Track overall results ----
declare -A SEED_STATUS
declare -A SEED_BEST_MAE

OVERALL_START=$(date +%s)

# ---- Train + Evaluate each seed ----
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${EXP_BASE}_seed${SEED}"
    SEED_START=$(date +%s)

    log ""
    log "=========================================================="
    log "SEED $SEED: $EXP_NAME"
    log "=========================================================="

    # ---- Training ----
    log "Starting training..."
    TRAIN_LOG="$LOG_DIR/${EXP_NAME}_training.log"

    if python scripts/train_baseline_unet.py \
        --data_dir "$DATA_DIR" \
        --exp_name "$EXP_NAME" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --patch_size "$PATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --rx_dose_gy "$RX_DOSE_GY" \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$TRAIN_LOG"; then
        log "Training completed successfully"
    else
        log "ERROR: Training FAILED for seed $SEED"
        SEED_STATUS[$SEED]="FAILED (training)"
        continue
    fi

    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$(( TRAIN_END - SEED_START ))
    log "Training duration: $(( TRAIN_DURATION / 3600 ))h $(( (TRAIN_DURATION % 3600) / 60 ))m"

    # ---- Find best checkpoint ----
    BEST_CKPT=$(ls -t "$LOG_DIR/$EXP_NAME/checkpoints/best-"*.ckpt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        log "ERROR: No best checkpoint found for seed $SEED"
        SEED_STATUS[$SEED]="FAILED (no checkpoint)"
        continue
    fi
    log "Best checkpoint: $BEST_CKPT"

    # Extract best val MAE from filename (format: best-{epoch:03d}-{val/mae_gy:.3f}.ckpt)
    BEST_MAE=$(echo "$BEST_CKPT" | grep -oP '[\d.]+(?=\.ckpt)' || echo "unknown")
    SEED_BEST_MAE[$SEED]="$BEST_MAE"
    log "Best val MAE: $BEST_MAE Gy"

    # ---- Prepare test set ----
    TEST_JSON="$LOG_DIR/$EXP_NAME/test_cases.json"
    TEST_DIR=$(mktemp -d "/tmp/baseline_v23_test_seed${SEED}_XXXX")

    if [ -f "$TEST_JSON" ]; then
        log "Creating test directory from $TEST_JSON"
        python -c "
import json
from pathlib import Path

with open('$TEST_JSON') as f:
    test_data = json.load(f)

print(f\"Test cases ({len(test_data['test_files'])}): {[Path(f).stem for f in test_data['test_files']]}\")
for fpath in test_data['test_files']:
    src = Path(fpath)
    dst = Path('$TEST_DIR') / src.name
    dst.symlink_to(src)
" 2>&1 | tee -a "$RUN_LOG"
    else
        log "WARNING: No test_cases.json — using all data for inference"
        for f in "$DATA_DIR"/*.npz; do
            ln -s "$f" "$TEST_DIR/$(basename "$f")"
        done
    fi

    N_TEST=$(ls -1 "$TEST_DIR"/*.npz 2>/dev/null | wc -l)
    log "Test cases: $N_TEST"

    # ---- Inference + Evaluation ----
    PRED_OUTDIR="$PRED_DIR/${EXP_NAME}_test"
    mkdir -p "$PRED_OUTDIR"
    INFERENCE_LOG="$LOG_DIR/${EXP_NAME}_inference.log"

    log "Starting inference + evaluation (gamma_subsample=$GAMMA_SUBSAMPLE)..."

    if python scripts/inference_baseline_unet.py \
        --checkpoint "$BEST_CKPT" \
        --input_dir "$TEST_DIR" \
        --output_dir "$PRED_OUTDIR" \
        --compute_metrics \
        --gamma_subsample "$GAMMA_SUBSAMPLE" \
        --rx_dose_gy "$RX_DOSE_GY" \
        2>&1 | tee "$INFERENCE_LOG"; then
        log "Inference completed successfully"
    else
        log "ERROR: Inference FAILED for seed $SEED"
        SEED_STATUS[$SEED]="FAILED (inference)"
        rm -rf "$TEST_DIR"
        continue
    fi

    # Cleanup temp dir
    rm -rf "$TEST_DIR"

    SEED_END=$(date +%s)
    SEED_DURATION=$(( SEED_END - SEED_START ))
    log "Seed $SEED total duration: $(( SEED_DURATION / 3600 ))h $(( (SEED_DURATION % 3600) / 60 ))m"

    SEED_STATUS[$SEED]="SUCCESS"

    # ---- Print per-seed summary ----
    RESULTS_JSON="$PRED_OUTDIR/baseline_evaluation_results.json"
    if [ -f "$RESULTS_JSON" ]; then
        log "Results saved to: $RESULTS_JSON"
        python -c "
import json
with open('$RESULTS_JSON') as f:
    data = json.load(f)
agg = data.get('aggregate_metrics', {})
print(f\"  MAE: {agg.get('mae_gy_mean', 'N/A'):.2f} +/- {agg.get('mae_gy_std', 'N/A'):.2f} Gy\")
if 'gamma_pass_rate_mean' in agg:
    print(f\"  Gamma: {agg['gamma_pass_rate_mean']:.1f} +/- {agg['gamma_pass_rate_std']:.1f}%\")
" 2>&1 | tee -a "$RUN_LOG"
    fi

done

# ---- Overall Summary ----
OVERALL_END=$(date +%s)
OVERALL_DURATION=$(( OVERALL_END - OVERALL_START ))

log ""
log "=========================================================="
log "EXPERIMENT COMPLETE"
log "=========================================================="
log "Total duration: $(( OVERALL_DURATION / 3600 ))h $(( (OVERALL_DURATION % 3600) / 60 ))m"
log ""
log "Per-seed status:"
for SEED in "${SEEDS[@]}"; do
    STATUS="${SEED_STATUS[$SEED]:-UNKNOWN}"
    MAE="${SEED_BEST_MAE[$SEED]:-N/A}"
    log "  Seed $SEED: $STATUS (best val MAE: $MAE Gy)"
done

# ---- Cross-seed aggregate ----
log ""
log "Cross-seed aggregate results:"
python -c "
import json
import numpy as np
from pathlib import Path

seeds = [42, 123, 456]
all_mae = []
all_gamma = []

for seed in seeds:
    results_path = Path('$PRED_DIR') / f'${EXP_BASE}_seed{seed}_test' / 'baseline_evaluation_results.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        agg = data.get('aggregate_metrics', {})
        if 'mae_gy_mean' in agg:
            all_mae.append(agg['mae_gy_mean'])
        if 'gamma_pass_rate_mean' in agg:
            all_gamma.append(agg['gamma_pass_rate_mean'])
        print(f'  Seed {seed}: MAE={agg.get(\"mae_gy_mean\", \"N/A\"):.2f} Gy, Gamma={agg.get(\"gamma_pass_rate_mean\", \"N/A\"):.1f}%' if 'gamma_pass_rate_mean' in agg else f'  Seed {seed}: MAE={agg.get(\"mae_gy_mean\", \"N/A\"):.2f} Gy')
    else:
        print(f'  Seed {seed}: MISSING results')

if all_mae:
    print(f'')
    print(f'  Cross-seed MAE: {np.mean(all_mae):.2f} +/- {np.std(all_mae):.2f} Gy')
if all_gamma:
    print(f'  Cross-seed Gamma: {np.mean(all_gamma):.1f} +/- {np.std(all_gamma):.1f}%')

# Save cross-seed summary
summary = {
    'experiment': '${EXP_BASE}',
    'git_hash': '$(git rev-parse HEAD)',
    'n_seeds': len(seeds),
    'seeds': seeds,
    'note': 'Pipeline validation on all 74 v2.3 cases (SIB + single-Rx). Test splits differ per seed.',
    'cross_seed_metrics': {}
}
if all_mae:
    summary['cross_seed_metrics']['mae_gy'] = {'mean': float(np.mean(all_mae)), 'std': float(np.std(all_mae)), 'per_seed': all_mae}
if all_gamma:
    summary['cross_seed_metrics']['gamma_pass_rate'] = {'mean': float(np.mean(all_gamma)), 'std': float(np.std(all_gamma)), 'per_seed': all_gamma}

out_path = Path('$PRED_DIR') / '${EXP_BASE}_cross_seed_summary.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'')
print(f'  Summary saved to: {out_path}')
" 2>&1 | tee -a "$RUN_LOG"

log ""
log "Log file: $RUN_LOG"
log "Next steps: Create experiment notebook, update EXPERIMENTS_INDEX.md"
log "DONE."
