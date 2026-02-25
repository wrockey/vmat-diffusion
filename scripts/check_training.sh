#!/bin/bash
# Quick training status check for baseline_v23_seed123
RUN_DIR="/home/wrockey/projects/vmat-diffusion/scripts/runs/baseline_v23_seed123"
METRICS="$RUN_DIR/version_1/metrics.csv"

echo "=== Training Check $(date) ==="

# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU: unavailable"

# Latest epoch from metrics
if [ -f "$METRICS" ]; then
    LAST_VAL=$(grep -v "^epoch" "$METRICS" | grep "val/mae_gy" | tail -1 2>/dev/null)
    if [ -z "$LAST_VAL" ]; then
        # Parse: find rows with non-empty val/mae_gy (column 10)
        LAST_EPOCH=$(awk -F',' 'NR>1 && $10!="" {print "Epoch "$1" val_MAE="$10" Gamma="$8}' "$METRICS" | tail -1)
        echo "Latest validation: $LAST_EPOCH"
    fi
    TOTAL_ROWS=$(wc -l < "$METRICS")
    echo "Metrics rows: $TOTAL_ROWS"
else
    echo "No metrics file yet"
fi

# Check if process is still running
if pgrep -f "baseline_v23_seed123" > /dev/null; then
    echo "Process: RUNNING"
else
    echo "Process: NOT RUNNING (may have completed or crashed)"
fi

# Checkpoints
echo "Checkpoints: $(ls "$RUN_DIR/checkpoints/" 2>/dev/null | grep "best" | tail -1)"
