#!/bin/bash
# Training Watchdog Script for VMAT Diffusion Project
#
# Monitors DDPM training for hangs and captures diagnostics before restarting.
# Creates a diagnostic log that can be analyzed to identify hang patterns.
#
# Usage:
#   ./scripts/training_watchdog.sh [--check-interval 60] [--hang-threshold 180]
#
# Options:
#   --check-interval  Seconds between checks (default: 60)
#   --hang-threshold  Seconds without progress before declaring hang (default: 180)
#
# Output:
#   - runs/hang_diagnostics.log - Detailed diagnostics from each hang
#   - runs/watchdog.log - Watchdog activity log

set -euo pipefail

# Configuration
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"      # Check every 60 seconds
HANG_THRESHOLD="${HANG_THRESHOLD:-180}"     # 3 minutes without progress = hang
PROJECT_DIR="$HOME/vmat-diffusion-project"
METRICS_PATTERN="$PROJECT_DIR/runs/vmat_dose_ddpm/version_*/metrics.csv"
DIAGNOSTICS_LOG="$PROJECT_DIR/runs/hang_diagnostics.log"
WATCHDOG_LOG="$PROJECT_DIR/runs/watchdog.log"
CONDA_ENV="vmat-diffusion"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --hang-threshold)
            HANG_THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

# Get the latest metrics file
get_latest_metrics() {
    ls -t $METRICS_PATTERN 2>/dev/null | head -1
}

# Get line count of metrics file
get_metrics_count() {
    local metrics_file="$1"
    if [[ -f "$metrics_file" ]]; then
        wc -l < "$metrics_file"
    else
        echo "0"
    fi
}

# Get training process PIDs
get_training_pids() {
    pgrep -f "train_dose_ddpm_v2.py" 2>/dev/null || true
}

# Capture comprehensive diagnostics
capture_diagnostics() {
    local reason="$1"
    local metrics_file="$2"
    local last_count="$3"

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "========================================" >> "$DIAGNOSTICS_LOG"
    echo "HANG DETECTED: $(date '+%Y-%m-%d %H:%M:%S')" >> "$DIAGNOSTICS_LOG"
    echo "Reason: $reason" >> "$DIAGNOSTICS_LOG"
    echo "Metrics file: $metrics_file" >> "$DIAGNOSTICS_LOG"
    echo "Last known line count: $last_count" >> "$DIAGNOSTICS_LOG"
    echo "========================================" >> "$DIAGNOSTICS_LOG"

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- LAST 10 METRICS ENTRIES ---" >> "$DIAGNOSTICS_LOG"
    if [[ -f "$metrics_file" ]]; then
        tail -10 "$metrics_file" >> "$DIAGNOSTICS_LOG" 2>&1
    else
        echo "No metrics file found" >> "$DIAGNOSTICS_LOG"
    fi

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- TRAINING PROCESSES ---" >> "$DIAGNOSTICS_LOG"
    ps aux | grep -E "python.*train_dose_ddpm" | grep -v grep >> "$DIAGNOSTICS_LOG" 2>&1 || echo "No processes found" >> "$DIAGNOSTICS_LOG"

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- PROCESS STATES (detailed) ---" >> "$DIAGNOSTICS_LOG"
    for pid in $(get_training_pids); do
        echo "PID $pid:" >> "$DIAGNOSTICS_LOG"
        if [[ -d "/proc/$pid" ]]; then
            echo "  State: $(cat /proc/$pid/status 2>/dev/null | grep -E '^State:' || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
            echo "  Threads: $(cat /proc/$pid/status 2>/dev/null | grep -E '^Threads:' || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
            echo "  VmRSS: $(cat /proc/$pid/status 2>/dev/null | grep -E '^VmRSS:' || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
            echo "  voluntary_ctxt_switches: $(cat /proc/$pid/status 2>/dev/null | grep voluntary_ctxt_switches || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
            echo "  nonvoluntary_ctxt_switches: $(cat /proc/$pid/status 2>/dev/null | grep nonvoluntary_ctxt_switches || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
            echo "  FD count: $(ls /proc/$pid/fd 2>/dev/null | wc -l || echo 'unknown')" >> "$DIAGNOSTICS_LOG"
        else
            echo "  Process no longer exists" >> "$DIAGNOSTICS_LOG"
        fi
    done

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- GPU STATE ---" >> "$DIAGNOSTICS_LOG"
    nvidia-smi >> "$DIAGNOSTICS_LOG" 2>&1 || echo "nvidia-smi failed" >> "$DIAGNOSTICS_LOG"

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- SYSTEM MEMORY ---" >> "$DIAGNOSTICS_LOG"
    free -h >> "$DIAGNOSTICS_LOG" 2>&1

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- DMESG (last 20 lines) ---" >> "$DIAGNOSTICS_LOG"
    dmesg 2>/dev/null | tail -20 >> "$DIAGNOSTICS_LOG" 2>&1 || echo "dmesg not available" >> "$DIAGNOSTICS_LOG"

    echo "" >> "$DIAGNOSTICS_LOG"
    echo "--- END DIAGNOSTICS ---" >> "$DIAGNOSTICS_LOG"
    echo "" >> "$DIAGNOSTICS_LOG"
}

# Kill training processes
kill_training() {
    log "Killing training processes..."
    pkill -f "train_dose_ddpm_v2.py" 2>/dev/null || true
    sleep 5
    # Force kill if still running
    pkill -9 -f "train_dose_ddpm_v2.py" 2>/dev/null || true
    sleep 2
}

# Start training
start_training() {
    log "Starting training..."
    cd "$PROJECT_DIR"

    # Source conda
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"

    # Start training in background
    nohup python scripts/train_dose_ddpm_v2.py \
        --data_dir ./processed \
        --epochs 200 \
        > runs/training_output.log 2>&1 &

    local new_pid=$!
    log "Training started with PID: $new_pid"

    # Wait for training to initialize
    sleep 30
}

# Main monitoring loop
main() {
    log "========================================"
    log "Training Watchdog Started"
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Hang threshold: ${HANG_THRESHOLD}s"
    log "========================================"

    local last_metrics_count=0
    local last_change_time=$(date +%s)
    local restart_count=0

    # Check if training is already running
    if [[ -z "$(get_training_pids)" ]]; then
        log "No training process found, starting training..."
        start_training
    fi

    while true; do
        sleep "$CHECK_INTERVAL"

        local metrics_file=$(get_latest_metrics)
        local current_count=$(get_metrics_count "$metrics_file")
        local current_time=$(date +%s)

        # Check if metrics have updated
        if [[ "$current_count" -gt "$last_metrics_count" ]]; then
            last_metrics_count="$current_count"
            last_change_time="$current_time"
            # Only log occasionally to avoid spam
            if (( current_count % 10 == 0 )); then
                log "Training progressing: $current_count metrics entries"
            fi
        else
            local stall_duration=$((current_time - last_change_time))

            if [[ "$stall_duration" -ge "$HANG_THRESHOLD" ]]; then
                log "HANG DETECTED! No progress for ${stall_duration}s"

                # Check if processes are still running
                local pids=$(get_training_pids)
                if [[ -z "$pids" ]]; then
                    log "Training processes died unexpectedly"
                    capture_diagnostics "Process died" "$metrics_file" "$last_metrics_count"
                else
                    log "Training processes still running but not progressing"
                    capture_diagnostics "Hang - processes alive but stuck" "$metrics_file" "$last_metrics_count"
                fi

                # Kill and restart
                kill_training
                restart_count=$((restart_count + 1))
                log "Restart count: $restart_count"

                start_training

                # Reset tracking
                last_change_time=$(date +%s)
                # Get new metrics file after restart
                sleep 10
                metrics_file=$(get_latest_metrics)
                last_metrics_count=$(get_metrics_count "$metrics_file")

            elif [[ "$stall_duration" -ge 60 ]]; then
                log "Warning: No progress for ${stall_duration}s (threshold: ${HANG_THRESHOLD}s)"
            fi
        fi

        # Check if training process is still alive
        if [[ -z "$(get_training_pids)" ]]; then
            log "Training process not found - may have completed or crashed"

            # Check if training completed (epoch 200)
            if [[ -f "$metrics_file" ]] && grep -q "^199," "$metrics_file" 2>/dev/null; then
                log "Training appears to have completed (epoch 199 found)"
                log "Watchdog exiting."
                exit 0
            else
                log "Training crashed or stopped unexpectedly"
                capture_diagnostics "Process not found" "$metrics_file" "$last_metrics_count"
                start_training
                restart_count=$((restart_count + 1))
                last_change_time=$(date +%s)
            fi
        fi
    done
}

# Run main function
main
