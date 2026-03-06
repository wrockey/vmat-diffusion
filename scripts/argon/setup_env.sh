#!/bin/bash
# =============================================================================
# setup_env.sh — One-time Argon environment setup for VMAT ablation study
#
# Run this on an Argon login node BEFORE submitting jobs.
# Sets up: conda environment, directory structure, data staging.
#
# Usage: bash setup_env.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_config.sh"

echo "=============================================="
echo "VMAT Ablation Study — Environment Setup"
echo "=============================================="
echo ""

# =============================================================================
# 1. CONDA ENVIRONMENT
# =============================================================================
echo "[1/4] Setting up conda environment ..."

if conda info --envs 2>/dev/null | grep -q "vmat-diffusion"; then
    echo "  Environment 'vmat-diffusion' already exists."
    echo "  To recreate: conda env remove -n vmat-diffusion && bash setup_env.sh"
else
    if [[ -f "${PROJECT_ROOT}/environment.yml" ]]; then
        echo "  Creating from environment.yml ..."
        conda env create -f "${PROJECT_ROOT}/environment.yml"
        echo "  Done."
    else
        echo "  WARNING: environment.yml not found at ${PROJECT_ROOT}/environment.yml"
        echo "  Create manually:"
        echo "    conda create -n vmat-diffusion python=3.11"
        echo "    conda activate vmat-diffusion"
        echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        echo "    pip install -r ${PROJECT_ROOT}/requirements.txt"
    fi
fi

echo ""

# =============================================================================
# 2. DIRECTORY STRUCTURE
# =============================================================================
echo "[2/4] Creating directory structure on /nfsscratch ..."

mkdir -p "${SCRATCH_BASE}"
mkdir -p "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}/runs"
mkdir -p "${RESULTS_DIR}/predictions"
mkdir -p "${RESULTS_DIR}/logs"

echo "  ${SCRATCH_BASE}/"
echo "  ├── data/              (stage NPZ files here)"
echo "  │   ├── *.npz          (all training/val/test cases)"
echo "  │   ├── test/          (symlink or copy of test-only cases)"
echo "  │   └── loss_normalization_calib.json"
echo "  ├── repo/              (git clone of vmat-diffusion)"
echo "  └── results/"
echo "      ├── runs/          (training outputs)"
echo "      ├── predictions/   (inference outputs)"
echo "      └── logs/          (job logs)"
echo ""

# =============================================================================
# 3. PROJECT REPO
# =============================================================================
echo "[3/4] Checking project repository ..."

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
    echo "  Repo exists at ${PROJECT_ROOT}"
    cd "${PROJECT_ROOT}"
    echo "  Branch: $(git branch --show-current)"
    echo "  Commit: $(git rev-parse --short HEAD)"
    echo "  Status: $(git status --short | wc -l) uncommitted files"
else
    echo "  Repo not found at ${PROJECT_ROOT}"
    echo "  Clone it:"
    echo "    git clone https://github.com/wrockey/vmat-diffusion.git ${PROJECT_ROOT}"
fi

echo ""

# =============================================================================
# 4. DATA CHECK
# =============================================================================
echo "[4/4] Checking data ..."

NPZ_COUNT=$(find "${DATA_DIR}" -name "*.npz" 2>/dev/null | wc -l)

if [[ "${NPZ_COUNT}" -gt 0 ]]; then
    echo "  Found ${NPZ_COUNT} NPZ files in ${DATA_DIR}"
    DATA_SIZE=$(du -sh "${DATA_DIR}" | cut -f1)
    echo "  Total size: ${DATA_SIZE}"
else
    echo "  No NPZ files found in ${DATA_DIR}"
    echo "  Transfer data from workstation:"
    echo "    scp -r -P 40 /path/to/processed_npz/*.npz \\"
    echo "        hawkid@argon.hpc.uiowa.edu:${DATA_DIR}/"
fi

if [[ -d "${DATA_DIR}/test" ]]; then
    TEST_COUNT=$(find "${DATA_DIR}/test" -name "*.npz" | wc -l)
    echo "  Test directory: ${TEST_COUNT} cases"
else
    echo "  No test/ subdirectory. Create it with test case symlinks:"
    echo "    mkdir -p ${DATA_DIR}/test"
fi

if [[ -f "${CALIBRATION_JSON}" ]]; then
    echo "  Calibration JSON: found"
else
    echo "  Calibration JSON: NOT FOUND"
    echo "  Run: python scripts/calibrate_loss_normalization.py --data_dir ${DATA_DIR}"
fi

echo ""
echo "=============================================="
echo "SETUP COMPLETE"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Transfer NPZ data to ${DATA_DIR}/"
echo "  2. Create test/ subdirectory with test cases"
echo "  3. Run calibration: python scripts/calibrate_loss_normalization.py --data_dir ${DATA_DIR}"
echo "  4. Submit: cd ${SCRIPT_DIR} && bash submit_ablation.sh --dry-run"
echo ""
