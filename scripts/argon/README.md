# Argon HPC Batch Scripts for VMAT Ablation Study

SGE job scripts for running the full pre-registered ablation study on the UIowa Argon HPC cluster.

## Overview

- **13 conditions** (C1-C10 loss ablation + C11-C13 architecture) × **3 seeds** = **39 training runs**
- Each training run chains to an inference run = **78 total jobs**
- Estimated GPU time: **~175 GPU-hours** (A100)
- Estimated wall-clock: **~5-7 hours** if all tasks run concurrently

## Files

| File | Purpose |
|------|---------|
| `ablation_config.sh` | Maps task ID → experiment name, seed, CLI flags |
| `train_ablation.job` | SGE array job for training (39 tasks) |
| `infer_ablation.job` | SGE array job for inference (39 tasks, chained) |
| `submit_ablation.sh` | Master submission script with preflight checks |
| `setup_env.sh` | One-time environment setup on Argon |

## Quick Start

```bash
# 1. SSH to Argon
ssh -p 40 hawkid@argon.hpc.uiowa.edu

# 2. One-time setup
cd /nfsscratch/$USER/vmat-diffusion/repo/scripts/argon
bash setup_env.sh

# 3. Transfer data (from workstation)
scp -r -P 40 /path/to/processed_npz/*.npz \
    hawkid@argon.hpc.uiowa.edu:/nfsscratch/$USER/vmat-diffusion/data/

# 4. Run calibration
cd /nfsscratch/$USER/vmat-diffusion/repo
conda activate vmat-diffusion
python scripts/calibrate_loss_normalization.py --data_dir ../data

# 5. Dry run (verify task mapping)
bash scripts/argon/submit_ablation.sh --dry-run

# 6. Submit all
bash scripts/argon/submit_ablation.sh

# 7. Submit loss ablation only (C1-C10)
bash scripts/argon/submit_ablation.sh --conditions 1-30
```

## Condition Mapping

| Tasks | Condition | Description |
|-------|-----------|-------------|
| 1-3 | C01 | Baseline MSE only |
| 4-6 | C02 | MSE + Gradient |
| 7-9 | C03 | MSE + DVH |
| 10-12 | C04 | MSE + Structure-weighted |
| 13-15 | C05 | MSE + Asymmetric PTV |
| 16-18 | C06 | Full combined (all 5 + uncertainty weighting) |
| 19-21 | C07 | Full - Gradient (leave-one-out) |
| 22-24 | C08 | Full - DVH (leave-one-out) |
| 25-27 | C09 | Full - Structure (leave-one-out) |
| 28-30 | C10 | Full - AsymPTV (leave-one-out) |
| 31-33 | C11 | AttentionUNet + MSE |
| 34-36 | C12 | AttentionUNet + Full |
| 37-39 | C13 | BottleneckAttn + MSE |

Each group of 3 tasks = seeds 42, 123, 456.

## Monitoring

```bash
# All your jobs
qstat -u $USER

# Per-task status for array job
qstat -t -j <JOB_ID>

# Check completed logs
ls /nfsscratch/$USER/vmat-diffusion/results/logs/

# Check results
ls /nfsscratch/$USER/vmat-diffusion/results/runs/
ls /nfsscratch/$USER/vmat-diffusion/results/predictions/
```

## Directory Structure on Argon

```
/nfsscratch/$USER/vmat-diffusion/
├── data/                          # NPZ files (staged from workstation)
│   ├── case_0001.npz ...
│   ├── test/                      # Test cases only (for inference)
│   └── loss_normalization_calib.json
├── repo/                          # Git clone of vmat-diffusion
│   └── scripts/argon/             # These scripts
└── results/
    ├── runs/                      # Training outputs
    │   ├── C01_baseline_mse_seed42/
    │   └── ...
    ├── predictions/               # Inference outputs
    │   ├── C01_baseline_mse_seed42_test/
    │   └── ...
    └── logs/                      # SGE job logs
```

## GPU Budget

| Component | Tasks | Hours/Task | Total GPU-Hours |
|-----------|-------|------------|-----------------|
| C1-C10 Training | 30 | ~4 | ~120 |
| C11-C13 Training | 9 | ~4 | ~36 |
| Inference | 39 | ~0.5 | ~19.5 |
| **Total** | **78** | | **~175.5** |

## Troubleshooting

**Job stuck in queue:** Check `qstat -j <ID>`. A100s are popular — consider removing `gpu_a100=true` constraint if wait times are long.

**Job fails immediately:** Check log in `results/logs/`. Common: missing data, conda env not found, disk quota.

**Inference can't find checkpoint:** Training may have failed. Check `results/runs/<name>/checkpoints/`.

**Preempted (exit 143):** If using `all.q`, jobs can be preempted. Use `UI-GPU` queue.

## Customization

Edit `ablation_config.sh` to change paths, conditions, or seeds.
Edit `.job` files to change email (`-M hawkid@uiowa.edu`).
