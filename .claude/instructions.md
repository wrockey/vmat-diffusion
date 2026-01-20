# Claude Code Instructions for VMAT Diffusion Project

**IMPORTANT: Keep this file updated as work progresses!**

---

## Project Overview

This project implements a generative AI model using diffusion techniques to create deliverable **Volumetric Modulated Arc Therapy (VMAT)** plans for radiation therapy (specifically prostate cancer).

### The Core Idea
Frame VMAT planning as a generative task analogous to AI image generation:
- **Input (Prompt):** CT scan + contoured structures + dose constraints
- **Output:** 3D dose distribution (and eventually MLC sequences, beam parameters)

### Key Goals
1. Train diffusion models (DDPM) to predict clinically acceptable dose distributions
2. Condition on patient anatomy (CT, structures as SDFs) and planning constraints
3. Achieve metrics: MAE < 3 Gy, Gamma pass rate > 95% (3%/3mm)
4. Eventually extend to predict deliverable MLC sequences

### Disease Site
- **Prostate cancer** with SIB (Simultaneous Integrated Boost)
- PTV70: 70 Gy in 28 fractions
- PTV56: 56 Gy in 28 fractions
- OARs: Rectum, Bladder, Femurs, Bowel

### Dataset Scale
| Phase | Cases | Status |
|-------|-------|--------|
| Current | 24 (23 usable) | Available |
| Near-term | 100-150 | Expected soon |
| Final target | 750-1000 | Ultimate goal |

**Implications:**
- Current results are preliminary (small test set = limited statistical power)
- Model architecture and pipeline should scale to larger datasets
- Will need to re-evaluate with larger test sets for publication
- Consider k-fold cross-validation when dataset grows

---

## Progress Log

### Completed ✅

**2026-01-18: Preprocessing Pipeline**
- Fixed SDF computation bug (uint8 bitwise NOT issue)
- Processed 23/24 cases successfully (case_0013 skipped - non-SIB)
- All validation checks pass
- Data: 4.6 GB total, ~200 MB per case
- Script: `preprocess_dicom_rt_v2.2.py`

**2026-01-18/19: Baseline U-Net Training**
- Trained BaselineUNet3D (23.7M parameters)
- **Best result: 3.73 Gy MAE** at epoch 12
- Early stopped at epoch 62/200 (patience=50)
- Training time: 2.55 hours on RTX 3090
- Checkpoint: `runs/baseline_unet_run1/checkpoints/best-epoch=012-val/mae_gy=3.735.ckpt`
- Notebook: `notebooks/2026-01-19_baseline_unet_experiment.ipynb`

**2026-01-19: Documentation Setup**
- Created experiment notebook template
- Created EXPERIMENTS_INDEX.md for tracking
- Created SCIENTIFIC_BEST_PRACTICES.md
- Set up publication-ready figure standards

**2026-01-19: Baseline U-Net Test Set Evaluation**
- Evaluated on 2 held-out test cases (case_0007, case_0021)
- **Test MAE: 1.43 ± 0.24 Gy** (excellent, well below 3 Gy target)
- **Test Gamma: 14.2 ± 5.7%** (poor, far below 95% target)
- **Key Finding:** Model underdoses PTVs (D95 ~20 Gy below target)
- OAR constraints all pass (model is conservative)
- Results: `predictions/baseline_unet_test/baseline_evaluation_results.json`

**Analysis:** The baseline learns a "blurred" dose pattern - good overall magnitude but misses sharp gradients needed for clinical acceptability. This motivates the DDPM approach.

### In Progress 🔄

**2026-01-19: DDPM Training (Running)**
- Training DDPM model on RTX 3090
- **Data location:** `./data/processed_npz/` (local SSD copy - NOT Windows mount!)
- Test cases held out: case_0007, case_0021 (same as baseline)
- **To monitor:**
  ```bash
  # Check metrics
  cat runs/vmat_dose_ddpm/version_*/metrics.csv | tail -20
  # Check GPU
  nvidia-smi
  # Check watchdog (if running)
  tail -20 runs/watchdog.log
  ```
- **To restart if needed:**
  ```bash
  cd ~/vmat-diffusion-project
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate vmat-diffusion
  python scripts/train_dose_ddpm_v2.py --data_dir ./processed --epochs 200
  ```
- **To run with watchdog (auto-recovery):**
  ```bash
  cd ~/vmat-diffusion-project
  nohup ./scripts/training_watchdog.sh > runs/watchdog_output.log 2>&1 &
  ```
- **If hangs occurred:** Check `runs/hang_diagnostics.log` for diagnostic info

### Next Steps 📋

1. **Train DDPM model** - Main diffusion model (~8-16 hours)
2. **Compare baseline vs DDPM** - Document improvements in gamma/DVH
3. **Create visualization notebook** - DVH plots, dose slices comparison
4. **Analyze why baseline underdoses PTVs** - Loss function investigation

### Future Work 📝

- Ablation studies (no SDF, no constraints)
- Hyperparameter tuning
- Larger dataset (process case_0013, acquire more data)
- MLC sequence prediction (Phase 2)
- Clinical validation

---

## Key Documentation (READ THESE FIRST)

Before starting work, review:
1. `docs/SCIENTIFIC_BEST_PRACTICES.md` - Reproducibility and publication guidelines
2. `notebooks/EXPERIMENTS_INDEX.md` - Experiment tracking and naming conventions
3. `notebooks/TEMPLATE_experiment.ipynb` - Template for new experiments

---

## Experiment Documentation Guidelines

**IMPORTANT: Every experiment MUST be documented in a notebook and tracked in the index.**

### BEFORE Running Any Experiment:

1. **COMMIT ALL CHANGES TO GIT** (Critical for reproducibility!):
   ```bash
   git add -A
   git commit -m "Pre-experiment commit: <brief description of experiment>"
   ```
   - This ensures you can reproduce results by checking out the exact commit
   - Record the commit hash in the experiment notebook
   - Never run experiments with uncommitted changes

2. **Verify clean working directory**:
   ```bash
   git status  # Should show "nothing to commit, working tree clean"
   ```

### Required Documentation Steps:

1. **Create experiment notebook** using the template:
   - Copy `notebooks/TEMPLATE_experiment.ipynb`
   - Rename to `YYYY-MM-DD_<experiment_name>.ipynb`
   - Fill in all sections (reproducibility, dataset, model, results, analysis)
   - **Record the git commit hash** in Section 2

2. **Update EXPERIMENTS_INDEX.md**:
   - Add new row to the Experiment Log table
   - Include: Date, Experiment ID, Notebook link, Model, Best Metric, Status

3. **Save all figures**:
   - Create `<output_dir>/figures/` directory
   - Save as PNG (300 DPI) AND PDF (vector)
   - Use publication-quality settings (see SCIENTIFIC_BEST_PRACTICES.md)

4. **Archive artifacts**:
   - Checkpoints, configs, metrics CSV, predictions
   - Document paths in notebook Section 10

5. **After experiment completes**, commit the results:
   ```bash
   git add notebooks/<experiment_notebook>.ipynb
   git add <output_dir>/figures/
   git commit -m "Results: <experiment_name> - <key metrics>"
   ```

### Template Location:
- `notebooks/TEMPLATE_experiment.ipynb` - Copy and customize for each experiment
- `notebooks/EXPERIMENTS_INDEX.md` - Central experiment tracking
- `docs/SCIENTIFIC_BEST_PRACTICES.md` - Full documentation standards

---

## Best Practices Summary

### For Every Experiment:
1. **BEFORE running:** `git add -A && git commit -m "Pre-experiment: <name>"` (REQUIRED!)
2. **Use seed:** 42 (or document if different)
3. **Create notebook:** `YYYY-MM-DD_<experiment>_experiment.ipynb` from template
4. **Record git hash:** In notebook Section 2 (Reproducibility Information)
5. **After completion:** Update EXPERIMENTS_INDEX.md
6. **Save figures:** 300 DPI, PNG + PDF in `figures/` subdirectory
7. **Commit results:** `git commit -m "Results: <name> - <metrics>"`

### Reproducibility Requirements:
- Git hash, Python version, PyTorch version, CUDA version, GPU model
- Random seeds for all stochastic operations
- Exact command to reproduce

### Figure Standards:
- 300 DPI minimum for publication
- Colorblind-friendly palettes
- Save as PNG (raster) and PDF (vector)
- Font size 12pt minimum

### Medical Physics Metrics:
- MAE in Gy (target: < 3 Gy)
- Gamma pass rate 3%/3mm (target: > 95%)
- DVH comparison for all structures
- Structure-specific dose statistics

---

## Data Locations

| Data | Primary Location | Fallback |
|------|------------------|----------|
| Raw DICOM | `/mnt/i/anonymized_dicom/` | `./data/raw/` |
| Processed NPZ (for training) | `./data/processed_npz/` (local SSD) | `/mnt/i/processed_npz/` |
| Training runs | `./runs/` | - |
| Predictions | `./predictions/` | - |

**IMPORTANT (WSL Performance):**
- **Always train from local SSD** (`./data/processed_npz/`), NOT from Windows mounts (`/mnt/`)
- Windows mounts are slow (9p filesystem) and cause I/O bottlenecks during training
- The symlink `./processed` should point to `./data/processed_npz/` (local)
- Copy data to local if needed: `cp -r /mnt/i/processed_npz ./data/`

Scripts auto-detect paths (check local first, then external drive).

---

## Environment

```bash
# Recreate environment from scratch
conda env create -f environment.yml
conda activate vmat-diffusion

# Or install with pip (after creating conda env with Python 3.12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Current environment:**
- Python: 3.12.12
- PyTorch: 2.4.1
- CUDA: 12.4
- GPU: NVIDIA GeForce RTX 3090 (24 GB)

**Key files:**
- `environment.yml` - Full conda environment export
- `requirements.txt` - Minimal pip dependencies

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 12 GB VRAM | 24 GB VRAM (RTX 3090) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 200 GB (for full dataset) |
| CUDA | 12.0+ | 12.4 |

**Notes:**
- Baseline U-Net: ~8 GB VRAM during training
- DDPM: ~23 GB VRAM during training (uses most of RTX 3090)
- Gamma computation is CPU/RAM intensive (~3 GB per case)

### WSL2-Specific Settings (Critical for Stability)

The training scripts have been tuned for WSL2 stability:

**DataLoader settings** (in `train_dose_ddpm_v2.py`):
- `num_workers=2` (not 4) - prevents dataloader deadlocks
- `persistent_workers=False` - disabled for WSL stability
- `prefetch_factor=2` - limits memory queue size

**Why these matter:**
- WSL2 has limited shared memory and IPC can deadlock with many workers
- `persistent_workers=True` caused worker processes to hang on WSL
- Higher worker counts (4+) led to training stalls after several epochs

**If training hangs:**
1. Kill training: `pkill -f train_dose_ddpm`
2. Restart WSL: `wsl --shutdown` (from PowerShell)
3. Reopen terminal and restart training

**DO NOT:**
- Cache all training data in RAM (causes WSL memory pressure → GPU driver errors)
- Use more than 2 dataloader workers on WSL
- Train from Windows mounts (`/mnt/`) - copy to local SSD first

---

## Structure Naming Conventions

The preprocessing expects these DICOM structure names (case-insensitive, partial match):

| Structure | Expected Names | Required |
|-----------|---------------|----------|
| PTV70 | `PTV70`, `PTV_70`, `PTV 70Gy` | Yes (SIB) |
| PTV56 | `PTV56`, `PTV_56`, `PTV 56Gy` | Yes (SIB) |
| Prostate | `Prostate`, `CTV_Prostate` | Yes |
| Rectum | `Rectum`, `Rect` | Yes |
| Bladder | `Bladder`, `Blad` | Yes |
| Femur_L | `Femur_L`, `FemoralHead_L`, `Left Femur` | Yes |
| Femur_R | `Femur_R`, `FemoralHead_R`, `Right Femur` | Yes |
| Bowel | `Bowel`, `BowelBag`, `SmallBowel` | Optional |

**Cases skipped if:** Missing PTV70 or PTV56 (non-SIB protocol)

See `oar_mapping.json` for the full mapping configuration.

---

## Adding New DICOM-RT Data

When you receive new cases (100+, 750+):

1. **Place anonymized DICOM in** `/mnt/i/anonymized_dicom/case_XXXX/`
   - Ensure PHI is removed
   - Each case needs: CT, RTStruct, RTDose, RTPlan

2. **Run preprocessing**:
   ```bash
   python scripts/preprocess_dicom_rt_v2.2.py --skip_plots
   ```
   - Skipped cases are logged (check for missing structures)
   - ~2-3 minutes per case

3. **Validate the batch**:
   ```bash
   # Quick check
   ls /mnt/i/processed_npz/*.npz | wc -l

   # Full validation (run verify_npz.ipynb or):
   python -c "import numpy as np; from pathlib import Path; [print(f'{p.name}: {np.load(p)[\"dose\"].shape}') for p in Path('/mnt/i/processed_npz').glob('*.npz')]"
   ```

4. **Update train/val/test splits** if needed (currently in training scripts)

---

## Known Issues & Troubleshooting

### Preprocessing Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Case skipped | Missing PTV56/PTV70 | Non-SIB case, expected |
| SDF all positive | Old bug (fixed) | Use `preprocess_dicom_rt_v2.2.py` |
| Structure not found | Naming mismatch | Update `oar_mapping.json` |

### Training Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `deterministic=True` error | Trilinear upsample | Use `deterministic="warn"` |
| OOM during training | Batch/patch too large | Reduce `batch_size` or `patch_size` |
| Loss goes NaN | Learning rate too high | Reduce `lr` by 10x |
| Training hangs/stalls | Dataloader deadlock on WSL | Use `num_workers=2`, disable `persistent_workers` |
| WSL crashes / GPU errors | RAM caching + WSL memory pressure | Don't cache data in RAM; use disk loading |
| Slow I/O, bursty GPU util | Data on Windows mount (`/mnt/`) | Copy data to WSL filesystem (`~/` or `./data/`) |
| CUDA driver errors | WSL2 GPU passthrough issues | Run `wsl --shutdown` from PowerShell, restart |

### Training Watchdog (Auto-Recovery)

A diagnostic watchdog script monitors training and auto-restarts on hangs while capturing diagnostic info.

**Location:** `scripts/training_watchdog.sh`

**What it does:**
1. Monitors `metrics.csv` for progress every 60 seconds
2. If no progress for 3 minutes, captures detailed diagnostics:
   - Process states (main + workers)
   - CPU/memory per process
   - GPU state
   - System memory
   - dmesg output
3. Logs diagnostics to `runs/hang_diagnostics.log`
4. Kills hung processes and restarts training
5. Logs all activity to `runs/watchdog.log`

**To start watchdog:**
```bash
cd ~/vmat-diffusion-project
nohup ./scripts/training_watchdog.sh > runs/watchdog_output.log 2>&1 &
```

**To check watchdog status:**
```bash
tail -20 runs/watchdog.log
```

**To analyze hang patterns:**
```bash
cat runs/hang_diagnostics.log
```

**To stop watchdog:**
```bash
pkill -f training_watchdog.sh
```

**IMPORTANT:** After a hang, check `runs/hang_diagnostics.log` for patterns before dismissing. Common patterns:
- Workers in `D` (uninterruptible sleep) state → I/O issue
- Workers in `S` state but not progressing → deadlock
- GPU memory full → OOM during validation
- Process died → crash (check dmesg)

### Inference Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Gamma computation fails | Missing numba | `pip install numba` |
| pymedphys import error | Old version | `pip install pymedphys>=0.40` |
| OOM during gamma | Full resolution | Use `--gamma_subsample 4` |

### Data Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| case_0013 skipped | Non-SIB protocol | Expected, not an error |
| Dose range varies | Different Rx doses | Normalized in preprocessing |

---

## Key Commands

```bash
# Preprocessing (all cases)
python scripts/preprocess_dicom_rt_v2.2.py --skip_plots

# Train baseline U-Net
python scripts/train_baseline_unet.py --data_dir /mnt/i/processed_npz --epochs 200

# Train DDPM
python scripts/train_dose_ddpm_v2.py --data_dir /mnt/i/processed_npz --epochs 200

# Run inference
python scripts/inference_baseline_unet.py --checkpoint <path> --input_dir <dir>
```

---

## Updating This File

**IMPORTANT:** After completing significant work, update this file:

1. Move items from "Next Steps" to "Completed" with date and details
2. Add new items to "Next Steps" as they emerge
3. Update "In Progress" when starting new work
4. Keep the progress log as a running history

This ensures continuity across sessions and after context compaction.

---

*Last updated: 2026-01-19 (Added training watchdog script for hang detection and auto-recovery)*
