# Claude Code Instructions for VMAT Diffusion Project

**IMPORTANT: Keep this file updated as work progresses!**

---

## ⚠️ PRIMARY PROJECT LOCATION

**All work should be done in:** `C:\Users\Bill\vmat-diffusion-project`

| Item | Path |
|------|------|
| Project | `C:\Users\Bill\vmat-diffusion-project` |
| Data | `I:\processed_npz` |
| Conda env | `vmat-win` (activate via Pinokio miniconda) |

**DEPRECATED:** `~/wrockey/vmat-diffusion-project` (WSL) - do not use

All logging, results, notebooks, and git commits should be in the Windows directory.

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

**What 24 cases CAN tell us:**
- **Relative model comparison** - Both models trained on identical data/splits, so comparing baseline vs DDPM is valid
- **Fundamental architectural issues** - Extreme volatility (MAE 12-64 Gy) isn't just small-sample noise; it indicates real problems
- **Workflow validation** - Training completes, checkpoints save, metrics log correctly
- **Loss vs metric disconnect** - If diffusion loss decreases but dose quality doesn't improve, more data won't fix that

**What 24 cases CAN'T tell us:**
- **Absolute performance** - Both models will likely improve with more data
- **Publication-quality claims** - Need larger test set (n≥50) for statistical significance
- **Whether DDPM could eventually win** - Diffusion models are more data-hungry than direct regression

**Recommendation:** Investigate issues now with lightweight experiments, but document as "preliminary" and re-evaluate when 100+ cases arrive. Don't spend weeks tuning hyperparameters on 24 cases.

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

**2026-01-20: DDPM v1 Training Complete**
- Platform: Native Windows/Pinokio (stable after WSL2 issues)
- Training time: 1.94 hours, 37 epochs (early stopped, patience=20)
- **Best MAE: 12.19 Gy** at epoch 15 (target: <3 Gy)
- **Result: Underperformed baseline** (baseline: 3.73 Gy val MAE)
- Checkpoint: `runs/vmat_dose_ddpm/checkpoints/best-epoch=015-val/mae_gy=12.19.ckpt`

**Key Finding: Loss vs MAE Disconnect**
- val_loss decreased steadily (0.108 → 0.004) ✓
- val_mae was extremely volatile (12-64 Gy range) ✗
- Diffusion model learns denoising well but produces unstable dose predictions
- Likely cause: sampling process introduces variability not captured by noise prediction loss

**Note on dataset size:** Results are preliminary (24 cases). However, the baseline vs DDPM comparison is valid since both used identical data. The extreme MAE volatility indicates a fundamental issue worth investigating now, not just small-sample noise. Re-evaluate with 100+ cases. See "Dataset Scale" section for details.

**Archived runs:**
- `runs/vmat_dose_ddpm/` - Completed DDPM v1 (this run)
- `runs/vmat_dose_ddpm_wsl_hangs/` - WSL runs with repeated hangs
- `runs/vmat_dose_ddpm_pinokio_crashed/` - Pinokio run that crashed with 0x113

### In Progress 🔄

None currently.

### Next Steps 📋

**See `docs/DDPM_OPTIMIZATION_PLAN.md` for detailed systematic plan.**

Summary of phases:
1. **Phase 1: Quick Wins** (no retraining) - sampling steps ablation, ensemble averaging
2. **Phase 2: Schedule/Loss** - try different schedules, add dose-aware loss terms
3. **Phase 3: Architecture** - conditioning ablation, capacity scaling
4. **Phase 4: Alternatives** - if DDPM still underperforms, try flow matching or improved baseline

**Immediate next action:** Run Phase 1 experiments using existing checkpoint (inference only, ~1 hour)

**Environment note:** numba/pymedphys[tests] is now installed - gamma metrics should work.

### Future Work 📝

- Ablation studies (no SDF, no constraints)
- Hyperparameter tuning
- Larger dataset (process case_0013, acquire more data)
- MLC sequence prediction (Phase 2)
- Clinical validation

---

## Key Documentation (READ THESE FIRST)

Before starting work, review these in order:

1. **`notebooks/EXPERIMENTS_INDEX.md`** - MASTER experiment list (single source of truth)
2. **`docs/DDPM_OPTIMIZATION_PLAN.md`** - Current focus: systematic plan to improve DDPM
3. **`docs/EXPERIMENT_STRUCTURE.md`** - How to organize experiments, notebooks, outputs
4. **`docs/SCIENTIFIC_BEST_PRACTICES.md`** - Reproducibility and publication guidelines
5. **`notebooks/TEMPLATE_experiment.ipynb`** - Template for new experiments

**Documentation hierarchy:**
```
notebooks/EXPERIMENTS_INDEX.md   → What experiments exist (MASTER LIST)
docs/EXPERIMENT_STRUCTURE.md     → How to organize new experiments
docs/DDPM_OPTIMIZATION_PLAN.md   → Current optimization work plan
```

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

| Data | Location |
|------|----------|
| Project | `C:\Users\Bill\vmat-diffusion-project` |
| Processed NPZ | `I:\processed_npz` (23 cases) |
| Training runs | `C:\Users\Bill\vmat-diffusion-project\runs` |
| Notebooks | `C:\Users\Bill\vmat-diffusion-project\notebooks` |
| Experiments | `C:\Users\Bill\vmat-diffusion-project\experiments` |
| Raw DICOM | `I:\anonymized_dicom` |

**Note:** WSL2 (`~/wrockey/...`) is deprecated due to stability issues.

---

## Environment

### Native Windows (Pinokio) - PREFERRED

```cmd
:: Activate the vmat-win environment
call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win

:: Environment was created with:
:: conda create -n vmat-win python=3.10
:: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
:: pip install pytorch-lightning numpy scipy tensorboard rich pymedphys
```

### WSL2 (Legacy)

```bash
# Recreate environment from scratch
conda env create -f environment.yml
conda activate vmat-diffusion

# Or install with pip (after creating conda env with Python 3.12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Current environments:**
| Platform | Env Name | Python | PyTorch | CUDA |
|----------|----------|--------|---------|------|
| Windows (Pinokio) | vmat-win | 3.10 | 2.x | 12.1 |
| WSL2 | vmat-diffusion | 3.12 | 2.4.1 | 12.4 |

**Hardware:** NVIDIA GeForce RTX 3090 (24 GB)

**Key files:**
- `environment.yml` - Full conda environment export (WSL)
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
| **0x113 TDR / dxgkrnl crash** | GPU overheating or driver timeout | Use native Windows with `--gpu_cooling` |
| **Windows unresponsive** | GPU at 100% sustained | Enable `--gpu_cooling`, reduce batch_size |
| **Repeated WSL hangs** | WSL2/CUDA bridge issues | Switch to native Windows/Pinokio |

### GPU Stability Options (for native Windows)

The training script supports GPU cooling pauses to prevent thermal throttling and TDR crashes:

```cmd
python scripts\train_dose_ddpm_v2.py ^
    --data_dir I:\processed_npz ^
    --gpu_cooling ^
    --cooling_interval 10 ^
    --cooling_pause 0.5 ^
    --batch_size 1 ^
    --base_channels 32
```

Or use the pre-configured safe mode: `start_training_safe.bat`

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
| "PyMedPhys unable to import numba.njit" | numba not installed in vmat-win env | `pip install pymedphys[tests]` (includes numba) |

### Data Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| case_0013 skipped | Non-SIB protocol | Expected, not an error |
| Dose range varies | Different Rx doses | Normalized in preprocessing |

---

## Key Commands

### Native Windows (Pinokio) - PREFERRED

```cmd
:: From C:\Users\Bill\vmat-diffusion-project

:: Train DDPM (safe mode - recommended)
start_training_safe.bat

:: Train DDPM (manual)
call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win
python scripts\train_dose_ddpm_v2.py --data_dir I:\processed_npz --epochs 200 --gpu_cooling
```

### WSL2 (Legacy)

```bash
# Preprocessing (all cases)
python scripts/preprocess_dicom_rt_v2.2.py --skip_plots

# Train baseline U-Net
python scripts/train_baseline_unet.py --data_dir /mnt/i/processed_npz --epochs 200

# Train DDPM
python scripts/train_dose_ddpm_v2.py --data_dir ./processed --epochs 200

# Run inference
python scripts/inference_baseline_unet.py --checkpoint <path> --input_dir <dir>
```

---

## Documentation Maintenance

### When to Update Each Document

| Document | Update When | What to Update |
|----------|-------------|----------------|
| `.claude/instructions.md` | After any significant work | Progress Log, Next Steps, In Progress |
| `notebooks/EXPERIMENTS_INDEX.md` | After EVERY experiment | Add row with date, ID, git hash, metrics |
| `docs/CHANGELOG.md` | New features, bug fixes, breaking changes | Version entry with date and details |
| `docs/DDPM_OPTIMIZATION_PLAN.md` | After optimization experiments | Results Tracking table, conclusions |
| `docs/README.md` | API changes, new scripts, architecture changes | Relevant sections |

### CHANGELOG Guidelines

Update `docs/CHANGELOG.md` when:
- **Adding new features** (new scripts, new model architectures, new metrics)
- **Fixing bugs** (preprocessing issues, training fixes)
- **Breaking changes** (file format changes, API changes)
- **Deprecating code** (moving scripts to `deprecated/`)

Format:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature description

### Changed
- Modified behavior

### Fixed
- Bug fix description

### Deprecated
- What's deprecated and what to use instead
```

### File Organization Rules

**New scripts:**
- Production scripts → `scripts/`
- One-off utilities → `scripts/` with clear docstring
- Deprecated versions → `scripts/deprecated/`

**When to deprecate:**
- When a v2/v3 version replaces the original
- Add `DEPRECATED: Use <new_script> instead` as first line of docstring
- Move to `scripts/deprecated/` after confirming new version works

**Naming conventions:**
- Scripts: `snake_case.py` (e.g., `train_dose_ddpm_v2.py`)
- Notebooks: `YYYY-MM-DD_<experiment_name>.ipynb`
- Configs: `<purpose>_config.json`

### Git Workflow

**Commit message format:**
```
<type>: <short description>

<optional body>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code change that neither fixes nor adds
- `test:` - Adding tests
- `chore:` - Maintenance (deps, cleanup)

**Tagging milestones:**
```bash
git tag -a v1.0-baseline -m "Baseline U-Net complete, 3.73 Gy MAE"
git push origin --tags
```

Tag when:
- Major model milestone achieved
- Before breaking changes
- Publication submission points

### Cleanup Tasks

**Periodic cleanup (weekly or before major work):**

1. **Log files:** Delete `*.log` files from root (they're gitignored)
   ```bash
   rm -f *.log
   ```

2. **Old checkpoints:** Keep only `best.ckpt` and `last.ckpt` per run
   ```bash
   # Review before deleting
   ls runs/*/checkpoints/
   ```

3. **Archived runs:** Move failed/test runs to archive or delete
   - Keep runs referenced in EXPERIMENTS_INDEX.md
   - Archive naming: `runs/<name>_archived/` or delete if not needed

4. **Disk space check:**
   ```bash
   du -sh runs/* experiments/* predictions/*
   ```

### Cross-Document Consistency Checklist

Before ending a work session, verify:

- [ ] `instructions.md` Progress Log updated
- [ ] `EXPERIMENTS_INDEX.md` has all experiments with git hashes
- [ ] `CHANGELOG.md` updated if any code/features changed
- [ ] `DDPM_OPTIMIZATION_PLAN.md` Results table updated (if running optimization)
- [ ] All changes committed and pushed
- [ ] `*Last updated*` timestamp updated in modified docs

---

## Updating This File

**IMPORTANT:** After completing significant work, update this file:

1. Move items from "Next Steps" to "Completed" with date and details
2. Add new items to "Next Steps" as they emerge
3. Update "In Progress" when starting new work
4. Keep the progress log as a running history
5. Update the `*Last updated*` line at the bottom

This ensures continuity across sessions and after context compaction.

---

*Last updated: 2026-01-20 (Added Documentation Maintenance section with housekeeping guidelines)*
