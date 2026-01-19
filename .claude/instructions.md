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

*Nothing currently in progress*

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
| Processed NPZ | `/mnt/i/processed_npz/` | `./processed/` |
| Training runs | `./runs/` | - |
| Predictions | `./predictions/` | - |

Scripts auto-detect paths (check local first, then external drive).

---

## Environment

```bash
# Activate environment
conda activate vmat-diffusion

# Key packages
Python: 3.12.12
PyTorch: 2.4.1
CUDA: 12.4
GPU: NVIDIA GeForce RTX 3090 (24 GB)
```

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

*Last updated: 2026-01-19 (Added pre-experiment git commit requirement)*
