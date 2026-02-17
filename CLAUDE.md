# CLAUDE.md

## Project Overview

VMAT Diffusion is a deep learning research project for automated **Volumetric Modulated Arc Therapy (VMAT) dose prediction** in radiation therapy. It uses diffusion models (DDPM) and baseline U-Net architectures to predict 3D dose distributions from patient CT scans, organ contours, and clinical dose constraints.

**Disease site:** Prostate cancer with SIB (70 Gy PTV70=prostate / 56 Gy PTV56=seminal vesicles in 28 fractions)
**Current dataset:** 24 cases (23 usable), expecting 100-150 near-term
**Clinical targets (updated 2026-02-17):** PTV70 D95 >= 66.5 Gy, PTV56 D95 >= 53.2 Gy, OAR DVH compliance, PTV-region Gamma > 95%. Global Gamma tracked as diagnostic only — see `.claude/instructions.md` for full priority table.

## Repository Structure

```
vmat-diffusion/
├── scripts/                          # All Python scripts
│   ├── train_baseline_unet.py        # Baseline U-Net trainer (primary model)
│   ├── train_dose_ddpm_v2.py         # DDPM trainer (not recommended currently)
│   ├── inference_baseline_unet.py    # Baseline model inference + evaluation
│   ├── inference_dose_ddpm.py        # DDPM inference + evaluation
│   ├── preprocess_dicom_rt_v2.2.py   # DICOM-RT → NPZ preprocessing
│   ├── run_phase1_experiments.py     # Sampling/ensemble ablation
│   ├── analyze_gamma_metric_hypothesis.py
│   ├── compute_test_metrics.py       # Standalone test evaluation
│   ├── generate_*_figures.py         # Publication figure scripts (6 variants)
│   ├── uncertainty_loss.py           # UncertaintyWeightedLoss (Kendall 2018) for Phase 2
│   ├── calibrate_loss_normalization.py # Loss calibration for initial_log_sigma values
│   ├── training_watchdog.sh          # Auto-recovery for hung training
│   └── deprecated/                   # Old script versions
├── notebooks/                        # Jupyter experiment notebooks
│   ├── EXPERIMENTS_INDEX.md          # MASTER experiment tracking (source of truth)
│   ├── TEMPLATE_experiment.ipynb     # Template for new experiments
│   ├── 2026-01-*_*.ipynb             # Dated experiment notebooks
│   ├── verify_npz.ipynb              # Data validation
│   └── analyze_results.ipynb         # General analysis
├── docs/                             # Documentation
│   ├── CHANGELOG.md
│   ├── DDPM_OPTIMIZATION_PLAN.md
│   ├── EXPERIMENT_STRUCTURE.md
│   ├── SCIENTIFIC_BEST_PRACTICES.md
│   ├── QUICKSTART.md
│   ├── preprocessing_guide.md
│   ├── training_guide.md
│   └── README.md
├── experiments/                      # Experiment output artifacts
├── .claude/instructions.md           # Detailed project state and session notes
├── oar_mapping.json                  # DICOM structure name → canonical mapping
├── environment.yml                   # Conda environment (WSL/Linux)
├── environment_vmat-win.yml          # Conda environment (Windows)
├── requirements.txt                  # Minimal pip dependencies
├── start_training_safe.bat           # Windows safe-mode training launcher
└── start_training.bat                # Windows training launcher
```

**Gitignored (not in repo):**
- `runs/` — Training outputs, checkpoints, TensorBoard logs
- `predictions/` — Inference outputs
- `processed/`, `data/` — Medical imaging data (.npz, .dcm)
- `*.ckpt`, `*.pth` — Model weights

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10-3.12 |
| Deep Learning | PyTorch 2.4+, PyTorch Lightning 2.0+ |
| Medical Imaging | pydicom, SimpleITK, rt-utils |
| Evaluation | pymedphys (gamma analysis, DVH), numba |
| Scientific | numpy, scipy, scikit-image, pandas |
| Visualization | matplotlib, seaborn, TensorBoard |
| Infrastructure | rich, tqdm, Jupyter |
| GPU | CUDA 12.4, cuDNN 9.10 |

## Environment Setup

**Conda (preferred):**
```bash
conda env create -f environment.yml
conda activate vmat-diffusion
```

**Pip (minimal):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Hardware:** NVIDIA RTX 3090 (24 GB) primary. Baseline U-Net uses ~8 GB VRAM; DDPM uses ~23 GB.

## Key Commands

```bash
# Preprocessing: DICOM-RT → NPZ
python scripts/preprocess_dicom_rt_v2.2.py --skip_plots

# Train baseline U-Net (primary model)
python scripts/train_baseline_unet.py \
    --data_dir /path/to/processed_npz \
    --exp_name my_experiment \
    --epochs 200

# Train with advanced losses (examples)
python scripts/train_baseline_unet.py \
    --data_dir /path/to/processed_npz \
    --exp_name dvh_aware_loss \
    --use_gradient_loss --gradient_loss_weight 0.1 \
    --use_dvh_loss --dvh_loss_weight 0.01 \
    --epochs 100

# Run inference + evaluation
python scripts/inference_baseline_unet.py \
    --checkpoint runs/<exp>/checkpoints/best-*.ckpt \
    --input_dir /path/to/processed_npz

# Train DDPM (NOT recommended — see notes below)
python scripts/train_dose_ddpm_v2.py --data_dir /path/to/processed_npz --epochs 200
```

## Architecture Overview

### Data Pipeline
- **Input:** CT (1 channel) + Signed Distance Fields for 8 structures (8 channels) = 9 input channels
- **Output:** 3D dose distribution (1 channel)
- **Constraints:** 13-dimensional vector (prescription doses + OAR limits)
- **Patch size:** 128³ voxels during training, sliding window for full-volume inference
- **Data split:** 80/10/10 (train/val/test)

### 8 Anatomical Structures (SDF channels)
0: PTV70, 1: PTV56, 2: Prostate, 3: Rectum, 4: Bladder, 5: Femur_L, 6: Femur_R, 7: Bowel

### NPZ Data Format (v2.2.0)
```python
{
    'ct': (D, H, W) float32,              # Normalized [0,1]
    'dose': (D, H, W) float32,            # Normalized to Rx
    'masks': (8, D, H, W) uint8,          # Binary masks
    'masks_sdf': (8, D, H, W) float32,    # Signed distance fields
    'constraints': (13,) float32,          # [Rx, OAR limits...]
    'metadata': dict,                      # Case info
}
```

### Model: BaselineUNet3D (primary)
- 3D U-Net with constraint conditioning via embedding
- 4 encoder/decoder levels (32→64→128→256 channels)
- Skip connections, SiLU activations
- ~23.7M parameters

### Model: DoseDDPM (experimental, not recommended)
- Conditional DDPM with cosine noise schedule
- DDIM sampling (50 steps optimal)
- Same U-Net backbone with time embedding
- Matches but does not beat baseline; added complexity without benefit

## Code Conventions

### Naming
- **Classes:** PascalCase (`DoseDDPM`, `SimpleUNet3D`, `VMATDosePatchDataset`)
- **Functions/variables:** snake_case (`cosine_beta_schedule`, `patch_size`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_SPACING_MM`, `PRIMARY_PRESCRIPTION_GY`)

### Import Order
```python
# Standard library
import os, sys, json, argparse

# Third-party scientific
import numpy as np
import torch

# Deep learning
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Project-internal
from train_baseline_unet import BaselineUNet3D
```

### Type Hints
Extensively used throughout (Python 3.8+ style):
```python
def __init__(self, data_dir: str, patch_size: int = 128) -> None:
```

### Docstrings
Module-level + class/method docstrings in descriptive style.

### Logging
- PyTorch Lightning `self.log()` for training metrics (TensorBoard backend)
- `PublicationLoggingCallback` for per-epoch CSV logs
- rich library for terminal progress

### Git Commit Messages
```
<type>: <short description>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
Types: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## Testing and Evaluation

There is no pytest suite. Evaluation is done through inference scripts with medical physics metrics:

| Metric | Target | How |
|--------|--------|-----|
| MAE (Gy) | < 3 Gy | `inference_baseline_unet.py` |
| Gamma (3%/3mm) | > 95% | pymedphys, `--gamma_subsample 4` for speed |
| DVH (D95, Dmean, Vx) | Per-structure clinical constraints | Per-structure evaluation |
| QUANTEC compliance | All OARs within limits | Clinical constraint checking |

## Experiment Documentation Requirements

**MANDATORY: Every experiment MUST be fully documented, logged, reproducible, and publishable. No exceptions.** Treat every experiment as if it will appear in a peer-reviewed journal submission. Incomplete documentation is equivalent to an experiment that never happened.

### Pre-Experiment (REQUIRED before any training run)

1. **Commit all code changes to git** — never run experiments on uncommitted code:
   ```bash
   git add -A
   git commit -m "Pre-experiment: <experiment_name>"
   git status  # Must show "nothing to commit, working tree clean"
   ```
2. **Record the exact git commit hash** — this is the only reliable way to reproduce results later. Without it, the experiment cannot be reproduced and has no scientific value.

### During the Experiment

3. **Run training** with appropriate CLI flags. All hyperparameters must be captured in the training config JSON that the scripts auto-save to `runs/<exp_name>/training_config.json`.
4. **Run test-set inference and evaluation** when training completes. Record MAE, Gamma (3%/3mm), DVH metrics, and QUANTEC compliance.

### Post-Experiment Documentation (ALL steps REQUIRED)

5. **Create a figure generation script** (`scripts/generate_<exp_name>_figures.py`):
   - Use existing scripts as templates (e.g., `scripts/generate_grad_loss_figures.py`, `scripts/generate_dvh_loss_figures.py`)
   - **Every figure must be publication-ready from the start** — there is no "draft" quality:
     ```python
     plt.rcParams.update({
         'font.family': 'serif',
         'font.size': 12,
         'figure.dpi': 150,
         'savefig.dpi': 300,
         'savefig.bbox': 'tight',
     })
     ```
   - Save every figure as both PNG (300 DPI raster) and PDF (vector) to `runs/<exp_name>/figures/`
   - Use colorblind-friendly color palettes (see existing scripts for standard palette)
   - Minimum font size: 12pt. Axes must be labeled with units. Legends must be present.

6. **Create an experiment notebook** (`notebooks/YYYY-MM-DD_<exp_name>.ipynb`):
   - Copy from `notebooks/TEMPLATE_experiment.ipynb`
   - Reference existing notebooks for format (e.g., `notebooks/2026-01-20_grad_loss_experiment.ipynb`)
   - **All 10 sections are required:**
     1. **Overview** — Objective, hypothesis, key results summary, conclusion
     2. **Reproducibility Information** — Git commit hash, Python/PyTorch/CUDA versions, GPU model, exact command to reproduce
     3. **Dataset Information** — Number of cases, split, preprocessing version
     4. **Model/Method Configuration** — Architecture, loss functions, conditioning
     5. **Training Configuration** — Epochs, LR, batch size, patch size, augmentations
     6. **Results** — Embedded publication-ready figures with descriptive captions. Every figure must have a caption explaining what it shows and why it matters.
     7. **Analysis** — Observations, comparison to all prior experiments, statistical assessment, limitations
     8. **Conclusions and Recommendations** — What worked, what didn't, and why
     9. **Next Steps** — What this experiment motivates
     10. **Artifacts** — Table of file paths (checkpoints, configs, predictions, figures)
   - **Figures in notebooks must include captions and written assessments** — a figure without interpretation is useless. Explain what the reader should observe, what it means clinically, and how it compares to prior results.

7. **Update the experiment index** (`notebooks/EXPERIMENTS_INDEX.md`):
   - This is the **single source of truth** for all experiments ever run
   - Add a row with: Date, Experiment ID, Git commit hash, Notebook link, Model type, Key metrics (MAE, Gamma, D95), Status
   - If an experiment is not in this index, it does not exist

8. **Commit all documentation**:
   ```bash
   git add scripts/generate_<exp_name>_figures.py
   git add notebooks/YYYY-MM-DD_<exp_name>.ipynb
   git add notebooks/EXPERIMENTS_INDEX.md
   git commit -m "Results: <exp_name> - <key metric summary>"
   ```

### Reproducibility Requirements (NON-NEGOTIABLE)

Every experiment must record:
- **Git commit hash** of the exact code used
- **Python version**, PyTorch version, CUDA version, GPU model
- **Random seed** (use 42 unless documented otherwise)
- **Exact CLI command** to reproduce the run
- **Data split** used (case IDs for train/val/test)
- **All hyperparameters** (saved automatically to `training_config.json`)

If any of these are missing, the experiment is not reproducible and cannot be cited in a publication.

### Publication Figure Standards

All figures — whether in scripts, notebooks, or standalone — must meet these standards:

| Requirement | Standard |
|-------------|----------|
| Resolution | 300 DPI minimum |
| Font | Serif family, 12pt minimum |
| Format | Both PNG (raster) and PDF (vector) |
| Colors | Colorblind-friendly palette |
| Axes | Labeled with units (e.g., "MAE (Gy)", "Epoch") |
| Legends | Present and readable |
| Captions | Required in notebooks — describe what is shown and what to conclude |

### Experiment Output Structure

```
runs/<exp_name>/
├── checkpoints/              # Model checkpoints (best + last)
├── figures/                  # Publication-ready figures (PNG + PDF)
├── version_*/                # PyTorch Lightning logs
├── training_config.json      # All hyperparameters
├── training_summary.json     # Final metrics
└── metrics.csv               # Per-epoch metrics

predictions/<exp_name>_test/
├── case_XXXX_pred.npz        # Per-case predictions
└── evaluation_results.json   # Aggregate test metrics
```

### What "Publishable" Means

Every experiment notebook should be ready to drop into a journal supplementary section as-is. This means:
- A reader unfamiliar with the project can understand what was done and why
- All results are quantified with proper metrics, not just "it looks better"
- Comparisons to prior experiments use consistent metrics and methodology
- Limitations and failure modes are honestly documented
- Figures tell a clear story without requiring external explanation

## Current Project Status

**For current strategy, model performance, decisions, and next steps: see `.claude/instructions.md`** (the living project state document, auto-loaded every session).

**For experiment history: see `notebooks/EXPERIMENTS_INDEX.md`** (the master experiment log).

## Important Notes

- **No CI/CD pipeline** — experiments are tracked manually via git + notebooks
- **No linter/formatter configured** — code follows PEP8 informally
- **No pytest tests** — validation is through medical physics metrics
- **Documentation hierarchy:**
  - `.claude/instructions.md` — living project state (strategy, decisions, next steps)
  - `CLAUDE.md` — static reference (this file: conventions, architecture, experiment protocol)
  - `notebooks/EXPERIMENTS_INDEX.md` — master experiment log
- **DataLoader:** Use `num_workers=2`, `persistent_workers=False` to avoid deadlocks (especially on WSL)
- **OAR name mapping:** `oar_mapping.json` maps 100+ clinical naming variations to 8 canonical structures
