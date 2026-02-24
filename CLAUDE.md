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
│   ├── setup_github_project.sh       # One-time GitHub labels/milestones/issues setup
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
# Preprocessing: DICOM-RT → NPZ (v2.3 crop pipeline)
python scripts/preprocess_dicom_rt_v2.2.py --skip_plots
# Options: --inplane_size 300 --z_margin_mm 30 (defaults)

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
- **Preprocessing (v2.3):** CT and masks kept at native resolution; dose resampled to CT grid (B-spline); all volumes cropped to ~300×300×Z centered on prostate. Variable output shape per patient.
- **Patch size:** 128³ voxels during training, sliding window for full-volume inference
- **Data split:** 80/10/10 (train/val/test)

### 8 Anatomical Structures (SDF channels)
0: PTV70, 1: PTV56, 2: Prostate, 3: Rectum, 4: Bladder, 5: Femur_L, 6: Femur_R, 7: Bowel

### NPZ Data Format (v2.3.0)
```python
{
    'ct': (Y, X, Z) float32,              # Normalized [0,1], native resolution cropped
    'dose': (Y, X, Z) float32,            # Normalized to Rx, on CT grid cropped
    'masks': (8, Y, X, Z) uint8,          # Binary masks, native grid cropped
    'masks_sdf': (8, Y, X, Z) float32,    # Signed distance fields [-1,1]
    'constraints': (13,) float32,          # [Rx, OAR limits...]
    'metadata': dict,                      # Case info, spacing, crop, validation
}
# Metadata includes: voxel_spacing_mm, volume_shape, crop_box, dose_grid_spacing_mm
# Typical output: ~300×300×160 (vs old 512×512×256)
```

### Model: BaselineUNet3D (primary)
- 3D U-Net with constraint conditioning via FiLM embedding
- 5 resolution levels, base_channels=48 (48→96→192→384→768)
- Skip connections, SiLU activations, GroupNorm
- ~23.7M parameters

### Model: DoseDDPM (experimental, not recommended)
- Conditional DDPM with cosine noise schedule, DDIM sampling (50 steps)
- SimpleUNet3D backbone with time embedding, base_channels=32 (32→64→128→256)
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

### Experiment Types

Not all experiments are training runs. Use the appropriate template variant:

| Type | When | Required Sections |
|------|------|-------------------|
| **Training experiment** | Training a model, comparing losses, etc. | All 10 sections |
| **Analysis experiment** | Metric analysis, data exploration, hypothesis testing | Sections 1-3, 6-10 (skip Model/Training config) |

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
   - Use existing scripts as templates (e.g., `scripts/generate_grad_loss_figures.py`)
   - Uses the standard plot configuration (see Publication Figure Standards below)
   - Save every figure as both PNG (300 DPI raster) and PDF (vector) to `runs/<exp_name>/figures/`
   - The notebook **loads** these saved figures — it does NOT regenerate them. This ensures the script is the single source for figures and the notebook stays lightweight.

6. **Create an experiment notebook** (`notebooks/YYYY-MM-DD_<exp_name>.ipynb`):
   - Copy from `notebooks/TEMPLATE_experiment.ipynb`
   - **All sections must be fully filled in** — no `None`, `[UPDATE]`, or placeholder values may remain in the final notebook.
   - **Required sections for training experiments:**
     1. **Overview** — Objective, hypothesis, key results summary, 1-2 sentence conclusion
     2. **What Changed** — Exactly one table: "Compared to [prior experiment], this experiment changes X. Everything else is identical." A reader must immediately know the single variable under test.
     3. **Reproducibility** — Git commit hash, Python/PyTorch/CUDA versions, GPU model, random seed, exact CLI command, conda environment snapshot
     4. **Dataset** — Case count, train/val/test case IDs (not just counts), preprocessing version, data file checksums or batch_summary.json reference
     5. **Model & Training Configuration** — Architecture, loss functions, all hyperparameters
     6. **Results** — Required figures (see Medical Physics Figure Set below), each with caption and "Key observations" bullets
     7. **Statistical Analysis** — For n>=10: confidence intervals (95% CI), effect sizes vs baseline. For multi-seed: mean +/- std across seeds. For comparisons: paired test (Wilcoxon signed-rank) with p-values. Box plots, not bar charts.
     8. **Cross-Experiment Comparison** — Table comparing this experiment to ALL prior experiments on the same standardized metrics. Same format every time.
     9. **Conclusions, Limitations, and Next Steps** — What worked, what didn't, what this motivates, honest limitations
     10. **Artifacts** — Table of file paths (checkpoints, configs, predictions, figures)
   - **Figure captions are mandatory.** Every figure must have: (a) what it shows, (b) what the reader should observe, (c) what it means clinically, (d) how it compares to prior results.

7. **Update the experiment index** (`notebooks/EXPERIMENTS_INDEX.md`):
   - This is the **single source of truth** for all experiments ever run
   - Add a row with standardized columns (see EXPERIMENTS_INDEX format below)
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
- **Data split** used (**actual case IDs** for train/val/test — not just counts)
- **Preprocessing version** and reference to `batch_summary.json` (data provenance)
- **All hyperparameters** (saved automatically to `training_config.json`)
- **Conda environment** (`conda list --export > environment_snapshot.txt`, saved in run directory)

If any of these are missing, the experiment is not reproducible and cannot be cited in a publication.

### Medical Physics Figure Set (Required for Training Experiments)

Every training experiment must include these standard figures. Use the standard plot configuration for all figures.

| # | Figure | What It Shows | Why It Matters |
|---|--------|--------------|----------------|
| 1 | **Training curves** | Loss and val MAE vs epoch | Training convergence and stability |
| 2 | **Dose colorwash** | Predicted vs ground truth dose overlaid on CT (axial, coronal, sagittal) for representative case | Visual dose accuracy — what reviewers look at first |
| 3 | **Dose difference map** | (Predicted - GT) overlaid on CT, with colorbar in Gy | Spatial pattern of errors |
| 4 | **DVH comparison** | Predicted vs GT DVH curves for all structures, one representative case | Clinical metric accuracy — what clinicians evaluate |
| 5 | **Gamma map** | 3%/3mm gamma index map overlaid on CT, with pass/fail coloring | Spatial distribution of pass/fail |
| 6 | **Per-case box plots** | Box plot of MAE, Gamma, D95 error across all test cases | Distribution, not just means — shows outliers |
| 7 | **Cross-experiment comparison** | Grouped bar or table comparing this experiment to all prior on MAE, Gamma, PTV D95 | Context — is this better or worse? |

For multi-seed experiments, add:
| 8 | **Seed variability** | Error bars or violin plots across seeds | Reproducibility of results |

Optional but recommended:
| 9 | **Dose profile** | 1D dose along a line through PTV center (predicted vs GT) | Penumbra accuracy, gradient realism |
| 10 | **Loss component breakdown** | Individual loss terms vs epoch (for multi-component losses) | Which loss dominates, convergence balance |

### Publication Figure Standards

All figures use a single standard configuration defined once and imported everywhere:

```python
# Standard plot configuration — use in ALL figure scripts and notebooks
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'savefig.pad_inches': 0.05,
}

# Standard colorblind-friendly palette (Wong 2011, Nature Methods)
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
}

# Standard experiment comparison order
COLOR_ORDER = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']
```

| Requirement | Standard |
|-------------|----------|
| Resolution | 300 DPI minimum |
| Font | Serif family, 12pt minimum (see config above) |
| Format | Both PNG (raster) and PDF (vector) |
| Colors | Wong 2011 colorblind-friendly palette (see above) |
| Axes | Labeled with units (e.g., "MAE (Gy)", "Epoch") |
| Legends | Present and readable |
| Captions | Required — what it shows, what to observe, clinical meaning, comparison to prior |
| Error representation | Box plots (not bar charts) for distributions. 95% CI for summary statistics. |

### EXPERIMENTS_INDEX.md Format

The experiment index uses standardized metric columns for at-a-glance comparison:

```markdown
| Date | ID | Git Hash | Notebook | Model | MAE (Gy) | Gamma 3%/3mm | PTV70 D95 Gap | Status |
```

- **MAE**: Test set mean +/- std (Gy)
- **Gamma**: Test set mean +/- std (%)
- **PTV70 D95 Gap**: Mean predicted D95 minus GT D95 (Gy). Negative = underdose.
- **Status**: `Pilot (v2.2.0)` for old data, `Complete` for v2.3+ data

### Experiment Output Structure

```
runs/<exp_name>/
├── checkpoints/              # Model checkpoints (best + last)
├── figures/                  # Publication-ready figures (PNG + PDF)
├── version_*/                # PyTorch Lightning logs
├── training_config.json      # All hyperparameters
├── training_summary.json     # Final metrics
├── metrics.csv               # Per-epoch metrics
└── environment_snapshot.txt  # conda list --export

predictions/<exp_name>_test/
├── case_XXXX_pred.npz        # Per-case predictions
└── evaluation_results.json   # Aggregate test metrics
```

### What "Publishable" Means

Every experiment notebook should be ready to drop into a journal supplementary section as-is. This means:
- A reader unfamiliar with the project can understand what was done, why, and what changed from prior work
- All results are quantified with proper metrics and statistics, not just "it looks better"
- Comparisons to prior experiments use consistent metrics, the same standardized table format, and statistical tests
- Limitations and failure modes are honestly documented
- Figures tell a clear story without requiring external explanation
- No placeholders, no `None` values, no `[UPDATE]` markers remain

## Current Project Status

**The single authoritative project plan is: `.claude/instructions.md`**
It contains the phased roadmap (Phases 0-3 + Parking Lot), strategic direction, current state, decisions log, and all planning content. Do NOT create separate plan files — all planning goes there.

**For experiment history: see `notebooks/EXPERIMENTS_INDEX.md`** (the master experiment log).

**For individual tasks: see [GitHub Issues](https://github.com/wrockey/vmat-diffusion/issues)** with phase labels and milestones.

## GitHub Issue Workflow

### Labels

| Label | Purpose |
|-------|---------|
| `phase/0-setup` | Phase 0: Work machine setup, data collection, pipeline fixes |
| `phase/1-eval` | Phase 1: Clinical evaluation framework |
| `phase/2-combined` | Phase 2: Combined loss experiment |
| `phase/3-iterate` | Phase 3: Iteration, publication prep |
| `type/experiment` | A specific experiment to design, run, or analyze |
| `type/decision` | Decision branch point with rationale |
| `type/backburner` | Revisit later — not blocking current work |
| `type/if-stuck` | Alternative approach to try if current path plateaus |
| `type/pipeline` | Data preprocessing or evaluation pipeline work |
| `type/publication` | Publication-related task |
| `priority/critical` | Blocking progress |
| `priority/high` | Important but not immediately blocking |
| `priority/low` | Nice to have |
| `status/blocked` | Waiting on external dependency |
| `status/needs-data` | Cannot proceed until dataset is available |

### Milestones

Four milestones track phase-level progress: `Phase 0: Setup`, `Phase 1: Evaluation Framework`, `Phase 2: Combined Loss`, `Phase 3: Iterate & Publish`.

### Workflow conventions

- **Before starting work:** Check open issues for the current phase
- **When starting a task:** Assign yourself to the issue
- **When work completes:** Close the issue with a commit reference (`Closes #N`)
- **When new tasks emerge:** Create an issue with appropriate phase + type labels
- **Decision points:** Create a `type/decision` issue documenting alternatives considered, rationale, and status (ACCEPTED/REJECTED/PENDING)
- **Backburner ideas:** Create a `type/backburner` issue — these are the parking lot
- **Setup script:** `scripts/setup_github_project.sh` creates all labels, milestones, and initial issues (run once with `gh` authenticated)

## Important Notes

- **No CI/CD pipeline** — experiments are tracked manually via git + notebooks + GitHub Issues
- **No linter/formatter configured** — code follows PEP8 informally
- **No pytest tests** — validation is through medical physics metrics
- **Documentation hierarchy:**
  - `.claude/instructions.md` — **THE PLAN:** living project state, strategy, phased roadmap overview, decisions summary. Updated every session.
  - `CLAUDE.md` — static reference (this file: conventions, architecture, experiment protocol, GitHub workflow). Rarely updated.
  - `notebooks/EXPERIMENTS_INDEX.md` — master experiment log. Updated after every experiment.
  - **GitHub Issues** — individual tasks, bugs, backburner ideas, decision records. Updated as work progresses.
  - **GitHub Milestones** — phase-level progress. Updated when issues are closed.
- **No separate plan files.** If a sub-plan is needed, it must be referenced from `.claude/instructions.md`. Currently one archived sub-plan exists: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED).
- **DataLoader:** Use `num_workers=2`, `persistent_workers=False` to avoid deadlocks (especially on WSL)
- **OAR name mapping:** `oar_mapping.json` maps 100+ clinical naming variations to 8 canonical structures
