# CLAUDE.md

## Project Overview

VMAT Diffusion is a deep learning research project for automated **Volumetric Modulated Arc Therapy (VMAT) dose prediction** in radiation therapy. It uses diffusion models (DDPM) and baseline U-Net architectures to predict 3D dose distributions from patient CT scans, organ contours, and clinical dose constraints.

**Disease site:** Prostate cancer with SIB (70 Gy / 56 Gy in 28 fractions)
**Current dataset:** 24 cases (23 usable), expecting 100-150 near-term
**Clinical target:** Gamma pass rate > 95% (3%/3mm), MAE < 3 Gy

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

## Experiment Workflow

1. **Before running:** Commit all changes (`git commit -m "Pre-experiment: <name>"`)
2. **Record git hash** for reproducibility
3. **Run training** with appropriate CLI flags
4. **Run inference** on test set
5. **Generate figures:** Create `scripts/generate_<exp>_figures.py` (300 DPI, PNG + PDF)
6. **Create notebook:** `notebooks/YYYY-MM-DD_<exp_name>.ipynb` from template
7. **Update tracking:** Add entry to `notebooks/EXPERIMENTS_INDEX.md`
8. **Commit results:** `git commit -m "Results: <name> - <key metrics>"`

Publication figure standards: 300 DPI, serif font, 12pt minimum, colorblind-friendly palette.

## Current Project Status (as of 2026-01-23)

### Model Performance
| Model | Val MAE | Test MAE | Gamma (3%/3mm) | D95 Gap |
|-------|---------|----------|----------------|---------|
| Baseline U-Net | 3.73 Gy | 1.43 Gy | 14.2% | ~-20 Gy |
| Gradient Loss 0.1 | 3.67 Gy | 1.44 Gy | 27.9% | ~-7 Gy |
| DVH-Aware | 3.61 Gy | **0.95 Gy** | 27.7% | ~-7 Gy |
| Structure-Weighted | **2.91 Gy** | 1.40 Gy | **31.2%** | ~-7 Gy |
| Asymmetric PTV | 3.36 Gy | 1.89 Gy | — | **-5.95 Gy** |

### Key Findings
- **DDPM is NOT recommended** for this task — matches baseline but adds complexity
- **Gradient loss** nearly doubled Gamma (14.2% → 27.9%)
- **DVH-aware loss** achieved best test MAE (0.95 Gy)
- **Structure-weighted loss** achieved best Gamma (31.2%)
- **Asymmetric PTV loss** best D95 coverage (-5.95 Gy gap)
- Ground truth itself fails clinical D95 threshold by ~11.5 Gy
- VGG perceptual loss does NOT help Gamma — do not use

### What NOT to Pursue
- DDPM tuning (structural mismatch with deterministic dose prediction)
- VGG perceptual loss (no Gamma improvement)
- Pure MSE/MAE optimization (leads to PTV underdosing)

## Important Notes

- **No CI/CD pipeline** — experiments are tracked manually via git + notebooks
- **No linter/formatter configured** — code follows PEP8 informally
- **No pytest tests** — validation is through medical physics metrics
- **Experiment tracking:** `notebooks/EXPERIMENTS_INDEX.md` is the authoritative source
- **Detailed session state:** `.claude/instructions.md` has extensive project context, progress logs, and next steps
- **DataLoader:** Use `num_workers=2`, `persistent_workers=False` to avoid deadlocks (especially on WSL)
- **OAR name mapping:** `oar_mapping.json` maps 100+ clinical naming variations to 8 canonical structures
