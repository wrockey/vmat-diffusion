# VMAT Diffusion

Loss-function engineering for clinically acceptable prostate VMAT dose prediction using deep learning.

## Overview

This project predicts 3D radiation dose distributions for prostate cancer VMAT (Volumetric Modulated Arc Therapy) treatment planning. Given a patient's CT scan, organ contours, and clinical dose constraints, a 3D U-Net predicts the full dose volume.

**Disease site:** Prostate cancer with SIB (70 Gy PTV70 / 56 Gy PTV56 in 28 fractions)
**Current dataset:** 74 cases (v2.3 preprocessing), expecting ~161 from 2 institutions
**Research focus:** Systematic evaluation of loss functions (gradient, DVH-aware, structure-weighted, asymmetric PTV) with learned uncertainty weighting (Kendall 2018), optimized for clinical acceptability rather than global pixel-wise accuracy.

**Primary model:** BaselineUNet3D (~23.7M parameters) with constraint conditioning via FiLM embedding.

## Key Results

### Baseline (v2.3 data, 3-seed aggregate, n=74 train/val, 7 test per seed)

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 4.22 ± 0.53 Gy | Mean of 3 seed means |
| Gamma 3%/3mm (global) | 33.8 ± 4.6% | Low due to valid low-dose diversity |
| Gamma 3%/3mm (PTV-region) | 80.2 ± 5.3% | Clinically relevant region |
| PTV70 D95 Gap | -1.76 ± 0.69 Gy | Systematic underdosing — target for Phase 2 |

Seeds: 42, 123, 456. MSE-only loss. This establishes the baseline for the 16-condition loss ablation study.

### Loss Components (implemented, Phase 2 combined experiment pending)

| Component | Purpose |
|-----------|---------|
| MSE (baseline) | Voxel-wise accuracy |
| Gradient Loss (3D Sobel) | Dose edge and penumbra realism |
| DVH-Aware Loss | Clinical metric optimization (D95, Vx, Dmean) |
| Structure-Weighted Loss | Prioritize PTV and OAR boundary regions |
| Asymmetric PTV Loss | Penalize underdosing 3x more than overdosing |
| Uncertainty Weighting | Learned per-component σ (Kendall 2018) |

## Quick Start

```bash
# Environment setup
conda env create -f environment.yml
conda activate vmat-diffusion

# Preprocess DICOM-RT data (v2.3: crop, B-spline dose interpolation)
python scripts/preprocess_dicom_rt_v2.3.py --skip_plots

# Train baseline U-Net
python scripts/train_baseline_unet.py \
    --data_dir /path/to/processed_npz \
    --exp_name my_experiment \
    --epochs 200

# Train with combined loss + uncertainty weighting (Phase 2)
python scripts/train_baseline_unet.py \
    --data_dir /path/to/processed_npz \
    --exp_name combined_loss \
    --use_gradient_loss --use_dvh_loss \
    --use_structure_weighted --use_asymmetric_ptv \
    --use_uncertainty_weighting \
    --calibration_json /path/to/loss_normalization_calib.json \
    --epochs 200

# Run inference + evaluation
python scripts/inference_baseline_unet.py \
    --checkpoint runs/<exp>/checkpoints/best-*.ckpt \
    --input_dir /path/to/test_npz \
    --compute_metrics --overlap 64 --gamma_subsample 4
```

## Data Pipeline

- **Input:** CT (1 ch) + Signed Distance Fields for 8 structures (8 ch) = 9 input channels
- **Output:** 3D dose distribution (1 channel)
- **Conditioning:** 13-dimensional clinical constraint vector
- **Preprocessing (v2.3):** Native resolution crop (~300×300×Z), B-spline dose interpolation
- **Patch size:** 128³ voxels (training), sliding window with overlap=64 (inference)

### 8 Anatomical Structures

| Channel | Structure | Role |
|---------|-----------|------|
| 0 | PTV70 | High-dose target (70 Gy) |
| 1 | PTV56 | Intermediate target (56 Gy) |
| 2 | Prostate | Clinical target volume |
| 3-4 | Rectum, Bladder | Organs at risk |
| 5-6 | Femur L/R | Organs at risk |
| 7 | Bowel | Organ at risk |

## Repository Structure

```
vmat-diffusion/
├── scripts/                        # Training, inference, preprocessing, figures
│   ├── train_baseline_unet.py      # Primary model trainer (all loss components)
│   ├── inference_baseline_unet.py  # Inference + evaluation
│   ├── preprocess_dicom_rt_v2.3.py # DICOM-RT to NPZ (v2.3 crop pipeline)
│   ├── uncertainty_loss.py         # UncertaintyWeightedLoss (Kendall 2018)
│   ├── calibrate_loss_normalization.py # Loss calibration for initial sigmas
│   └── generate_*_figures.py       # Publication figure scripts
├── notebooks/                      # Experiment notebooks (dated)
│   ├── EXPERIMENTS_INDEX.md        # Master experiment log
│   └── TEMPLATE_experiment.ipynb   # Template for new experiments
├── docs/                           # Reference documentation
├── .claude/instructions.md         # Project plan and roadmap
├── CLAUDE.md                       # Code conventions and protocols
├── environment.yml                 # Conda environment
└── oar_mapping.json                # Structure name mapping
```

Training outputs (`runs/`), predictions (`predictions/`), and data (`processed/`, `data/`) are gitignored.

## Documentation

| Document | Purpose |
|----------|---------|
| `.claude/instructions.md` | Project plan, roadmap, pre-registered analysis plan |
| `CLAUDE.md` | Code conventions, architecture, experiment protocol |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log |
| [GitHub Issues](https://github.com/wrockey/vmat-diffusion/issues) | Task tracking |

## Requirements

- Python 3.10+
- PyTorch 2.4+ with CUDA 12.4+
- NVIDIA GPU with 16+ GB VRAM (24 GB recommended)
- See `environment.yml` for full dependency list

## License

MIT License. For research purposes; clinical deployment requires regulatory validation.

## Acknowledgments

- Anthropic Claude for development assistance
- PyTorch Lightning for training infrastructure
- pymedphys for gamma evaluation and DVH analysis
