# VMAT Diffusion

Loss-function engineering for clinically acceptable prostate VMAT dose prediction using deep learning.

## Overview

This project predicts 3D radiation dose distributions for prostate cancer VMAT (Volumetric Modulated Arc Therapy) treatment planning. Given a patient's CT scan, organ contours, and clinical dose constraints, a 3D U-Net predicts the full dose volume.

**Disease site:** Prostate cancer with SIB (70 Gy PTV70 / 56 Gy PTV56 in 28 fractions)

**Research focus:** Systematic evaluation of loss functions (gradient, DVH-aware, structure-weighted, asymmetric PTV) optimized for clinical acceptability rather than global pixel-wise accuracy.

**Primary model:** BaselineUNet3D (~23.7M parameters) with constraint conditioning via FiLM embedding.

## Key Results (Pilot Study, n=23)

| Loss Function | Val MAE (Gy) | Test MAE (Gy) | Gamma 3%/3mm | Key Strength |
|---------------|-------------|--------------|-------------|-------------|
| Baseline U-Net | 3.73 | 1.43 | 14.2% | Starting point |
| + Gradient Loss | 3.67 | 1.44 | 27.9% | Nearly doubled Gamma |
| + DVH-Aware | 3.61 | **0.95** | 27.7% | Best test MAE |
| + Structure-Weighted | **2.91** | 1.40 | **31.2%** | Best Gamma |
| + Asymmetric PTV | 3.36 | 1.89 | -- | Best D95 coverage |

Pilot validated methodology on 23 cases. Production training on 100+ cases is the next phase.

## Quick Start

```bash
# Environment setup
conda env create -f environment.yml
conda activate vmat-diffusion

# Preprocess DICOM-RT data
python scripts/preprocess_dicom_rt_v2.2.py --skip_plots

# Train baseline U-Net
python scripts/train_baseline_unet.py \
    --data_dir /path/to/processed_npz \
    --exp_name my_experiment \
    --epochs 200

# Run inference + evaluation
python scripts/inference_baseline_unet.py \
    --checkpoint runs/<exp>/checkpoints/best-*.ckpt \
    --input_dir /path/to/processed_npz
```

## Data Pipeline

- **Input:** CT (1 ch) + Signed Distance Fields for 8 structures (8 ch) = 9 input channels
- **Output:** 3D dose distribution (1 channel)
- **Conditioning:** 13-dimensional clinical constraint vector
- **Patch size:** 128x128x128 voxels (sliding window for full-volume inference)

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
│   ├── train_baseline_unet.py      # Primary model trainer
│   ├── inference_baseline_unet.py  # Inference + evaluation
│   ├── preprocess_dicom_rt_v2.2.py # DICOM-RT to NPZ
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
| `.claude/instructions.md` | Project plan, roadmap, decisions log |
| `CLAUDE.md` | Code conventions, architecture, experiment protocol |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log |

## Requirements

- Python 3.10+
- PyTorch 2.4+ with CUDA 12.4
- NVIDIA GPU with 16+ GB VRAM (24 GB recommended)
- See `environment.yml` for full dependency list

## License

MIT License. For research purposes; clinical deployment requires regulatory validation.

## Acknowledgments

- Anthropic Claude for development assistance
- PyTorch Lightning for training infrastructure
- pymedphys for gamma evaluation and DVH analysis
