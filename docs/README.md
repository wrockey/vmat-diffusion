> **SUPERSEDED (2026-02-17)** — This document is historical. It describes the original DDPM-focused project plan.
> The DDPM approach was abandoned in favor of a baseline U-Net with loss-function engineering.
> For current project state: see `.claude/instructions.md`. For code conventions: see `CLAUDE.md`.

# VMAT Dose Prediction using Conditional Diffusion Models

A deep learning pipeline for predicting 3D radiation dose distributions from CT scans and anatomical structures using Denoising Diffusion Probabilistic Models (DDPM).

## Project Overview

This project aims to predict clinically acceptable VMAT (Volumetric Modulated Arc Therapy) dose distributions for prostate cancer treatment planning. The pipeline consists of two main phases:

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | Predict dose from anatomy (CT + structures + constraints) | 🔧 Implementation Complete |
| **Phase 2** | Predict deliverable MLC sequences (planned) | 🔜 Future |

### Phase 1 Performance Targets (Not Yet Validated)

1. **MAE < 2 Gy** - Mean absolute error in dose prediction
2. **Gamma (3%/3mm) > 95%** - Clinical pass rate
3. **Training in < 24 hours** - On RTX 3090
4. **Feasibility demonstration** - Prove the approach works

---

## Quick Start

### 1. Preprocess DICOM-RT Data

```bash
# Auto-detects ./data/raw and ./processed
python preprocess_dicom_rt_v2.2.py --mapping_file ./oar_mapping.json

# Or specify paths explicitly
python preprocess_dicom_rt_v2.2.py \
    --input_dir ./data/raw \
    --output_dir ./processed \
    --mapping_file ./oar_mapping.json
```

### 2. Train the Model

```bash
# Auto-detects ./processed
python train_dose_ddpm_v2.py --epochs 200 --batch_size 2

# Or specify data path
python train_dose_ddpm_v2.py --data_dir ./processed --epochs 200
```

### 3. Run Inference

```bash
# Evaluate on test set (computes full-volume gamma)
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions \
    --compute_metrics

# Options for gamma computation:
#   --gamma_subsample 2  (default: 256×256×128 grid)
#   --gamma_subsample 1  (full resolution, slower but most accurate)
```

---

## Repository Structure

```
vmat-diffusion/
├── preprocess_dicom_rt_v2.2.py   # DICOM-RT preprocessing (Phase 2 ready)
├── train_dose_ddpm_v2.py         # DDPM training script
├── inference_dose_ddpm.py        # Inference and evaluation
├── train_baseline_unet.py        # Baseline model (for comparison)
├── inference_baseline_unet.py    # Baseline inference
├── verify_npz.ipynb              # Data verification notebook
│
├── processed_npz/                # Preprocessed .npz files
│   ├── case_0001.npz
│   ├── case_0002.npz
│   └── ...
│
├── runs/                         # Training outputs
│   ├── vmat_dose_ddpm/           # Diffusion model
│   │   ├── checkpoints/
│   │   └── ...
│   └── baseline_unet/            # Baseline model
│       ├── checkpoints/
│       └── ...
│
└── docs/
    ├── README.md                 # This file
    ├── CHANGELOG.md              # Version history
    ├── preprocessing_guide.md    # Preprocessing documentation
    ├── preprocessing_assumptions.md  # Data assumptions
    └── training_guide.md         # Training documentation
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | A100 (40GB+) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB SSD | 200 GB NVMe |
| CPU | 8 cores | 16+ cores |

### Training Time Estimates

| Dataset Size | 3090 (24GB) | A100 (40GB) |
|--------------|-------------|-------------|
| 25 cases | ~5 hours | ~2 hours |
| 100 cases | ~20 hours | ~8 hours |
| 500 cases | ~4 days | ~1.5 days |

---

## Software Requirements

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision
pip install pytorch-lightning>=2.0.0
pip install numpy scipy scikit-image
pip install pydicom rt-utils
pip install tensorboard rich tqdm

# Optional (for gamma evaluation)
pip install pymedphys

# Development
pip install jupyter matplotlib
```

---

## Data Format

### Input: DICOM-RT Files

Each case requires:
- **CT**: DICOM CT series
- **RTSTRUCT**: Structure set with contours
- **RTDOSE**: 3D dose distribution
- **RTPLAN**: Treatment plan (for MLC extraction)

### Output: .npz Files (v2.2)

```python
data = np.load('case.npz', allow_pickle=True)

# Core arrays
data['ct']           # (512, 512, 256) float32 - CT [0,1]
data['dose']         # (512, 512, 256) float32 - Dose normalized to Rx
data['masks']        # (8, 512, 512, 256) uint8 - Binary masks
data['masks_sdf']    # (8, 512, 512, 256) float32 - Signed distance fields
data['constraints']  # (13,) float32 - Planning constraints

# MLC data (Phase 2 ready)
data['beam0_mlc_a']  # (n_cp, 60) float32 - MLC bank A positions
data['beam0_mlc_b']  # (n_cp, 60) float32 - MLC bank B positions

# Metadata
data['metadata']     # dict - Case info, beam geometry, validation
```

### Structure Channels

| Channel | Structure | Description |
|---------|-----------|-------------|
| 0 | PTV70 | High-dose PTV (70 Gy) |
| 1 | PTV56 | Intermediate PTV (56 Gy) |
| 2 | Prostate | Clinical target |
| 3 | Rectum | OAR |
| 4 | Bladder | OAR |
| 5 | Femur_L | OAR |
| 6 | Femur_R | OAR |
| 7 | Bowel | OAR |

---

## Model Architecture

### SimpleUNet3D

```
Input: [noisy_dose (1) + CT (1) + SDFs (8)] = 10 channels
                        ↓
              Encoder (4 levels)
           32 → 64 → 128 → 256 ch
                        ↓
              Bottleneck (256 ch)
           + FiLM conditioning
                        ↓
              Decoder (4 levels)
           256 → 128 → 64 → 32 ch
                        ↓
Output: Predicted noise (1 channel)
```

### Key Features

- **Patch-based training**: 128³ patches from 512×512×256 volumes
- **FiLM conditioning**: Timestep and constraints via scale/shift
- **Cosine noise schedule**: Better than linear for images
- **DDIM sampling**: 50 steps for fast inference
- **Sliding window inference**: Gaussian-weighted overlap averaging

---

## Performance Targets

### Expected Performance (Not Yet Validated)

These are targets based on literature, not measured results:

| Metric | Target | Typical Result |
|--------|--------|----------------|
| MAE | < 2.0 Gy | 1.5-2.0 Gy |
| Gamma (3%/3mm) | > 95% | 92-98% |
| Max Error | < 10 Gy | 5-8 Gy |
| Training Time | < 24h | 5-6h (25 cases) |

**Note on Gamma Evaluation:**
- Training script computes gamma on small patches (fast proxy for monitoring)
- **Inference script computes gamma on full 512×512×256 volumes** (proper evaluation)
- Always report inference gamma on the held-out test set for publications

### Outputs for Publication

Training automatically generates:

1. **training_config.json** - Complete reproducibility info
2. **training_summary.json** - Results summary
3. **epoch_metrics.csv** - Learning curves data
4. **checkpoints/** - Model weights

---

## Baseline Comparison

A direct regression baseline model is provided to answer: **"Is diffusion actually helping?"**

### Train Both Models

```bash
# 1. Train diffusion model (auto-detects data path)
python train_dose_ddpm_v2.py --epochs 200

# 2. Train baseline (same data, same split via seed)
python train_baseline_unet.py --epochs 200
```

### Compare on Test Set

```bash
# Evaluate diffusion
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input_dir ./test_npz --output_dir ./predictions_ddpm --compute_metrics

# Evaluate baseline
python inference_baseline_unet.py \
    --checkpoint ./runs/baseline_unet/checkpoints/best.ckpt \
    --input_dir ./test_npz --output_dir ./predictions_baseline --compute_metrics
```

### Expected Differences

| Metric | Diffusion | Baseline | Notes |
|--------|-----------|----------|-------|
| MAE | Likely lower | Higher | Diffusion should win |
| Gamma | Likely higher | Lower | Diffusion captures detail |
| Inference time | ~5-10 min | ~10-30 sec | Baseline is ~50-100× faster |
| Uncertainty | Yes (sample multiple times) | No | Diffusion advantage |

If the baseline performs nearly as well as diffusion, the added complexity may not be justified.

---

## Documentation

| Document | Description |
|----------|-------------|
| [preprocessing_guide.md](preprocessing_guide.md) | Preprocessing pipeline details |
| [preprocessing_assumptions.md](preprocessing_assumptions.md) | Data assumptions and requirements |
| [training_guide.md](training_guide.md) | Training script documentation |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Reproducibility

### Exact Reproduction

```bash
# Same seed = same train/val split = same results
python train_dose_ddpm_v2.py \
    --seed 42 \
    --epochs 200 \
    --batch_size 2 \
    --patch_size 128 \
    --lr 1e-4
```

### Logged Information

All runs automatically log:
- Random seed and train/val case IDs
- PyTorch, CUDA, Python versions
- GPU model and memory
- All hyperparameters
- Per-epoch metrics
- Training time

---

## Troubleshooting

### Out of Memory

```bash
# Reduce memory usage
python train_dose_ddpm_v2.py \
    --batch_size 1 \
    --patch_size 96 \
    --base_channels 24
```

### Resume Training

```bash
python train_dose_ddpm_v2.py \
    --resume ./runs/vmat_dose_ddpm/checkpoints/last.ckpt
```

### Multi-GPU

```bash
python train_dose_ddpm_v2.py \
    --devices 4 \
    --strategy ddp
```

---

## Known Limitations

### Phase 1→2 Gap: Deliverability

Phase 1 predicts dose distributions **without guaranteeing deliverability**. The predicted dose may require MLC configurations that:

- Violate leaf travel speed constraints
- Exceed dose rate limits
- Are physically impossible to deliver

**Mitigation Strategy:**

1. **Sequential Validation**: Train Phase 1, evaluate quality, then proceed to Phase 2
2. **Measure the Gap**: Run Phase 2 on both ground-truth and predicted doses to quantify degradation
3. **Joint Fine-tuning** (if needed): Add consistency loss between predicted dose and dose recomputed from predicted MLC

Since training data comes from real delivered plans, the model should learn the distribution of deliverable doses. The gap becomes problematic only if the model generates out-of-distribution predictions.

---

## Citation

```bibtex
@software{vmat_dose_ddpm,
  title = {VMAT Dose Prediction using Conditional Diffusion Models},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[username]/vmat-diffusion}
}
```

---

## License

[Specify license]

---

## Acknowledgments

- Anthropic Claude for code development assistance
- PyTorch Lightning for training infrastructure
- pymedphys for gamma evaluation
