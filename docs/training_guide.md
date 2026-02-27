> **PARTIALLY SUPERSEDED (2026-02-17)** — The DDPM-specific sections (architecture, diffusion process,
> DDIM sampling) are historical — the primary model is now the baseline U-Net. However, the general
> training infrastructure sections (patch-based training, augmentation, troubleshooting, GPU stability,
> monitoring, reproducibility) are still applicable. For current project state: see `.claude/instructions.md`.

# VMAT Dose DDPM Training Guide

## Overview

This document describes the training pipeline for the VMAT dose prediction diffusion model (Phase 1). The model learns to predict 3D dose distributions from CT scans and anatomical structures using a conditional Denoising Diffusion Probabilistic Model (DDPM).

**Script:** `train_dose_ddpm_v2.py`  
**Version:** 2.2.0  
**Compatible with:** `preprocess_dicom_rt_v2.3.py` output

---

## Quick Start

```bash
# Basic training (3090, 24GB)
python train_dose_ddpm_v2.py \
    --data_dir ./processed_npz \
    --epochs 200 \
    --batch_size 2

# Monitor training
tensorboard --logdir ./runs

# Resume interrupted training
python train_dose_ddpm_v2.py \
    --data_dir ./processed_npz \
    --resume ./runs/vmat_dose_ddpm/checkpoints/last.ckpt
```

---

## Requirements

### Hardware

| GPU | batch_size | patch_size | Memory Used | Time (200 epochs, 25 cases) |
|-----|------------|------------|-------------|----------------------------|
| RTX 3090 (24GB) | 2 | 128 | ~8-10 GB | ~5 hours |
| RTX 4090 (24GB) | 4 | 128 | ~12-14 GB | ~3 hours |
| A100 (40GB) | 8 | 160 | ~25-30 GB | ~2 hours |

### Software

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision
pip install pytorch-lightning>=2.0.0
pip install numpy scipy
pip install tensorboard rich

# Optional (for gamma evaluation)
pip install pymedphys
```

### Data

Input: Preprocessed `.npz` files from `preprocess_dicom_rt_v2.3.py`

Required keys in each `.npz`:
- `ct`: (512, 512, 256) float32
- `dose`: (512, 512, 256) float32
- `masks_sdf`: (8, 512, 512, 256) float32
- `constraints`: (13,) float32
- `metadata`: dict

---

## Architecture

### Model: SimpleUNet3D

```
Input: [noisy_dose (1) + CT (1) + SDFs (8)] = 10 channels
                        ↓
    ┌─────────────────────────────────────┐
    │         Encoder (4 levels)          │
    │   32 → 64 → 128 → 256 channels      │
    │   MaxPool3d (2×2×2) between levels  │
    └─────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────┐
    │            Bottleneck               │
    │         256 channels                │
    │   + FiLM conditioning               │
    └─────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────┐
    │         Decoder (4 levels)          │
    │   256 → 128 → 64 → 32 channels      │
    │   Trilinear upsampling (2×)         │
    │   Skip connections from encoder     │
    └─────────────────────────────────────┘
                        ↓
Output: Predicted noise (1 channel)
```

### Model Statistics

| Property | Value |
|----------|-------|
| Parameters | ~2.8 million |
| Base channels | 32 |
| Encoder levels | 4 |
| Activation | SiLU |
| Normalization | GroupNorm (8 groups) |

### Conditioning

**Spatial Conditioning (concatenated to input):**

| Input | Channels | Description |
|-------|----------|-------------|
| Noisy dose | 1 | Dose with added Gaussian noise at timestep t |
| CT | 1 | Normalized CT [0, 1] |
| SDFs | 8 | Signed distance fields for 8 structures |
| **Total** | **10** | Concatenated spatial input |

**Non-Spatial Conditioning (via FiLM):**

| Input | Dimension | Description |
|-------|-----------|-------------|
| Timestep | 128 | Sinusoidal embedding → MLP |
| Constraints | 128 | Linear → MLP |
| **Combined** | **256** | Concatenated, applied as scale/shift |

**FiLM (Feature-wise Linear Modulation):**
```python
# Applied at each conv block
scale, shift = mlp(concat(time_emb, constraint_emb)).chunk(2)
features = features * (1 + scale) + shift
```

### Diffusion Process

| Parameter | Value | Notes |
|-----------|-------|-------|
| Timesteps | 1000 | Number of diffusion steps |
| Schedule | Cosine | From Nichol & Dhariwal 2021 |
| Prediction target | ε (noise) | Model predicts added noise |
| Training sampling | Uniform | t ~ U(0, 1000) |
| Inference sampling | DDIM | 50 steps (20× faster than DDPM) |

---

## Training Process

### Patch-Based Training

Full 512×512×256 volumes don't fit in GPU memory. Solution:

```
Full volume:  512 × 512 × 256 = 67 million voxels
Patch:        128 × 128 × 128 = 2 million voxels
Reduction:    33× fewer voxels per forward pass
```

**Patch Sampling Strategy:**
- 4 patches extracted per volume per epoch
- 50% centered on high-dose regions (dose > 10% Rx)
- 50% random locations (for diversity)
- Different patches each epoch (not deterministic)

### Data Augmentation

Applied only during training:

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| Flip X | 50% | Left-right flip (with femur channel swap) |
| Translation | 50% | Random shift ±16 voxels in each axis |
| ~~Flip Y~~ | ~~50%~~ | ~~Anterior-posterior flip~~ — **Removed: beam entry matters** |
| ~~Flip Z~~ | ~~50%~~ | ~~Superior-inferior flip~~ — **Removed: anatomy not symmetric** |
| ~~Rotate 90°~~ | ~~75%~~ | ~~Axial rotation~~ — **Removed: beam angles are meaningful** |

**Why only X-flip?** Prostate anatomy has bilateral symmetry (left-right), but:
- Anterior-posterior flip changes beam entry direction
- Superior-inferior flip creates anatomically impossible configurations  
- Rotations change the relationship between beam angles and anatomy

**Why translation?** 
- Simulates inter-fraction patient positioning variation (±16-32mm)
- Preserves spatial relationships (CT, dose, SDFs shifted together)
- Uses nearest-edge filling to avoid wrap-around artifacts

**Effective diversity:** 2× (flip) × ~33³ (translation positions) = substantial variation

**Note:** When X-flip is applied, Femur_L and Femur_R SDF channels are swapped to maintain anatomical correctness.

### Training Loop (One Epoch)

```
For each training case:
    Load .npz file (~2-3 sec from SSD)
    
    For each of 4 patches:
        1. Sample patch center (dose-biased or random)
        2. Extract 128³ patch from CT, SDFs, dose
        3. Apply augmentation (X-flip only, 50%)
        4. Sample random timestep t ~ U(0, 1000)
        5. Add noise: x_t = √ᾱ_t × dose + √(1-ᾱ_t) × ε
        6. Concatenate [x_t, CT, SDFs] → 10 channels
        7. Forward pass: ε_pred = model(x_input, t, constraints)
        8. Compute loss: MSE(ε_pred, ε) + 0.1 × negative_dose_penalty
        9. Backward pass and optimizer step

Validation (after each epoch):
    1. Compute validation loss on val patches
    2. Run DDIM sampling on first val batch (50 steps)
    3. Compute MAE in Gy
    4. Compute gamma pass rate (subsampled 4×)
    5. Log metrics to TensorBoard
```

### Memory Usage (RTX 3090, batch_size=2, patch_size=128)

| Component | Memory |
|-----------|--------|
| Input tensor (10 × 128³) | ~80 MB |
| Model weights | ~50 MB |
| Gradients + optimizer states | ~100 MB |
| Activations (mixed precision) | ~6-8 GB |
| **Total** | **~8-10 GB** |

Headroom on 24GB: ~14-16 GB available for safety.

---

## Command Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--data_dir` | Path to directory containing `.npz` files |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Maximum training epochs |
| `--batch_size` | 2 | Batch size (reduce if OOM) |
| `--patch_size` | 128 | Patch size in voxels |
| `--patches_per_volume` | 4 | Patches sampled per case per epoch |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-2 | AdamW weight decay |
| `--val_split` | 0.1 | Fraction of data for validation |
| `--test_split` | 0.1 | Fraction of data for held-out test set |

### Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 1000 | Diffusion timesteps |
| `--schedule` | cosine | Noise schedule (cosine/linear) |
| `--base_channels` | 48 | Base filter count (reduce to 32 if OOM) |
| `--use_simple_unet` | True | Use memory-efficient U-Net |

### Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--log_dir` | ./runs | Output directory |
| `--exp_name` | vmat_dose_ddpm | Experiment name |

### Misc

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | 42 | Random seed for reproducibility |
| `--num_workers` | 2 | DataLoader workers (keep low on WSL2, use 0 on native Windows) |
| `--resume` | None | Checkpoint path to resume from |
| `--rx_dose_gy` | 70.0 | Prescription dose (Gy) |
| `--devices` | 1 | Number of GPUs |
| `--strategy` | auto | Distributed strategy (ddp for multi-GPU) |

### GPU Stability

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu_cooling` | False | Enable pauses between batches to prevent overheating |
| `--cooling_interval` | 10 | Pause every N batches (if --gpu_cooling enabled) |
| `--cooling_pause` | 0.5 | Pause duration in seconds (if --gpu_cooling enabled) |

---

## Output Structure

```
runs/vmat_dose_ddpm/
├── checkpoints/
│   ├── best-epoch=134-val_mae_gy=1.82.ckpt   # Best model
│   ├── best-epoch=098-val_mae_gy=1.95.ckpt   # 2nd best
│   ├── best-epoch=112-val_mae_gy=1.91.ckpt   # 3rd best
│   └── last.ckpt                              # Most recent
├── training_config.json      # Full configuration
├── training_summary.json     # Results summary
├── epoch_metrics.csv         # Per-epoch metrics
├── version_0/
│   ├── events.out.tfevents.* # TensorBoard logs
│   ├── hparams.yaml          # Hyperparameters
│   └── metrics.csv           # CSVLogger output
└── version_1/                # Created if run again
```

### training_config.json

Saved at training start. Contains everything needed for reproducibility:

```json
{
  "script_version": "2.2.0",
  "timestamp_start": "2025-01-10T14:30:00",
  "pytorch_version": "2.1.0",
  "cuda_version": "12.1",
  "python_version": "3.10.12",
  "platform": "Linux-5.15.0-x86_64",
  
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "gpu_count": 1,
  "gpu_memory_gb": 24.0,
  
  "model_class": "SimpleUNet3D",
  "total_parameters": 12500000,
  "trainable_parameters": 12500000,
  
  "hyperparameters": {
    "in_channels": 10,
    "timesteps": 1000,
    "schedule": "cosine",
    "lr": 0.0001,
    "weight_decay": 0.01,
    "base_channels": 48,
    "rx_dose_gy": 70.0
  },
  
  "train_cases": 20,
  "val_cases": 3,
  "test_cases": 2,
  "train_case_ids": ["case_0001", "case_0003", ...],
  "val_case_ids": ["case_0002", "case_0015", "case_0021"],
  "test_case_ids": ["case_0010", "case_0025"],
  
  "max_epochs": 200,
  "precision": "16-mixed",
  "accumulate_grad_batches": 1,
  "gradient_clip_val": 1.0
}
```

### training_summary.json

Saved at training end:

```json
{
  "timestamp_end": "2025-01-09T19:45:00",
  "total_training_time_hours": 5.25,
  "total_epochs_completed": 156,
  "avg_epoch_time_sec": 92.3,
  
  "best_val_mae_gy": 1.82,
  "best_epoch": 134,
  "converged_epoch": 98,
  
  "final_train_loss": 0.0182,
  "final_val_loss": 0.0245,
  "final_val_mae_gy": 1.89,
  
  "early_stopped": true
}
```

### epoch_metrics.csv

For plotting learning curves:

```csv
epoch,epoch_time_sec,train_loss,val_loss,val_mae_gy,learning_rate,val_gamma
0,95.2,0.1523,0.1102,8.45,0.0001,
1,93.8,0.0842,0.0756,6.23,0.0001,
2,94.1,0.0621,0.0598,5.12,0.0001,
...
134,91.5,0.0178,0.0231,1.82,0.000024,96.2
```

---

## Inference

### Sliding Window Full-Volume Prediction

The model trains on 128³ patches but must predict full 512×512×256 volumes:

```python
pred_dose = model.predict_full_volume(
    condition=condition,      # (1, 9, H, W, D)
    constraints=constraints,  # (1, 13)
    patch_size=128,          # Process in 128³ chunks
    overlap=32,              # Overlap between patches
    ddim_steps=50,           # DDIM sampling steps
)
```

**How it works:**

1. Divide volume into overlapping 128³ patches (stride = 96)
2. For each patch: run DDIM denoising (50 steps)
3. Apply Gaussian weighting kernel to each patch
4. Accumulate weighted predictions
5. Normalize by total weights → seamless blending

**Patch grid for 512×512×256 volume:**
- 5 × 5 × 2 = 50 patches
- ~5-10 minutes on 3090

### Standalone Inference Script

```bash
# Single case
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input ./processed_npz/case_0001.npz \
    --output ./predictions/case_0001_pred.npz

# Batch with metrics
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions \
    --compute_metrics
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./runs
# Open http://localhost:6006
```

**Key metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| `train/loss` | Training MSE loss | Decreasing |
| `val/loss` | Validation MSE loss | Decreasing, close to train |
| `val/mae_gy` | Mean absolute error (Gy) | < 2.0 Gy |
| `val/gamma_3mm3pct` | Gamma pass rate (%) | > 95% |
| `misc/gpu_memory_gb` | GPU memory usage | < 20 GB |
| `misc/eta_hours` | Estimated time remaining | - |
| `lr-AdamW` | Current learning rate | Decreasing (cosine) |

### Important: Training vs. Inference Gamma

**Training gamma** is a fast proxy computed on a single 128³ patch:

| Stage | Volume | Subsampling | Grid Evaluated | Purpose |
|-------|--------|-------------|----------------|---------|
| Training | 128³ patch | 4× | 32×32×32 | Quick feedback, ~1 sec |
| **Inference** | **512×512×256 full** | **2× (default)** | **256×256×128** | **Final evaluation** |

**Why the difference?**
- Training: Speed matters (every epoch). Patch-based gamma is a reasonable proxy.
- Inference: Accuracy matters. Full-volume gamma is the proper clinical metric.

**Inference gamma command:**
```bash
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions \
    --gamma_subsample 2  # Default; use 1 for full resolution (slower)
```

**Do not** use training gamma values for publication. Always report inference gamma on the held-out test set.

### Expected Learning Curve

```
Epoch   train/loss   val/mae_gy   Status
─────────────────────────────────────────
  1       0.15         8.5        Starting
 10       0.08         5.5        Learning basic features
 50       0.04         3.2        Learning dose gradients
100       0.02         2.1        Approaching target
150       0.018        1.8        Target met ✓
200       0.017        1.7        Converged
```

### Warning Signs

| Symptom | Cause | Solution |
|---------|-------|----------|
| `val/loss` increases, `train/loss` decreases | Overfitting | More augmentation, early stopping, reduce model size |
| Loss becomes NaN | Gradient explosion | Reduce learning rate, check data |
| Loss stuck at high value | Learning rate too low | Increase learning rate |
| `val/mae_gy` stuck > 5 Gy | Model not learning | Check data loading, increase model capacity |
| OOM error | Memory exceeded | Reduce batch_size or patch_size |
| Training very slow | I/O bottleneck | Use local SSD (not `/mnt/`), keep num_workers=2 on WSL |
| Training hangs/stalls | Dataloader deadlock (WSL) | Use num_workers=2, disable persistent_workers |
| WSL crashes / GPU errors | Memory pressure | Don't cache data in RAM; restart WSL |
| System crash (0x113 / TDR) | GPU overheating or driver timeout | Use native Windows, enable --gpu_cooling |
| Windows becomes unresponsive | GPU at 100% sustained | Enable --gpu_cooling, reduce batch_size |

---

## Reproducibility

### Exact Reproduction

```bash
python train_dose_ddpm_v2.py \
    --data_dir ./processed_npz \
    --seed 42 \
    --epochs 200 \
    --batch_size 2 \
    --patch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --timesteps 1000 \
    --schedule cosine
```

Same seed → same train/val split → same initialization → same results (within floating point variance).

### Logged for Reproducibility

| Factor | Location | Field |
|--------|----------|-------|
| Random seed | training_config.json | CLI argument |
| Train/val split | training_config.json | train_case_ids, val_case_ids |
| PyTorch version | training_config.json | pytorch_version |
| CUDA version | training_config.json | cuda_version |
| GPU model | training_config.json | gpu_name |
| All hyperparameters | training_config.json | hyperparameters |
| Training time | training_summary.json | total_training_time_hours |
| Best results | training_summary.json | best_val_mae_gy, best_epoch |

---

## Native Windows / Pinokio Setup

If you experience stability issues with WSL2 (hangs, crashes, dxgkrnl errors), running natively on Windows can be more stable.

### Setup

1. Install Miniconda in Pinokio: `C:\pinokio\bin\miniconda`
2. Create the environment:
   ```cmd
   conda create -n vmat-win python=3.10
   conda activate vmat-win
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install pytorch-lightning numpy scipy tensorboard rich pymedphys
   ```

3. Copy project files to Windows:
   ```cmd
   mkdir C:\Users\<username>\vmat-diffusion-project
   copy scripts\train_dose_ddpm_v2.py C:\Users\<username>\vmat-diffusion-project\scripts\
   ```

### Safe Training Mode

For systems experiencing GPU overheating or driver crashes, use the safe training configuration:

```cmd
:: Run from C:\Users\<username>\vmat-diffusion-project
start_training_safe.bat
```

This batch file uses reduced settings:
- **Batch size: 1** (reduced from 2)
- **Base channels: 32** (reduced from 48, ~50% fewer parameters)
- **Workers: 0** (avoids Windows multiprocessing issues)
- **GPU cooling enabled** (0.5s pause every 10 batches + 2s between epochs)

### Monitor GPU Temperature

Use GPU-Z or HWiNFO64 to monitor:
- **Target temp:** < 80°C during training
- **Danger zone:** > 85°C (may trigger thermal throttling or crashes)

If temps are too high:
1. Increase cooling_pause (e.g., 1.0s instead of 0.5s)
2. Reduce cooling_interval (e.g., 5 batches instead of 10)
3. Check GPU fan curve / case airflow

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Option 1: Reduce batch size
python train_dose_ddpm_v2.py --batch_size 1

# Option 2: Reduce patch size
python train_dose_ddpm_v2.py --patch_size 96

# Option 3: Reduce model capacity
python train_dose_ddpm_v2.py --base_channels 24

# Aggressive memory saving
python train_dose_ddpm_v2.py \
    --batch_size 1 \
    --patch_size 96 \
    --base_channels 24
```

### Slow Training

```bash
# Increase batch size (if memory allows)
python train_dose_ddpm_v2.py --batch_size 4

# Use local SSD storage (CRITICAL for WSL!)
# Copy data from /mnt/ to local WSL filesystem:
cp -r /mnt/i/processed_npz ./data/processed_npz
ln -sf ./data/processed_npz ./processed

# NOTE: On WSL2, keep num_workers=2 (default) to avoid deadlocks
# Do NOT use num_workers=8 on WSL - it causes training to hang
```

### Resume After Crash

```bash
python train_dose_ddpm_v2.py \
    --data_dir ./processed_npz \
    --resume ./runs/vmat_dose_ddpm/checkpoints/last.ckpt
```

This restores:
- Model weights
- Optimizer state (momentum, etc.)
- Learning rate scheduler position
- Epoch counter
- Random state

### Multi-GPU Training

```bash
# 4 GPUs with DDP
python train_dose_ddpm_v2.py \
    --data_dir ./processed_npz \
    --devices 4 \
    --strategy ddp \
    --batch_size 8  # 2 per GPU
```

---

## Scaling to Larger Datasets

| Dataset Size | Patches/Epoch | Time/Epoch | 200 Epochs | Recommendation |
|--------------|---------------|------------|------------|----------------|
| 25 cases | 88 | ~1.5 min | ~5 hours | 3090 fine |
| 100 cases | 360 | ~6 min | ~20 hours | 3090 fine |
| 500 cases | 1,800 | ~30 min | ~4 days | Consider HPC |
| 750 cases | 2,700 | ~45 min | ~6 days | Use HPC |

For 500+ cases, consider:
- Multi-GPU training (4-8 GPUs)
- Pre-caching patches to disk
- Memory-mapped data loading

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | 2025-01-10 | Fixed augmentation (X-flip only), base_channels=48, test split, physics loss |
| 2.1.0 | 2025-01-09 | Publication logging, sliding window inference, bug fixes |
| 2.0.0 | 2025-01-09 | Patch-based training, SDF support, FiLM conditioning |
| 1.0.0 | 2025-01-08 | Initial implementation (replaced) |
