> **PARTIALLY SUPERSEDED (2026-02-17)** — The DDPM-specific assumptions (diffusion process, noise schedule,
> DDIM sampling) are historical. Data assumptions, augmentation assumptions, and hardware requirements
> are still applicable to the baseline U-Net. For current project state: see `.claude/instructions.md`.

# Training Assumptions and Requirements

This document lists all assumptions made by the training pipeline. Understanding these assumptions is critical for reproducing results and troubleshooting issues.

---

## Data Assumptions

### Input Format

The training script expects `.npz` files from `preprocess_dicom_rt_v2.3.py`:

| Key | Shape | Dtype | Range | Required |
|-----|-------|-------|-------|----------|
| `ct` | (512, 512, 256) | float32 | [0, 1] | Yes |
| `dose` | (512, 512, 256) | float32 | [0, ~1.1] | Yes |
| `masks_sdf` | (8, 512, 512, 256) | float32 | [-1, 1] | Yes |
| `constraints` | (13,) | float32 | Various | Yes |
| `metadata` | dict | - | - | Yes |
| `masks` | (8, 512, 512, 256) | uint8 | {0, 1} | Optional |

### Data Quality

- **No NaN or Inf values** in any array
- **Dose normalized to prescription** (PTV70 mean ≈ 1.0)
- **SDFs correctly computed** (negative inside, positive outside)
- **Structures registered to dose** (verified during preprocessing)

### Dataset Size

| Size | Training Cases | Validation Cases | Expected Outcome |
|------|----------------|------------------|------------------|
| Minimum | 20 | 3 | Pipeline validation only |
| Recommended | 90 | 10 | Feasibility demonstration |
| Optimal | 450+ | 50+ | Publication-quality results |

### Data Distribution

- All cases should be **prostate VMAT** (same treatment site)
- **Similar fractionation** (28 fx, 70/56 Gy SIB)
- **Single institution** preferred for initial training
- Validation cases should be **representative** of training distribution

---

## Model Assumptions

### Architecture

- **3D U-Net** with 4 encoder/decoder levels
- **Patch-based**: 128³ input, not full volume
- **FiLM conditioning** for timestep and constraints
- **Cosine noise schedule** (1000 timesteps)

### Input/Output

| Property | Value |
|----------|-------|
| Input channels | 10 (1 noisy dose + 1 CT + 8 SDFs) |
| Output channels | 1 (predicted noise) |
| Spatial size | 128 × 128 × 128 (patches) |
| Batch size | 2 (default for 3090) |

### Conditioning

**Assumptions about constraint vector:**
- Index 0: PTV70 exists (1.0) or not (0.0)
- Index 1: PTV56 dose level (0.8 = 56/70) or 0.0
- Index 2: PTV50.4 dose level (0.72) or 0.0
- Indices 3-12: DVH constraint values [0, ~1.1]

The model learns to use these constraints, but **does not enforce them**. Predicted doses may violate constraints.

---

## Training Assumptions

### Optimization

| Setting | Value | Assumption |
|---------|-------|------------|
| Optimizer | AdamW | Standard choice for transformers/diffusion |
| Learning rate | 1e-4 | Conservative start, with cosine decay |
| Weight decay | 1e-2 | Regularization for generalization |
| Gradient clipping | 1.0 | Prevents gradient explosion |
| Precision | 16-mixed | Assumes Tensor Cores available |

### Patch Sampling

**Assumption:** High-dose regions are most important

- 50% of patches centered on voxels with dose > 10% of Rx
- 50% random patches for background/low-dose learning
- Different patches sampled each epoch (not cached)

### Augmentation

**Assumption:** Prostate anatomy has bilateral (left-right) symmetry only

- Left-right (X) flips are valid augmentation ✓
- Femur SDF channels swapped on X-flip to maintain correctness ✓
- Random translation ±16 voxels is valid ✓ (simulates positioning variation)
- Anterior-posterior (Y) flips are **NOT** used — beam entry direction matters
- Superior-inferior (Z) flips are **NOT** used — anatomy not symmetric
- 90° rotations are **NOT** used — beam angles are meaningful

**Rationale:** Dose distributions have physical directionality from beam angles. Only left-right symmetry is valid for prostate due to bilateral anatomy. Translation simulates realistic inter-fraction positioning variation.

**Not used:**
- Elastic deformations (not implemented)
- Intensity augmentation (CT values are meaningful; dose was computed from original CT)
- Scaling augmentation (fixed anatomy size)

### Train/Validation/Test Split

- **File-level split** (not patch-level)
- **Random shuffle** with fixed seed for reproducibility
- **80/10/10 split** by default (train/val/test)
- Test set completely held out, saved to `test_cases.json`
- No patient-level grouping (assumes one .npz per patient)

---

## Diffusion Process Assumptions

### Forward Process

```python
x_t = sqrt(α_cumprod_t) * x_0 + sqrt(1 - α_cumprod_t) * ε
```

**Assumption:** Gaussian noise is appropriate for dose data

- Dose values are continuous
- Dose distributions are relatively smooth
- Adding Gaussian noise creates meaningful training signal

### Reverse Process

**Assumption:** Model can learn to predict noise at any timestep

- Training samples uniform timesteps t ~ U(0, 1000)
- Model should generalize across all noise levels
- Cosine schedule provides good SNR progression

### Sampling

**Assumption:** DDIM provides faithful reconstructions with fewer steps

- 50 DDIM steps ≈ 1000 DDPM steps in quality
- Deterministic sampling (eta=0) is sufficient
- No need for classifier-free guidance (conditioning is strong)

---

## Hardware Assumptions

### GPU

- **CUDA-capable GPU** with 16+ GB VRAM
- **Tensor Cores** available for mixed precision (Volta+)
- **Sufficient bandwidth** for 3D convolutions

### Memory

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB |
| System RAM | 32 GB | 64 GB |
| Storage speed | HDD | NVMe SSD |

### Multi-GPU

- **DDP (Distributed Data Parallel)** strategy
- All GPUs should be identical
- Linear speedup assumed (not always achieved)

---

## Inference Assumptions

### Sliding Window

**Assumption:** Patch-based predictions can be seamlessly combined

- Gaussian weighting eliminates seam artifacts
- Overlap of 32 voxels is sufficient
- Model produces consistent predictions across patch boundaries

### Dose Range

- Predicted dose is clipped to [0, 1.5] (normalized)
- Values > 1.0 are valid (hot spots)
- Negative predictions are clipped to 0

### Timing

| Volume Size | Patches | DDIM Steps | Estimated Time (3090) |
|-------------|---------|------------|----------------------|
| 512×512×256 | 50 | 50 | 5-10 min |
| 512×512×128 | 25 | 50 | 2-5 min |
| 256×256×128 | 8 | 50 | 1-2 min |

---

## Metric Assumptions

### MAE (Mean Absolute Error)

- Computed in **Gy** (denormalized)
- Averaged over **all voxels** (not just PTV or high-dose)
- **Lower is better**
- Target: < 2.0 Gy

### Gamma

- **3%/3mm** criteria (clinical standard for IMRT QA)
- **10% dose threshold** (ignore low-dose regions)
- **Global normalization** (percent of max dose)
- Target: > 95% pass rate

**Training vs. Inference Gamma:**

| Context | Volume | Subsampling | Notes |
|---------|--------|-------------|-------|
| Training | Single 128³ patch | 4× → 32³ | Fast proxy for monitoring (~1 sec) |
| Inference | Full 512×512×256 | 2× → 256³ | Proper final evaluation (~minutes) |

**Assumption:** Training gamma on patches is a reasonable proxy for full-volume gamma. This is generally true because:
- Patches are sampled from high-dose regions (where gamma matters most)
- Subsampling is uniform across the volume
- Final evaluation always uses full volume

**Important:** Report only inference gamma (from `inference_dose_ddpm.py`) in publications.

### DVH Metrics

- Computed per structure
- Compared to ground truth DVH
- D95, D50, D5, Vxx values
- Errors in Gy or percentage points

---

## Limitations and Known Issues

### Not Modeled

- **Deliverability**: Model predicts dose, not MLC sequences
- **Physics constraints**: No beam model, no scatter calculation
- **Hard constraints**: DVH constraints are not enforced

### Potential Failure Modes

| Issue | Symptom | Mitigation |
|-------|---------|------------|
| Distribution shift | Poor val performance | Ensure consistent data |
| Overfitting | Train >>> val | More data, augmentation, early stopping |
| Mode collapse | Same prediction for all inputs | Check conditioning, increase model capacity |
| Boundary artifacts | Visible seams | Increase overlap, check Gaussian weights |

### Edge Cases

- **Very large patients**: May truncate at boundaries
- **Unusual anatomy**: May perform poorly (e.g., hip prosthesis)
- **Different fractionation**: Model trained on 70/56 Gy only
- **Non-prostate**: Not expected to generalize

---

## Reproducibility Checklist

For exact reproduction of results:

- [ ] Same `.npz` files (preprocess with same version)
- [ ] Same random seed (`--seed 42`)
- [ ] Same PyTorch version
- [ ] Same CUDA version
- [ ] Same GPU model (for determinism)
- [ ] Same hyperparameters (all CLI arguments)
- [ ] `deterministic=True` in trainer

**Note:** Even with all the above, floating-point non-determinism may cause small variations (<0.1% in metrics).

---

## Validation Before Training

Run these checks before starting training:

```python
import numpy as np
from pathlib import Path

data_dir = Path('./processed_npz')
files = list(data_dir.glob('*.npz'))

print(f"Found {len(files)} files")

# Check first file
data = np.load(files[0], allow_pickle=True)

# Required keys
required = ['ct', 'dose', 'masks_sdf', 'constraints', 'metadata']
for key in required:
    assert key in data.files, f"Missing {key}"

# Shapes
assert data['ct'].shape == (512, 512, 256), f"CT shape: {data['ct'].shape}"
assert data['dose'].shape == (512, 512, 256), f"Dose shape: {data['dose'].shape}"
assert data['masks_sdf'].shape == (8, 512, 512, 256), f"SDF shape: {data['masks_sdf'].shape}"
assert data['constraints'].shape == (13,), f"Constraints shape: {data['constraints'].shape}"

# Ranges
assert 0 <= data['ct'].min() and data['ct'].max() <= 1, f"CT range: [{data['ct'].min()}, {data['ct'].max()}]"
assert data['dose'].min() >= 0, f"Dose min: {data['dose'].min()}"
assert -1 <= data['masks_sdf'].min() and data['masks_sdf'].max() <= 1, f"SDF range: [{data['masks_sdf'].min()}, {data['masks_sdf'].max()}]"

# No NaN/Inf
assert np.isfinite(data['ct']).all(), "CT has NaN/Inf"
assert np.isfinite(data['dose']).all(), "Dose has NaN/Inf"
assert np.isfinite(data['masks_sdf']).all(), "SDF has NaN/Inf"

print("All checks passed!")
```
