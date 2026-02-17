> **SUPERSEDED (2026-02-17)** — This changelog was last maintained through January 2025 and does not
> cover the loss function experiments (gradient, DVH-aware, structure-weighted, asymmetric PTV)
> or the pilot-to-production transition. For current project state: see `.claude/instructions.md`.
> For experiment history: see `notebooks/EXPERIMENTS_INDEX.md`. Change tracking is now done
> via git log and the experiment index rather than this file.

# Changelog

All notable changes to the VMAT Diffusion project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Fluence map generation from MLC sequences (Phase 2)
- MLC trajectory prediction model (Phase 2)
- Joint dose + MLC prediction (Phase 2)
- PTV50.4 support for 3-level SIB cases
- Multi-institution validation
- Clinical deployment pipeline

---

## Inference Script (`inference_dose_ddpm.py`)

### [1.1.0] - 2025-01-10

Added clinical constraint checking against QUANTEC-based limits.

#### Added

**Clinical Constraint Checking**
- QUANTEC-based constraints for prostate VMAT (70/56 Gy SIB)
- PTV constraints: D95_min, V95_min
- OAR constraints: Rectum/Bladder V70, V60, V50, Dmax; Femur Dmax
- `check_clinical_constraints()` function compares predicted DVH to limits
- Per-case violation reporting
- Aggregate statistics: cases passing, total violations, most common violations

**Enhanced Output**
- Each case now reports clinical constraint pass/fail
- Summary shows percentage of cases meeting all constraints
- Lists most common constraint violations across batch

**Customization**
- `CLINICAL_CONSTRAINTS` dict at top of file can be modified
- Users should adjust limits based on institutional protocols

---

### [1.0.0] - 2025-01-09

## Training Script (`train_dose_ddpm_v2.py`)

### [2.2.0] - 2025-01-10

Improved training pipeline with physics-informed loss and corrected augmentation.

#### Changed

**Fixed Augmentation Strategy**
- Removed Y-flip (anterior-posterior) - beam entry direction matters
- Removed Z-flip (superior-inferior) - anatomy not symmetric in this direction
- Removed 90° rotations - beam angles are meaningful
- Kept only X-flip (left-right) - valid for prostate bilateral symmetry
- Added femur SDF channel swap on X-flip (Femur_L ↔ Femur_R)
- Added random translation ±16 voxels (~±16-32mm) for robustness
  - Simulates inter-fraction patient positioning variation
  - Uses scipy.ndimage.shift with nearest-edge filling

**Increased Default Model Capacity**
- Default base_channels changed from 32 to 48
- Results in ~2× more parameters
- Better capacity for complex dose gradients

#### Added

**Train/Val/Test Split**
- New `--test_split` argument (default: 0.1)
- Test cases saved to `test_cases.json` for later evaluation
- Test set completely held out from training
- Default split: 80% train / 10% val / 10% test

**Physics-Informed Loss**
- Added negative dose penalty term
- Penalizes predictions that would result in negative dose values
- Loss = MSE(noise) + 0.1 × mean(ReLU(-predicted_dose))
- Improves physical plausibility without constraining deliverability

#### Files
- `train_dose_ddpm_v2.py` (updated, ~1,370 lines)

---

### [2.1.0] - 2025-01-09

Complete training pipeline for Phase 1 dose prediction with publication-ready logging.

#### Added

**Patch-Based Training**
- 128³ patches extracted from 512×512×256 volumes
- Fits on RTX 3090 (24GB) with batch_size=2
- Dose-biased sampling (50% high-dose regions, 50% random)
- 3D augmentation (flips along all axes, 90° axial rotations)
- ~8× effective diversity per case

**Model Architecture: SimpleUNet3D**
- 4-level encoder-decoder with skip connections
- Base channels: 32 → 64 → 128 → 256
- ~2.8M trainable parameters
- FiLM conditioning for timestep and constraints
- GroupNorm + SiLU activations throughout

**Diffusion Process**
- Cosine noise schedule (1000 timesteps)
- Epsilon (noise) prediction target
- Proper buffer registration for schedule tensors
- DDIM sampling for fast inference (50 steps)

**Sliding Window Inference**
- `predict_full_volume()` method for full-volume prediction
- Gaussian-weighted overlap averaging
- Seamless blending across patch boundaries
- Configurable patch size (default 128) and overlap (default 32)

**Publication Logging**
- `PublicationLoggingCallback`: Comprehensive training logging
- `training_config.json`: Environment, hyperparameters, train/val split
- `training_summary.json`: Final results, timing, convergence info
- `epoch_metrics.csv`: Per-epoch metrics for plotting
- `GPUMemoryCallback`: GPU memory usage tracking
- ETA estimation during training

**Metrics**
- Training loss (MSE on predicted noise)
- Validation loss (MSE on predicted noise)
- Validation MAE in Gy (denormalized)
- Gamma pass rate (3%/3mm) via pymedphys (subsampled for speed)
- Automatic convergence epoch detection

**Multi-GPU Support**
- `--devices` argument for number of GPUs
- `--strategy` argument for distributed strategy (ddp, etc.)
- Compatible with PyTorch Lightning DDP

**Reproducibility**
- Fixed random seed (default 42)
- Deterministic training mode
- File-level train/val split (not patch-level)
- Complete case ID logging

#### Files
- `train_dose_ddpm_v2.py` (1,335 lines)

---

### [1.0.0] - 2025-01-08

Initial training script (replaced by v2.1.0).

#### Issues Fixed in v2.1.0
- Used binary masks instead of SDFs
- Full-volume training (didn't fit on 3090)
- Broken DDPM implementation (device issues)
- Missing FiLM conditioning
- 2D augmentation (incompatible with 3D data)
- Incorrect pymedphys gamma API

---

## Inference Script (`inference_dose_ddpm.py`)

### [1.0.0] - 2025-01-09

Standalone inference and evaluation script.

#### Added
- Single case and batch inference
- Sliding window full-volume prediction
- Comprehensive evaluation metrics:
  - MAE (overall and per-threshold)
  - RMSE
  - Gamma pass rate (3%/3mm)
  - DVH metrics per structure (D95, D50, D5, V70, etc.)
- JSON results export
- Goal assessment (MAE < 2 Gy, Gamma > 95%)

#### Usage
```bash
# Single case
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input ./processed_npz/case_0001.npz \
    --output ./predictions/case_0001_pred.npz

# Batch with metrics
python inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_dose_ddpm/checkpoints/best.ckpt \
    --input_dir ./processed_npz \
    --output_dir ./predictions \
    --compute_metrics
```

#### Files
- `inference_dose_ddpm.py` (350 lines)

---

## Baseline Model (`train_baseline_unet.py`)

### [1.0.0] - 2025-01-10

Direct regression baseline for comparison with diffusion model.

#### Purpose
Answer the question: "Is diffusion actually helping?"

#### Key Differences from Diffusion Model

| Aspect | Diffusion (DDPM) | Baseline (Direct) |
|--------|------------------|-------------------|
| Output | Predicts noise | Predicts dose directly |
| Training | Sample timestep, add noise | Simple MSE on dose |
| Inference | 50 DDIM steps | Single forward pass |
| Time/case | ~5-10 min | ~10-30 sec |
| Uncertainty | Multiple samples possible | Point estimate only |

#### Architecture
- Same U-Net structure as diffusion model (comparable capacity)
- Same base_channels=48 default
- FiLM conditioning for constraints (no time embedding)
- Same augmentation (X-flip, translation)
- Same train/val/test split logic

#### Usage
```bash
# Train baseline
python train_baseline_unet.py --data_dir ./processed_npz --epochs 200

# Evaluate
python inference_baseline_unet.py \
    --checkpoint ./runs/baseline_unet/checkpoints/best.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions \
    --compute_metrics
```

#### Expected Comparison
Run both models on same test set, compare:
- MAE (Gy)
- Gamma pass rate (%)
- DVH metrics
- Inference time

#### Files
- `train_baseline_unet.py` (~650 lines)
- `inference_baseline_unet.py` (~180 lines)

---

## Preprocessing Script (`preprocess_dicom_rt_v2.2.py`)

### [2.2.0] - 2025-01-09

Phase 2-ready preprocessing with full MLC extraction.

#### Added

**Full MLC Leaf Position Extraction**
- Complete MLC positions for both banks (A and B) at every control point
- Typical VMAT arc: 178 control points × 60 leaf pairs = 21,360 positions/arc
- Stored as numpy arrays: `beam0_mlc_a`, `beam0_mlc_b`, etc.
- Shape: `(n_control_points, n_leaves_per_bank)` float32

**Control Point Data**
- `gantry_angles`: Gantry angle at each CP (degrees)
- `cumulative_meterset_weight`: Cumulative MU at each CP
- `dose_rates`: Dose rate (MU/min) at each CP
- `jaw_x1`, `jaw_x2`, `jaw_y1`, `jaw_y2`: Jaw positions at each CP (mm)

**Additional Beam Information**
- `nominal_energy`: Beam energy (e.g., 6.0 for 6MV)
- `treatment_machine`: Linac name
- `beam_number`: DICOM beam number
- `mlc_type`: MLC device type (e.g., "MLCX")
- `num_leaves`: Total number of leaves (typically 120)
- `leaf_pair_count`: Number of leaf pairs (typically 60)
- `leaf_boundaries`: Leaf position boundaries (mm)

**Storage Optimization**
- MLC arrays stored as separate npz keys (not nested in metadata)
- Uses `np.savez_compressed` for ~50% file size reduction
- Typical file: ~400-500 MB compressed (was ~800 MB)

#### Changed
- Script version: `2.2.0`
- Default output uses compressed npz format

#### Files
- `preprocess_dicom_rt_v2.2.py`

---

### [2.1.1] - 2025-01-09

SDF validation and configurable clip distance.

#### Added
- `validate_sdf()`: Checks SDF correctness (inside negative, outside positive)
- `--sdf_clip_mm`: Configurable SDF clip distance (default 50mm)
- SDF validation results stored in `metadata['sdf_validation']`

---

### [2.1.0] - 2025-01-09

Added SDFs, beam geometry, and enhanced validation.

#### Added

**Signed Distance Fields (SDFs)**
- `compute_sdf()`: Generates SDFs from binary masks using scipy EDT
- SDFs clipped at ±50mm and normalized to [-1, 1]
- Convention: negative inside, positive outside, zero at boundary
- Stored as `masks_sdf` key in .npz files
- Provides smooth gradients for neural network training

**Beam Geometry Extraction**
- `extract_beam_geometry()`: Extracts VMAT arc parameters from RP file
- Per-beam: arc start/stop angles, gantry direction, collimator, couch, MU
- Plan-level: number of beams, total MU, plan label

**Improved Validation**
- `quantify_truncation()`: Measures boundary truncation per structure
- HU clipping detection (tracks original range before normalization)
- Registration validation (dose-in-PTV > dose-outside)

**Command Line Options**
- `--no_sdf`: Skip SDF computation
- `--no_beams`: Skip beam geometry extraction  
- `--strict_validation`: Fail on validation issues

#### Changed
- Script version: `2.1.0`
- Added `masks_sdf` to output arrays
- Enhanced metadata structure with beam_geometry and validation info

---

### [2.0.0] - 2025-01-08

Major rewrite with SIB support and constraints vector.

#### Added

**Prescription Extraction**
- Automatic extraction from DoseReferenceSequence
- Fallback to ReferencedDoseReferenceSequence
- Manual fallback with warnings

**SIB Support**
- Detects 2-level SIB (PTV70 + PTV56)
- Detects 3-level SIB (PTV70 + PTV56 + PTV50.4)
- Case type stored in metadata

**Validation Suite**
- CT range validation [0, 1]
- Dose non-negativity check
- PTV70 mean dose check (~1.0 normalized)
- Registration validation (dose aligns with structures)

**Constraints Vector**
- 13-element conditioning vector
- PTV targets (indices 0-2)
- OAR constraints (indices 3-12)
- Normalized appropriately for each constraint type

#### Changed
- Output format: .npz with multiple arrays
- Dose normalization: Divided by prescription dose
- Volume centering: Prostate/PTV70 centroid

---

### [1.0.0] - 2025-01-07

Initial preprocessing pipeline.

#### Added
- CT/dose/mask loading from DICOM-RT
- Resampling to fixed grid (512×512×256 @ 1×1×2mm)
- 8-channel structure mask extraction
- Basic validation checks
- .npz output format

---

## .npz File Format

### Version 2.2 Structure

```python
data = np.load('case.npz', allow_pickle=True)

# === Core Arrays ===
data['ct']           # (512, 512, 256) float32 - CT normalized [0,1]
data['dose']         # (512, 512, 256) float32 - Dose normalized to Rx
data['masks']        # (8, 512, 512, 256) uint8 - Binary structure masks
data['masks_sdf']    # (8, 512, 512, 256) float32 - SDFs [-1, 1]
data['constraints']  # (13,) float32 - Conditioning vector

# === MLC Arrays (Phase 2) ===
data['beam0_mlc_a']  # (n_cp, n_leaves) float32 - MLC bank A positions
data['beam0_mlc_b']  # (n_cp, n_leaves) float32 - MLC bank B positions
data['beam1_mlc_a']  # (n_cp, n_leaves) float32 - If 2+ arcs
data['beam1_mlc_b']  # (n_cp, n_leaves) float32 - If 2+ arcs

# === Metadata ===
metadata = data['metadata'].item()
metadata['patient_id']
metadata['case_type']           # 'sib_2level' or 'sib_3level'
metadata['prescription_gy']     # e.g., 70.0
metadata['original_spacing']
metadata['beam_geometry']       # Dict with beam info
metadata['truncation_info']     # Boundary truncation stats
metadata['sdf_validation']      # SDF correctness checks
metadata['script_version']      # '2.2.0'
```

### Structure Channels (8 total)

| Channel | Structure | Description |
|---------|-----------|-------------|
| 0 | PTV70 | High-dose PTV (70 Gy in 28 fx) |
| 1 | PTV56 | Intermediate PTV (56 Gy in 28 fx) |
| 2 | Prostate | Clinical target volume |
| 3 | Rectum | OAR - anterior rectal wall |
| 4 | Bladder | OAR |
| 5 | Femur_L | OAR - left femoral head |
| 6 | Femur_R | OAR - right femoral head |
| 7 | Bowel | OAR - small/large bowel |

### Constraints Vector (13 values)

| Index | Constraint | Description | Normalization |
|-------|------------|-------------|---------------|
| 0 | PTV70 target | High-dose PTV | 1.0 if exists, 0 otherwise |
| 1 | PTV56 target | Intermediate PTV | 0.8 (56/70) if exists |
| 2 | PTV50.4 target | Low-dose PTV | 0.72 (50.4/70) if exists |
| 3 | Rectum V50 | Volume receiving 50 Gy | Fraction [0,1] |
| 4 | Rectum V60 | Volume receiving 60 Gy | Fraction [0,1] |
| 5 | Rectum V70 | Volume receiving 70 Gy | Fraction [0,1] |
| 6 | Rectum Dmax | Maximum rectum dose | Normalized to Rx |
| 7 | Bladder V65 | Volume receiving 65 Gy | Fraction [0,1] |
| 8 | Bladder V70 | Volume receiving 70 Gy | Fraction [0,1] |
| 9 | Bladder V75 | Volume receiving 75 Gy | Fraction [0,1] |
| 10 | Femur V50 | Femoral head V50 | Fraction [0,1] |
| 11 | Bowel V45 | Bowel bag constraint | Normalized |
| 12 | Cord Dmax | Spinal cord max (if present) | Normalized to Rx |

---

## Verification Notebook (`verify_npz.ipynb`)

### [1.0.0] - 2025-01-08

Jupyter notebook for visual verification of preprocessed data.

#### Features
- Load and inspect .npz file structure
- Visualize CT slices (axial, sagittal, coronal)
- Overlay dose on CT
- Display structure contours
- SDF visualization
- Constraint vector inspection
- Beam geometry summary
- MLC position plots
