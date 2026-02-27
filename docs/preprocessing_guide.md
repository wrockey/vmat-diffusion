> **CURRENT REFERENCE** — This preprocessing guide is active and accurate for `preprocess_dicom_rt_v2.3.py`.
> For project strategy: see `.claude/instructions.md`. For data assumptions: see `docs/preprocessing_assumptions.md`.

# VMAT Preprocessing Guide

## Overview

This document describes the preprocessing pipeline that converts DICOM-RT data into the standardized `.npz` format required for model training.

**Script:** `preprocess_dicom_rt_v2.3.py`  
**Version:** 2.2.0  
**Output:** Phase 2-ready `.npz` files with full MLC data

---

## Quick Start

```bash
# Single case
python preprocess_dicom_rt_v2.3.py \
    --input_dir /path/to/dicom/case \
    --output_dir ./processed_npz \
    --output_name case_0001

# Batch processing
for dir in /path/to/all/cases/*/; do
    name=$(basename "$dir")
    python preprocess_dicom_rt_v2.3.py \
        --input_dir "$dir" \
        --output_dir ./processed_npz \
        --output_name "$name"
done
```

---

## Input Requirements

### Required DICOM Files

| File Type | Description | Required Keys |
|-----------|-------------|---------------|
| **CT** | CT image series | PixelData, ImagePositionPatient, etc. |
| **RTSTRUCT** | Structure set | ROIContourSequence |
| **RTDOSE** | Dose distribution | DoseGridScaling, PixelData |
| **RTPLAN** | Treatment plan | BeamSequence (for MLC extraction) |

### Expected Directory Structure

```
case_directory/
├── CT.*.dcm           # CT slices (multiple files)
├── RS.*.dcm           # Structure set (1 file)
├── RD.*.dcm           # Dose file (1 file)
└── RP.*.dcm           # Plan file (1 file)
```

### Required Structures

The following structures must be present (case-insensitive matching):

| Structure | Aliases Recognized |
|-----------|-------------------|
| PTV70 | PTV_70, PTV70Gy, PTV 70, ptvhigh |
| PTV56 | PTV_56, PTV56Gy, PTV 56, ptvlow |
| Prostate | prostate, CTV, ctv_prostate |
| Rectum | rectum, rect, rectal |
| Bladder | bladder, blad |
| Femur_L | femur_l, left_femur, femoral_l, l_femur |
| Femur_R | femur_r, right_femur, femoral_r, r_femur |
| Bowel | bowel, bowel_bag, small_bowel, large_bowel |

---

## Output Format

### .npz File Contents (v2.2)

```python
import numpy as np
data = np.load('case.npz', allow_pickle=True)

# Core arrays
print(data['ct'].shape)          # (512, 512, 256)
print(data['dose'].shape)        # (512, 512, 256)
print(data['masks'].shape)       # (8, 512, 512, 256)
print(data['masks_sdf'].shape)   # (8, 512, 512, 256)
print(data['constraints'].shape) # (13,)

# MLC arrays
print(data['beam0_mlc_a'].shape) # (178, 60) typical
print(data['beam0_mlc_b'].shape) # (178, 60) typical

# Metadata
metadata = data['metadata'].item()
print(metadata.keys())
```

### Array Specifications

| Array | Shape | Dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `ct` | (512, 512, 256) | float32 | [0, 1] | Normalized CT |
| `dose` | (512, 512, 256) | float32 | [0, ~1.1] | Dose / Rx |
| `masks` | (8, 512, 512, 256) | uint8 | {0, 1} | Binary masks |
| `masks_sdf` | (8, 512, 512, 256) | float32 | [-1, 1] | Signed distance fields |
| `constraints` | (13,) | float32 | [0, ~1.1] | Planning constraints |
| `beam*_mlc_a/b` | (n_cp, n_leaves) | float32 | [-200, 200] mm | MLC positions |

### Coordinate System

- **Origin**: Centered on Prostate/PTV70 centroid
- **Axes**: 
  - X: Patient left (+) to right (-)
  - Y: Patient anterior (+) to posterior (-)
  - Z: Patient superior (+) to inferior (-)
- **Spacing**: 1.0 × 1.0 × 2.0 mm (fixed)
- **Grid**: 512 × 512 × 256 voxels

---

## Processing Steps

### 1. DICOM Loading

```python
# CT loading
ct_volume = load_ct_series(ct_files)
ct_spacing = get_spacing(ct_files[0])

# Structure loading via rt-utils
rtstruct = RTStructBuilder.create_from(dicom_path, rt_struct_path)
mask = rtstruct.get_roi_mask_by_name(structure_name)

# Dose loading
dose = dcmread(dose_file)
dose_volume = dose.pixel_array * dose.DoseGridScaling
```

### 2. Resampling

All volumes resampled to fixed grid:

```python
OUTPUT_SHAPE = (512, 512, 256)
OUTPUT_SPACING = (1.0, 1.0, 2.0)  # mm

# Using scipy.ndimage.zoom
zoom_factors = original_spacing / output_spacing * original_shape / output_shape
ct_resampled = zoom(ct_volume, zoom_factors, order=1)      # Linear
dose_resampled = zoom(dose_volume, zoom_factors, order=1)  # Linear
mask_resampled = zoom(mask_volume, zoom_factors, order=0)  # Nearest
```

### 3. Centering

Volume centered on prostate/PTV70 centroid:

```python
# Find centroid
centroid = np.array(np.where(prostate_mask > 0)).mean(axis=1)

# Shift to center
shift = np.array(OUTPUT_SHAPE) // 2 - centroid
ct_centered = ndimage.shift(ct_resampled, shift)
```

### 4. CT Normalization

```python
# Clip to soft tissue range
ct_clipped = np.clip(ct_volume, -1000, 2000)  # HU

# Normalize to [0, 1]
ct_normalized = (ct_clipped - (-1000)) / (2000 - (-1000))
```

### 5. Dose Normalization

```python
# Normalize to prescription dose
prescription_gy = extract_prescription(rtplan)  # e.g., 70.0 Gy
dose_normalized = dose_volume / prescription_gy

# Result: PTV70 should have mean ~1.0
```

### 6. SDF Computation

Signed Distance Fields provide smooth gradients for neural network training:

```python
from scipy.ndimage import distance_transform_edt

def compute_sdf(binary_mask, spacing=(1.0, 1.0, 2.0), clip_mm=50.0):
    # Distance outside structure (positive)
    dist_outside = distance_transform_edt(~binary_mask, sampling=spacing)
    
    # Distance inside structure (negative)
    dist_inside = distance_transform_edt(binary_mask, sampling=spacing)
    
    # Combine: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    # Clip and normalize to [-1, 1]
    sdf = np.clip(sdf, -clip_mm, clip_mm) / clip_mm
    
    return sdf.astype(np.float32)
```

**SDF Properties:**
- Inside structure: negative values (approaching -1 at center)
- Outside structure: positive values (approaching +1 far away)
- At boundary: zero
- Smooth gradients everywhere (unlike binary step function)

### 7. Constraint Extraction

```python
def extract_constraints(dose, masks, prescription):
    constraints = np.zeros(13, dtype=np.float32)
    
    # PTV targets
    constraints[0] = 1.0 if masks['PTV70'].any() else 0.0
    constraints[1] = 56/70 if masks['PTV56'].any() else 0.0
    constraints[2] = 50.4/70 if masks['PTV50.4'].any() else 0.0
    
    # Rectum DVH constraints
    rectum = masks['Rectum']
    constraints[3] = (dose[rectum] >= 50/prescription).mean()  # V50
    constraints[4] = (dose[rectum] >= 60/prescription).mean()  # V60
    constraints[5] = (dose[rectum] >= 70/prescription).mean()  # V70
    constraints[6] = dose[rectum].max()  # Dmax
    
    # ... similar for bladder, femur, bowel
    
    return constraints
```

### 8. MLC Extraction

```python
def extract_mlc_data(rtplan):
    for beam in rtplan.BeamSequence:
        if beam.TreatmentDeliveryType != 'TREATMENT':
            continue
            
        mlc_a_positions = []
        mlc_b_positions = []
        
        for cp in beam.ControlPointSequence:
            # Find MLC device
            for device in cp.BeamLimitingDevicePositionSequence:
                if 'MLC' in device.RTBeamLimitingDeviceType:
                    positions = device.LeafJawPositions
                    n_leaves = len(positions) // 2
                    
                    # Bank A (leaves 0 to n-1)
                    mlc_a_positions.append(positions[:n_leaves])
                    
                    # Bank B (leaves n to 2n-1)
                    mlc_b_positions.append(positions[n_leaves:])
        
        mlc_a = np.array(mlc_a_positions, dtype=np.float32)
        mlc_b = np.array(mlc_b_positions, dtype=np.float32)
        
        return mlc_a, mlc_b  # Shape: (n_control_points, n_leaves)
```

---

## Command Line Options

### Required

| Option | Description |
|--------|-------------|
| `--input_dir` | Path to DICOM directory |
| `--output_dir` | Output directory for .npz files |
| `--output_name` | Output filename (without .npz) |

### Optional - Processing

| Option | Default | Description |
|--------|---------|-------------|
| `--no_sdf` | False | Skip SDF computation |
| `--no_beams` | False | Skip beam geometry extraction |
| `--sdf_clip_mm` | 50.0 | SDF clip distance in mm |

### Optional - Validation

| Option | Default | Description |
|--------|---------|-------------|
| `--strict_validation` | False | Fail on validation warnings |
| `--skip_plots` | False | Skip visualization plots |

### Optional - Override

| Option | Default | Description |
|--------|---------|-------------|
| `--prescription_gy` | Auto | Override prescription dose |
| `--force` | False | Overwrite existing output |

---

## Validation Checks

The script performs these validation checks:

### 1. CT Validation
- Range check: Values should be in [0, 1] after normalization
- HU range tracking: Original range logged for reference

### 2. Dose Validation
- Non-negativity: All dose values >= 0
- PTV70 mean: Should be approximately 1.0 (normalized)
- Maximum dose: Logged for reference

### 3. Structure Validation
- All required structures present
- Non-empty masks (at least 1 voxel)
- Boundary truncation < 10% of structure

### 4. Registration Validation
- Mean dose in PTV70 > mean dose outside PTV70
- Detects potential misregistration between dose and structures

### 5. SDF Validation
- Inside voxels have negative SDF
- Outside voxels have positive SDF
- Boundary voxels near zero

---

## Troubleshooting

### Missing Structures

```
ERROR: Required structure 'PTV70' not found
Available structures: ['PTV_High', 'CTV', 'Rectum', ...]
```

**Solution:** The script searches for common aliases. Add your structure name to the alias list in the script, or rename structures in the DICOM.

### Dose-Structure Misregistration

```
WARNING: Registration validation failed
  Mean dose in PTV70: 0.45
  Mean dose outside: 0.52
```

**Solution:** Check that RTDOSE and RTSTRUCT reference the same CT series. May need to re-export from TPS.

### Boundary Truncation

```
WARNING: Structure 'Bowel' truncated at boundary
  Truncation: 15.3% of voxels at edge
```

**Solution:** This is informational. Some truncation is expected for large structures. The `truncation_info` in metadata tracks this.

### Memory Issues

For very large CT series:
```bash
# Process with lower memory usage
python preprocess_dicom_rt_v2.3.py \
    --input_dir ./case \
    --output_dir ./output \
    --output_name case \
    --skip_plots
```

---

## Output Verification

Use the provided notebook to verify preprocessing:

```bash
jupyter notebook verify_npz.ipynb
```

Or quick command-line check:

```python
import numpy as np

data = np.load('case.npz', allow_pickle=True)

# Check shapes
print(f"CT: {data['ct'].shape}")           # (512, 512, 256)
print(f"Dose: {data['dose'].shape}")       # (512, 512, 256)
print(f"Masks: {data['masks'].shape}")     # (8, 512, 512, 256)
print(f"SDFs: {data['masks_sdf'].shape}")  # (8, 512, 512, 256)

# Check ranges
print(f"CT range: [{data['ct'].min():.2f}, {data['ct'].max():.2f}]")     # [0, 1]
print(f"Dose range: [{data['dose'].min():.2f}, {data['dose'].max():.2f}]") # [0, ~1.1]

# Check metadata
meta = data['metadata'].item()
print(f"Prescription: {meta['prescription_gy']} Gy")
print(f"Script version: {meta['script_version']}")
print(f"Beams: {len(meta['beam_geometry']['beams'])}")

# Check MLC data
if 'beam0_mlc_a' in data.files:
    print(f"MLC shape: {data['beam0_mlc_a'].shape}")  # (178, 60) typical
```

---

## Version History

| Version | Changes |
|---------|---------|
| 2.2.0 | Full MLC extraction, compressed npz, Phase 2 ready |
| 2.1.1 | SDF validation, configurable clip distance |
| 2.1.0 | SDFs, beam geometry, enhanced validation |
| 2.0.0 | SIB support, constraints vector, major rewrite |
| 1.0.0 | Initial implementation |

---

## File Size Reference

| Content | Approximate Size |
|---------|-----------------|
| CT (512×512×256 float32) | 256 MB |
| Dose (512×512×256 float32) | 256 MB |
| Masks (8×512×512×256 uint8) | 512 MB |
| SDFs (8×512×512×256 float32) | 2 GB |
| MLC data | ~1 MB |
| **Total uncompressed** | **~3 GB** |
| **Total compressed (.npz)** | **~400-500 MB** |

Compression ratio is approximately 6:1 due to sparse structure masks and smooth SDF fields.
