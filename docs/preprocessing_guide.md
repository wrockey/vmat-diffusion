# Preprocessing Guide

## Overview

The preprocessing pipeline converts DICOM-RT treatment plans into normalized `.npz` files suitable for training the VMAT diffusion model.

**Current Version:** 2.0.0  
**Script:** `preprocess_dicom_rt_v2.py`

---

## Quick Start

```bash
# Basic usage
python preprocess_dicom_rt_v2.py \
    --input_dir ./data/raw \
    --output_dir ./processed_npz

# Include single-prescription cases (no PTV56)
python preprocess_dicom_rt_v2.py --relax_filter

# Strict QA mode (fail on validation issues)
python preprocess_dicom_rt_v2.py --strict_validation

# Fast mode for HPC (no debug images)
python preprocess_dicom_rt_v2.py --skip_plots
```

---

## Input Requirements

### Directory Structure

```
data/raw/
├── case_0001/
│   ├── CT.1.2.3....dcm      # CT slices (multiple files)
│   ├── RS.1.2.3....dcm      # Structure set (one file)
│   ├── RD.1.2.3....dcm      # Dose grid (one file)
│   └── RP.1.2.3....dcm      # RT Plan (one file, optional but recommended)
├── case_0002/
│   └── ...
└── ...
```

### Required DICOM Files

| File Prefix | Type | Required | Used For |
|-------------|------|----------|----------|
| `CT.*` | CT Image | Yes | Anatomy |
| `RS.*` | RT Structure Set | Yes | Contours |
| `RD.*` | RT Dose | Yes | Dose distribution |
| `RP.*` | RT Plan | Recommended | Prescription extraction |

### Required Structures

The following structures must be present in the RS file (name variations handled via `oar_mapping.json`):

| Structure | Required | Channel |
|-----------|----------|---------|
| PTV70 | Yes | 0 |
| PTV56 | Yes* | 1 |
| Prostate | No | 2 |
| Rectum | Recommended | 3 |
| Bladder | Recommended | 4 |
| Femur_L | No | 5 |
| Femur_R | No | 6 |
| Bowel | No | 7 |

*Use `--relax_filter` to process cases without PTV56

---

## Output Structure

### .npz File Contents

```python
data = np.load('case_0001.npz', allow_pickle=True)

# Arrays
data['ct']          # (512, 512, 256) float32 - Normalized CT [0, 1]
data['dose']        # (512, 512, 256) float32 - Normalized dose [0, ~1.1]
data['masks']       # (8, 512, 512, 256) uint8 - Binary structure masks
data['constraints'] # (13,) float32 - Conditioning vector

# Metadata (new in v2.0)
data['metadata']    # dict - Rich case information
```

### Metadata Structure

```python
metadata = data['metadata'].item()

{
    'case_id': 'case_0001',
    'processed_date': '2025-01-09T14:30:00',
    'prescription_info': {
        'primary_dose': 70.0,           # Gy
        'fractions': 28,
        'dose_per_fraction': 2.5,       # Gy
        'source': 'DoseReferenceSequence'  # or 'FractionGroupSequence' or 'default'
    },
    'case_type': {
        'type': 'sib_2level',           # or 'single_rx' or 'sib_3level'
        'ptv70_exists': True,
        'ptv56_exists': True,
        'ptv50_exists': False
    },
    'normalization_dose_gy': 70.0,
    'target_shape': (512, 512, 256),
    'target_spacing_mm': (1.0, 1.0, 2.0),
    'validation': {
        'ct_range_valid': True,
        'dose_nonneg': True,
        'ptv70_dose_mean': 0.98,
        'registration_valid': True,
        'all_critical_passed': True
    },
    'boundary_warnings': [],
    'structure_channels': {0: 'PTV70', 1: 'PTV56', ...}
}
```

### Constraints Vector Structure

```python
constraints = data['constraints']  # Shape: (13,)

# Indices 0-2: Prescription targets (normalized to primary Rx)
constraints[0]  # PTV70 target: 1.0 if exists, else 0.0
constraints[1]  # PTV56 target: 0.8 (56/70) if exists, else 0.0
constraints[2]  # PTV50.4 target: 0.72 (50.4/70) if exists, else 0.0

# Indices 3-12: OAR constraints (normalized)
constraints[3]  # Rectum V50 (fraction)
constraints[4]  # Rectum V60 (fraction)
constraints[5]  # Rectum V70 (fraction)
constraints[6]  # Rectum max (normalized to Rx)
constraints[7]  # Bladder V65 (fraction)
constraints[8]  # Bladder V70 (fraction)
constraints[9]  # Bladder V75 (fraction)
constraints[10] # Femur V50 (fraction)
constraints[11] # Bowel V45 (normalized)
constraints[12] # Spinal cord max (normalized to Rx)
```

### Debug Outputs

```
processed_npz/
├── case_0001.npz           # Preprocessed data
├── debug_case_0001.png     # 2x3 visualization grid
├── case_0002.npz
├── debug_case_0002.png
└── batch_summary.json      # Batch processing summary
```

---

## Normalization Details

### CT Normalization

```python
ct_normalized = np.clip(ct_hu, -1000, 3000) / 4000 + 0.5
```

| HU Value | Tissue | Normalized |
|----------|--------|------------|
| -1000 | Air | 0.0 |
| 0 | Water | 0.5 |
| +1000 | Soft tissue/bone | 0.75 |
| +3000 | Dense bone | 1.0 |

**To recover HU:**
```python
ct_hu = (ct_normalized - 0.5) * 4000
```

### Dose Normalization

```python
dose_normalized = dose_gy / prescription_dose_gy
```

**v2.0 change:** Prescription is now extracted from RP file, not hardcoded.

| Dose (Gy) | Normalized (70 Gy Rx) |
|-----------|----------------------|
| 0 | 0.0 |
| 56 | 0.8 |
| 70 | 1.0 |
| 77 | 1.1 (hotspot) |

**To recover Gy:**
```python
prescription = metadata['normalization_dose_gy']
dose_gy = dose_normalized * prescription
```

---

## Validation Checks

The v2.0 script performs these checks before saving:

| Check | Pass Criteria | Failure Impact |
|-------|---------------|----------------|
| `ct_range_valid` | CT in [0, 1] | Normalization error |
| `dose_nonneg` | Dose ≥ -0.01 | Interpolation artifact |
| `dose_reasonable` | Max < 1.5 | Hotspot or scaling error |
| `ptv70_exists` | Voxels > 0 | Missing contour |
| `ptv70_dose_adequate` | Mean 0.85-1.15 | Wrong Rx or registration |
| `registration_valid` | Dose in PTV > outside | **Critical**: Misalignment |

### Strict Mode

With `--strict_validation`, cases that fail any critical check are skipped:

```bash
python preprocess_dicom_rt_v2.py --strict_validation
```

### Validation Results

Check validation in the notebook:
```python
metadata = data['metadata'].item()
print(f"Passed: {metadata['validation']['all_critical_passed']}")
print(f"PTV70 dose: {metadata['validation']['ptv70_dose_mean']:.3f}")
```

---

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | `~/vmat-diffusion-project/data/raw` | Input directory with case folders |
| `--output_dir` | `~/vmat-diffusion-project/processed_npz` | Output directory for .npz files |
| `--mapping_file` | `oar_mapping.json` | Structure name mapping file |
| `--relax_filter` | False | Process cases without PTV56 |
| `--skip_plots` | False | Skip debug PNG generation |
| `--strict_validation` | False | Fail cases with validation issues |

---

## Troubleshooting

### "No RP file found, using default prescription"

**Cause:** RT Plan file missing from case directory.

**Impact:** Prescription defaults to 70 Gy. May be incorrect for non-standard cases.

**Solution:** Ensure RP.*.dcm file is present, or verify 70 Gy is correct.

### "Prescription X Gy differs from expected 70 Gy"

**Cause:** Extracted prescription doesn't match expected value.

**Impact:** Warning only. Dose normalized to extracted value.

**Action:** Verify this case should be included in the cohort.

### "Skipping case: Missing PTV70"

**Cause:** No contour matching PTV70 variations found.

**Solution:** 
1. Check structure names in RS file
2. Add variation to `oar_mapping.json`

### "Skipping case: Missing PTV56"

**Cause:** Single-prescription case without PTV56.

**Solution:** Use `--relax_filter` to include these cases.

### "CRITICAL: Registration check failed"

**Cause:** Mean dose outside PTV is higher than inside.

**Impact:** Data unusable - CT and dose grids misaligned.

**Solution:**
1. Check dose grid origin in RD file
2. Verify GridFrameOffsetVector
3. Compare with TPS visualization

### "Structure touches boundary"

**Cause:** Structure extends to edge of resampled grid.

**Impact:** 
- Femoral heads: Usually acceptable
- PTV/Rectum/Bladder: May need larger grid

**Solution:** Review in notebook; consider grid expansion if critical.

---

## Migration from v1.0

### Key Differences

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Dose normalization | Hardcoded /70.0 | Extracted from RP |
| Constraints | Fixed AAPM values | Case-specific + Rx targets |
| Metadata | None | Rich dict with validation |
| Validation | None | Comprehensive checks |
| SIB handling | Implicit | Explicit case type |

### Migration Steps

1. **Re-run preprocessing**
   ```bash
   python preprocess_dicom_rt_v2.py --input_dir ./data/raw --output_dir ./processed_npz_v2
   ```

2. **Update data loading**
   ```python
   # Add allow_pickle=True for metadata
   data = np.load('case.npz', allow_pickle=True)
   metadata = data['metadata'].item()
   ```

3. **Update constraints usage**
   ```python
   # First 3 elements are now prescription targets
   ptv70_target = constraints[0]  # 1.0 or 0.0
   ptv56_target = constraints[1]  # 0.8 or 0.0
   ```

4. **Verify with notebook**
   ```bash
   jupyter notebook notebooks/verify_npz.ipynb
   ```

---

## Best Practices

### Before Training (~100 cases)

1. Run preprocessing with defaults
2. Run `verify_npz.ipynb` on 5-10 random cases
3. Check batch_summary.json for failures
4. Investigate any validation warnings

### Before HPC Scaling (1000+ cases)

1. Run with `--strict_validation`
2. Batch validate all cases with notebook
3. Document any excluded cases
4. Verify case type distribution matches expectations

### Recommended Workflow

```bash
# Step 1: Initial run
python preprocess_dicom_rt_v2.py --input_dir ./data/raw --output_dir ./processed_npz

# Step 2: Review summary
cat ./processed_npz/batch_summary.json

# Step 3: Investigate failures (if any)
# Check debug PNGs and re-run individual cases

# Step 4: Validate in notebook
jupyter notebook notebooks/verify_npz.ipynb

# Step 5: Production run for HPC
python preprocess_dicom_rt_v2.py --strict_validation --skip_plots
```

---

## References

- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [preprocessing_assumptions.md](preprocessing_assumptions.md) - Known limitations
- [verify_npz.md](verify_npz.md) - Notebook documentation
- [oar_mapping.json](../oar_mapping.json) - Structure name mappings
