# Changelog

All notable changes to the VMAT Diffusion project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: Beam geometry extraction (arc angles, MU, collimator)
- Phase 2: MLC leaf sequence extraction
- Signed distance fields (SDFs) for mask encoding
- PTV50.4 support (channel 8) for 3-level SIB cases
- pymedphys integration for DVH/gamma validation

---

## [2.0.0] - 2025-01-09

### Added

#### Prescription Extraction
- New `extract_prescription_dose()` function extracts actual prescription from RP DICOM files
- Attempts `DoseReferenceSequence.TargetPrescriptionDose` first (most reliable)
- Falls back to `FractionGroupSequence` beam dose calculation
- Returns prescription dose, fractions, dose per fraction, and extraction source
- Dose normalization now uses extracted prescription instead of hardcoded 70 Gy

#### Case Type Classification
- New `determine_case_type()` function automatically classifies cases:
  - `single_rx`: PTV70 only
  - `sib_2level`: PTV70 + PTV56
  - `sib_3level`: PTV70 + PTV56 + PTV50.4 (placeholder for future)
- Case type stored in metadata for cohort analysis

#### Improved Constraints Vector
- New `build_constraints_vector()` function creates case-specific conditioning
- **Prescription targets**: Encodes normalized dose targets per PTV level
  - PTV70: 1.0 (70/70)
  - PTV56: 0.8 (56/70) when present
  - PTV50.4: 0.72 (50.4/70) when present
- **OAR constraints**: Properly normalized to [0,1] range
- Model can now learn prescription-conditional dose generation

#### Validation System
- New `validate_preprocessed_data()` function runs QA checks before saving:
  - CT range valid [0, 1]
  - Dose non-negative (allows -0.01 for interpolation artifacts)
  - Dose max < 150% prescription
  - PTV70 exists with adequate mean dose (0.85-1.15)
  - **Registration check**: Verifies dose is higher inside PTV than outside
  - Boundary truncation detection for all structures
- New `--strict_validation` flag to fail cases that don't pass checks
- Validation results stored in metadata

#### Rich Metadata
- Comprehensive metadata now stored in each .npz file:
  ```python
  metadata = {
      'case_id': str,
      'processed_date': ISO timestamp,
      'prescription_info': {
          'primary_dose': float,
          'fractions': int,
          'dose_per_fraction': float,
          'source': str
      },
      'case_type': {
          'type': str,
          'ptv70_exists': bool,
          'ptv56_exists': bool,
          'ptv50_exists': bool
      },
      'normalization_dose_gy': float,
      'target_shape': tuple,
      'target_spacing_mm': tuple,
      'validation': dict,
      'boundary_warnings': list,
      'structure_channels': dict
  }
  ```

#### Improved Debug Plots
- Expanded from 1×3 to 2×3 grid layout
- New panels: CT+dose overlay, rectum/bladder contours, validation summary
- Case type and prescription shown in figure title
- Validation status (PASS/FAIL) displayed

#### Batch Processing Improvements
- New `batch_summary.json` saved after batch runs containing:
  - Total cases, processed count, failed count
  - List of failed case IDs
  - Processing settings used
- Failed cases listed at end of batch run for easy review

#### Documentation
- New `docs/preprocessing_assumptions.md` - Documents all assumptions and limitations
- New `docs/verify_npz.md` - Notebook usage guide
- New `CHANGELOG.md` - This file

### Changed

- **Dose normalization**: Now uses extracted prescription (was hardcoded `/70.0`)
- **Constraints vector**: Now case-specific with prescription targets (was fixed AAPM values)
- **PTV type vector**: Replaced with meaningful prescription targets
- **.npz structure**: Added `metadata` key with comprehensive case information
- **Debug plots**: Enhanced with more panels and validation info

### Fixed

- SIB cases now properly encode that PTV56 target is 0.8, not 1.0
- Prescription source is now traceable (stored in metadata)
- Registration errors now detected before training

### Deprecated

- Original `ptv_type = np.array([1.0, 0.0, 0.0])` - replaced with prescription targets

### Migration Guide

#### Updating from v1.0 to v2.0

1. **Re-run preprocessing** - The .npz structure has changed
   ```bash
   python preprocess_dicom_rt_v2.py --input_dir ./data/raw --output_dir ./processed_npz_v2
   ```

2. **Update data loading code** - New metadata structure
   ```python
   # Old
   data = np.load('case.npz')
   ct, dose, masks, constraints = data['ct'], data['dose'], data['masks'], data['constraints']
   
   # New (backward compatible, but metadata available)
   data = np.load('case.npz', allow_pickle=True)
   ct, dose, masks, constraints = data['ct'], data['dose'], data['masks'], data['constraints']
   metadata = data['metadata'].item()  # Access rich metadata
   print(f"Case type: {metadata['case_type']['type']}")
   print(f"Prescription: {metadata['prescription_info']['primary_dose']} Gy")
   ```

3. **Update constraints interpretation**
   ```python
   # Constraints vector structure (13 elements):
   # [0]: PTV70 target (1.0 if exists, 0.0 if not)
   # [1]: PTV56 target (0.8 if exists, 0.0 if not)  
   # [2]: PTV50.4 target (0.72 if exists, 0.0 if not)
   # [3-12]: OAR constraints (normalized)
   ```

4. **Verify with notebook**
   ```bash
   jupyter notebook notebooks/verify_npz.ipynb
   ```

---

## [1.0.0] - 2025-01-01

### Added
- Initial preprocessing script `preprocess_dicom_rt.py`
- DICOM-RT loading (CT, RS, RD files)
- SimpleITK-based resampling to fixed grid (512×512×256 @ 1×1×2mm)
- Contour-to-mask conversion with polygon filling
- OAR mapping via `oar_mapping.json`
- Basic debug PNG output
- Batch processing capability

### Known Limitations (v1.0)
- Hardcoded dose normalization to 70 Gy
- Fixed AAPM constraints (not case-specific)
- No validation checks before saving
- No prescription extraction from RP file
- Limited metadata storage
- No SIB case type classification

---

## File Versions

| File | Current Version | Last Updated |
|------|-----------------|--------------|
| `preprocess_dicom_rt.py` | 1.0.0 | 2025-01-01 |
| `preprocess_dicom_rt_v2.py` | 2.0.0 | 2025-01-09 |
| `oar_mapping.json` | 1.0.0 | 2025-01-01 |
| `verify_npz.ipynb` | 1.0.0 | 2025-01-09 |

---

## Links

- [Preprocessing Assumptions](docs/preprocessing_assumptions.md)
- [Verification Notebook Guide](docs/verify_npz.md)
- [Project README](README.md)
