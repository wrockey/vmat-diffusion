> **CURRENT REFERENCE** — This document is active and accurate for the NPZ verification notebook.
> For project strategy: see `.claude/instructions.md`.

# NPZ Verification Notebook Documentation

## Overview

The `verify_npz.ipynb` notebook provides visual and quantitative verification of preprocessed `.npz` files before training the VMAT diffusion model. It is designed to catch preprocessing errors, registration issues, and data quality problems before scaling to HPC.

## Quick Start

```bash
# Navigate to notebooks directory
cd ~/vmat-diffusion-project/notebooks

# Launch Jupyter
jupyter notebook verify_npz.ipynb
```

1. Update `NPZ_DIR` path in Cell 2 to point to your `processed_npz` folder
2. Run all cells sequentially
3. Change `CASE_INDEX` to inspect different cases
4. Set `RUN_BATCH = True` in the final cell to validate all files

## Dependencies

```bash
pip install numpy matplotlib ipywidgets
```

Optional for interactive sliders:
```bash
jupyter nbextension enable --py widgetsnbextension
```

## Expected .npz Structure

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `ct` | (512, 512, 256) | float32 | CT volume, normalized [0, 1] |
| `dose` | (512, 512, 256) | float32 | Dose volume, normalized to 70 Gy |
| `masks` | (8, 512, 512, 256) | uint8 | Binary masks for structures |
| `constraints` | (13,) | float32 | AAPM constraints + PTV type vector |

### Mask Channel Mapping

| Channel | Structure | Expected in All Cases |
|---------|-----------|----------------------|
| 0 | PTV70 | ✓ Yes (required) |
| 1 | PTV56 | ~90% of cases (SIB) |
| 2 | Prostate | Most cases |
| 3 | Rectum | ✓ Yes |
| 4 | Bladder | ✓ Yes |
| 5 | Femur_L | Most cases |
| 6 | Femur_R | Most cases |
| 7 | Bowel | Variable |

### Normalization Reference

| Data | Normalization | To Convert Back |
|------|---------------|-----------------|
| CT | `(HU + 1000) / 4000 + 0.5` → [0, 1] | `(val - 0.5) * 4000 - 1000` |
| Dose | `Gy / 70.0` | `val * 70.0` |

## Notebook Sections

### 1. Load and Inspect (Cells 3-4)

Lists available `.npz` files and loads a selected case with basic shape validation.

**What to check:**
- All expected keys present (`ct`, `dose`, `masks`, `constraints`)
- Shapes match expected dimensions

### 2. Validation Checks (Cell 5)

Automated QA checks with pass/fail indicators.

| Check | Pass Criteria | Failure Indicates |
|-------|---------------|-------------------|
| `ct_range_valid` | CT values in [0, 1] | Normalization error |
| `dose_nonneg` | Dose ≥ -0.01 | Interpolation artifact |
| `dose_reasonable` | Max dose < 1.5 (105 Gy) | Hotspot or scaling error |
| `ptv70_exists` | PTV70 voxels > 0 | Missing contour or mapping error |
| `ptv70_dose_adequate` | Mean dose 0.9-1.1 | Underdose, wrong Rx, or registration error |
| `dose_higher_in_ptv` | Mean in PTV > mean outside | **Critical**: Registration failure |

**SIB-specific checks:**
- `ptv56_only_dose_mean`: Mean dose in PTV56 excluding PTV70 overlap
- Expected: ~0.8 (56/70 Gy) for properly planned SIB cases

### 3. Structure Statistics (Cell 6)

Calculates for each structure:
- **Voxels**: Total voxel count
- **Volume (cc)**: Physical volume (assumes 1×1×2 mm spacing)
- **Dose Mean/Min/Max**: Normalized dose statistics
- **D95**: Dose to 95% of volume (minimum dose to most of structure)
- **D5**: Dose to 5% of volume (near-maximum dose)

**Expected values for well-planned prostate VMAT:**

| Structure | Mean Dose | D95 | Max Dose |
|-----------|-----------|-----|----------|
| PTV70 | ~1.0 (70 Gy) | >0.95 | <1.10 |
| PTV56 (excl PTV70) | ~0.8 (56 Gy) | >0.75 | <0.95 |
| Rectum | <0.5 (35 Gy) | - | <1.05 |
| Bladder | <0.5 (35 Gy) | - | <1.05 |

### 4. Visual Inspection: Axial Slices (Cells 7-8)

Four-panel display for each slice:
1. **CT only**: Grayscale anatomy
2. **Dose only**: Colorwash (jet colormap, 0-75 Gy)
3. **CT + Dose overlay**: Registration verification
4. **CT + Structure contours**: Contour accuracy check

**What to look for:**
- Dose hotspot centered on PTV70
- Dose gradient falling off in OARs
- Contours matching visible anatomy
- No obvious misregistration (dose shifted from anatomy)

### 5. Interactive Slice Viewer (Cell 9)

Scroll through all Z slices with a slider. Requires `ipywidgets`.

**Best for:**
- Checking superior/inferior extent of structures
- Finding slices with anomalies
- Verifying dose coverage through entire PTV

### 6. Orthogonal Views (Cell 10)

Six-panel display showing axial, coronal, and sagittal views:
- Row 1: CT + dose overlay
- Row 2: CT + structure contours

Yellow crosshairs indicate the slice positions.

**What to look for:**
- PTV coverage in all three planes
- Dose gradient around rectum (posterior) and bladder (superior)
- No unexpected dose in femoral heads

### 7. Dose-Volume Histogram (Cell 11)

Cumulative DVH curves for all structures with prescription reference lines.

**How to interpret:**
- PTV70 curve should be steep and pass through (70 Gy, ~95%)
- PTV56 curve should pass through (56 Gy, ~95%)
- OAR curves should fall off rapidly at lower doses
- Rectum V70 should be low (<20%)

### 8. Dose Profiles (Cell 12)

Line profiles through PTV center in X (L-R), Y (A-P), and Z (S-I) directions.

**What to look for:**
- Flat dose plateau within PTV (shaded red region)
- Sharp falloff at PTV edges
- Prescription lines (70 Gy, 56 Gy) for reference

### 9. Summary Report (Cell 13)

Text summary including:
- Overall pass/fail status
- Case type (single Rx vs. SIB)
- Key dose metrics
- Any warnings

### 10. Batch Validation (Cell 14)

Quick validation across all `.npz` files. Set `RUN_BATCH = True` to enable.

**Output:**
- Table with pass/fail status for each file
- PTV70 mean dose
- Warning count
- Overall summary (X/Y cases passed)

## Common Issues and Solutions

### Issue: PTV70 mean dose < 0.9

**Possible causes:**
1. Wrong prescription assumed (not 70 Gy)
2. Plan was underdosed
3. Dose grid misregistration

**Solution:** Check original DICOM RD file prescription; verify dose grid alignment in TPS.

### Issue: Dose higher outside PTV than inside

**This is critical** - indicates registration failure.

**Possible causes:**
1. Dose grid origin/spacing mismatch
2. CT and dose from different coordinate systems
3. Incorrect GridFrameOffsetVector handling

**Solution:** Review `preprocess_dicom_rt.py` dose loading section; compare with TPS visualization.

### Issue: Structure touches boundary

**Possible causes:**
1. Patient anatomy extends beyond 512mm FOV
2. Centering on prostate missed peripheral structures

**Solution:** 
- If femoral heads only: usually acceptable
- If PTV/rectum/bladder: may need to expand grid or adjust centering

### Issue: Empty structure (0 voxels)

**Possible causes:**
1. Structure not contoured in original plan
2. Contour name not in `oar_mapping.json` variations
3. Contour entirely outside resampled grid

**Solution:** Check `oar_mapping.json`; verify structure exists in RS file.

## Validation Workflow

### Before Training (~100 cases)

1. Run notebook on 5-10 random cases manually
2. Check all visualizations carefully
3. Run batch validation on all cases
4. Investigate any failures
5. Document known acceptable deviations

### Before Scaling to HPC (1000+ cases)

1. All cases should pass batch validation
2. Manual review of any edge cases
3. Confirm SIB dose levels are as expected
4. No critical registration failures
5. Document case exclusions with reasons

## Extending the Notebook

### Adding PTV50.4 Support

When you add channel 8 for PTV50.4:

```python
# In STRUCTURE_NAMES
8: 'PTV50.4',

# In STRUCTURE_COLORS
8: (0.5, 0.0, 0.5, 0.5),  # Purple

# In validation
ptv50_mask = masks[8] > 0
if ptv50_mask.sum() > 0:
    ptv50_only = ptv50_mask & ~ptv70_mask & ~ptv56_mask
    # Expected: ~0.72 (50.4/70)
```

### Adding pymedphys DVH

For clinical-grade DVH calculation:

```python
from pymedphys import dvh

# Requires dose in Gy and structure mask
dvh_calc = dvh.calculate(dose * 70, masks[ch], voxel_size=(1, 1, 2))
```

## File Locations

```
vmat-diffusion-project/
├── notebooks/
│   └── verify_npz.ipynb      # This notebook
├── docs/
│   └── verify_npz.md         # This documentation
├── data/
│   └── processed_npz/        # Input .npz files
└── oar_mapping.json          # Structure name mappings
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial release with 10 sections |

## Contact

For issues with this notebook, open a GitHub issue or contact the project maintainers.
