> **CURRENT REFERENCE** — These preprocessing assumptions are active and accurate for `preprocess_dicom_rt_v2.3.py`.
> For project strategy: see `.claude/instructions.md`.

# Preprocessing Assumptions and Requirements

This document lists all assumptions made by the preprocessing pipeline. Data that violates these assumptions may produce incorrect results or errors.

---

## Treatment Protocol Assumptions

### Disease Site
- **Prostate cancer** (intact prostate, not post-prostatectomy)
- May include seminal vesicles in PTV

### Fractionation
- **28 fractions** standard
- SIB (Simultaneous Integrated Boost) protocol:
  - PTV70: 70 Gy in 28 fx (2.5 Gy/fx)
  - PTV56: 56 Gy in 28 fx (2.0 Gy/fx)
  - PTV50.4: 50.4 Gy in 28 fx (1.8 Gy/fx) - if applicable

### Treatment Technique
- **VMAT** (Volumetric Modulated Arc Therapy)
- Typically 2 full arcs (clockwise + counter-clockwise)
- Single isocenter
- 6 MV photons (some cases may use 10 MV)

---

## Structure Assumptions

### Required Structures (8 channels)

| Channel | Structure | Required | Notes |
|---------|-----------|----------|-------|
| 0 | PTV70 | **Yes** | High-dose PTV |
| 1 | PTV56 | Yes* | *Can be empty if not SIB |
| 2 | Prostate | **Yes** | Used for centering |
| 3 | Rectum | **Yes** | - |
| 4 | Bladder | **Yes** | - |
| 5 | Femur_L | Recommended | May be absent |
| 6 | Femur_R | Recommended | May be absent |
| 7 | Bowel | Recommended | May be absent |

### Structure Naming

The script searches for common aliases (case-insensitive):

```python
STRUCTURE_ALIASES = {
    'PTV70': ['ptv70', 'ptv_70', 'ptv70gy', 'ptv 70', 'ptvhigh', 'ptv_high'],
    'PTV56': ['ptv56', 'ptv_56', 'ptv56gy', 'ptv 56', 'ptvlow', 'ptv_low', 'ptv_intermediate'],
    'Prostate': ['prostate', 'ctv', 'ctv_prostate', 'gland'],
    'Rectum': ['rectum', 'rect', 'rectal', 'rectum_wall'],
    'Bladder': ['bladder', 'blad', 'bladder_wall'],
    'Femur_L': ['femur_l', 'left_femur', 'femoral_l', 'l_femur', 'lt_femur', 'fem_l'],
    'Femur_R': ['femur_r', 'right_femur', 'femoral_r', 'r_femur', 'rt_femur', 'fem_r'],
    'Bowel': ['bowel', 'bowel_bag', 'small_bowel', 'large_bowel', 'bowel_cavity'],
}
```

### Structure Geometry

- Structures should be **closed contours** (not open lines)
- Structures should **not extend beyond CT FOV** significantly
- PTV should be **inside the body** (not extending into air)

---

## Dose Assumptions

### Dose Grid
- 3D dose distribution covering PTV with margin
- May be lower resolution than CT (common: 2.5-3mm)
- Will be resampled to 1×1×2mm

### Dose Values
- **Physical dose** (not biologically corrected)
- In **Gy** after applying DoseGridScaling
- **Sum of all beams/arcs** (not individual beam doses)

### Prescription
- Prescription extracted from RTPLAN DoseReferenceSequence
- If not found, falls back to maximum dose in PTV70 × 1.02
- Can be overridden with `--prescription_gy`

### Expected Dose Characteristics
- PTV70 mean dose: ~70 Gy (normalized to ~1.0)
- PTV70 D95: >66.5 Gy (>95% prescription)
- Hot spots: <107% of prescription (75 Gy)
- OAR doses: Within clinical constraints

---

## CT Assumptions

### Acquisition
- **Helical CT** with patient supine
- **Treatment planning CT** (not diagnostic)
- Includes **full pelvis** (L4 to below ischial tuberosities)

### Technical Parameters
- Slice thickness: 2-3 mm typical
- In-plane resolution: ~1 mm typical
- Field of view: Includes entire pelvis
- HU range: -1000 (air) to +3000 (bone/metal)

### Artifacts
- **No significant metal artifacts** in treatment region
- Minor hip prosthesis artifacts may be acceptable if away from PTV
- Table should not be in the CT data (or will be clipped)

---

## Geometric Assumptions

### Patient Position
- **Head-first supine (HFS)**
- Standard DICOM coordinate system

### Field of View
- CT FOV should include:
  - All target structures (PTV70, PTV56, Prostate)
  - All OARs (Rectum, Bladder, Femurs, Bowel)
  - At least 5cm margin around PTV in all directions

### Centering
- Output volume centered on **Prostate/PTV70 centroid**
- If prostate extends beyond output grid, truncation will occur
- Truncation percentage is logged and stored in metadata

---

## DICOM Assumptions

### File Organization
- All files for one patient in **single directory** or subdirectories
- CT series: Multiple .dcm files (one per slice)
- RTSTRUCT: Single .dcm file
- RTDOSE: Single .dcm file
- RTPLAN: Single .dcm file

### DICOM Compliance
- Files should be **valid DICOM** (not proprietary formats)
- Required DICOM tags must be present:
  - CT: PixelData, ImagePositionPatient, ImageOrientationPatient, PixelSpacing, SliceThickness
  - RTSTRUCT: ROIContourSequence, StructureSetROISequence
  - RTDOSE: PixelData, DoseGridScaling, GridFrameOffsetVector
  - RTPLAN: BeamSequence, DoseReferenceSequence

### References
- RTSTRUCT must reference the CT series
- RTDOSE must reference the RTPLAN
- All files should share the same FrameOfReferenceUID

---

## MLC Assumptions (Phase 2 Data)

### MLC Type
- **120-leaf MLC** typical (60 leaf pairs)
- Varian Millennium or TrueBeam style
- Central 40 pairs: 5mm width
- Outer 20 pairs: 10mm width (if applicable)

### Control Points
- Typically **178 control points** per arc
- 2° gantry spacing
- First and last control points may have zero MU

### Data Format
- LeafJawPositions in mm from isocenter
- Bank A (negative X): First 60 values
- Bank B (positive X): Last 60 values
- Negative values = leaves on patient left side

---

## Validation Checks

The preprocessing script checks these assumptions and warns if violated:

### Critical (will fail)
- Missing required structures (PTV70, Prostate, Rectum, Bladder)
- Empty dose grid
- CT/Dose size mismatch after resampling

### Warning (will proceed with caution)
- Prescription not found in DICOM (uses fallback)
- Structure extends beyond FOV (>10% truncation)
- SDF validation fails
- Registration check fails (dose-structure misalignment suspected)

### Informational (logged only)
- Optional structures missing (Femurs, Bowel)
- Original HU range outside expected [-1000, 3000]
- Dose hot spots >110% of prescription

---

## Output Guarantees

If preprocessing completes successfully, the output guarantees:

| Property | Guarantee |
|----------|-----------|
| CT shape | (512, 512, 256) |
| CT range | [0, 1] |
| Dose shape | (512, 512, 256) |
| Dose range | [0, max] where max ~1.0-1.1 |
| Masks shape | (8, 512, 512, 256) |
| Masks values | {0, 1} uint8 |
| SDF shape | (8, 512, 512, 256) |
| SDF range | [-1, 1] float32 |
| Constraints shape | (13,) |
| Spacing | (1.0, 1.0, 2.0) mm |
| Centering | Prostate centroid at grid center |

---

## Known Limitations

### Not Supported
- Post-prostatectomy cases (no prostate structure)
- Brachytherapy boost cases
- Stereotactic treatments (SBRT)
- Non-prostate disease sites
- Multiple isocenters
- Electron or proton treatments

### Limited Support
- 3-level SIB (PTV50.4 extraction basic)
- Non-standard MLC configurations
- FFF (flattening-filter-free) beams
- Very large patients (>50cm AP diameter)

### Planned Improvements
- PTV50.4 full support for 3-level SIB
- Automatic structure name mapping via ML
- Support for post-prostatectomy cases
- Multi-institution structure naming conventions
