# Preprocessing Assumptions and Limitations

## Document Purpose

This document explicitly states the assumptions, limitations, and known issues in the `preprocess_dicom_rt.py` script. Understanding these is critical before training the diffusion model or scaling to HPC.

**Last Updated:** January 2025  
**Phase:** 1 (Dose Prediction)

---

## Phase 1 Scope

The current preprocessing is designed for **Phase 1: Dose Prediction**, which aims to:
- Generate anatomically-plausible dose distributions from CT + contours
- Validate the diffusion model architecture on prostate VMAT cases
- Establish feasibility before adding beam/MLC parameters

**Phase 1 does NOT include:**
- Beam geometry extraction (arc angles, MU, collimator)
- MLC leaf sequences
- Fluence maps
- Full VMAT plan generation

These will be added in Phase 2+ via modular append scripts (no re-preprocessing required).

---

## Critical Assumptions

### 1. Dose Normalization

**Current Implementation:**
```python
dose_volume = sitk.GetArrayFromImage(dose_resampled).transpose(1, 2, 0) / 70.0
```

**Assumption:** All cases have a primary prescription of 70 Gy to PTV70.

**Limitation:** For SIB cases with PTV56 (56 Gy) or PTV50.4 (50.4 Gy), these regions will have normalized values of ~0.8 and ~0.72 respectively. The model must learn that these are "correctly dosed" rather than "underdosed."

**Impact:**
- Model sees PTV56 at ~0.8 normalized dose
- Model sees PTV50.4 at ~0.72 normalized dose
- Without explicit prescription encoding, model cannot distinguish between "56 Gy target" and "underdosed 70 Gy target"

**Mitigation (Recommended):**
```python
# Store prescription metadata
prescription_info = {
    'primary_rx': 70.0,
    'ptv56_rx': 56.0 if ptv56_exists else None,
    'ptv50_rx': 50.4 if ptv50_exists else None,
}
np.savez(..., prescription_info=prescription_info)
```

**TODO:** Extract actual prescription from RP file `DoseReferenceSequence` or `FractionGroupSequence`.

---

### 2. Fixed Output Grid

**Current Implementation:**
```python
target_shape = (512, 512, 256)
target_spacing = (1.0, 1.0, 2.0)  # mm
```

**Physical Coverage:**
- X (Left-Right): 512 mm
- Y (Anterior-Posterior): 512 mm  
- Z (Superior-Inferior): 512 mm

**Assumption:** 512 mm FOV centered on prostate is sufficient for all pelvic anatomy.

**Known Risks:**
- Large patients may have femoral heads cropped
- Extended pelvic node cases (PTV50.4) may have superior nodes cropped
- Bladder superior extent may be truncated if very full

**Validation:** The `verify_npz.ipynb` notebook includes boundary truncation checks.

**Acceptable Truncations:**
- Femoral head edges (not dose-relevant)
- Bowel superior extent (if outside treatment field)

**Unacceptable Truncations:**
- PTV70 or PTV56
- Rectum
- Bladder (if clinically relevant portion)

---

### 3. Grid Centering

**Current Implementation:**
```python
# Center on prostate (or PTV70 if prostate not contoured)
prostate_mask = masks[2] if masks[2].sum() > 0 else masks[0]
centroid_phys_x = np.mean(phys_x)  # Physical centroid
centroid_phys_y = np.mean(phys_y)
centroid_phys_z = np.mean(phys_z)
```

**Assumption:** Centering on prostate/PTV70 captures all relevant anatomy.

**Limitation:** If prostate is positioned asymmetrically (e.g., shifted left), the resampled grid may crop contralateral structures.

---

### 4. CT Normalization

**Current Implementation:**
```python
ct_volume = np.clip(ct_array, -1000, 3000) / 4000 + 0.5
```

**Mapping:**
| HU Value | Tissue | Normalized |
|----------|--------|------------|
| -1000 | Air | 0.0 |
| 0 | Water | 0.5 |
| +3000 | Dense bone | 1.0 |

**Limitation:** HU > 3000 (metal implants, fiducials) will be clipped.

**Impact:** Hip prostheses or fiducial markers will appear as uniform dense bone. This is generally acceptable for diffusion model training but may affect dose calculation accuracy in hybrid physics extensions.

---

### 5. Constraints Vector

**Current Implementation:**
```python
AAPM_CONSTRAINTS = np.array([
    70.0,  # PTV70 mean (Gy)
    50.0,  # Rectum V50 (%)
    35.0,  # Rectum V60 (%)
    20.0,  # Rectum V70 (%)
    50.0,  # Bladder V65 (%)
    35.0,  # Bladder V70 (%)
    25.0,  # Bladder V75 (%)
    10.0,  # Femur V50 (%)
    195.0, # Bowel V45 (cc)
    45.0   # Spinal cord max (Gy)
]) / 100.0

ptv_type = np.array([1.0, 0.0, 0.0])
constraints = np.concatenate([AAPM_CONSTRAINTS, ptv_type])
```

**Assumption:** All cases use identical AAPM/QUANTEC constraints.

**Limitations:**
1. Constraints are fixed, not case-specific
2. Model cannot learn constraint-conditional generation
3. `ptv_type` vector meaning is undefined
4. DVH constraints (V50, V60) are scalars, not curves
5. Absolute vs. relative volumes mixed (cc vs. %)

**Impact:** The model will learn unconditional dose prediction for "typical prostate VMAT" rather than constraint-driven optimization.

**Phase 2 TODO:** Extract case-specific constraints from RP file and encode properly.

---

### 6. Missing Beam Parameters

**Not Currently Extracted:**
- Arc start/stop angles
- Gantry rotation direction (CW/CCW)
- Collimator angles
- Couch angles
- Number of arcs
- MU per arc
- MLC leaf positions
- Control point sequences

**Impact:** Cannot train for deliverable VMAT plan generation in Phase 1.

**Phase 2 Plan:** Add `extract_beam_geometry()` function to pull from RP file `BeamSequence` and `ControlPointSequence`.

---

### 7. Interpolation Methods

**Current Implementation:**
| Data | Interpolator | Rationale |
|------|--------------|-----------|
| CT | `sitkLinear` | Smooth HU gradients |
| Dose | `sitkLinear` | Smooth dose gradients |
| Masks | `sitkNearestNeighbor` | Preserve binary boundaries |

**Known Issue:** Linear interpolation of dose may create small negative values at sharp gradients.

**Mitigation:**
```python
dose_volume = np.maximum(dose_volume, 0)  # Clip negative artifacts
```

---

### 8. Coordinate System

**Assumption:** All DICOM files use standard IEC 61217 patient coordinate system:
- X: Patient left (+) to right (-)
- Y: Patient posterior (+) to anterior (-)
- Z: Patient inferior (+) to superior (-)

**SimpleITK Handling:** Origin and spacing preserved through resampling.

**Validation:** Visual inspection in `verify_npz.ipynb` confirms anatomy orientation.

---

## Data Filtering

### Required Structures

Cases are processed only if:
```python
ptv70_sum > 0 and ptv56_sum > 0  # Unless --relax_filter
```

**Rationale:** Ensure consistent training data for SIB cases.

**Override:** Use `--relax_filter` flag for PTV70-only cases.

### Prescription Filter (External)

Cases should be pre-filtered to:
- 28 fractions
- 70 Gy to PTV70 (2.5 Gy/fx)
- 56 Gy to PTV56 if present (2.0 Gy/fx)
- 50.4 Gy to PTV50.4 if present (1.8 Gy/fx)

This filtering is done externally before preprocessing.

---

## Known Issues

### Issue #1: SIB Dose Interpretation

**Problem:** Model may interpret PTV56 dose of 0.8 as "80% of target" rather than "correct 56 Gy prescription."

**Status:** Documented; requires prescription encoding in Phase 2.

**Workaround:** Include prescription targets in constraints vector.

### Issue #2: No Registration Validation

**Problem:** No automated check that dose grid aligns with CT/contours.

**Status:** Added to `verify_npz.ipynb` validation checks.

**Check:** Mean dose inside PTV should be higher than outside.

### Issue #3: Variable Contour Names

**Problem:** Institutions use different naming conventions.

**Status:** Handled via `oar_mapping.json` with extensive variations.

**Maintenance:** Add new variations as encountered.

---

## Recommendations Before HPC Scaling

### Must Do
- [ ] Run `verify_npz.ipynb` batch validation on all cases
- [ ] Manual review of any failed cases
- [ ] Confirm no critical registration failures
- [ ] Document any excluded cases with reasons

### Should Do
- [ ] Add prescription metadata to npz files
- [ ] Implement boundary truncation warnings in preprocessing
- [ ] Create case manifest with SIB type classification

### Nice to Have
- [ ] Extract beam geometry (prep for Phase 2)
- [ ] Add signed distance fields for masks
- [ ] Implement pymedphys DVH validation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial documentation |

---

## References

- AAPM TG-101: Stereotactic Body Radiation Therapy
- QUANTEC: Quantitative Analysis of Normal Tissue Effects in the Clinic
- DICOM RT Standard: PS3.3 Information Object Definitions
- SimpleITK Documentation: https://simpleitk.readthedocs.io/
