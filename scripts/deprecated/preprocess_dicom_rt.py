"""
DEPRECATED: Use preprocess_dicom_rt_v2.2.py instead.

This script is kept for reference only. The v2.2 version includes:
- Signed Distance Fields (SDFs) for neural network training
- Full MLC extraction for Phase 2
- Compressed .npz output (~50% smaller)
- Enhanced validation and beam geometry extraction

Preprocess DICOM-RT VMAT plans to .npz for diffusion model training.

Functionality: Resamples/aligns CT, masks, dose to fixed grid (512x512x256 @ 1x1x2mm),
centers on prostate/PTV70, normalizes, and saves with constraints and metadata.

Version: 2.0 - Adds prescription extraction, SIB support, validation checks

Assumptions: See docs/preprocessing_assumptions.md

Usage: python preprocess_dicom_rt_v2.2.py [flags]
"""
import warnings
warnings.warn(
    "preprocess_dicom_rt.py is deprecated. Use preprocess_dicom_rt_v2.2.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import json
import numpy as np
import pydicom
from skimage.draw import polygon
import warnings
import argparse
import SimpleITK as sitk
from datetime import datetime

# Force non-interactive backend to avoid Qt/Wayland/X11 issues on headless systems
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Relax pydicom validation for Pinnacle DS VR precision issues
import pydicom.config
pydicom.config.settings.reading_validation_mode = pydicom.config.WARN

# =============================================================================
# CONSTANTS
# =============================================================================

# Primary prescription for normalization (all cases should be 70 Gy to PTV70)
PRIMARY_PRESCRIPTION_GY = 70.0

# Expected SIB prescription levels
SIB_PRESCRIPTIONS = {
    'PTV70': 70.0,
    'PTV56': 56.0,
    'PTV50.4': 50.4,
}

# Default AAPM/QUANTEC constraints for prostate VMAT
# Format: (name, type, value, unit)
DEFAULT_CONSTRAINTS = {
    'ptv70_d95': 66.5,      # Gy - D95 >= 95% of Rx
    'rectum_v50': 50.0,     # % volume receiving 50 Gy
    'rectum_v60': 35.0,     # % volume receiving 60 Gy
    'rectum_v70': 20.0,     # % volume receiving 70 Gy
    'rectum_max': 75.0,     # Gy - max point dose
    'bladder_v65': 50.0,    # % volume receiving 65 Gy
    'bladder_v70': 35.0,    # % volume receiving 70 Gy
    'bladder_v75': 25.0,    # % volume receiving 75 Gy
    'femur_v50': 10.0,      # % volume receiving 50 Gy
    'bowel_v45': 195.0,     # cc receiving 45 Gy
    'spinal_cord_max': 45.0,  # Gy
}

# Structure channel mapping (must match oar_mapping.json)
STRUCTURE_CHANNELS = {
    0: 'PTV70',
    1: 'PTV56', 
    2: 'Prostate',
    3: 'Rectum',
    4: 'Bladder',
    5: 'Femur_L',
    6: 'Femur_R',
    7: 'Bowel',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_mapping(mapping_file='oar_mapping.json'):
    """
    Load the OAR mapping JSON file for contour variations.

    Args:
        mapping_file (str): Path to the JSON file.

    Returns:
        dict: Loaded mapping.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"{mapping_file} not found; run generate_mapping_json.py and edit")
    with open(mapping_file, 'r') as f:
        return json.load(f)


def get_ct_volume_and_metadata(plan_dir):
    """
    Load and stack CT slices, sort by Z, compute metadata.

    Args:
        plan_dir (str): Path to the plan directory.

    Returns:
        tuple: (ct_volume, spacing, position, slice_z, ct_ds)

    Raises:
        ValueError: If no CT files found.
    """
    ct_files = [f for f in os.listdir(plan_dir) if f.startswith('CT')]
    if not ct_files:
        raise ValueError(f"No CT files in {plan_dir}")
    ct_paths = [os.path.join(plan_dir, f) for f in ct_files]
    ct_ds = [pydicom.dcmread(p) for p in ct_paths]
    # Sort by increasing z-position
    sorted_indices = np.argsort([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    ct_ds = [ct_ds[i] for i in sorted_indices]
    slice_z = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    print(f"CT z range: min {slice_z.min():.2f}, max {slice_z.max():.2f}, slices {len(slice_z)}, spacing {np.mean(np.diff(slice_z)):.2f}")
    ct_volume = np.stack([ds.pixel_array.astype(np.float32) for ds in ct_ds], axis=2)
    spacing = np.array([float(ct_ds[0].PixelSpacing[0]), float(ct_ds[0].PixelSpacing[1]), np.mean(np.diff(slice_z))])
    position = np.array([float(x) for x in ct_ds[0].ImagePositionPatient])
    return ct_volume, spacing, position, slice_z, ct_ds


def extract_prescription_dose(plan_dir):
    """
    Extract prescription dose from RT Plan file.
    
    Attempts multiple DICOM tags to find the prescription:
    1. DoseReferenceSequence.TargetPrescriptionDose
    2. FractionGroupSequence with BeamDose * NumberOfFractionsPlanned
    3. Falls back to default 70 Gy if not found
    
    Args:
        plan_dir (str): Path to the plan directory.
        
    Returns:
        dict: Prescription information including:
            - primary_dose: Primary prescription dose in Gy
            - fractions: Number of fractions
            - dose_per_fraction: Dose per fraction in Gy
            - source: Where the prescription was found
    """
    result = {
        'primary_dose': PRIMARY_PRESCRIPTION_GY,
        'fractions': 28,
        'dose_per_fraction': PRIMARY_PRESCRIPTION_GY / 28,
        'source': 'default'
    }
    
    try:
        rp_files = [f for f in os.listdir(plan_dir) if f.startswith('RP')]
        if not rp_files:
            warnings.warn(f"No RP file found in {plan_dir}, using default prescription")
            return result
            
        rp = pydicom.dcmread(os.path.join(plan_dir, rp_files[0]))
        
        # Method 1: DoseReferenceSequence (most reliable)
        if hasattr(rp, 'DoseReferenceSequence'):
            for dose_ref in rp.DoseReferenceSequence:
                if hasattr(dose_ref, 'TargetPrescriptionDose'):
                    result['primary_dose'] = float(dose_ref.TargetPrescriptionDose)
                    result['source'] = 'DoseReferenceSequence'
                    break
        
        # Method 2: FractionGroupSequence
        if result['source'] == 'default' and hasattr(rp, 'FractionGroupSequence'):
            for fg in rp.FractionGroupSequence:
                if hasattr(fg, 'NumberOfFractionsPlanned'):
                    result['fractions'] = int(fg.NumberOfFractionsPlanned)
                    
                if hasattr(fg, 'ReferencedBeamSequence'):
                    total_beam_dose = 0
                    for ref_beam in fg.ReferencedBeamSequence:
                        if hasattr(ref_beam, 'BeamDose'):
                            total_beam_dose += float(ref_beam.BeamDose)
                    
                    if total_beam_dose > 0:
                        result['primary_dose'] = total_beam_dose * result['fractions']
                        result['source'] = 'FractionGroupSequence'
                        break
        
        # Calculate dose per fraction
        if result['fractions'] > 0:
            result['dose_per_fraction'] = result['primary_dose'] / result['fractions']
            
        print(f"Prescription: {result['primary_dose']:.1f} Gy in {result['fractions']} fx "
              f"({result['dose_per_fraction']:.2f} Gy/fx) [source: {result['source']}]")
              
    except Exception as e:
        warnings.warn(f"Error extracting prescription: {e}, using default")
    
    return result


def determine_case_type(masks):
    """
    Determine the case type based on which PTVs are present.
    
    Args:
        masks: Mask array with shape (n_structures, Y, X, Z)
        
    Returns:
        dict: Case type information including:
            - type: 'single_rx', 'sib_2level', or 'sib_3level'
            - ptv70_exists: bool
            - ptv56_exists: bool
            - ptv50_exists: bool (placeholder for future)
    """
    ptv70_exists = masks[0].sum() > 0 if masks.shape[0] > 0 else False
    ptv56_exists = masks[1].sum() > 0 if masks.shape[0] > 1 else False
    # PTV50.4 would be channel 8 if added
    ptv50_exists = False  # Placeholder for future
    
    if ptv50_exists:
        case_type = 'sib_3level'
    elif ptv56_exists:
        case_type = 'sib_2level'
    else:
        case_type = 'single_rx'
    
    return {
        'type': case_type,
        'ptv70_exists': ptv70_exists,
        'ptv56_exists': ptv56_exists,
        'ptv50_exists': ptv50_exists,
    }


def build_constraints_vector(case_type_info, prescription_info):
    """
    Build the constraints vector for model conditioning.
    
    This encodes both the prescription targets and OAR constraints
    in a format suitable for model conditioning.
    
    Args:
        case_type_info: Dict from determine_case_type()
        prescription_info: Dict from extract_prescription_dose()
        
    Returns:
        np.array: Constraints vector
        dict: Human-readable constraint descriptions
    """
    # Prescription targets (normalized to primary Rx)
    primary_rx = prescription_info['primary_dose']
    
    prescription_targets = np.array([
        SIB_PRESCRIPTIONS['PTV70'] / primary_rx if case_type_info['ptv70_exists'] else 0.0,
        SIB_PRESCRIPTIONS['PTV56'] / primary_rx if case_type_info['ptv56_exists'] else 0.0,
        SIB_PRESCRIPTIONS.get('PTV50.4', 0) / primary_rx if case_type_info['ptv50_exists'] else 0.0,
    ])
    
    # OAR constraints (normalized to [0, 1] range for model)
    oar_constraints = np.array([
        DEFAULT_CONSTRAINTS['rectum_v50'] / 100.0,   # Volume constraints as fractions
        DEFAULT_CONSTRAINTS['rectum_v60'] / 100.0,
        DEFAULT_CONSTRAINTS['rectum_v70'] / 100.0,
        DEFAULT_CONSTRAINTS['rectum_max'] / primary_rx,  # Dose constraints normalized to Rx
        DEFAULT_CONSTRAINTS['bladder_v65'] / 100.0,
        DEFAULT_CONSTRAINTS['bladder_v70'] / 100.0,
        DEFAULT_CONSTRAINTS['bladder_v75'] / 100.0,
        DEFAULT_CONSTRAINTS['femur_v50'] / 100.0,
        DEFAULT_CONSTRAINTS['bowel_v45'] / 500.0,  # Normalize cc to reasonable range
        DEFAULT_CONSTRAINTS['spinal_cord_max'] / primary_rx,
    ])
    
    constraints = np.concatenate([prescription_targets, oar_constraints])
    
    descriptions = {
        'prescription_targets': {
            'ptv70': prescription_targets[0],
            'ptv56': prescription_targets[1],
            'ptv50': prescription_targets[2],
        },
        'oar_constraints': DEFAULT_CONSTRAINTS.copy(),
    }
    
    return constraints.astype(np.float32), descriptions


def get_roi_numbers(rtstruct, variations):
    """
    Get ROI numbers matching variations in the structure set.

    Args:
        rtstruct: pydicom Dataset for RS.
        variations (list): List of name variations.

    Returns:
        list: Matching ROI numbers.
    """
    roi_nums = []
    for roi in rtstruct.StructureSetROISequence:
        name = roi.ROIName.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '')
        if name in [v.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '') for v in variations]:
            roi_nums.append(int(roi.ROINumber))
    return roi_nums


def create_mask_from_contours(rtstruct, roi_nums, ct_shape, position, spacing, slice_z, ct_ds):
    """
    Create binary mask from contours for given ROIs.

    Args:
        rtstruct: pydicom Dataset for RS.
        roi_nums (list): ROI numbers to include.
        ct_shape (tuple): Shape of CT volume.
        position (np.array): Image position.
        spacing (np.array): Pixel spacing.
        slice_z (np.array): Z positions of slices.
        ct_ds (list): List of CT slice Datasets.

    Returns:
        np.array: Binary mask (uint8).
    """
    mask = np.zeros(ct_shape, dtype=np.uint8)
    for roi_num in roi_nums:
        try:
            roi_contour = next(c for c in rtstruct.ROIContourSequence if c.ReferencedROINumber == roi_num)
            for cont_seq in roi_contour.ContourSequence:
                points = np.array(cont_seq.ContourData).reshape(-1, 3)
                z = points[0, 2]
                slice_idx = np.argmin(np.abs(slice_z - z))
                if not 0 <= slice_idx < ct_shape[2]:
                    warnings.warn(f"Slice idx {slice_idx} out of bounds for z {z}; clamping")
                    slice_idx = np.clip(slice_idx, 0, ct_shape[2] - 1)
                slice_ds = ct_ds[slice_idx]
                slice_position = np.array([float(x) for x in slice_ds.ImagePositionPatient])
                slice_spacing = np.array([float(x) for x in slice_ds.PixelSpacing])
                pix_x = (points[:, 0] - slice_position[0]) / slice_spacing[0]
                pix_y = (points[:, 1] - slice_position[1]) / slice_spacing[1]
                rr, cc = polygon(pix_y, pix_x, (ct_shape[0], ct_shape[1]))
                rr = np.clip(rr, 0, ct_shape[0] - 1)
                cc = np.clip(cc, 0, ct_shape[1] - 1)
                mask[rr, cc, slice_idx] = 1
        except StopIteration:
            warnings.warn(f"No contours found for ROI {roi_num}")
    return mask


def resample_image(image, reference, interpolator=sitk.sitkLinear):
    """
    Resample an image to a reference grid using SimpleITK.

    Args:
        image: sitk.Image to resample.
        reference: sitk.Image reference grid.
        interpolator: sitk interpolator (default linear).

    Returns:
        sitk.Image: Resampled image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def validate_preprocessed_data(ct_volume, dose_volume, masks_resampled, prescription_dose):
    """
    Validate preprocessed data before saving.
    
    Args:
        ct_volume: Normalized CT array
        dose_volume: Normalized dose array
        masks_resampled: Resampled mask array
        prescription_dose: Primary prescription in Gy
        
    Returns:
        dict: Validation results with pass/fail for each check
    """
    checks = {}
    
    # CT checks
    checks['ct_range_valid'] = (ct_volume.min() >= 0) and (ct_volume.max() <= 1)
    checks['ct_min'] = float(ct_volume.min())
    checks['ct_max'] = float(ct_volume.max())
    
    # Dose checks
    checks['dose_nonneg'] = dose_volume.min() >= -0.01
    checks['dose_min'] = float(dose_volume.min())
    checks['dose_max'] = float(dose_volume.max())
    checks['dose_reasonable'] = dose_volume.max() < 1.5  # Max < 150% Rx
    
    # PTV70 checks
    ptv70_mask = masks_resampled[0] > 0
    checks['ptv70_exists'] = ptv70_mask.sum() > 0
    
    if checks['ptv70_exists']:
        ptv70_dose = dose_volume[ptv70_mask]
        checks['ptv70_dose_mean'] = float(ptv70_dose.mean())
        checks['ptv70_dose_std'] = float(ptv70_dose.std())
        checks['ptv70_dose_adequate'] = 0.85 < ptv70_dose.mean() < 1.15
    else:
        checks['ptv70_dose_mean'] = None
        checks['ptv70_dose_adequate'] = False
    
    # Registration check: dose should be higher inside PTV
    any_ptv = (masks_resampled[0] > 0) | (masks_resampled[1] > 0)
    if any_ptv.sum() > 0 and (~any_ptv).sum() > 0:
        dose_in_ptv = dose_volume[any_ptv].mean()
        dose_outside_ptv = dose_volume[~any_ptv].mean()
        checks['registration_valid'] = dose_in_ptv > dose_outside_ptv
        checks['dose_in_ptv'] = float(dose_in_ptv)
        checks['dose_outside_ptv'] = float(dose_outside_ptv)
    else:
        checks['registration_valid'] = None
    
    # Boundary truncation check
    boundary_warnings = []
    for ch, name in STRUCTURE_CHANNELS.items():
        if ch < masks_resampled.shape[0]:
            mask = masks_resampled[ch]
            if mask.sum() == 0:
                continue
            
            # Check Z boundaries (most important for superior/inferior)
            if mask[:, :, 0].sum() > 0:
                boundary_warnings.append(f"{name} touches inferior boundary")
            if mask[:, :, -1].sum() > 0:
                boundary_warnings.append(f"{name} touches superior boundary")
    
    checks['boundary_warnings'] = boundary_warnings
    
    # Overall pass/fail
    critical_checks = ['ct_range_valid', 'dose_nonneg', 'ptv70_exists', 'ptv70_dose_adequate']
    checks['all_critical_passed'] = all(checks.get(c, False) for c in critical_checks)
    
    if checks.get('registration_valid') == False:
        checks['all_critical_passed'] = False
        warnings.warn("CRITICAL: Registration check failed - dose higher outside PTV!")
    
    return checks


def preprocess_dicom_rt(plan_dir, output_dir, structure_map, target_shape=(512, 512, 256), 
                        target_spacing=(1.0, 1.0, 2.0), relax_filter=False, skip_plots=False,
                        strict_validation=False):
    """
    Process a single DICOM-RT plan directory.

    Args:
        plan_dir (str): Path to case dir with DICOM files.
        output_dir (str): Output directory for .npz and PNGs.
        structure_map (dict): OAR mapping from JSON.
        target_shape (tuple): Fixed output shape (default 512x512x256).
        target_spacing (tuple): Target voxel spacing (default 1x1x2 mm).
        relax_filter (bool): Process even with zero PTV sums.
        skip_plots (bool): Skip debug PNGs.
        strict_validation (bool): Fail on any validation warning.

    Returns:
        str: Path to output .npz or None on error.
    """
    os.makedirs(output_dir, exist_ok=True)
    plan_id = os.path.basename(plan_dir)
    
    try:
        # =================================================================
        # Load CT
        # =================================================================
        ct_volume, ct_spacing, position, slice_z, ct_ds = get_ct_volume_and_metadata(plan_dir)
        
        # =================================================================
        # Extract prescription
        # =================================================================
        prescription_info = extract_prescription_dose(plan_dir)
        normalization_dose = prescription_info['primary_dose']
        
        # Warn if prescription differs significantly from expected 70 Gy
        if abs(normalization_dose - PRIMARY_PRESCRIPTION_GY) > 1.0:
            warnings.warn(f"Prescription {normalization_dose:.1f} Gy differs from expected "
                         f"{PRIMARY_PRESCRIPTION_GY:.1f} Gy - normalizing to extracted value")
        
        # =================================================================
        # Load structure set and create masks
        # =================================================================
        rtstruct_file = [f for f in os.listdir(plan_dir) if f.startswith('RS')][0]
        rtstruct = pydicom.dcmread(os.path.join(plan_dir, rtstruct_file))
        
        masks = np.zeros((len(structure_map),) + ct_volume.shape, dtype=np.uint8)
        for ch_str, info in structure_map.items():
            ch = int(ch_str)
            roi_nums = get_roi_numbers(rtstruct, info['variations'])
            channel_mask = create_mask_from_contours(rtstruct, roi_nums, ct_volume.shape, 
                                                      position, ct_spacing, slice_z, ct_ds)
            masks[ch] = channel_mask
        
        # Check PTV existence
        ptv70_sum = masks[0].sum()
        ptv56_sum = masks[1].sum()
        print(f"PTV70 sum: {ptv70_sum}, PTV56 sum: {ptv56_sum}")
        
        if ptv70_sum == 0:
            warnings.warn(f"Skipping {plan_dir}: Missing PTV70 (required)")
            return None
            
        if ptv56_sum == 0 and not relax_filter:
            warnings.warn(f"Skipping {plan_dir}: Missing PTV56 (use --relax_filter to override)")
            return None
        
        # =================================================================
        # Determine case type
        # =================================================================
        case_type_info = determine_case_type(masks)
        print(f"Case type: {case_type_info['type']}")
        
        # =================================================================
        # Build constraints vector
        # =================================================================
        constraints, constraint_descriptions = build_constraints_vector(case_type_info, prescription_info)
        
        # =================================================================
        # Compute centering (use PTV70 if prostate empty)
        # =================================================================
        prostate_mask = masks[2] if masks[2].sum() > 0 else masks[0]
        
        nonzero = np.nonzero(prostate_mask)
        if len(nonzero[0]) == 0:
            warnings.warn("No prostate/PTV mask; centering on mid-volume")
            mid_y = prostate_mask.shape[0] / 2
            mid_x = prostate_mask.shape[1] / 2
            mid_z = prostate_mask.shape[2] / 2
            centroid_phys_x = position[0] + mid_x * ct_spacing[0]
            centroid_phys_y = position[1] + mid_y * ct_spacing[1]
            centroid_phys_z = position[2] + mid_z * ct_spacing[2]
        else:
            phys_x = position[0] + nonzero[1] * ct_spacing[0]
            phys_y = position[1] + nonzero[0] * ct_spacing[1]
            phys_z = slice_z[nonzero[2]]
            centroid_phys_x = np.mean(phys_x)
            centroid_phys_y = np.mean(phys_y)
            centroid_phys_z = np.mean(phys_z)
        
        # =================================================================
        # Create CT SimpleITK image
        # =================================================================
        ct_array_sitk = ct_volume.transpose(2, 0, 1)  # z y x
        ct_image = sitk.GetImageFromArray(ct_array_sitk)
        ct_image.SetSpacing((ct_spacing[0], ct_spacing[1], ct_spacing[2]))
        ct_image.SetOrigin((position[0], position[1], position[2]))
        
        # =================================================================
        # Load dose
        # =================================================================
        rtdose_file = [f for f in os.listdir(plan_dir) if f.startswith('RD')][0]
        dose_ds = pydicom.dcmread(os.path.join(plan_dir, rtdose_file))
        dose_scaling = float(dose_ds.DoseGridScaling)
        dose_array = dose_ds.pixel_array.astype(np.float32) * dose_scaling
        if dose_array.ndim != 3:
            raise ValueError("Dose not 3D")
        print(f"Dose original shape: {dose_array.shape}")
        
        dose_array_sitk = dose_array  # (z, y, x)
        dose_position = [float(x) for x in dose_ds.ImagePositionPatient]
        dose_pixel_spacing = [float(x) for x in dose_ds.PixelSpacing]
        
        if 'GridFrameOffsetVector' in dose_ds:
            offsets = [float(o) for o in dose_ds.GridFrameOffsetVector]
            if len(offsets) != dose_array.shape[0]:
                raise ValueError("GridFrameOffsetVector length mismatch with dose frames")
            dose_z_positions = [dose_position[2] + offsets[i] for i in range(len(offsets))]
        else:
            warnings.warn("No GridFrameOffsetVector; assuming single frame or CT-like z")
            dose_z_positions = [dose_position[2]]
            if dose_array.shape[0] != 1:
                raise ValueError("Multi-frame dose without GridFrameOffsetVector")
        
        # Sort z for robustness
        sort_idx = np.argsort(dose_z_positions)
        if not np.all(sort_idx == np.arange(len(sort_idx))):
            warnings.warn("Sorting non-increasing dose z-positions")
            dose_array_sitk = dose_array_sitk[sort_idx]
            dose_z_positions = [dose_z_positions[i] for i in sort_idx]
        
        if len(dose_z_positions) > 1:
            diffs = np.diff(dose_z_positions)
            dose_z_spacing = np.mean(diffs)
            if np.std(diffs) > 0.1:
                warnings.warn(f"Non-uniform dose z-spacing (std={np.std(diffs):.2f}); using mean {dose_z_spacing:.2f}")
        else:
            dose_z_spacing = ct_spacing[2]
        
        dose_spacing_sitk = (dose_pixel_spacing[0], dose_pixel_spacing[1], dose_z_spacing)
        dose_origin_sitk = (dose_position[0], dose_position[1], dose_z_positions[0])
        dose_image = sitk.GetImageFromArray(dose_array_sitk)
        dose_image.SetSpacing(dose_spacing_sitk)
        dose_image.SetOrigin(dose_origin_sitk)
        
        # =================================================================
        # Define reference image centered on prostate
        # =================================================================
        target_size_sitk = (target_shape[1], target_shape[0], target_shape[2])  # x y z
        target_spacing_sitk = (target_spacing[0], target_spacing[1], target_spacing[2])  # x y z
        target_origin_sitk = (
            centroid_phys_x - (target_size_sitk[0] * target_spacing_sitk[0]) / 2,
            centroid_phys_y - (target_size_sitk[1] * target_spacing_sitk[1]) / 2,
            centroid_phys_z - (target_size_sitk[2] * target_spacing_sitk[2]) / 2
        )
        reference_image = sitk.Image(target_size_sitk, sitk.sitkFloat32)
        reference_image.SetSpacing(target_spacing_sitk)
        reference_image.SetOrigin(target_origin_sitk)
        
        # =================================================================
        # Resample CT
        # =================================================================
        ct_resampled = resample_image(ct_image, reference_image, sitk.sitkLinear)
        ct_array = sitk.GetArrayFromImage(ct_resampled).transpose(1, 2, 0)  # y x z
        ct_volume_normalized = np.clip(ct_array, -1000, 3000) / 4000 + 0.5
        
        # =================================================================
        # Resample dose (normalize to extracted prescription)
        # =================================================================
        dose_resampled = resample_image(dose_image, reference_image, sitk.sitkLinear)
        dose_volume = sitk.GetArrayFromImage(dose_resampled).transpose(1, 2, 0) / normalization_dose
        dose_volume = np.maximum(dose_volume, 0)  # Clip negative interpolation artifacts
        
        # =================================================================
        # Resample masks
        # =================================================================
        masks_resampled = np.zeros((len(structure_map),) + target_shape, dtype=np.uint8)
        for ch_str, _ in structure_map.items():
            ch = int(ch_str)
            mask_original = masks[ch]  # y x z
            mask_array_sitk = mask_original.transpose(2, 0, 1)  # z y x
            mask_image = sitk.GetImageFromArray(mask_array_sitk.astype(np.float32))
            mask_image.SetSpacing(ct_image.GetSpacing())
            mask_image.SetOrigin(ct_image.GetOrigin())
            mask_res = resample_image(mask_image, reference_image, sitk.sitkNearestNeighbor)
            mask_array = sitk.GetArrayFromImage(mask_res).transpose(1, 2, 0)
            masks_resampled[ch] = (mask_array > 0.5).astype(np.uint8)
        
        print(f"Resampled PTV70 sum: {masks_resampled[0].sum()}, PTV56 sum: {masks_resampled[1].sum()}")
        
        # =================================================================
        # Validate before saving
        # =================================================================
        validation = validate_preprocessed_data(ct_volume_normalized, dose_volume, 
                                                 masks_resampled, normalization_dose)
        
        if not validation['all_critical_passed']:
            warning_msg = f"Validation failed for {plan_id}: "
            if not validation.get('ct_range_valid', True):
                warning_msg += f"CT range [{validation['ct_min']:.2f}, {validation['ct_max']:.2f}]; "
            if not validation.get('dose_nonneg', True):
                warning_msg += f"Negative dose {validation['dose_min']:.4f}; "
            if not validation.get('ptv70_exists', True):
                warning_msg += "PTV70 missing; "
            if not validation.get('ptv70_dose_adequate', True):
                warning_msg += f"PTV70 dose {validation.get('ptv70_dose_mean', 'N/A')}; "
            if validation.get('registration_valid') == False:
                warning_msg += "Registration failed; "
            
            warnings.warn(warning_msg)
            
            if strict_validation:
                return None
        
        if validation['boundary_warnings']:
            for bw in validation['boundary_warnings']:
                warnings.warn(f"Boundary: {bw}")
        
        # =================================================================
        # Build metadata
        # =================================================================
        metadata = {
            'case_id': plan_id,
            'processed_date': datetime.now().isoformat(),
            'prescription_info': prescription_info,
            'case_type': case_type_info,
            'normalization_dose_gy': normalization_dose,
            'target_shape': target_shape,
            'target_spacing_mm': target_spacing,
            'validation': {k: v for k, v in validation.items() if k != 'boundary_warnings'},
            'boundary_warnings': validation['boundary_warnings'],
            'structure_channels': STRUCTURE_CHANNELS,
        }
        
        # =================================================================
        # Save .npz
        # =================================================================
        output_path = os.path.join(output_dir, f'{plan_id}.npz')
        np.savez(output_path, 
                 ct=ct_volume_normalized.astype(np.float32),
                 masks=masks_resampled,
                 dose=dose_volume.astype(np.float32),
                 constraints=constraints,
                 metadata=metadata)
        
        # =================================================================
        # Save debug plots
        # =================================================================
        if not skip_plots:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            mid = target_shape[2] // 2
            
            # Row 1: CT, Dose, PTV70
            axs[0, 0].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axs[0, 0].set_title('CT (mid)')
            
            axs[0, 1].imshow(dose_volume[:, :, mid] * normalization_dose, cmap='hot', vmin=0, vmax=75)
            axs[0, 1].set_title(f'Dose (Gy, normalized to {normalization_dose:.0f})')
            
            axs[0, 2].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axs[0, 2].contour(masks_resampled[0][:, :, mid], colors='red', linewidths=1)
            if masks_resampled[1][:, :, mid].sum() > 0:
                axs[0, 2].contour(masks_resampled[1][:, :, mid], colors='orange', linewidths=1)
            axs[0, 2].set_title('PTV70 (red) / PTV56 (orange)')
            
            # Row 2: CT+Dose overlay, Rectum/Bladder, Validation info
            axs[1, 0].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            dose_masked = np.ma.masked_where(dose_volume[:, :, mid] < 0.1, dose_volume[:, :, mid])
            axs[1, 0].imshow(dose_masked * normalization_dose, cmap='jet', alpha=0.5, vmin=0, vmax=75)
            axs[1, 0].set_title('CT + Dose overlay')
            
            axs[1, 1].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            if masks_resampled[3][:, :, mid].sum() > 0:
                axs[1, 1].contour(masks_resampled[3][:, :, mid], colors='brown', linewidths=1)
            if masks_resampled[4][:, :, mid].sum() > 0:
                axs[1, 1].contour(masks_resampled[4][:, :, mid], colors='yellow', linewidths=1)
            axs[1, 1].set_title('Rectum (brown) / Bladder (yellow)')
            
            # Validation summary
            axs[1, 2].axis('off')
            val_text = f"Case: {plan_id}\n"
            val_text += f"Type: {case_type_info['type']}\n"
            val_text += f"Rx: {normalization_dose:.1f} Gy\n"
            val_text += f"PTV70 mean: {validation.get('ptv70_dose_mean', 'N/A'):.3f}\n" if validation.get('ptv70_dose_mean') else ""
            val_text += f"Validation: {'PASS' if validation['all_critical_passed'] else 'FAIL'}\n"
            if validation['boundary_warnings']:
                val_text += f"Warnings: {len(validation['boundary_warnings'])}"
            axs[1, 2].text(0.1, 0.5, val_text, fontsize=12, verticalalignment='center',
                          family='monospace')
            axs[1, 2].set_title('Summary')
            
            for ax in axs.flat:
                if ax != axs[1, 2]:
                    ax.axis('off')
            
            plt.suptitle(f'{plan_id} - {case_type_info["type"]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'debug_{plan_id}.png'), dpi=150)
            plt.close()
        
        # =================================================================
        # Print summary
        # =================================================================
        stats = {
            'mask_sums': [int(masks_resampled[ch].sum()) for ch in range(len(structure_map))],
            'dose_mean': float(dose_volume.mean()),
            'ptv70_dose_mean': validation.get('ptv70_dose_mean'),
            'validation_passed': validation['all_critical_passed'],
        }
        print(f"Processed {plan_dir} -> {output_path}")
        print(f"  Case type: {case_type_info['type']}, Rx: {normalization_dose:.1f} Gy")
        print(f"  PTV70 dose mean: {stats['ptv70_dose_mean']:.3f}" if stats['ptv70_dose_mean'] else "")
        print(f"  Validation: {'PASS' if stats['validation_passed'] else 'FAIL'}")
        
        return output_path
        
    except Exception as e:
        warnings.warn(f"Error processing {plan_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_preprocess(input_base_dir, output_dir, mapping_file='oar_mapping.json', 
                     relax_filter=False, skip_plots=False, strict_validation=False):
    """
    Batch process multiple plan directories.

    Args:
        input_base_dir (str): Base raw data directory.
        output_dir (str): Output directory.
        mapping_file (str): OAR mapping file.
        relax_filter (bool): Relax PTV filter.
        skip_plots (bool): Skip PNGs.
        strict_validation (bool): Fail on validation warnings.

    Prints:
        Batch completion stats.
    """
    structure_map = load_json_mapping(mapping_file)
    plan_dirs = sorted([os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir) 
                        if os.path.isdir(os.path.join(input_base_dir, d))])
    
    processed = []
    failed = []
    
    for plan_dir in plan_dirs:
        result = preprocess_dicom_rt(plan_dir, output_dir, structure_map, 
                                      relax_filter=relax_filter, 
                                      skip_plots=skip_plots,
                                      strict_validation=strict_validation)
        if result:
            processed.append(result)
        else:
            failed.append(os.path.basename(plan_dir))
    
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE: {len(processed)}/{len(plan_dirs)} cases processed")
    print("=" * 60)
    
    if failed:
        print(f"\nFailed cases ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    
    # Save batch summary
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    summary = {
        'processed_date': datetime.now().isoformat(),
        'total_cases': len(plan_dirs),
        'processed': len(processed),
        'failed': len(failed),
        'failed_cases': failed,
        'settings': {
            'relax_filter': relax_filter,
            'strict_validation': strict_validation,
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nBatch summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM-RT to .npz for VMAT diffusion model")
    parser.add_argument("--input_dir", default="~/vmat-diffusion-project/data/raw",
                        help="Input directory containing case subdirectories")
    parser.add_argument("--output_dir", default="~/vmat-diffusion-project/processed_npz",
                        help="Output directory for .npz files")
    parser.add_argument("--mapping_file", default="oar_mapping.json",
                        help="Path to OAR mapping JSON file")
    parser.add_argument("--relax_filter", action="store_true",
                        help="Process cases even without PTV56")
    parser.add_argument("--skip_plots", action="store_true", 
                        help="Skip saving debug PNGs")
    parser.add_argument("--strict_validation", action="store_true",
                        help="Fail cases that don't pass validation checks")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    batch_preprocess(input_dir, output_dir, args.mapping_file, 
                     args.relax_filter, args.skip_plots, args.strict_validation)


# =============================================================================
# FUTURE TODOs (Phase 2+)
# =============================================================================
# TODO: Add beam geometry extraction from RP file (arc angles, MU, collimator)
# TODO: Add MLC leaf sequence extraction
# TODO: Implement signed distance fields (SDFs) for mask encoding
# TODO: Add pymedphys integration for DVH/gamma validation
# TODO: Dynamic target_shape based on dataset max extents
# TODO: Support for PTV50.4 (channel 8) for 3-level SIB
