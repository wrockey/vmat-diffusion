"""
Preprocess DICOM-RT VMAT plans to .npz for diffusion model training.

Functionality: Keeps CT and masks at native resolution, resamples only dose
(coarse grid) to CT grid using B-spline interpolation, then crops to a fixed
physical extent centered on the prostate. Computes SDFs, extracts beam geometry,
and saves with constraints and metadata.

Version: 2.3.0 - Crop-based pipeline (fixes D95 artifact)

Changes from v2.2.0:
- CRITICAL FIX: Replaced fixed-grid resampling (512x512x256 @ 1x1x2mm) with
  crop-based pipeline. Masks are NEVER resampled — generated on CT grid, then
  cropped. Dose resampled to CT grid only (B-spline). Eliminates mask/dose
  boundary mismatch that caused PTV70 D95 to read ~55 Gy instead of ≥66.5 Gy.
- Dose interpolation changed from sitkLinear to sitkBSpline (less boundary smoothing)
- In-plane crop: 300 voxels centered on prostate centroid (~30cm)
- Axial crop: dynamic based on structure bounding box + 30mm margin
- SDFs computed with actual CT spacing (not hardcoded 1x1x2mm)
- Metadata: voxel_spacing_mm, volume_shape, crop_box, dose_grid_spacing_mm
- DVH validation: warns if PTV70 D95 < 64 Gy after preprocessing
- Dose coverage check: warns if >5% of PTV voxels have zero dose
- New CLI args: --inplane_size (default 300), --z_margin_mm (default 30)
- Removed CLI args: --target_shape, --target_spacing
- get_spacing_from_metadata() utility for backwards-compatible spacing lookup

Key Features:
- Signed Distance Fields (SDFs) for all structures
- FULL beam geometry from RP file (MLC, dose rate, jaws) - Phase 2 ready
- Improved truncation quantification
- HU clipping detection
- DVH validation per case

.npz File Structure (v2.3.0):
- ct: (Y, X, Z) float32 - CT normalized [0,1], native resolution cropped
- dose: (Y, X, Z) float32 - Dose normalized to Rx, on CT grid cropped
- masks: (8, Y, X, Z) uint8 - Binary structure masks, native grid cropped
- masks_sdf: (8, Y, X, Z) float32 - Signed distance fields [-1,1]
- constraints: (13,) float32 - Prescription + OAR constraints
- metadata: dict - Case info, spacing, crop, beam geometry, validation
- beam0_mlc_a: (n_cp, n_leaves) float32 - MLC bank A positions per control point
- beam0_mlc_b: (n_cp, n_leaves) float32 - MLC bank B positions per control point
- beam1_mlc_a, beam1_mlc_b: (if 2+ arcs)

Assumptions: See docs/preprocessing_assumptions.md

Usage: python preprocess_dicom_rt_v2.2.py [flags]
"""

import os
import json
import numpy as np
import pydicom
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
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
    # 8: 'PTV50.4',  # Reserved for Phase 2 / 3-level SIB
}

# Default spacing for backwards compatibility with v2.2 NPZ files
DEFAULT_SPACING_MM = (1.0, 1.0, 2.0)

# Crop parameters (v2.3+)
DEFAULT_INPLANE_SIZE = 300  # Voxels: 300 × ~1mm = 30cm, covers pelvis + margin
DEFAULT_Z_MARGIN_MM = 30.0  # mm beyond structure bounding box (dose falloff margin)

# SDF parameters (can be overridden via command line)
DEFAULT_SDF_CLIP_MM = 50.0  # Default clip at ±50mm (captures relevant dose gradients)
# Note: For tighter structures or faster falloff, consider 30mm


def get_spacing_from_metadata(metadata):
    """
    Extract voxel spacing from NPZ metadata with backwards-compatible fallback.

    Fallback chain:
        1. voxel_spacing_mm (v2.3+ native spacing)
        2. target_spacing_mm (v2.2 resampled spacing)
        3. DEFAULT_SPACING_MM (1.0, 1.0, 2.0 — last resort)

    Args:
        metadata: dict from npz['metadata'].item()

    Returns:
        tuple: (y_spacing, x_spacing, z_spacing) in mm
    """
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()

    if 'voxel_spacing_mm' in metadata:
        spacing = metadata['voxel_spacing_mm']
        return tuple(float(s) for s in spacing)

    if 'target_spacing_mm' in metadata:
        spacing = metadata['target_spacing_mm']
        return tuple(float(s) for s in spacing)

    return DEFAULT_SPACING_MM

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_mapping(mapping_file='oar_mapping.json'):
    """Load the OAR mapping JSON file for contour variations."""
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"{mapping_file} not found; run generate_mapping_json.py and edit")
    with open(mapping_file, 'r') as f:
        return json.load(f)


def find_dicom_files(plan_dir, prefix_patterns, recursive=False):
    """Find DICOM files matching any of the given prefix patterns.

    Skips Zone.Identifier files (Windows download artifacts).
    If recursive=True, searches subdirectories as well.

    Args:
        plan_dir: Directory to search
        prefix_patterns: List of filename prefixes to match (e.g., ['CT', 'ct'])
        recursive: Whether to search subdirectories

    Returns:
        List of full file paths matching the patterns
    """
    matches = []
    if recursive:
        for root, dirs, files in os.walk(plan_dir):
            for f in files:
                if 'Zone.Identifier' in f:
                    continue
                if any(f.startswith(p) for p in prefix_patterns):
                    matches.append(os.path.join(root, f))
    else:
        for f in os.listdir(plan_dir):
            if 'Zone.Identifier' in f:
                continue
            if any(f.startswith(p) for p in prefix_patterns):
                matches.append(os.path.join(plan_dir, f))
    return matches


def get_ct_volume_and_metadata(plan_dir):
    """Load and stack CT slices, sort by Z, compute metadata."""
    ct_paths = find_dicom_files(plan_dir, ['CT', 'ct'], recursive=True)
    if not ct_paths:
        raise ValueError(f"No CT files in {plan_dir}")
    # Filter out non-image files (RTSTRUCT etc. can start with CT prefix in paths)
    ct_paths = [p for p in ct_paths if not any(
        kw in os.path.basename(p).upper() for kw in ['RTSTRUCT', 'RTDOSE', 'RTPLAN']
    )]
    if not ct_paths:
        raise ValueError(f"No CT image files in {plan_dir}")
    ct_ds = [pydicom.dcmread(p) for p in ct_paths]
    sorted_indices = np.argsort([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    ct_ds = [ct_ds[i] for i in sorted_indices]
    slice_z = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    print(f"CT z range: min {slice_z.min():.2f}, max {slice_z.max():.2f}, slices {len(slice_z)}, spacing {np.mean(np.diff(slice_z)):.2f}")
    # Apply RescaleSlope/RescaleIntercept to convert stored values to HU
    def _to_hu(ds):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        return arr * slope + intercept
    ct_volume = np.stack([_to_hu(ds) for ds in ct_ds], axis=2)
    spacing = np.array([float(ct_ds[0].PixelSpacing[0]), float(ct_ds[0].PixelSpacing[1]), np.mean(np.diff(slice_z))])
    position = np.array([float(x) for x in ct_ds[0].ImagePositionPatient])
    return ct_volume, spacing, position, slice_z, ct_ds


def extract_prescription_dose(plan_dir):
    """
    Extract prescription dose from RT Plan file.
    
    Returns:
        dict: Prescription information
    """
    result = {
        'primary_dose': PRIMARY_PRESCRIPTION_GY,
        'fractions': 28,
        'dose_per_fraction': PRIMARY_PRESCRIPTION_GY / 28,
        'source': 'default'
    }
    
    try:
        rp_paths = find_dicom_files(plan_dir, ['RP', 'RTPLAN', 'rtplan'])
        if not rp_paths:
            warnings.warn(f"No RP file found in {plan_dir}, using default prescription")
            return result

        rp = pydicom.dcmread(rp_paths[0])
        
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
        
        if result['fractions'] > 0:
            result['dose_per_fraction'] = result['primary_dose'] / result['fractions']
            
        print(f"Prescription: {result['primary_dose']:.1f} Gy in {result['fractions']} fx "
              f"({result['dose_per_fraction']:.2f} Gy/fx) [source: {result['source']}]")
              
    except Exception as e:
        warnings.warn(f"Error extracting prescription: {e}, using default")
    
    return result


def extract_beam_geometry(plan_dir, include_mlc=True):
    """
    Extract comprehensive beam geometry from RT Plan file for Phase 2 deliverability.
    
    Extracts:
    - Plan-level: num_beams, total_mu, plan_label
    - Beam-level: name, type, energy, arc angles, collimator, couch
    - Control point-level: gantry angles, MLC positions, jaw positions, 
                           cumulative MU, dose rate
    
    Args:
        plan_dir: Path to directory containing RP file
        include_mlc: Whether to extract full MLC sequences (adds ~1-2MB per case)
    
    Returns:
        dict: Comprehensive beam geometry (None if RP not found or error)
    """
    try:
        rp_paths = find_dicom_files(plan_dir, ['RP', 'RTPLAN', 'rtplan'])
        if not rp_paths:
            return None

        rp = pydicom.dcmread(rp_paths[0])
        
        if not hasattr(rp, 'BeamSequence'):
            return None
        
        beam_params = []
        total_mu = 0
        
        for beam in rp.BeamSequence:
            beam_info = {
                # Basic beam info
                'beam_name': str(beam.BeamName) if hasattr(beam, 'BeamName') else None,
                'beam_number': int(beam.BeamNumber) if hasattr(beam, 'BeamNumber') else None,
                'beam_type': str(beam.BeamType) if hasattr(beam, 'BeamType') else None,
                'radiation_type': str(beam.RadiationType) if hasattr(beam, 'RadiationType') else None,
                'treatment_machine': str(beam.TreatmentMachineName) if hasattr(beam, 'TreatmentMachineName') else None,
                
                # Arc geometry
                'num_control_points': None,
                'arc_start_angle': None,
                'arc_stop_angle': None,
                'arc_direction': None,
                'gantry_rotation_direction': None,
                
                # Static angles
                'collimator_angle': None,
                'couch_angle': None,
                
                # Beam energy
                'nominal_energy': None,
                
                # MU
                'final_mu': None,
                
                # MLC info
                'mlc_type': None,
                'num_leaves': None,
                'leaf_pair_count': None,
                
                # Control point sequences (Phase 2 data)
                'control_points': None,
            }
            
            # Extract nominal beam energy
            if hasattr(beam, 'ControlPointSequence') and len(beam.ControlPointSequence) > 0:
                cp0 = beam.ControlPointSequence[0]
                if hasattr(cp0, 'NominalBeamEnergy'):
                    beam_info['nominal_energy'] = float(cp0.NominalBeamEnergy)
            
            # Extract MLC device info from BeamLimitingDeviceSequence
            if hasattr(beam, 'BeamLimitingDeviceSequence'):
                for bld in beam.BeamLimitingDeviceSequence:
                    bld_type = str(bld.RTBeamLimitingDeviceType) if hasattr(bld, 'RTBeamLimitingDeviceType') else ''
                    if 'MLC' in bld_type.upper():
                        beam_info['mlc_type'] = bld_type
                        if hasattr(bld, 'NumberOfLeafJawPairs'):
                            beam_info['leaf_pair_count'] = int(bld.NumberOfLeafJawPairs)
                            beam_info['num_leaves'] = beam_info['leaf_pair_count'] * 2  # A and B banks
                        if hasattr(bld, 'LeafPositionBoundaries'):
                            beam_info['leaf_boundaries'] = [float(x) for x in bld.LeafPositionBoundaries]
            
            # Extract from ControlPointSequence (VMAT arcs)
            if hasattr(beam, 'ControlPointSequence') and len(beam.ControlPointSequence) > 0:
                cp_seq = beam.ControlPointSequence
                beam_info['num_control_points'] = len(cp_seq)
                
                # First control point - static geometry
                cp0 = cp_seq[0]
                if hasattr(cp0, 'GantryAngle'):
                    beam_info['arc_start_angle'] = float(cp0.GantryAngle)
                if hasattr(cp0, 'BeamLimitingDeviceAngle'):
                    beam_info['collimator_angle'] = float(cp0.BeamLimitingDeviceAngle)
                if hasattr(cp0, 'PatientSupportAngle'):
                    beam_info['couch_angle'] = float(cp0.PatientSupportAngle)
                if hasattr(cp0, 'GantryRotationDirection'):
                    beam_info['gantry_rotation_direction'] = str(cp0.GantryRotationDirection)
                
                # Last control point
                cp_last = cp_seq[-1]
                if hasattr(cp_last, 'GantryAngle'):
                    beam_info['arc_stop_angle'] = float(cp_last.GantryAngle)
                
                # Arc direction
                if beam_info['gantry_rotation_direction'] == 'CW':
                    beam_info['arc_direction'] = 'clockwise'
                elif beam_info['gantry_rotation_direction'] == 'CC':
                    beam_info['arc_direction'] = 'counter_clockwise'
                
                # Extract full control point sequence data
                if include_mlc:
                    control_points = []
                    
                    for cp_idx, cp in enumerate(cp_seq):
                        cp_data = {
                            'index': cp_idx,
                            'gantry_angle': float(cp.GantryAngle) if hasattr(cp, 'GantryAngle') else None,
                            'cumulative_meterset_weight': float(cp.CumulativeMetersetWeight) if hasattr(cp, 'CumulativeMetersetWeight') else None,
                            'dose_rate': float(cp.DoseRateSet) if hasattr(cp, 'DoseRateSet') else None,
                            'gantry_rotation_direction': str(cp.GantryRotationDirection) if hasattr(cp, 'GantryRotationDirection') else None,
                            
                            # Will be populated below
                            'mlc_positions_a': None,  # Bank A (negative X)
                            'mlc_positions_b': None,  # Bank B (positive X)
                            'jaw_positions': None,
                        }
                        
                        # Extract MLC and jaw positions from BeamLimitingDevicePositionSequence
                        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                            jaw_x = [None, None]  # X1, X2
                            jaw_y = [None, None]  # Y1, Y2
                            
                            for bld_pos in cp.BeamLimitingDevicePositionSequence:
                                bld_type = str(bld_pos.RTBeamLimitingDeviceType) if hasattr(bld_pos, 'RTBeamLimitingDeviceType') else ''
                                
                                if hasattr(bld_pos, 'LeafJawPositions'):
                                    positions = [float(x) for x in bld_pos.LeafJawPositions]
                                    
                                    if 'MLC' in bld_type.upper():
                                        # MLC positions: first half is bank A, second half is bank B
                                        n_leaves = len(positions) // 2
                                        cp_data['mlc_positions_a'] = positions[:n_leaves]
                                        cp_data['mlc_positions_b'] = positions[n_leaves:]
                                    elif bld_type in ['X', 'ASYMX']:
                                        jaw_x = positions
                                    elif bld_type in ['Y', 'ASYMY']:
                                        jaw_y = positions
                            
                            cp_data['jaw_positions'] = {
                                'x1': jaw_x[0] if len(jaw_x) > 0 else None,
                                'x2': jaw_x[1] if len(jaw_x) > 1 else None,
                                'y1': jaw_y[0] if len(jaw_y) > 0 else None,
                                'y2': jaw_y[1] if len(jaw_y) > 1 else None,
                            }
                        
                        control_points.append(cp_data)
                    
                    beam_info['control_points'] = control_points
            
            # MU from FinalCumulativeMetersetWeight
            if hasattr(beam, 'FinalCumulativeMetersetWeight'):
                beam_info['final_mu'] = float(beam.FinalCumulativeMetersetWeight)
                total_mu += beam_info['final_mu']
            
            beam_params.append(beam_info)
        
        # Convert MLC data to numpy arrays for efficient storage
        for beam in beam_params:
            if beam['control_points'] is not None:
                n_cp = len(beam['control_points'])
                
                # Extract arrays
                gantry_angles = np.array([cp['gantry_angle'] for cp in beam['control_points']], dtype=np.float32)
                cumulative_mu = np.array([cp['cumulative_meterset_weight'] for cp in beam['control_points']], dtype=np.float32)
                dose_rates = np.array([cp['dose_rate'] if cp['dose_rate'] is not None else 0.0 
                                       for cp in beam['control_points']], dtype=np.float32)
                
                # MLC arrays: (n_control_points, n_leaves_per_bank)
                if beam['control_points'][0]['mlc_positions_a'] is not None:
                    n_leaves = len(beam['control_points'][0]['mlc_positions_a'])
                    mlc_a = np.array([cp['mlc_positions_a'] for cp in beam['control_points']], dtype=np.float32)
                    mlc_b = np.array([cp['mlc_positions_b'] for cp in beam['control_points']], dtype=np.float32)
                else:
                    mlc_a = None
                    mlc_b = None
                
                # Jaw arrays
                jaw_x1 = np.array([cp['jaw_positions']['x1'] if cp['jaw_positions'] and cp['jaw_positions']['x1'] is not None else 0.0 
                                   for cp in beam['control_points']], dtype=np.float32)
                jaw_x2 = np.array([cp['jaw_positions']['x2'] if cp['jaw_positions'] and cp['jaw_positions']['x2'] is not None else 0.0 
                                   for cp in beam['control_points']], dtype=np.float32)
                jaw_y1 = np.array([cp['jaw_positions']['y1'] if cp['jaw_positions'] and cp['jaw_positions']['y1'] is not None else 0.0 
                                   for cp in beam['control_points']], dtype=np.float32)
                jaw_y2 = np.array([cp['jaw_positions']['y2'] if cp['jaw_positions'] and cp['jaw_positions']['y2'] is not None else 0.0 
                                   for cp in beam['control_points']], dtype=np.float32)
                
                # Replace control_points list with efficient numpy structure
                beam['control_point_data'] = {
                    'n_control_points': n_cp,
                    'gantry_angles': gantry_angles.tolist(),  # Convert to list for JSON serialization in metadata
                    'cumulative_meterset_weight': cumulative_mu.tolist(),
                    'dose_rates': dose_rates.tolist(),
                    'jaw_x1': jaw_x1.tolist(),
                    'jaw_x2': jaw_x2.tolist(),
                    'jaw_y1': jaw_y1.tolist(),
                    'jaw_y2': jaw_y2.tolist(),
                }
                
                # Store MLC as numpy arrays (these go in separate npz keys for efficiency)
                if mlc_a is not None:
                    beam['mlc_bank_a'] = mlc_a  # Shape: (n_cp, n_leaves)
                    beam['mlc_bank_b'] = mlc_b  # Shape: (n_cp, n_leaves)
                
                # Remove the verbose control_points list
                del beam['control_points']
        
        result = {
            'num_beams': len(beam_params),
            'total_mu': total_mu,
            'beams': beam_params,
            'plan_label': str(rp.RTPlanLabel) if hasattr(rp, 'RTPlanLabel') else None,
            'mlc_extracted': include_mlc,
        }
        
        # Calculate total MLC data size for logging
        mlc_size_mb = 0
        for beam in beam_params:
            if 'mlc_bank_a' in beam and beam['mlc_bank_a'] is not None:
                mlc_size_mb += beam['mlc_bank_a'].nbytes / 1e6
                mlc_size_mb += beam['mlc_bank_b'].nbytes / 1e6
        
        print(f"Beam geometry: {result['num_beams']} beams, {total_mu:.1f} total MU")
        if include_mlc:
            n_cp_total = sum(b.get('num_control_points', 0) or 0 for b in beam_params)
            print(f"  MLC data: {n_cp_total} control points, {mlc_size_mb:.2f} MB")
        
        return result
        
    except Exception as e:
        warnings.warn(f"Error extracting beam geometry: {e}")
        import traceback
        traceback.print_exc()
        return None


def determine_case_type(masks):
    """Determine the case type based on which PTVs are present."""
    ptv70_exists = masks[0].sum() > 0 if masks.shape[0] > 0 else False
    ptv56_exists = masks[1].sum() > 0 if masks.shape[0] > 1 else False
    ptv50_exists = False  # Reserved for channel 8
    
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
    
    Structure (13 elements):
    - [0-2]: Prescription targets (PTV70, PTV56, PTV50.4) normalized to primary Rx
    - [3-12]: OAR constraints normalized appropriately
    """
    primary_rx = prescription_info['primary_dose']
    
    prescription_targets = np.array([
        SIB_PRESCRIPTIONS['PTV70'] / primary_rx if case_type_info['ptv70_exists'] else 0.0,
        SIB_PRESCRIPTIONS['PTV56'] / primary_rx if case_type_info['ptv56_exists'] else 0.0,
        SIB_PRESCRIPTIONS.get('PTV50.4', 0) / primary_rx if case_type_info['ptv50_exists'] else 0.0,
    ])
    
    oar_constraints = np.array([
        DEFAULT_CONSTRAINTS['rectum_v50'] / 100.0,
        DEFAULT_CONSTRAINTS['rectum_v60'] / 100.0,
        DEFAULT_CONSTRAINTS['rectum_v70'] / 100.0,
        DEFAULT_CONSTRAINTS['rectum_max'] / primary_rx,
        DEFAULT_CONSTRAINTS['bladder_v65'] / 100.0,
        DEFAULT_CONSTRAINTS['bladder_v70'] / 100.0,
        DEFAULT_CONSTRAINTS['bladder_v75'] / 100.0,
        DEFAULT_CONSTRAINTS['femur_v50'] / 100.0,
        DEFAULT_CONSTRAINTS['bowel_v45'] / 500.0,
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


def compute_sdf(binary_mask, spacing_mm=(1.0, 1.0, 2.0), clip_mm=DEFAULT_SDF_CLIP_MM):
    """
    Compute signed distance field from binary mask.
    
    Args:
        binary_mask: Binary mask array (Y, X, Z)
        spacing_mm: Voxel spacing in mm (Y, X, Z)
        clip_mm: Clip SDF values at ±clip_mm
        
    Returns:
        np.array: SDF with negative inside, positive outside, zero at boundary
                  Normalized to [-1, 1] range
    """
    if binary_mask.sum() == 0:
        # Empty mask - return all positive (outside) at max distance
        return np.full(binary_mask.shape, 1.0, dtype=np.float32)

    # Cast to bool to ensure ~ is logical NOT, not bitwise NOT (uint8 bug)
    mask_bool = binary_mask.astype(bool)

    # Distance transform for outside (where mask is 0)
    dist_outside = distance_transform_edt(~mask_bool, sampling=spacing_mm)

    # Distance transform for inside (where mask is 1)
    dist_inside = distance_transform_edt(mask_bool, sampling=spacing_mm)
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    # Clip to reasonable range
    sdf = np.clip(sdf, -clip_mm, clip_mm)
    
    # Normalize to [-1, 1] for model input
    sdf_normalized = sdf / clip_mm
    
    return sdf_normalized.astype(np.float32)


def validate_sdf(masks_sdf, masks_binary, structure_names):
    """
    Validate SDF computation for correctness.
    
    Checks:
    - SDF negative inside structure (where binary=1)
    - SDF positive outside structure (where binary=0)
    - SDF near zero at boundaries
    
    Args:
        masks_sdf: SDF array (C, Y, X, Z)
        masks_binary: Binary mask array (C, Y, X, Z)
        structure_names: Dict mapping channel to name
        
    Returns:
        dict: Validation results per structure
    """
    results = {}
    
    for ch, name in structure_names.items():
        if ch >= masks_sdf.shape[0]:
            continue
            
        sdf = masks_sdf[ch]
        binary = masks_binary[ch]
        
        if binary.sum() == 0:
            results[name] = {'exists': False, 'valid': True}
            continue
        
        inside_mask = binary > 0
        outside_mask = ~inside_mask
        
        # Check: SDF should be negative inside
        sdf_inside = sdf[inside_mask]
        inside_negative = (sdf_inside <= 0).mean()
        
        # Check: SDF should be positive outside (excluding boundary region)
        # Use eroded outside to avoid boundary
        from scipy.ndimage import binary_erosion
        outside_eroded = binary_erosion(outside_mask, iterations=2)
        if outside_eroded.sum() > 0:
            sdf_outside = sdf[outside_eroded]
            outside_positive = (sdf_outside >= 0).mean()
        else:
            outside_positive = 1.0  # Small structure, can't check
        
        valid = inside_negative > 0.95 and outside_positive > 0.95
        
        results[name] = {
            'exists': True,
            'valid': valid,
            'inside_negative_pct': float(inside_negative),
            'outside_positive_pct': float(outside_positive),
        }
        
        if not valid:
            warnings.warn(f"SDF validation issue for {name}: "
                         f"inside_neg={inside_negative:.1%}, outside_pos={outside_positive:.1%}")
    
    return results


def get_roi_numbers(rtstruct, variations):
    """Get ROI numbers matching variations in the structure set."""
    roi_nums = []
    for roi in rtstruct.StructureSetROISequence:
        name = roi.ROIName.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '')
        if name in [v.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '') for v in variations]:
            roi_nums.append(int(roi.ROINumber))
    return roi_nums


def create_mask_from_contours(rtstruct, roi_nums, ct_shape, position, spacing, slice_z, ct_ds):
    """Create binary mask from contours for given ROIs."""
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
    """Resample an image to a reference grid using SimpleITK."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def quantify_truncation(mask, structure_name):
    """
    Quantify how much of a structure is truncated at boundaries.
    
    Returns:
        dict: Truncation information including percentage at each boundary
    """
    if mask.sum() == 0:
        return {'exists': False, 'truncated': False, 'truncation_percent': 0.0}
    
    total_voxels = mask.sum()
    
    # Count voxels at each boundary
    boundary_voxels = {
        'y_min': mask[0, :, :].sum(),      # Anterior
        'y_max': mask[-1, :, :].sum(),     # Posterior
        'x_min': mask[:, 0, :].sum(),      # Right
        'x_max': mask[:, -1, :].sum(),     # Left
        'z_min': mask[:, :, 0].sum(),      # Inferior
        'z_max': mask[:, :, -1].sum(),     # Superior
    }
    
    total_boundary = sum(boundary_voxels.values())
    truncation_percent = (total_boundary / total_voxels) * 100 if total_voxels > 0 else 0
    
    return {
        'exists': True,
        'truncated': total_boundary > 0,
        'truncation_percent': truncation_percent,
        'total_voxels': int(total_voxels),
        'boundary_voxels': boundary_voxels,
        'touching_boundaries': [k for k, v in boundary_voxels.items() if v > 0],
    }


def compute_crop_box(masks, ct_spacing, inplane_size=DEFAULT_INPLANE_SIZE,
                     z_margin_mm=DEFAULT_Z_MARGIN_MM, min_z=128):
    """
    Compute crop box centered on prostate/PTV70 centroid.

    In-plane: fixed `inplane_size` voxels centered on centroid.
    Axial (Z): bounding box of ALL structures + z_margin_mm, minimum min_z voxels.

    Args:
        masks: (C, Y, X, Z) uint8 binary masks on CT grid
        ct_spacing: (y_spacing, x_spacing, z_spacing) in mm
        inplane_size: number of in-plane voxels per dimension (default 300)
        z_margin_mm: margin beyond structure bounding box in mm (default 30)
        min_z: minimum Z extent in voxels (default 128, for patch extraction)

    Returns:
        dict with keys:
            y_start, y_end, x_start, x_end, z_start, z_end: crop indices
            centroid_voxel: (y, x, z) centroid in voxel coords
    """
    # Find centroid from prostate (channel 2) or PTV70 (channel 0) fallback
    ref_mask = masks[2] if masks[2].sum() > 0 else masks[0]
    nz = np.nonzero(ref_mask)
    if len(nz[0]) == 0:
        # Fallback: center of volume
        cy, cx, cz = [s // 2 for s in masks.shape[1:]]
    else:
        cy = int(np.mean(nz[0]))
        cx = int(np.mean(nz[1]))
        cz = int(np.mean(nz[2]))

    vol_shape = masks.shape[1:]  # (Y, X, Z)

    # --- In-plane crop (Y, X): fixed size centered on centroid ---
    half_ip = inplane_size // 2
    for dim_idx, (center, dim_size) in enumerate([(cy, vol_shape[0]), (cx, vol_shape[1])]):
        start = max(0, center - half_ip)
        end = start + inplane_size
        if end > dim_size:
            end = dim_size
            start = max(0, end - inplane_size)
        if dim_idx == 0:
            y_start, y_end = start, end
        else:
            x_start, x_end = start, end

    # --- Axial crop (Z): bounding box of all structures + margin ---
    # Union of all non-empty structure masks
    any_mask = np.any(masks > 0, axis=0)  # (Y, X, Z)
    z_projection = np.any(any_mask, axis=(0, 1))  # (Z,)
    z_indices = np.where(z_projection)[0]

    if len(z_indices) == 0:
        # No structures — use center ± min_z/2
        z_start = max(0, cz - min_z // 2)
        z_end = min(vol_shape[2], z_start + min_z)
    else:
        z_spacing = ct_spacing[2]
        margin_voxels = int(np.ceil(z_margin_mm / z_spacing))

        z_start = max(0, z_indices[0] - margin_voxels)
        z_end = min(vol_shape[2], z_indices[-1] + 1 + margin_voxels)

        # Enforce minimum Z extent (expand from center if needed)
        z_extent = z_end - z_start
        if z_extent < min_z:
            z_center = (z_start + z_end) // 2
            z_start = max(0, z_center - min_z // 2)
            z_end = min(vol_shape[2], z_start + min_z)
            # Re-check if we hit the boundary
            if z_end - z_start < min_z:
                z_start = max(0, z_end - min_z)

    return {
        'y_start': int(y_start), 'y_end': int(y_end),
        'x_start': int(x_start), 'x_end': int(x_end),
        'z_start': int(z_start), 'z_end': int(z_end),
        'centroid_voxel': (int(cy), int(cx), int(cz)),
    }


def validate_preprocessed_data(ct_volume, dose_volume, masks_resampled, prescription_dose,
                                ct_raw_min=None, ct_raw_max=None):
    """
    Validate preprocessed data before saving.
    
    Returns:
        dict: Validation results with pass/fail for each check
    """
    checks = {}
    
    # CT checks
    # CT normalized as clip(-1000,3000)/4000+0.5 → valid range [0.0, 1.25]
    checks['ct_range_valid'] = (ct_volume.min() >= -0.01) and (ct_volume.max() <= 1.26)
    checks['ct_min'] = float(ct_volume.min())
    checks['ct_max'] = float(ct_volume.max())
    
    # HU clipping detection
    if ct_raw_min is not None and ct_raw_max is not None:
        checks['hu_clipped_low'] = ct_raw_min < -1000
        checks['hu_clipped_high'] = ct_raw_max > 3000
        checks['hu_original_range'] = (float(ct_raw_min), float(ct_raw_max))
    
    # Dose checks
    checks['dose_nonneg'] = dose_volume.min() >= -0.01
    checks['dose_min'] = float(dose_volume.min())
    checks['dose_max'] = float(dose_volume.max())
    checks['dose_reasonable'] = dose_volume.max() < 1.5
    
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
    
    # Registration check
    any_ptv = (masks_resampled[0] > 0) | (masks_resampled[1] > 0)
    if any_ptv.sum() > 0 and (~any_ptv).sum() > 0:
        dose_in_ptv = dose_volume[any_ptv].mean()
        dose_outside_ptv = dose_volume[~any_ptv].mean()
        checks['registration_valid'] = dose_in_ptv > dose_outside_ptv
        checks['dose_in_ptv'] = float(dose_in_ptv)
        checks['dose_outside_ptv'] = float(dose_outside_ptv)
    else:
        checks['registration_valid'] = None
    
    # Truncation quantification for critical structures
    truncation_info = {}
    critical_structures = [0, 1, 3, 4]  # PTV70, PTV56, Rectum, Bladder
    checks['critical_truncation'] = False
    
    for ch in range(masks_resampled.shape[0]):
        name = STRUCTURE_CHANNELS.get(ch, f'channel_{ch}')
        trunc = quantify_truncation(masks_resampled[ch], name)
        truncation_info[name] = trunc
        
        # Flag if critical structure has >5% truncation
        if ch in critical_structures and trunc['truncation_percent'] > 5.0:
            checks['critical_truncation'] = True
            warnings.warn(f"Critical truncation: {name} has {trunc['truncation_percent']:.1f}% at boundary")
    
    checks['truncation_info'] = truncation_info
    
    # Overall pass/fail
    critical_checks = ['ct_range_valid', 'dose_nonneg', 'ptv70_exists', 'ptv70_dose_adequate']
    checks['all_critical_passed'] = all(checks.get(c, False) for c in critical_checks)
    
    if checks.get('registration_valid') == False:
        checks['all_critical_passed'] = False
        warnings.warn("CRITICAL: Registration check failed - dose higher outside PTV!")
    
    if checks.get('critical_truncation'):
        warnings.warn("WARNING: Critical structure truncation detected")
    
    return checks


def preprocess_dicom_rt(plan_dir, output_dir, structure_map, relax_filter=False,
                        skip_plots=False, strict_validation=False, compute_sdfs=True,
                        extract_beams=True, sdf_clip_mm=DEFAULT_SDF_CLIP_MM,
                        inplane_size=DEFAULT_INPLANE_SIZE, z_margin_mm=DEFAULT_Z_MARGIN_MM):
    """
    Process a single DICOM-RT plan directory.

    v2.3 pipeline: keeps CT and masks at native resolution, resamples only
    dose (coarse grid) to CT grid using B-spline interpolation, then crops
    to a fixed physical extent centered on the prostate.

    Args:
        plan_dir (str): Path to case dir with DICOM files.
        output_dir (str): Output directory for .npz and PNGs.
        structure_map (dict): OAR mapping from JSON.
        relax_filter (bool): Process cases without PTV56.
        skip_plots (bool): Skip debug PNGs.
        strict_validation (bool): Fail on validation issues.
        compute_sdfs (bool): Compute signed distance fields for masks.
        extract_beams (bool): Extract beam geometry from RP file.
        sdf_clip_mm (float): Clip SDF values at ±this distance in mm.
        inplane_size (int): In-plane crop size in voxels (default 300).
        z_margin_mm (float): Axial margin beyond structures in mm (default 30).

    Returns:
        str: Path to output .npz or None on error.
    """
    os.makedirs(output_dir, exist_ok=True)
    plan_id = os.path.basename(plan_dir)
    
    try:
        # =================================================================
        # Load CT
        # =================================================================
        ct_volume_raw, ct_spacing, position, slice_z, ct_ds = get_ct_volume_and_metadata(plan_dir)
        ct_raw_min, ct_raw_max = ct_volume_raw.min(), ct_volume_raw.max()
        
        # =================================================================
        # Extract prescription
        # =================================================================
        prescription_info = extract_prescription_dose(plan_dir)
        normalization_dose = prescription_info['primary_dose']
        
        if abs(normalization_dose - PRIMARY_PRESCRIPTION_GY) > 1.0:
            warnings.warn(f"Prescription {normalization_dose:.1f} Gy differs from expected "
                         f"{PRIMARY_PRESCRIPTION_GY:.1f} Gy")
        
        # =================================================================
        # Extract beam geometry (Phase 2 prep)
        # =================================================================
        beam_geometry = None
        if extract_beams:
            beam_geometry = extract_beam_geometry(plan_dir)
        
        # =================================================================
        # Load structure set and create masks
        # =================================================================
        rtstruct_paths = find_dicom_files(plan_dir, ['RS', 'RTSTRUCT', 'rtstruct'])
        if not rtstruct_paths:
            raise ValueError(f"No RTSTRUCT file in {plan_dir}")
        rtstruct = pydicom.dcmread(rtstruct_paths[0])
        
        masks = np.zeros((len(structure_map),) + ct_volume_raw.shape, dtype=np.uint8)
        for ch_str, info in structure_map.items():
            ch = int(ch_str)
            roi_nums = get_roi_numbers(rtstruct, info['variations'])
            channel_mask = create_mask_from_contours(rtstruct, roi_nums, ct_volume_raw.shape, 
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
            warnings.warn(f"Skipping {plan_dir}: Missing PTV56 (use --relax_filter)")
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
        # Load dose grid
        # =================================================================
        rtdose_paths = find_dicom_files(plan_dir, ['RD', 'RTDOSE', 'rtdose'])
        if not rtdose_paths:
            raise ValueError(f"No RTDOSE file in {plan_dir}")
        dose_ds = pydicom.dcmread(rtdose_paths[0])
        dose_scaling = float(dose_ds.DoseGridScaling)
        dose_array = dose_ds.pixel_array.astype(np.float32) * dose_scaling
        if dose_array.ndim != 3:
            raise ValueError("Dose not 3D")

        dose_position = [float(x) for x in dose_ds.ImagePositionPatient]
        dose_pixel_spacing = [float(x) for x in dose_ds.PixelSpacing]

        if 'GridFrameOffsetVector' in dose_ds:
            offsets = [float(o) for o in dose_ds.GridFrameOffsetVector]
            if len(offsets) != dose_array.shape[0]:
                raise ValueError("GridFrameOffsetVector length mismatch")
            dose_z_positions = [dose_position[2] + offsets[i] for i in range(len(offsets))]
        else:
            warnings.warn("No GridFrameOffsetVector")
            dose_z_positions = [dose_position[2]]
            if dose_array.shape[0] != 1:
                raise ValueError("Multi-frame dose without GridFrameOffsetVector")

        sort_idx = np.argsort(dose_z_positions)
        if not np.all(sort_idx == np.arange(len(sort_idx))):
            warnings.warn("Sorting dose z-positions")
            dose_array = dose_array[sort_idx]
            dose_z_positions = [dose_z_positions[i] for i in sort_idx]

        if len(dose_z_positions) > 1:
            diffs = np.diff(dose_z_positions)
            dose_z_spacing = float(np.mean(diffs))
        else:
            dose_z_spacing = ct_spacing[2]

        dose_grid_spacing = (dose_pixel_spacing[0], dose_pixel_spacing[1], dose_z_spacing)
        print(f"Dose grid: shape={dose_array.shape}, spacing={dose_grid_spacing}")

        # =================================================================
        # Resample dose to CT grid using B-spline (the one unavoidable
        # interpolation — upsampling coarse dose to finer CT resolution)
        # =================================================================
        # Create SimpleITK images for dose and CT reference
        dose_image = sitk.GetImageFromArray(dose_array)  # (Z, Y, X) ordering
        dose_image.SetSpacing((dose_pixel_spacing[0], dose_pixel_spacing[1], dose_z_spacing))
        dose_image.SetOrigin((dose_position[0], dose_position[1], dose_z_positions[0]))

        ct_array_sitk = ct_volume_raw.transpose(2, 0, 1)  # (Y, X, Z) → (Z, Y, X)
        ct_reference = sitk.GetImageFromArray(ct_array_sitk.astype(np.float32))
        ct_reference.SetSpacing((ct_spacing[0], ct_spacing[1], ct_spacing[2]))
        ct_reference.SetOrigin((position[0], position[1], position[2]))

        # B-spline interpolation: less smoothing at boundaries than trilinear
        dose_on_ct_grid = resample_image(dose_image, ct_reference, sitk.sitkBSpline)
        dose_on_ct = sitk.GetArrayFromImage(dose_on_ct_grid).transpose(1, 2, 0)  # (Z,Y,X)→(Y,X,Z)
        dose_on_ct = np.maximum(dose_on_ct, 0)  # Clamp negative artifacts from B-spline

        print(f"Dose resampled to CT grid: {dose_on_ct.shape} (native CT resolution)")

        # =================================================================
        # Compute crop box (centered on prostate, dynamic Z)
        # =================================================================
        crop = compute_crop_box(masks, ct_spacing, inplane_size=inplane_size,
                                z_margin_mm=z_margin_mm)
        ys, ye = crop['y_start'], crop['y_end']
        xs, xe = crop['x_start'], crop['x_end']
        zs, ze = crop['z_start'], crop['z_end']

        print(f"Crop box: Y=[{ys}:{ye}], X=[{xs}:{xe}], Z=[{zs}:{ze}]")
        print(f"  Output shape: ({ye-ys}, {xe-xs}, {ze-zs})")
        print(f"  Physical extent: {(ye-ys)*ct_spacing[1]:.0f} x {(xe-xs)*ct_spacing[0]:.0f} x {(ze-zs)*ct_spacing[2]:.0f} mm")

        # =================================================================
        # Crop all volumes (integer indexing — no interpolation)
        # =================================================================
        ct_cropped = ct_volume_raw[ys:ye, xs:xe, zs:ze]
        dose_cropped = dose_on_ct[ys:ye, xs:xe, zs:ze]
        masks_cropped = masks[:, ys:ye, xs:xe, zs:ze]

        output_shape = ct_cropped.shape

        # =================================================================
        # Normalize CT and dose on cropped volumes
        # =================================================================
        ct_volume_normalized = np.clip(ct_cropped, -1000, 3000) / 4000 + 0.5
        dose_volume = dose_cropped / normalization_dose
        masks_resampled = masks_cropped  # Rename for compatibility with downstream code

        print(f"Cropped PTV70 sum: {masks_resampled[0].sum()}, PTV56 sum: {masks_resampled[1].sum()}")

        # =================================================================
        # DVH validation: check PTV70 D95 on preprocessed data
        # =================================================================
        ptv70_mask_bool = masks_resampled[0] > 0
        if ptv70_mask_bool.sum() > 0:
            ptv70_dose_gy = dose_volume[ptv70_mask_bool] * normalization_dose
            ptv70_d95_gy = float(np.percentile(ptv70_dose_gy, 5))  # D95 = 5th percentile
            print(f"PTV70 D95 validation: {ptv70_d95_gy:.1f} Gy (target >= 64.0 Gy)")
            if ptv70_d95_gy < 64.0:
                warnings.warn(f"WARNING: PTV70 D95 = {ptv70_d95_gy:.1f} Gy < 64.0 Gy — "
                             f"possible pipeline artifact for {plan_id}")

            # Check dose coverage: warn if >5% of PTV voxels have zero dose
            zero_dose_frac = float((ptv70_dose_gy < 0.1).sum()) / ptv70_mask_bool.sum()
            if zero_dose_frac > 0.05:
                warnings.warn(f"WARNING: {zero_dose_frac:.1%} of PTV70 voxels have near-zero "
                             f"dose — dose grid may not fully cover PTV for {plan_id}")
        
        # =================================================================
        # Compute SDFs (using actual CT spacing, not hardcoded)
        # =================================================================
        masks_sdf = None
        sdf_validation = None
        if compute_sdfs:
            actual_spacing = (ct_spacing[1], ct_spacing[0], ct_spacing[2])  # Y, X, Z
            print(f"Computing SDFs (clip={sdf_clip_mm}mm, spacing={actual_spacing})...")
            masks_sdf = np.zeros((len(structure_map),) + output_shape, dtype=np.float32)
            for ch in range(len(structure_map)):
                masks_sdf[ch] = compute_sdf(masks_resampled[ch], spacing_mm=actual_spacing,
                                            clip_mm=sdf_clip_mm)
            print(f"SDFs computed: shape {masks_sdf.shape}")
            
            # Validate SDFs
            sdf_validation = validate_sdf(masks_sdf, masks_resampled, STRUCTURE_CHANNELS)
            sdf_valid = all(v.get('valid', True) for v in sdf_validation.values())
            if not sdf_valid:
                warnings.warn(f"SDF validation issues detected for {plan_id}")
        
        # =================================================================
        # Validate
        # =================================================================
        validation = validate_preprocessed_data(ct_volume_normalized, dose_volume, 
                                                 masks_resampled, normalization_dose,
                                                 ct_raw_min, ct_raw_max)
        
        if not validation['all_critical_passed']:
            warning_msg = f"Validation failed for {plan_id}"
            warnings.warn(warning_msg)
            if strict_validation:
                return None
        
        # =================================================================
        # Build metadata
        # =================================================================
        
        # Prepare beam_geometry for metadata (without numpy arrays)
        beam_geometry_metadata = None
        mlc_arrays = {}
        
        if beam_geometry is not None:
            beam_geometry_metadata = {
                'num_beams': beam_geometry['num_beams'],
                'total_mu': beam_geometry['total_mu'],
                'plan_label': beam_geometry['plan_label'],
                'mlc_extracted': beam_geometry.get('mlc_extracted', False),
                'beams': []
            }
            
            for beam_idx, beam in enumerate(beam_geometry['beams']):
                # Extract MLC arrays to save separately (more efficient)
                if 'mlc_bank_a' in beam and beam['mlc_bank_a'] is not None:
                    mlc_arrays[f'beam{beam_idx}_mlc_a'] = beam['mlc_bank_a']
                    mlc_arrays[f'beam{beam_idx}_mlc_b'] = beam['mlc_bank_b']
                
                # Create clean beam dict for metadata (no numpy arrays)
                beam_meta = {k: v for k, v in beam.items() 
                            if k not in ['mlc_bank_a', 'mlc_bank_b', 'control_points']}
                beam_geometry_metadata['beams'].append(beam_meta)
        
        metadata = {
            'case_id': plan_id,
            'processed_date': datetime.now().isoformat(),
            'script_version': '2.3.0',
            'prescription_info': prescription_info,
            'case_type': case_type_info,
            'normalization_dose_gy': normalization_dose,
            # v2.3: native spacing and crop (replaces target_shape/target_spacing)
            'voxel_spacing_mm': (float(ct_spacing[1]), float(ct_spacing[0]), float(ct_spacing[2])),
            'volume_shape': output_shape,
            'crop_box': crop,
            'dose_grid_spacing_mm': tuple(float(s) for s in dose_grid_spacing),
            'sdf_clip_mm': sdf_clip_mm if compute_sdfs else None,
            'sdf_validation': sdf_validation,
            'validation': {k: v for k, v in validation.items()
                          if k not in ['truncation_info']},
            'truncation_info': validation.get('truncation_info', {}),
            'structure_channels': STRUCTURE_CHANNELS,
            'beam_geometry': beam_geometry_metadata,  # Phase 2 data (no numpy arrays)
        }
        
        # =================================================================
        # Save .npz
        # =================================================================
        output_path = os.path.join(output_dir, f'{plan_id}.npz')
        
        save_dict = {
            'ct': ct_volume_normalized.astype(np.float32),
            'masks': masks_resampled,
            'dose': dose_volume.astype(np.float32),
            'constraints': constraints,
            'metadata': metadata,
        }
        
        if masks_sdf is not None:
            save_dict['masks_sdf'] = masks_sdf
        
        # Add MLC arrays as separate keys (efficient storage)
        for key, arr in mlc_arrays.items():
            save_dict[key] = arr
        
        np.savez_compressed(output_path, **save_dict)  # Use compressed to save space with MLC data
        
        # =================================================================
        # Debug plots
        # =================================================================
        if not skip_plots:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            mid = output_shape[2] // 2
            
            # Row 1: CT, Dose, PTVs, SDFs
            axs[0, 0].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axs[0, 0].set_title('CT')
            
            axs[0, 1].imshow(dose_volume[:, :, mid] * normalization_dose, cmap='hot', vmin=0, vmax=75)
            axs[0, 1].set_title(f'Dose (Gy)')
            
            axs[0, 2].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axs[0, 2].contour(masks_resampled[0][:, :, mid], colors='red', linewidths=1)
            if masks_resampled[1][:, :, mid].sum() > 0:
                axs[0, 2].contour(masks_resampled[1][:, :, mid], colors='orange', linewidths=1)
            axs[0, 2].set_title('PTV70/56')
            
            if masks_sdf is not None:
                axs[0, 3].imshow(masks_sdf[0][:, :, mid], cmap='RdBu', vmin=-1, vmax=1)
                axs[0, 3].set_title('PTV70 SDF')
            else:
                axs[0, 3].axis('off')
            
            # Row 2: Overlay, OARs, SDF detail, Summary
            axs[1, 0].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            dose_masked = np.ma.masked_where(dose_volume[:, :, mid] < 0.1, dose_volume[:, :, mid])
            axs[1, 0].imshow(dose_masked * normalization_dose, cmap='jet', alpha=0.5, vmin=0, vmax=75)
            axs[1, 0].set_title('CT + Dose')
            
            axs[1, 1].imshow(ct_volume_normalized[:, :, mid], cmap='gray', vmin=0, vmax=1)
            if masks_resampled[3][:, :, mid].sum() > 0:
                axs[1, 1].contour(masks_resampled[3][:, :, mid], colors='brown', linewidths=1)
            if masks_resampled[4][:, :, mid].sum() > 0:
                axs[1, 1].contour(masks_resampled[4][:, :, mid], colors='yellow', linewidths=1)
            axs[1, 1].set_title('Rectum/Bladder')
            
            if masks_sdf is not None:
                axs[1, 2].imshow(masks_sdf[3][:, :, mid], cmap='RdBu', vmin=-1, vmax=1)
                axs[1, 2].set_title('Rectum SDF')
            else:
                axs[1, 2].axis('off')
            
            # Summary
            axs[1, 3].axis('off')
            val_text = f"Case: {plan_id}\n"
            val_text += f"Type: {case_type_info['type']}\n"
            val_text += f"Rx: {normalization_dose:.1f} Gy\n"
            if validation.get('ptv70_dose_mean'):
                val_text += f"PTV70 mean: {validation['ptv70_dose_mean']:.3f}\n"
            val_text += f"Validation: {'PASS' if validation['all_critical_passed'] else 'FAIL'}\n"
            if beam_geometry:
                val_text += f"Beams: {beam_geometry['num_beams']}, MU: {beam_geometry['total_mu']:.0f}"
            axs[1, 3].text(0.1, 0.5, val_text, fontsize=11, verticalalignment='center', family='monospace')
            
            for ax in axs.flat:
                if ax != axs[1, 3]:
                    ax.axis('off')
            
            plt.suptitle(f'{plan_id} - {case_type_info["type"]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'debug_{plan_id}.png'), dpi=150)
            plt.close()
        
        # =================================================================
        # Summary
        # =================================================================
        print(f"Processed {plan_dir} -> {output_path}")
        print(f"  Validation: {'PASS' if validation['all_critical_passed'] else 'FAIL'}")
        if masks_sdf is not None:
            print(f"  SDFs: computed")
        if beam_geometry:
            n_beams = beam_geometry['num_beams']
            mlc_info = "with MLC" if beam_geometry.get('mlc_extracted') else "no MLC"
            print(f"  Beam geometry: {n_beams} beams extracted ({mlc_info})")
        if mlc_arrays:
            print(f"  MLC arrays: {len(mlc_arrays)} arrays saved")
        
        return output_path
        
    except Exception as e:
        warnings.warn(f"Error processing {plan_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_preprocess(input_base_dir, output_dir, mapping_file='oar_mapping.json',
                     relax_filter=False, skip_plots=False, strict_validation=False,
                     compute_sdfs=True, extract_beams=True, sdf_clip_mm=DEFAULT_SDF_CLIP_MM,
                     inplane_size=DEFAULT_INPLANE_SIZE, z_margin_mm=DEFAULT_Z_MARGIN_MM):
    """Batch process multiple plan directories."""
    structure_map = load_json_mapping(mapping_file)
    plan_dirs = sorted([os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir)
                        if os.path.isdir(os.path.join(input_base_dir, d))])

    processed = []
    failed = []

    for plan_dir in plan_dirs:
        result = preprocess_dicom_rt(plan_dir, output_dir, structure_map,
                                      relax_filter=relax_filter,
                                      skip_plots=skip_plots,
                                      strict_validation=strict_validation,
                                      compute_sdfs=compute_sdfs,
                                      extract_beams=extract_beams,
                                      sdf_clip_mm=sdf_clip_mm,
                                      inplane_size=inplane_size,
                                      z_margin_mm=z_margin_mm)
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
    
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    summary = {
        'processed_date': datetime.now().isoformat(),
        'script_version': '2.3.0',
        'total_cases': len(plan_dirs),
        'processed': len(processed),
        'failed': len(failed),
        'failed_cases': failed,
        'settings': {
            'relax_filter': relax_filter,
            'strict_validation': strict_validation,
            'compute_sdfs': compute_sdfs,
            'extract_beams': extract_beams,
            'extract_mlc': extract_beams,
            'sdf_clip_mm': sdf_clip_mm if compute_sdfs else None,
            'inplane_size': inplane_size,
            'z_margin_mm': z_margin_mm,
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nBatch summary saved to: {summary_path}")


def get_default_paths():
    """
    Auto-detect the best default paths based on environment.

    Priority order:
    1. Local ./data/raw and ./processed (if data exists)
    2. External drive fallback (if mounted)

    Returns:
        tuple: (input_dir, output_dir)
    """
    # Get script directory to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Local paths (clean workstation)
    local_input = os.path.join(project_root, "data", "raw")
    local_output = os.path.join(project_root, "processed")

    # External drive paths (WSL)
    external_input = "/mnt/i/anonymized_dicom"
    external_output = "/mnt/i/processed_npz"

    # Check for local data directory with case folders
    if os.path.isdir(local_input):
        # Check if it contains actual case folders (not just symlinks to /mnt/i)
        has_local_cases = any(
            os.path.isdir(os.path.join(local_input, d)) and
            not os.path.islink(os.path.join(local_input, d))
            for d in os.listdir(local_input) if d.startswith("case_")
        )
        if has_local_cases:
            print(f"[Auto-detect] Using local paths (found case folders in {local_input})")
            return local_input, local_output

    # Check for external drive
    if os.path.isdir(external_input):
        print(f"[Auto-detect] Using external drive paths ({external_input})")
        return external_input, external_output

    # Fallback to local paths (user will need to populate or specify)
    print(f"[Auto-detect] Defaulting to local paths (no data found yet)")
    return local_input, local_output


if __name__ == "__main__":
    # Get environment-appropriate defaults
    default_input, default_output = get_default_paths()

    parser = argparse.ArgumentParser(description="Preprocess DICOM-RT to .npz for VMAT diffusion model (v2.3.0 - Crop Pipeline)")
    parser.add_argument("--input_dir", default=default_input,
                        help=f"Directory containing DICOM-RT case folders (default: {default_input})")
    parser.add_argument("--output_dir", default=default_output,
                        help=f"Output directory for .npz files (default: {default_output})")
    parser.add_argument("--mapping_file", default="oar_mapping.json")
    parser.add_argument("--relax_filter", action="store_true", help="Process cases without PTV56")
    parser.add_argument("--skip_plots", action="store_true", help="Skip debug PNGs")
    parser.add_argument("--strict_validation", action="store_true", help="Fail on validation issues")
    parser.add_argument("--no_sdf", action="store_true", help="Skip SDF computation")
    parser.add_argument("--no_beams", action="store_true", help="Skip beam/MLC geometry extraction")
    parser.add_argument("--sdf_clip_mm", type=float, default=DEFAULT_SDF_CLIP_MM,
                        help=f"SDF clip distance in mm (default: {DEFAULT_SDF_CLIP_MM})")
    parser.add_argument("--inplane_size", type=int, default=DEFAULT_INPLANE_SIZE,
                        help=f"In-plane crop size in voxels (default: {DEFAULT_INPLANE_SIZE})")
    parser.add_argument("--z_margin_mm", type=float, default=DEFAULT_Z_MARGIN_MM,
                        help=f"Axial margin beyond structures in mm (default: {DEFAULT_Z_MARGIN_MM})")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    batch_preprocess(input_dir, output_dir, args.mapping_file,
                     args.relax_filter, args.skip_plots, args.strict_validation,
                     compute_sdfs=not args.no_sdf,
                     extract_beams=not args.no_beams,
                     sdf_clip_mm=args.sdf_clip_mm,
                     inplane_size=args.inplane_size,
                     z_margin_mm=args.z_margin_mm)


# =============================================================================
# FUTURE TODOs
# =============================================================================
# TODO: Parallelize SDF computation with joblib for HPC (1000+ cases)
# TODO: Add pymedphys gamma subsample in validation (defer to model evaluation)
# TODO: Fluence map generation from MLC sequences
# TODO: Support for PTV50.4 (channel 8) for 3-level SIB
