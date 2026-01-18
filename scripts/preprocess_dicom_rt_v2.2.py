"""
Preprocess DICOM-RT VMAT plans to .npz for diffusion model training.

Functionality: Resamples/aligns CT, masks, dose to fixed grid (512x512x256 @ 1x1x2mm),
centers on prostate/PTV70, normalizes, computes SDFs, extracts FULL beam geometry
including MLC sequences, and saves with constraints and metadata.

Version: 2.2.0 - Phase 2-ready with full MLC/dose rate extraction

Changes from v2.1.1:
- Full MLC leaf position extraction at every control point
- Dose rate extraction per control point
- Jaw positions (X1, X2, Y1, Y2) per control point
- Cumulative meterset weight (MU) per control point
- Gantry angles per control point
- Beam energy extraction
- Treatment machine name
- Leaf boundary information
- MLC arrays stored as separate npz keys for efficient access
- Uses np.savez_compressed for smaller file sizes

Key Features:
- Signed Distance Fields (SDFs) for all structures
- FULL beam geometry from RP file (MLC, dose rate, jaws) - Phase 2 ready
- Improved truncation quantification
- HU clipping detection

.npz File Structure (v2.2.0):
- ct: (512, 512, 256) float32 - CT normalized [0,1]
- dose: (512, 512, 256) float32 - Dose normalized to Rx
- masks: (8, 512, 512, 256) uint8 - Binary structure masks
- masks_sdf: (8, 512, 512, 256) float32 - Signed distance fields [-1,1]
- constraints: (13,) float32 - Prescription + OAR constraints
- metadata: dict - Case info, beam geometry summary, validation
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

# SDF parameters (can be overridden via command line)
DEFAULT_SDF_CLIP_MM = 50.0  # Default clip at ±50mm (captures relevant dose gradients)
# Note: For tighter structures or faster falloff, consider 30mm

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_mapping(mapping_file='oar_mapping.json'):
    """Load the OAR mapping JSON file for contour variations."""
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"{mapping_file} not found; run generate_mapping_json.py and edit")
    with open(mapping_file, 'r') as f:
        return json.load(f)


def get_ct_volume_and_metadata(plan_dir):
    """Load and stack CT slices, sort by Z, compute metadata."""
    ct_files = [f for f in os.listdir(plan_dir) if f.startswith('CT')]
    if not ct_files:
        raise ValueError(f"No CT files in {plan_dir}")
    ct_paths = [os.path.join(plan_dir, f) for f in ct_files]
    ct_ds = [pydicom.dcmread(p) for p in ct_paths]
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
        rp_files = [f for f in os.listdir(plan_dir) if f.startswith('RP')]
        if not rp_files:
            return None
            
        rp = pydicom.dcmread(os.path.join(plan_dir, rp_files[0]))
        
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


def validate_preprocessed_data(ct_volume, dose_volume, masks_resampled, prescription_dose, 
                                ct_raw_min=None, ct_raw_max=None):
    """
    Validate preprocessed data before saving.
    
    Returns:
        dict: Validation results with pass/fail for each check
    """
    checks = {}
    
    # CT checks
    checks['ct_range_valid'] = (ct_volume.min() >= 0) and (ct_volume.max() <= 1)
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


def preprocess_dicom_rt(plan_dir, output_dir, structure_map, target_shape=(512, 512, 256), 
                        target_spacing=(1.0, 1.0, 2.0), relax_filter=False, skip_plots=False,
                        strict_validation=False, compute_sdfs=True, extract_beams=True,
                        sdf_clip_mm=DEFAULT_SDF_CLIP_MM):
    """
    Process a single DICOM-RT plan directory.

    Args:
        plan_dir (str): Path to case dir with DICOM files.
        output_dir (str): Output directory for .npz and PNGs.
        structure_map (dict): OAR mapping from JSON.
        target_shape (tuple): Fixed output shape (default 512x512x256).
        target_spacing (tuple): Target voxel spacing (default 1x1x2 mm).
        relax_filter (bool): Process cases without PTV56.
        skip_plots (bool): Skip debug PNGs.
        strict_validation (bool): Fail on validation issues.
        compute_sdfs (bool): Compute signed distance fields for masks.
        extract_beams (bool): Extract beam geometry from RP file.
        sdf_clip_mm (float): Clip SDF values at ±this distance in mm.

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
        rtstruct_file = [f for f in os.listdir(plan_dir) if f.startswith('RS')][0]
        rtstruct = pydicom.dcmread(os.path.join(plan_dir, rtstruct_file))
        
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
        # Compute centering
        # =================================================================
        prostate_mask = masks[2] if masks[2].sum() > 0 else masks[0]
        
        nonzero = np.nonzero(prostate_mask)
        if len(nonzero[0]) == 0:
            warnings.warn("No prostate/PTV mask; centering on mid-volume")
            mid_y, mid_x, mid_z = [s / 2 for s in prostate_mask.shape]
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
        # Create SimpleITK images
        # =================================================================
        ct_array_sitk = ct_volume_raw.transpose(2, 0, 1)
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
        
        dose_array_sitk = dose_array
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
            dose_array_sitk = dose_array_sitk[sort_idx]
            dose_z_positions = [dose_z_positions[i] for i in sort_idx]
        
        if len(dose_z_positions) > 1:
            diffs = np.diff(dose_z_positions)
            dose_z_spacing = np.mean(diffs)
        else:
            dose_z_spacing = ct_spacing[2]
        
        dose_spacing_sitk = (dose_pixel_spacing[0], dose_pixel_spacing[1], dose_z_spacing)
        dose_origin_sitk = (dose_position[0], dose_position[1], dose_z_positions[0])
        dose_image = sitk.GetImageFromArray(dose_array_sitk)
        dose_image.SetSpacing(dose_spacing_sitk)
        dose_image.SetOrigin(dose_origin_sitk)
        
        # =================================================================
        # Define reference grid
        # =================================================================
        target_size_sitk = (target_shape[1], target_shape[0], target_shape[2])
        target_spacing_sitk = (target_spacing[0], target_spacing[1], target_spacing[2])
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
        ct_array = sitk.GetArrayFromImage(ct_resampled).transpose(1, 2, 0)
        ct_volume_normalized = np.clip(ct_array, -1000, 3000) / 4000 + 0.5
        
        # =================================================================
        # Resample dose
        # =================================================================
        dose_resampled = resample_image(dose_image, reference_image, sitk.sitkLinear)
        dose_volume = sitk.GetArrayFromImage(dose_resampled).transpose(1, 2, 0) / normalization_dose
        dose_volume = np.maximum(dose_volume, 0)
        
        # =================================================================
        # Resample masks
        # =================================================================
        masks_resampled = np.zeros((len(structure_map),) + target_shape, dtype=np.uint8)
        for ch_str, _ in structure_map.items():
            ch = int(ch_str)
            mask_original = masks[ch]
            mask_array_sitk = mask_original.transpose(2, 0, 1)
            mask_image = sitk.GetImageFromArray(mask_array_sitk.astype(np.float32))
            mask_image.SetSpacing(ct_image.GetSpacing())
            mask_image.SetOrigin(ct_image.GetOrigin())
            mask_res = resample_image(mask_image, reference_image, sitk.sitkNearestNeighbor)
            mask_array = sitk.GetArrayFromImage(mask_res).transpose(1, 2, 0)
            masks_resampled[ch] = (mask_array > 0.5).astype(np.uint8)
        
        print(f"Resampled PTV70 sum: {masks_resampled[0].sum()}, PTV56 sum: {masks_resampled[1].sum()}")
        
        # =================================================================
        # Compute SDFs
        # =================================================================
        masks_sdf = None
        sdf_validation = None
        if compute_sdfs:
            print(f"Computing SDFs (clip={sdf_clip_mm}mm)...")
            masks_sdf = np.zeros((len(structure_map),) + target_shape, dtype=np.float32)
            for ch in range(len(structure_map)):
                masks_sdf[ch] = compute_sdf(masks_resampled[ch], spacing_mm=target_spacing, 
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
            'script_version': '2.2.0',
            'prescription_info': prescription_info,
            'case_type': case_type_info,
            'normalization_dose_gy': normalization_dose,
            'target_shape': target_shape,
            'target_spacing_mm': target_spacing,
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
            mid = target_shape[2] // 2
            
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
                     compute_sdfs=True, extract_beams=True, sdf_clip_mm=DEFAULT_SDF_CLIP_MM):
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
                                      sdf_clip_mm=sdf_clip_mm)
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
        'script_version': '2.2.0',
        'total_cases': len(plan_dirs),
        'processed': len(processed),
        'failed': len(failed),
        'failed_cases': failed,
        'settings': {
            'relax_filter': relax_filter,
            'strict_validation': strict_validation,
            'compute_sdfs': compute_sdfs,
            'extract_beams': extract_beams,
            'extract_mlc': extract_beams,  # MLC is now always extracted with beams
            'sdf_clip_mm': sdf_clip_mm if compute_sdfs else None,
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

    parser = argparse.ArgumentParser(description="Preprocess DICOM-RT to .npz for VMAT diffusion model (v2.2.0 - Phase 2 Ready)")
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
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    batch_preprocess(input_dir, output_dir, args.mapping_file, 
                     args.relax_filter, args.skip_plots, args.strict_validation,
                     compute_sdfs=not args.no_sdf,
                     extract_beams=not args.no_beams,
                     sdf_clip_mm=args.sdf_clip_mm)


# =============================================================================
# FUTURE TODOs
# =============================================================================
# TODO: Parallelize SDF computation with joblib for HPC (1000+ cases)
# TODO: Add pymedphys gamma subsample in validation (defer to model evaluation)
# TODO: Fluence map generation from MLC sequences
# TODO: Support for PTV50.4 (channel 8) for 3-level SIB
