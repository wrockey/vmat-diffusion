"""
Centralized evaluation constants, data loading, and shared types.

Single source of truth for structure definitions, spacing defaults,
dose utilities, and standardized result types. All evaluation modules
import from here.

Version: 1.0.0
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Framework Version
# =============================================================================

FRAMEWORK_VERSION = "1.0.0"


# =============================================================================
# Structure Definitions — THE ONE DEFINITION
# =============================================================================

STRUCTURE_CHANNELS: Dict[int, str] = {
    0: 'PTV70',
    1: 'PTV56',
    2: 'Prostate',
    3: 'Rectum',
    4: 'Bladder',
    5: 'Femur_L',
    6: 'Femur_R',
    7: 'Bowel',
}

STRUCTURE_INDEX: Dict[str, int] = {v: k for k, v in STRUCTURE_CHANNELS.items()}

PTV_STRUCTURES: List[str] = ['PTV70', 'PTV56']
OAR_STRUCTURES: List[str] = ['Rectum', 'Bladder', 'Femur_L', 'Femur_R', 'Bowel']
ALL_STRUCTURES: List[str] = list(STRUCTURE_CHANNELS.values())


# =============================================================================
# Physical Constants
# =============================================================================

DEFAULT_SPACING_MM: Tuple[float, float, float] = (1.0, 1.0, 2.0)

PRIMARY_PRESCRIPTION_GY: float = 70.0
SECONDARY_PRESCRIPTION_GY: float = 56.0

# Minimum voxels for reliable percentile-based metrics
MIN_VOXELS_RELIABLE: int = 100


# =============================================================================
# Data Loading & Spacing
# =============================================================================

def get_spacing_from_metadata(metadata) -> Tuple[float, float, float]:
    """
    Extract voxel spacing from NPZ metadata with backwards-compatible fallback.

    Fallback chain:
        1. voxel_spacing_mm (v2.3+ native spacing)
        2. target_spacing_mm (v2.2 resampled spacing)
        3. DEFAULT_SPACING_MM (1.0, 1.0, 2.0)

    Args:
        metadata: dict or numpy array from npz['metadata']

    Returns:
        (y_spacing, x_spacing, z_spacing) in mm
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


@dataclass
class CaseData:
    """Loaded case data ready for evaluation."""
    case_id: str
    ct: np.ndarray              # (Y, X, Z) float32
    dose: np.ndarray            # (Y, X, Z) float32, normalized [0, 1]
    masks: np.ndarray           # (8, Y, X, Z) uint8 binary masks
    masks_sdf: np.ndarray       # (8, Y, X, Z) float32 signed distance fields
    spacing_mm: Tuple[float, float, float]
    metadata: dict
    constraints: Optional[np.ndarray] = None  # (13,) float32


def load_case(npz_path: str) -> CaseData:
    """
    Load a case from NPZ file.

    Args:
        npz_path: Path to the NPZ file

    Returns:
        CaseData with all fields populated
    """
    from pathlib import Path

    data = np.load(npz_path, allow_pickle=True)

    metadata = data['metadata'].item() if 'metadata' in data.files else {}
    spacing = get_spacing_from_metadata(metadata)

    case_id = Path(npz_path).stem

    return CaseData(
        case_id=case_id,
        ct=data['ct'],
        dose=data['dose'],
        masks=data['masks'],
        masks_sdf=data['masks_sdf'] if 'masks_sdf' in data.files else None,
        spacing_mm=spacing,
        metadata=metadata,
        constraints=data['constraints'] if 'constraints' in data.files else None,
    )


# =============================================================================
# Dose Utilities
# =============================================================================

def denormalize_dose(dose_normalized: np.ndarray,
                     rx_dose_gy: float = PRIMARY_PRESCRIPTION_GY) -> np.ndarray:
    """Convert normalized dose [0, 1] to absolute dose in Gy."""
    return dose_normalized * rx_dose_gy


def get_structure_mask(masks: np.ndarray, structure_name: str) -> np.ndarray:
    """
    Get boolean mask for a structure by name.

    Args:
        masks: (8, Y, X, Z) binary mask array
        structure_name: e.g. 'PTV70', 'Rectum'

    Returns:
        Boolean mask (Y, X, Z)
    """
    if structure_name not in STRUCTURE_INDEX:
        raise ValueError(
            f"Unknown structure '{structure_name}'. "
            f"Valid names: {list(STRUCTURE_INDEX.keys())}"
        )
    idx = STRUCTURE_INDEX[structure_name]
    if idx >= masks.shape[0]:
        raise ValueError(
            f"Mask array has {masks.shape[0]} channels, "
            f"but structure '{structure_name}' is at index {idx}"
        )
    return masks[idx] > 0


def validate_dose_array(dose_gy: np.ndarray, label: str = "dose") -> List[str]:
    """
    Validate a dose array, returning a list of warnings.

    Checks for NaN, Inf, negative values, and suspiciously high values.

    Args:
        dose_gy: Dose array in Gy
        label: Label for warning messages (e.g. "predicted", "target")

    Returns:
        List of warning strings (empty if all OK)
    """
    issues = []

    nan_count = np.isnan(dose_gy).sum()
    if nan_count > 0:
        issues.append(f"{label}: {nan_count} NaN values detected")

    inf_count = np.isinf(dose_gy).sum()
    if inf_count > 0:
        issues.append(f"{label}: {inf_count} Inf values detected")

    neg_count = (dose_gy < 0).sum()
    if neg_count > 0:
        issues.append(
            f"{label}: {neg_count} negative values "
            f"(min={float(dose_gy.min()):.3f} Gy)"
        )

    max_dose = float(np.nanmax(dose_gy)) if dose_gy.size > 0 else 0.0
    if max_dose > PRIMARY_PRESCRIPTION_GY * 1.2:
        issues.append(
            f"{label}: max dose {max_dose:.1f} Gy exceeds "
            f"120% of Rx ({PRIMARY_PRESCRIPTION_GY * 1.2:.1f} Gy)"
        )

    return issues


def sanitize_dose(dose_gy: np.ndarray) -> np.ndarray:
    """
    Sanitize dose array: clamp negatives to 0, replace NaN/Inf with 0.

    Issues a warning if any values were modified.

    Returns:
        Sanitized copy of the dose array
    """
    dose_clean = dose_gy.copy()
    modified = False

    nan_inf_mask = ~np.isfinite(dose_clean)
    if nan_inf_mask.any():
        dose_clean[nan_inf_mask] = 0.0
        modified = True

    neg_mask = dose_clean < 0
    if neg_mask.any():
        warnings.warn(
            f"Clamping {neg_mask.sum()} negative dose values to 0 "
            f"(min was {float(dose_gy[neg_mask].min()):.3f} Gy)"
        )
        dose_clean[neg_mask] = 0.0
        modified = True

    return dose_clean


# =============================================================================
# Standardized Result Types
# =============================================================================

@dataclass
class EvaluationResult:
    """Standardized result container for a single case evaluation."""
    case_id: str
    spacing_mm: Tuple[float, float, float]
    dose_metrics: Dict = field(default_factory=dict)
    gamma: Dict = field(default_factory=dict)
    dvh_metrics: Dict = field(default_factory=dict)
    clinical_constraints: Dict = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)
    framework_version: str = FRAMEWORK_VERSION

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'case_id': self.case_id,
            'spacing_mm': list(self.spacing_mm),
            'dose_metrics': self.dose_metrics,
            'gamma': self.gamma,
            'dvh_metrics': self.dvh_metrics,
            'clinical_constraints': self.clinical_constraints,
            'validation_warnings': self.validation_warnings,
            'framework_version': self.framework_version,
        }
