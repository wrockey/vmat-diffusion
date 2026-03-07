#!/usr/bin/env python3
"""
Interactive DICOM-RT QA Review Tool for VMAT Dose Prediction Dataset

Scans raw DICOM-RT case directories, matches structures to canonical names,
and provides an interactive review workflow:
  1. Displays CT slice through PTV70 centroid with dose colorwash + contour overlays
  2. Accept/reject cases, correct structure name mappings per case
  3. Outputs JSON manifest and publishable contact sheet PDF (4 cases per page)
  4. Resume support for interrupted sessions

Usage:
    # Interactive review
    python scripts/qa_review_dicom.py --input ~/data/raw_dicom --output ~/data/qa

    # Scan only (no interactive review, just build manifest)
    python scripts/qa_review_dicom.py --input ~/data/raw_dicom --output ~/data/qa --scan-only

    # Generate contact sheet from existing manifest
    python scripts/qa_review_dicom.py --input ~/data/raw_dicom --output ~/data/qa --contact-sheet-only

    # Resume interrupted review
    python scripts/qa_review_dicom.py --input ~/data/raw_dicom --output ~/data/qa --resume

See GitHub Issue #64 for background.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pydicom
import pydicom.config
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Relax pydicom validation for Pinnacle DS VR precision issues
pydicom.config.settings.reading_validation_mode = pydicom.config.WARN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# STRUCTURE DEFINITIONS
# =============================================================================

# 9 canonical structures for the VMAT dose prediction model.
# Patterns are tried in order; first match wins. Exclusion filters remove
# planning aids (rings, expansions, evaluation contours, etc.)

STRUCTURE_DEFS = {
    "PTV70": {
        "required": True,
        "patterns": [
            r"^ptv[\s_-]?7000\b",
            r"^ptv[\s_-]?70\b",
            r"^ptv70",
            r"^planningptv[\s_-]?7000",
            r"^pptv[\s_-]?7000",
            r"^treat[\s_-]?7000",
        ],
        "color": "red",
        "display_order": 0,
    },
    "PTV56": {
        "required": True,
        "patterns": [
            r"^ptv[\s_-]?5600\b",
            r"^ptv[\s_-]?56\b",
            r"^ptv56",
            r"^planningptv[\s_-]?5600",
            r"^pptv[\s_-]?5600",
            r"^treat[\s_-]?5600",
        ],
        "color": "orange",
        "display_order": 1,
    },
    "PTV50.4": {
        "required": False,
        "patterns": [
            r"^ptv[\s_-]?5040\b",
            r"^ptv[\s_-]?50[\s_.-]?4\b",
            r"^planningptv[\s_-]?5040",
            r"^pptv[\s_-]?5040",
            r"^treat[\s_-]?5040",
        ],
        "color": "yellow",
        "display_order": 2,
    },
    "Prostate": {
        "required": False,
        "patterns": [
            r"^prostate$",
            r"^prostate[\s_]",
            r"^ctv[\s_-]?prostate",
        ],
        "color": "cyan",
        "display_order": 3,
    },
    "Rectum": {
        "required": True,
        "patterns": [
            r"^rectum$",
            r"^rectum[\s_]",
            r"^rect\b",
        ],
        "color": "#8B4513",
        "display_order": 4,
    },
    "Bladder": {
        "required": True,
        "patterns": [
            r"^bladder$",
            r"^bladder[\s_]",
        ],
        "color": "#FFD700",
        "display_order": 5,
    },
    "Bowel": {
        "required": False,
        "patterns": [
            r"^bowel",
            r"^small[\s_]?bowel",
            r"^lg[\s_]?bowel",
            r"^large[\s_]?bowel",
            r"^bowel[\s_]?bag",
        ],
        "color": "green",
        "display_order": 6,
    },
    "Femur_L": {
        "required": False,
        "patterns": [
            r"femur[\s_]?head[\s_]?l\b",
            r"^femur[\s_]?l\b",
            r"^fem[\s_]?head[\s_]?l\b",
            r"^l[\s_]?femur",
            r"^l[\s_]?fem[\s_]?head",
            r"^lt[\s_]?femur",
            r"^left[\s_]?femur",
            r"^femoral[\s_]?head[\s_]?l",
        ],
        "color": "magenta",
        "display_order": 7,
    },
    "Femur_R": {
        "required": False,
        "patterns": [
            r"femur[\s_]?head[\s_]?r\b",
            r"^femur[\s_]?r\b",
            r"^fem[\s_]?head[\s_]?r\b",
            r"^r[\s_]?femur",
            r"^r[\s_]?fem[\s_]?head",
            r"^rt[\s_]?femur",
            r"^right[\s_]?femur",
            r"^femoral[\s_]?head[\s_]?r",
        ],
        "color": "#FF69B4",
        "display_order": 8,
    },
}

# Exclusion terms for PTV matching — planning aids, not real targets
PTV_EXCLUSIONS = [
    "ring", "anti", "xxx", "+", "mm", "eval", "expand", "opt",
    "avoid", "minus", "shell", "wall", "prv", "margin", "dose",
    "hot", "cold", "opti", "help", "inner", "outer",
]

# ROI names to always ignore (not real structures)
GLOBAL_EXCLUSIONS = [
    r"ptv[\s_-]?6160",  # PTV6160 — not a target we track
    r"^body$",
    r"^external$",
    r"^couch",
    r"^support",
    r"^fixation",
    r"^artifact",
]


# =============================================================================
# STRUCTURE MATCHING
# =============================================================================

def match_roi(canonical: str, roi_names: list[str],
              overrides: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    """Match a canonical structure name to an actual ROI name.

    Args:
        canonical: Canonical structure name (e.g. "PTV70")
        roi_names: List of all ROI names from RTSTRUCT
        overrides: Optional {canonical: actual_roi_name} corrections for this case

    Returns:
        (actual_roi_name, display_label) or (None, None) if not found
    """
    # Check overrides first
    if overrides and canonical in overrides:
        override_name = overrides[canonical]
        # Verify the override name exists in the ROI list
        for name in roi_names:
            if name.lower().strip() == override_name.lower().strip():
                return name, f"{canonical} ({name})*"
        # Override name not found — fall through to pattern matching
        logger.warning(f"Override '{override_name}' for {canonical} not found in ROI list")

    struct_def = STRUCTURE_DEFS.get(canonical)
    if not struct_def:
        return None, None

    for pattern in struct_def["patterns"]:
        candidates = []
        for name in roi_names:
            lower = name.lower().strip()

            # Check global exclusions
            if any(re.search(excl, lower) for excl in GLOBAL_EXCLUSIONS):
                continue

            # Check PTV-specific exclusions
            if canonical.startswith("PTV"):
                if any(excl in lower for excl in PTV_EXCLUSIONS):
                    continue

            if re.search(pattern, lower):
                candidates.append(name)

        if candidates:
            # Prefer shorter names (less likely to be derivatives)
            best = min(candidates, key=len)
            clean_canonical = canonical.lower().replace(" ", "").replace("_", "").replace(".", "")
            clean_best = best.lower().replace(" ", "").replace("_", "").replace(".", "")
            if clean_best == clean_canonical:
                label = canonical
            else:
                label = f"{canonical} ({best})"
            return best, label

    return None, None


def scan_case(case_dir: str, overrides: dict[str, str] | None = None) -> dict:
    """Scan a single case directory and return structure matching results.

    Returns dict with:
        - matched: {canonical: {actual_name, display_label}}
        - unmatched: [roi_names not matched to any canonical]
        - missing: [required canonical names not found]
        - ptv_config: str describing PTV configuration
        - all_rois: [all ROI names in RTSTRUCT]
        - error: str if case couldn't be scanned
    """
    result = {
        "matched": {},
        "unmatched": [],
        "missing": [],
        "ptv_config": "unknown",
        "all_rois": [],
        "error": None,
    }

    rtstruct_path = os.path.join(case_dir, "RTSTRUCT.dcm")
    if not os.path.exists(rtstruct_path):
        result["error"] = "No RTSTRUCT.dcm found"
        return result

    try:
        rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    except Exception as e:
        result["error"] = f"Failed to read RTSTRUCT: {e}"
        return result

    if not hasattr(rtstruct, "StructureSetROISequence"):
        result["error"] = "RTSTRUCT has no StructureSetROISequence"
        return result

    roi_names = [roi.ROIName for roi in rtstruct.StructureSetROISequence]
    result["all_rois"] = roi_names

    # Match each canonical structure
    matched_actual_names = set()
    for canonical in STRUCTURE_DEFS:
        actual_name, display_label = match_roi(canonical, roi_names, overrides)
        if actual_name:
            result["matched"][canonical] = {
                "actual_name": actual_name,
                "display_label": display_label,
            }
            matched_actual_names.add(actual_name)
        elif STRUCTURE_DEFS[canonical]["required"]:
            result["missing"].append(canonical)

    # Find unmatched ROIs (exclude globally-excluded names)
    for name in roi_names:
        if name in matched_actual_names:
            continue
        lower = name.lower().strip()
        if any(re.search(excl, lower) for excl in GLOBAL_EXCLUSIONS):
            continue
        result["unmatched"].append(name)

    # Determine PTV configuration
    has_ptv70 = "PTV70" in result["matched"]
    has_ptv56 = "PTV56" in result["matched"]
    has_ptv504 = "PTV50.4" in result["matched"]

    if has_ptv70 and has_ptv56 and has_ptv504:
        result["ptv_config"] = "PTV70/PTV56/PTV50.4"
    elif has_ptv70 and has_ptv56:
        result["ptv_config"] = "PTV70/PTV56"
    elif has_ptv70 and has_ptv504:
        result["ptv_config"] = "PTV70/PTV50.4"
    elif has_ptv70:
        result["ptv_config"] = "PTV70 only"
    elif not has_ptv70:
        result["ptv_config"] = "No PTV70"

    return result


# =============================================================================
# CASE DISCOVERY
# =============================================================================

def _looks_like_case_dir(path: str) -> bool:
    """Heuristic: does this directory look like a DICOM-RT case?"""
    if not os.path.isdir(path):
        return False
    # Must have at least one of: RTSTRUCT.dcm, RTDOSE.dcm, images/
    for marker in ["RTSTRUCT.dcm", "RTDOSE.dcm", "images"]:
        if os.path.exists(os.path.join(path, marker)):
            return True
    return False


def discover_cases(input_dir: str) -> list[tuple[str, str]]:
    """Discover case directories, supporting institution subdirectories.

    Returns list of (case_name, case_dir) tuples.
    case_name includes subdirectory prefix for institution subdirs,
    e.g. "institution_b/prostate70gy_0001".
    """
    cases = []

    # Check for institution subdirectories
    subdirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    # Detect layout: flat (cases directly in input_dir) or nested (institution subdirs)
    flat_cases = [d for d in subdirs if _looks_like_case_dir(os.path.join(input_dir, d))]
    institution_dirs = [d for d in subdirs if not _looks_like_case_dir(os.path.join(input_dir, d))]

    if flat_cases and not institution_dirs:
        # Flat layout — all cases directly in input_dir
        for d in sorted(flat_cases):
            cases.append((d, os.path.join(input_dir, d)))
    elif institution_dirs:
        # Nested layout — institution subdirectories
        # Also include any flat cases at root level
        for d in sorted(flat_cases):
            cases.append((d, os.path.join(input_dir, d)))

        for inst_dir in sorted(institution_dirs):
            inst_path = os.path.join(input_dir, inst_dir)
            for d in sorted(os.listdir(inst_path)):
                case_path = os.path.join(inst_path, d)
                if _looks_like_case_dir(case_path):
                    case_name = f"{inst_dir}/{d}"
                    cases.append((case_name, case_path))
    else:
        logger.warning(f"No case directories found in {input_dir}")

    return cases


# =============================================================================
# CT / DOSE / CONTOUR LOADING (adapted from dicom-collect contact_sheet.py)
# =============================================================================

def discover_ct_files(case_dir: str) -> list[str]:
    """Find all CT files under case_dir/images/CT.*/ directories."""
    images_dir = os.path.join(case_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    ct_files = []
    for series_dir_name in os.listdir(images_dir):
        series_path = os.path.join(images_dir, series_dir_name)
        if not os.path.isdir(series_path):
            continue
        for fname in os.listdir(series_path):
            fpath = os.path.join(series_path, fname)
            if "Zone.Identifier" in fname:
                continue
            if os.path.isfile(fpath):
                ct_files.append(fpath)
    return ct_files


def load_ct_slice_headers(ct_files: list[str]) -> list[tuple[float, str]]:
    """Load CT headers (no pixels) and return sorted (z_position, filepath)."""
    slices = []
    for path in ct_files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            z = float(ds.ImagePositionPatient[2])
            slices.append((z, path))
        except Exception:
            continue
    slices.sort(key=lambda x: x[0])
    return slices


def load_ct_at_z(ct_slices: list[tuple[float, str]], target_z: float):
    """Load the CT slice nearest to target_z. Returns (pixel_array_HU, ds)."""
    if not ct_slices:
        return None, None
    idx = int(np.argmin([abs(z - target_z) for z, _ in ct_slices]))
    _, path = ct_slices[idx]
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixels * slope + intercept
    return hu, ds


def interpolate_dose_to_ct(dose_ds, ct_ds, slice_z: float,
                           prescribed_gy: float = 70.0):
    """Interpolate dose grid onto CT pixel coordinates at slice_z.

    Returns normalized dose array (same shape as CT) in [0, 1],
    where 1 = prescribed_gy, or None if dose doesn't cover the slice.
    """
    dose_grid = dose_ds.pixel_array.astype(np.float64)
    dose_grid_scaling = float(getattr(dose_ds, "DoseGridScaling", 1.0))
    dose_gy = dose_grid * dose_grid_scaling

    # Dose grid geometry
    dose_ipp = [float(x) for x in dose_ds.ImagePositionPatient]
    dose_ps = [float(x) for x in dose_ds.PixelSpacing]

    # Dose Z positions via GridFrameOffsetVector
    gfov = [float(x) for x in dose_ds.GridFrameOffsetVector]
    dose_z_positions = [dose_ipp[2] + offset for offset in gfov]

    # Find closest dose frame
    frame_idx = int(np.argmin([abs(dz - slice_z) for dz in dose_z_positions]))
    if abs(dose_z_positions[frame_idx] - slice_z) > 5.0:
        return None

    dose_frame = dose_gy[frame_idx]

    # CT geometry
    ct_ipp = [float(x) for x in ct_ds.ImagePositionPatient]
    ct_ps = [float(x) for x in ct_ds.PixelSpacing]
    ct_rows, ct_cols = int(ct_ds.Rows), int(ct_ds.Columns)

    # Build CT pixel grid in patient coordinates
    ct_col_coords = ct_ipp[0] + np.arange(ct_cols) * ct_ps[1]
    ct_row_coords = ct_ipp[1] + np.arange(ct_rows) * ct_ps[0]

    # Map to dose grid coordinates
    dose_col_indices = (ct_col_coords - dose_ipp[0]) / dose_ps[1]
    dose_row_indices = (ct_row_coords - dose_ipp[1]) / dose_ps[0]

    row_grid, col_grid = np.meshgrid(dose_row_indices, dose_col_indices, indexing="ij")
    dose_on_ct = map_coordinates(dose_frame, [row_grid, col_grid],
                                 order=1, mode="constant", cval=0.0)

    return dose_on_ct / prescribed_gy


def get_contour_at_z(rtstruct_ds, roi_number: int, target_z: float,
                     tolerance: float = 2.0) -> list[np.ndarray]:
    """Get contour rings at the given Z slice.

    Returns list of Nx2 arrays (x, y patient coords).
    """
    rings = []
    if not hasattr(rtstruct_ds, "ROIContourSequence"):
        return rings
    for roi_contour in rtstruct_ds.ROIContourSequence:
        if getattr(roi_contour, "ReferencedROINumber", None) != roi_number:
            continue
        if not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            mean_z = pts[:, 2].mean()
            if abs(mean_z - target_z) <= tolerance:
                rings.append(pts[:, :2])
        break
    return rings


def get_ptv70_centroid_z(rtstruct_ds, roi_number: int) -> float | None:
    """Get the Z centroid of PTV70 for slice selection."""
    if not hasattr(rtstruct_ds, "ROIContourSequence"):
        return None
    for roi_contour in rtstruct_ds.ROIContourSequence:
        if getattr(roi_contour, "ReferencedROINumber", None) != roi_number:
            continue
        if not hasattr(roi_contour, "ContourSequence"):
            continue
        all_z = []
        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            all_z.extend(pts[:, 2].tolist())
        if all_z:
            return float(np.mean(all_z))
    return None


def patient_to_pixel(points_xy: np.ndarray, ct_ds) -> np.ndarray:
    """Convert patient (x, y) coords to CT pixel (col, row) coords."""
    origin_x = float(ct_ds.ImagePositionPatient[0])
    origin_y = float(ct_ds.ImagePositionPatient[1])
    ps_row = float(ct_ds.PixelSpacing[0])
    ps_col = float(ct_ds.PixelSpacing[1])
    cols = (points_xy[:, 0] - origin_x) / ps_col
    rows = (points_xy[:, 1] - origin_y) / ps_row
    return np.column_stack([cols, rows])


def build_roi_map(rtstruct_ds) -> dict[str, int]:
    """Build {ROIName: ROINumber} from StructureSetROISequence."""
    roi_map = {}
    if hasattr(rtstruct_ds, "StructureSetROISequence"):
        for roi in rtstruct_ds.StructureSetROISequence:
            roi_map[roi.ROIName] = roi.ROINumber
    return roi_map


# =============================================================================
# RENDERING
# =============================================================================

def render_review_figure(case_name: str, case_dir: str, scan_result: dict,
                         overrides: dict[str, str] | None = None) -> plt.Figure:
    """Render a full review figure for interactive review.

    Left panel: CT + dose colorwash + contour overlays
    Right panel: Info text (structures matched, PTV config, unmatched ROIs)
    """
    fig, (ax_img, ax_info) = plt.subplots(1, 2, figsize=(14, 7),
                                           gridspec_kw={"width_ratios": [2, 1]})

    if scan_result.get("error"):
        ax_img.text(0.5, 0.5, f"ERROR: {scan_result['error']}",
                    ha="center", va="center", transform=ax_img.transAxes,
                    fontsize=12, color="red")
        ax_img.set_facecolor("black")
        ax_img.axis("off")
        ax_info.axis("off")
        fig.suptitle(case_name, fontsize=14, fontweight="bold")
        return fig

    # Load RTSTRUCT
    rtstruct_ds = pydicom.dcmread(os.path.join(case_dir, "RTSTRUCT.dcm"))
    roi_map = build_roi_map(rtstruct_ds)

    # Find target Z from PTV70 centroid
    target_z = None
    ptv70_match = scan_result["matched"].get("PTV70")
    if ptv70_match:
        actual_name = ptv70_match["actual_name"]
        if actual_name in roi_map:
            target_z = get_ptv70_centroid_z(rtstruct_ds, roi_map[actual_name])

    # Load CT
    ct_files = discover_ct_files(case_dir)
    ct_slices = load_ct_slice_headers(ct_files)

    if not ct_slices:
        ax_img.text(0.5, 0.5, "No CT slices found", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=12, color="red")
        ax_img.set_facecolor("black")
        ax_img.axis("off")
        ax_info.axis("off")
        fig.suptitle(case_name, fontsize=14, fontweight="bold")
        return fig

    # Fallback to middle slice
    if target_z is None:
        mid_idx = len(ct_slices) // 2
        target_z = ct_slices[mid_idx][0]

    hu_array, ct_ds = load_ct_at_z(ct_slices, target_z)
    if hu_array is None:
        ax_img.text(0.5, 0.5, "CT load failed", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=12, color="red")
        ax_img.set_facecolor("black")
        ax_img.axis("off")
        ax_info.axis("off")
        fig.suptitle(case_name, fontsize=14, fontweight="bold")
        return fig

    actual_z = float(ct_ds.ImagePositionPatient[2])
    slice_thickness = float(getattr(ct_ds, "SliceThickness", 2.5))

    # Soft-tissue window: W=400, L=40
    w_min, w_max = -160, 240
    ct_display = np.clip((hu_array - w_min) / (w_max - w_min), 0, 1)
    ax_img.imshow(ct_display, cmap="gray", aspect="equal", interpolation="bilinear")

    # Dose colorwash overlay
    dose_path = os.path.join(case_dir, "RTDOSE.dcm")
    if os.path.exists(dose_path):
        try:
            dose_ds = pydicom.dcmread(dose_path)
            dose_norm = interpolate_dose_to_ct(dose_ds, ct_ds, actual_z)
            if dose_norm is not None:
                # Jet colormap with transparency
                cmap = plt.cm.jet
                dose_clipped = np.clip(dose_norm, 0, 1.2)
                dose_rgba = cmap(dose_clipped / 1.2)
                # Only show dose > 10% of Rx
                mask = dose_norm > 0.1
                dose_rgba[~mask, 3] = 0.0
                dose_rgba[mask, 3] = 0.4
                ax_img.imshow(dose_rgba, aspect="equal", interpolation="bilinear")
        except Exception as e:
            logger.debug(f"Dose overlay failed for {case_name}: {e}")

    # Draw contours
    for canonical in sorted(STRUCTURE_DEFS, key=lambda k: STRUCTURE_DEFS[k]["display_order"]):
        match_info = scan_result["matched"].get(canonical)
        if not match_info:
            continue
        actual_name = match_info["actual_name"]
        if actual_name not in roi_map:
            continue
        color = STRUCTURE_DEFS[canonical]["color"]
        roi_num = roi_map[actual_name]
        rings = get_contour_at_z(rtstruct_ds, roi_num, actual_z,
                                 tolerance=slice_thickness)
        for ring in rings:
            pixel_coords = patient_to_pixel(ring, ct_ds)
            closed = np.vstack([pixel_coords, pixel_coords[0:1]])
            ax_img.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.2)

    ax_img.axis("off")

    # Info panel
    ax_info.axis("off")
    info_lines = []
    info_lines.append(f"Case: {case_name}")
    info_lines.append(f"PTV config: {scan_result['ptv_config']}")
    info_lines.append(f"Slice Z: {actual_z:.1f} mm")
    info_lines.append("")
    info_lines.append("--- Matched Structures ---")
    for canonical in sorted(STRUCTURE_DEFS, key=lambda k: STRUCTURE_DEFS[k]["display_order"]):
        match_info = scan_result["matched"].get(canonical)
        color = STRUCTURE_DEFS[canonical]["color"]
        req = "*" if STRUCTURE_DEFS[canonical]["required"] else ""
        if match_info:
            info_lines.append(f"  [{canonical}{req}] = {match_info['actual_name']}")
        else:
            status = "MISSING" if STRUCTURE_DEFS[canonical]["required"] else "not found"
            info_lines.append(f"  [{canonical}{req}] {status}")

    if scan_result["unmatched"]:
        info_lines.append("")
        info_lines.append("--- Unmatched ROIs ---")
        for name in scan_result["unmatched"][:15]:
            info_lines.append(f"  {name}")
        if len(scan_result["unmatched"]) > 15:
            info_lines.append(f"  ... and {len(scan_result['unmatched']) - 15} more")

    info_text = "\n".join(info_lines)
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=9, fontfamily="monospace", va="top", ha="left")

    fig.suptitle(case_name, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def render_contact_panel(ax, case_name: str, case_dir: str, scan_result: dict,
                         case_status: str = "pending"):
    """Render a compact contact sheet panel (4 per page).

    Shows CT + dose + contours with ROI checklist overlay.
    """
    if scan_result.get("error"):
        ax.text(0.5, 0.5, f"{case_name}\n{scan_result['error']}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=6, color="red")
        ax.set_facecolor("black")
        ax.axis("off")
        return

    # Load RTSTRUCT
    rtstruct_path = os.path.join(case_dir, "RTSTRUCT.dcm")
    if not os.path.exists(rtstruct_path):
        ax.text(0.5, 0.5, f"{case_name}\nNo RTSTRUCT",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=6, color="red")
        ax.set_facecolor("black")
        ax.axis("off")
        return

    rtstruct_ds = pydicom.dcmread(rtstruct_path)
    roi_map = build_roi_map(rtstruct_ds)

    # Target Z from PTV70
    target_z = None
    ptv70_match = scan_result["matched"].get("PTV70")
    if ptv70_match and ptv70_match["actual_name"] in roi_map:
        target_z = get_ptv70_centroid_z(rtstruct_ds,
                                         roi_map[ptv70_match["actual_name"]])

    ct_files = discover_ct_files(case_dir)
    ct_slices = load_ct_slice_headers(ct_files)

    if not ct_slices:
        ax.text(0.5, 0.5, f"{case_name}\nNo CT", ha="center", va="center",
                transform=ax.transAxes, fontsize=6, color="red")
        ax.set_facecolor("black")
        ax.axis("off")
        return

    if target_z is None:
        target_z = ct_slices[len(ct_slices) // 2][0]

    hu_array, ct_ds = load_ct_at_z(ct_slices, target_z)
    if hu_array is None:
        ax.text(0.5, 0.5, f"{case_name}\nCT failed", ha="center", va="center",
                transform=ax.transAxes, fontsize=6, color="red")
        ax.set_facecolor("black")
        ax.axis("off")
        return

    actual_z = float(ct_ds.ImagePositionPatient[2])
    slice_thickness = float(getattr(ct_ds, "SliceThickness", 2.5))

    # Soft-tissue window
    w_min, w_max = -160, 240
    ct_display = np.clip((hu_array - w_min) / (w_max - w_min), 0, 1)
    ax.imshow(ct_display, cmap="gray", aspect="equal", interpolation="bilinear")

    # Dose overlay
    dose_path = os.path.join(case_dir, "RTDOSE.dcm")
    if os.path.exists(dose_path):
        try:
            dose_ds = pydicom.dcmread(dose_path)
            dose_norm = interpolate_dose_to_ct(dose_ds, ct_ds, actual_z)
            if dose_norm is not None:
                cmap = plt.cm.jet
                dose_clipped = np.clip(dose_norm, 0, 1.2)
                dose_rgba = cmap(dose_clipped / 1.2)
                mask = dose_norm > 0.1
                dose_rgba[~mask, 3] = 0.0
                dose_rgba[mask, 3] = 0.35
                ax.imshow(dose_rgba, aspect="equal", interpolation="bilinear")
        except Exception:
            pass

    # Contours
    for canonical in sorted(STRUCTURE_DEFS, key=lambda k: STRUCTURE_DEFS[k]["display_order"]):
        match_info = scan_result["matched"].get(canonical)
        if not match_info:
            continue
        actual_name = match_info["actual_name"]
        if actual_name not in roi_map:
            continue
        color = STRUCTURE_DEFS[canonical]["color"]
        rings = get_contour_at_z(rtstruct_ds, roi_map[actual_name],
                                 actual_z, tolerance=slice_thickness)
        for ring in rings:
            pixel_coords = patient_to_pixel(ring, ct_ds)
            closed = np.vstack([pixel_coords, pixel_coords[0:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=0.6)

    # Case label with status color
    status_colors = {"accepted": "lime", "rejected": "red", "pending": "yellow", "skipped": "gray"}
    status_color = status_colors.get(case_status, "white")
    label = f"{case_name} [{scan_result['ptv_config']}]"
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=4.5,
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="black",
                      alpha=0.7, edgecolor="none"))

    # Status badge
    ax.text(0.98, 0.98, case_status.upper(), transform=ax.transAxes,
            fontsize=5, color=status_color, va="top", ha="right",
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="black",
                      alpha=0.7, edgecolor="none"))

    # ROI checklist (right side, small)
    for i, canonical in enumerate(sorted(STRUCTURE_DEFS,
                                         key=lambda k: STRUCTURE_DEFS[k]["display_order"])):
        match_info = scan_result["matched"].get(canonical)
        if match_info:
            color = "lime"
            text = match_info["display_label"]
        else:
            req = STRUCTURE_DEFS[canonical]["required"]
            color = "red" if req else "#888888"
            text = canonical
        y_pos = 0.92 - i * 0.065
        ax.text(0.98, y_pos, text, transform=ax.transAxes, fontsize=3,
                color=color, va="top", ha="right",
                bbox=dict(boxstyle="square,pad=0.03", facecolor="black",
                          alpha=0.5, edgecolor="none"))

    ax.axis("off")


# =============================================================================
# CONTACT SHEET PDF
# =============================================================================

def generate_contact_sheet(manifest: dict, input_dir: str, output_path: str):
    """Generate a multi-page contact sheet PDF (4 cases per page).

    Includes a summary page at the beginning with dataset statistics.
    """
    cases = manifest.get("cases", {})
    if not cases:
        logger.error("No cases in manifest")
        return

    case_names = sorted(cases.keys())
    panels_per_page = 4  # 2x2 grid
    n_pages = (len(case_names) + panels_per_page - 1) // panels_per_page

    logger.info(f"Generating contact sheet: {len(case_names)} cases, {n_pages + 1} pages")

    with PdfPages(output_path) as pdf:
        # Summary page
        fig = plt.figure(figsize=(8.5, 11))  # portrait letter
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Compute summary stats
        status_counts = {}
        ptv_counts = {}
        for cn, info in cases.items():
            status = info.get("status", "pending")
            ptv = info.get("ptv_config", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            ptv_counts[ptv] = ptv_counts.get(ptv, 0) + 1

        summary_lines = [
            "VMAT Dose Prediction — DICOM-RT QA Review",
            "=" * 50,
            "",
            f"Total cases scanned: {len(cases)}",
            "",
            "Status Breakdown:",
        ]
        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status:12s}: {count}")
        summary_lines.append("")
        summary_lines.append("PTV Configuration:")
        for ptv, count in sorted(ptv_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {ptv:25s}: {count}")

        # Missing structures summary
        missing_counts = {}
        for cn, info in cases.items():
            for m in info.get("missing", []):
                missing_counts[m] = missing_counts.get(m, 0) + 1
        if missing_counts:
            summary_lines.append("")
            summary_lines.append("Missing Required Structures:")
            for struct, count in sorted(missing_counts.items(), key=lambda x: -x[1]):
                summary_lines.append(f"  {struct:12s}: {count} cases")

        summary_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, fontfamily="monospace", va="top", ha="left")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Case panels
        for page_idx in range(n_pages):
            start = page_idx * panels_per_page
            batch = case_names[start:start + panels_per_page]

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))  # landscape
            fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                                wspace=0.02, hspace=0.04)

            for i, ax in enumerate(axes.flat):
                if i < len(batch):
                    cn = batch[i]
                    case_info = cases[cn]
                    case_dir = case_info.get("case_dir", "")

                    # Rebuild scan_result from manifest
                    scan_result = {
                        "matched": case_info.get("matched", {}),
                        "unmatched": case_info.get("unmatched", []),
                        "missing": case_info.get("missing", []),
                        "ptv_config": case_info.get("ptv_config", "unknown"),
                        "error": case_info.get("error"),
                    }
                    status = case_info.get("status", "pending")

                    logger.info(f"  Rendering {cn} ({start + i + 1}/{len(case_names)})")
                    try:
                        render_contact_panel(ax, cn, case_dir, scan_result, status)
                    except Exception as e:
                        logger.error(f"  Error rendering {cn}: {e}")
                        ax.text(0.5, 0.5, f"{cn}\nERROR: {e}",
                                ha="center", va="center", transform=ax.transAxes,
                                fontsize=6, color="red", wrap=True)
                        ax.set_facecolor("black")
                        ax.axis("off")
                else:
                    ax.axis("off")
                    ax.set_facecolor("white")

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            if (page_idx + 1) % 10 == 0:
                logger.info(f"  Page {page_idx + 1}/{n_pages}")

    logger.info(f"Contact sheet saved to {output_path}")


# =============================================================================
# MANIFEST I/O
# =============================================================================

def load_manifest(path: str) -> dict:
    """Load existing manifest JSON, or return empty structure."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cases": {}, "version": 1}


def save_manifest(manifest: dict, path: str):
    """Save manifest to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {path}")


# =============================================================================
# INTERACTIVE REVIEW
# =============================================================================

def parse_corrections(text: str) -> dict[str, str]:
    """Parse correction input like 'PTV70=ActualName, Bladder=BladderFoo'.

    Returns {canonical: actual_name} dict.
    """
    corrections = {}
    if not text.strip():
        return corrections
    for part in text.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        canonical, actual = part.split("=", 1)
        canonical = canonical.strip()
        actual = actual.strip()
        if canonical in STRUCTURE_DEFS and actual:
            corrections[canonical] = actual
        else:
            logger.warning(f"Ignoring correction: '{canonical}={actual}' "
                          f"(canonical must be one of {list(STRUCTURE_DEFS.keys())})")
    return corrections


def interactive_review(manifest: dict, cases: list[tuple[str, str]],
                       output_dir: str):
    """Run interactive review loop.

    Commands:
        Enter / a   — Accept case
        r           — Reject case
        s / skip    — Skip (review later)
        b / back    — Go back to previous case
        c CORR      — Apply corrections (e.g. c PTV70=ActualPTV70, Bladder=BladFoo)
        list        — Show case list with statuses
        q / quit    — Save and quit
    """
    manifest_path = os.path.join(output_dir, "qa_manifest.json")

    # Build case index for navigation
    case_names = [cn for cn, _ in cases]
    case_dirs = {cn: cd for cn, cd in cases}

    # Find first unreviewed case
    current_idx = 0
    for i, cn in enumerate(case_names):
        case_info = manifest["cases"].get(cn, {})
        if case_info.get("status") in ("accepted", "rejected"):
            continue
        current_idx = i
        break

    reviewed = sum(1 for cn in case_names
                   if manifest["cases"].get(cn, {}).get("status") in ("accepted", "rejected"))
    logger.info(f"Starting review: {reviewed}/{len(case_names)} already reviewed")

    plt.ion()  # Interactive mode

    while 0 <= current_idx < len(case_names):
        cn = case_names[current_idx]
        case_dir = case_dirs[cn]
        case_info = manifest["cases"].get(cn, {})
        overrides = case_info.get("corrections", {})

        # Scan (or re-scan with corrections)
        scan_result = scan_case(case_dir, overrides)

        # Update manifest with scan data
        if cn not in manifest["cases"]:
            manifest["cases"][cn] = {}
        manifest["cases"][cn].update({
            "case_dir": case_dir,
            "matched": scan_result["matched"],
            "unmatched": scan_result["unmatched"],
            "missing": scan_result["missing"],
            "ptv_config": scan_result["ptv_config"],
            "all_rois": scan_result["all_rois"],
            "error": scan_result.get("error"),
        })
        if "status" not in manifest["cases"][cn]:
            manifest["cases"][cn]["status"] = "pending"
        if "corrections" not in manifest["cases"][cn]:
            manifest["cases"][cn]["corrections"] = {}

        # Render figure
        fig = render_review_figure(cn, case_dir, scan_result, overrides)
        fig.show()
        plt.pause(0.1)

        # Prompt
        status = manifest["cases"][cn]["status"]
        reviewed = sum(1 for c in case_names
                       if manifest["cases"].get(c, {}).get("status") in ("accepted", "rejected"))
        prompt = (f"\n[{current_idx + 1}/{len(case_names)}] "
                  f"({reviewed} reviewed) {cn} "
                  f"[{status}]\n"
                  f"  (Enter=accept, r=reject, s=skip, b=back, "
                  f"c CORR=correct, list, q=quit): ")

        try:
            user_input = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting...")
            save_manifest(manifest, manifest_path)
            plt.close("all")
            return

        plt.close(fig)

        if user_input in ("", "a", "accept"):
            manifest["cases"][cn]["status"] = "accepted"
            current_idx += 1
        elif user_input in ("r", "reject"):
            manifest["cases"][cn]["status"] = "rejected"
            current_idx += 1
        elif user_input in ("s", "skip"):
            manifest["cases"][cn]["status"] = "skipped"
            current_idx += 1
        elif user_input in ("b", "back"):
            if current_idx > 0:
                current_idx -= 1
            else:
                print("Already at first case.")
        elif user_input.startswith("c "):
            correction_text = user_input[2:]
            corrections = parse_corrections(correction_text)
            if corrections:
                manifest["cases"][cn].setdefault("corrections", {}).update(corrections)
                print(f"  Applied corrections: {corrections}")
                print("  Re-scanning with corrections...")
                # Don't advance — will re-render with corrections
            else:
                print("  No valid corrections parsed.")
                print("  Format: c PTV70=ActualName, Bladder=BladderFoo")
        elif user_input in ("list", "ls"):
            print(f"\n{'Case':<40s} {'Status':<10s} {'PTV Config':<25s}")
            print("-" * 75)
            for i, c in enumerate(case_names):
                ci = manifest["cases"].get(c, {})
                st = ci.get("status", "pending")
                ptv = ci.get("ptv_config", "?")
                marker = " >>>" if i == current_idx else "    "
                print(f"{marker}{c:<36s} {st:<10s} {ptv:<25s}")
        elif user_input in ("q", "quit"):
            save_manifest(manifest, manifest_path)
            print(f"Saved. {reviewed} reviewed out of {len(case_names)}.")
            plt.close("all")
            return
        else:
            print(f"  Unknown command: '{user_input}'")

        # Auto-save every 10 reviews
        if reviewed > 0 and reviewed % 10 == 0:
            save_manifest(manifest, manifest_path)

    # End of list
    save_manifest(manifest, manifest_path)
    reviewed = sum(1 for c in case_names
                   if manifest["cases"].get(c, {}).get("status") in ("accepted", "rejected"))
    print(f"\nReview complete. {reviewed}/{len(case_names)} reviewed.")
    plt.close("all")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive DICOM-RT QA review tool for VMAT dose prediction dataset"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Root directory containing case directories (flat or with institution subdirs)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for manifest JSON and contact sheet PDF"
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Scan all cases and save manifest without interactive review"
    )
    parser.add_argument(
        "--contact-sheet-only", action="store_true",
        help="Generate contact sheet PDF from existing manifest (no review)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted review session"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    manifest_path = os.path.join(args.output, "qa_manifest.json")

    # Discover cases
    cases = discover_cases(args.input)
    if not cases:
        logger.error(f"No case directories found in {args.input}")
        sys.exit(1)
    logger.info(f"Discovered {len(cases)} cases")

    # Load or create manifest
    if args.resume or args.contact_sheet_only:
        manifest = load_manifest(manifest_path)
        if not manifest.get("cases"):
            logger.warning("No existing manifest found, starting fresh scan")
            manifest = {"cases": {}, "version": 1}
    else:
        manifest = load_manifest(manifest_path)
        if not manifest.get("cases"):
            manifest = {"cases": {}, "version": 1}

    # Scan all cases (always, to pick up new cases)
    if not args.contact_sheet_only:
        logger.info("Scanning all cases...")
        for cn, cd in cases:
            overrides = manifest.get("cases", {}).get(cn, {}).get("corrections", {})
            scan_result = scan_case(cd, overrides)

            if cn not in manifest["cases"]:
                manifest["cases"][cn] = {"status": "pending", "corrections": {}}

            manifest["cases"][cn].update({
                "case_dir": cd,
                "matched": scan_result["matched"],
                "unmatched": scan_result["unmatched"],
                "missing": scan_result["missing"],
                "ptv_config": scan_result["ptv_config"],
                "all_rois": scan_result["all_rois"],
                "error": scan_result.get("error"),
            })

        save_manifest(manifest, manifest_path)

        # Print summary
        ptv_counts = {}
        missing_counts = {}
        for cn, info in manifest["cases"].items():
            ptv = info.get("ptv_config", "unknown")
            ptv_counts[ptv] = ptv_counts.get(ptv, 0) + 1
            for m in info.get("missing", []):
                missing_counts[m] = missing_counts.get(m, 0) + 1

        print(f"\n{'='*60}")
        print(f"SCAN SUMMARY — {len(manifest['cases'])} cases")
        print(f"{'='*60}")
        print("\nPTV Configuration:")
        for ptv, count in sorted(ptv_counts.items(), key=lambda x: -x[1]):
            print(f"  {ptv:25s}: {count:3d}")
        if missing_counts:
            print("\nMissing Required Structures:")
            for struct, count in sorted(missing_counts.items(), key=lambda x: -x[1]):
                print(f"  {struct:12s}: {count:3d} cases")
        print()

    if args.scan_only:
        logger.info("Scan complete. Manifest saved.")
        return

    if args.contact_sheet_only:
        pdf_path = os.path.join(args.output, "contact_sheet.pdf")
        generate_contact_sheet(manifest, args.input, pdf_path)
        return

    # Interactive review
    interactive_review(manifest, cases, args.output)

    # Generate contact sheet after review
    reviewed = sum(1 for info in manifest["cases"].values()
                   if info.get("status") in ("accepted", "rejected"))
    if reviewed > 0:
        generate = input("\nGenerate contact sheet PDF? [Y/n] ").strip().lower()
        if generate != "n":
            pdf_path = os.path.join(args.output, "contact_sheet.pdf")
            generate_contact_sheet(manifest, args.input, pdf_path)


if __name__ == "__main__":
    main()
