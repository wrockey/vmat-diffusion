#!/usr/bin/env python3
"""
Generate OAR Mapping JSON from DICOM Structure Sets

This script scans your DICOM data to discover all structure names and helps
create/update the oar_mapping.json file needed for preprocessing.

Usage:
    # Scan data and print discovered structures
    python generate_oar_mapping.py --input_dir ./data/raw --scan
    
    # Generate initial mapping file
    python generate_oar_mapping.py --input_dir ./data/raw --output oar_mapping.json
    
    # Update existing mapping with new structures
    python generate_oar_mapping.py --input_dir ./data/raw --update oar_mapping.json

Version: 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re

try:
    import pydicom
except ImportError:
    print("Error: pydicom not installed. Run: pip install pydicom")
    sys.exit(1)


# Default structure patterns to match
STRUCTURE_PATTERNS = {
    'PTV70': [
        r'ptv.*70', r'ptv.*high', r'ptv[_\s]?1(?![0-9])', r'ptv.*primary'
    ],
    'PTV56': [
        r'ptv.*56', r'ptv.*low', r'ptv[_\s]?2(?![0-9])', r'ptv.*nodal', r'ptv.*elective'
    ],
    'Prostate': [
        r'prostate', r'ctv[_\s]?p', r'gtv[_\s]?p', r'ctv.*prostate', r'gtv.*prostate'
    ],
    'Rectum': [
        r'rectum', r'rectal', r'^rect$'
    ],
    'Bladder': [
        r'bladder', r'^blad$'
    ],
    'Femur_L': [
        r'femur.*l', r'l.*femur', r'left.*femur', r'femoral.*head.*l', r'l.*femoral'
    ],
    'Femur_R': [
        r'femur.*r', r'r.*femur', r'right.*femur', r'femoral.*head.*r', r'r.*femoral'
    ],
    'Bowel': [
        r'bowel', r'small.*bowel', r'large.*bowel', r'sigmoid', r'colon', r'intestine'
    ],
}

CHANNEL_MAPPING = {
    'PTV70': 0,
    'PTV56': 1,
    'Prostate': 2,
    'Rectum': 3,
    'Bladder': 4,
    'Femur_L': 5,
    'Femur_R': 6,
    'Bowel': 7,
}


def scan_structure_names(input_dir: Path) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Scan all DICOM structure sets and collect unique structure names.
    
    Returns:
        Tuple of (structure_names_by_patient, all_unique_names)
    """
    structure_names_by_patient = defaultdict(set)
    all_names = set()
    
    # Find all RS files
    rs_files = list(input_dir.rglob("RS*.dcm"))
    if not rs_files:
        # Try alternative patterns
        rs_files = list(input_dir.rglob("*RTSTRUCT*.dcm"))
    
    if not rs_files:
        print(f"Warning: No structure set files found in {input_dir}")
        return {}, []
    
    print(f"Found {len(rs_files)} structure set files")
    
    for rs_path in rs_files:
        try:
            rs = pydicom.dcmread(str(rs_path), force=True)
            patient_id = rs_path.parent.name
            
            if hasattr(rs, 'StructureSetROISequence'):
                for roi in rs.StructureSetROISequence:
                    name = roi.ROIName
                    structure_names_by_patient[patient_id].add(name)
                    all_names.add(name)
        except Exception as e:
            print(f"  Warning: Could not read {rs_path}: {e}")
    
    return dict(structure_names_by_patient), sorted(all_names)


def normalize_name(name: str) -> str:
    """Normalize structure name for matching."""
    return name.lower().replace('_', '').replace(' ', '').replace('-', '').replace('gy', '')


def classify_structure(name: str) -> str:
    """
    Attempt to classify a structure name into one of our categories.
    
    Returns:
        Category name or 'Unknown'
    """
    normalized = normalize_name(name)
    
    for category, patterns in STRUCTURE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return category
    
    return 'Unknown'


def generate_mapping(all_names: List[str], existing_mapping: Dict = None) -> Dict:
    """Generate the oar_mapping.json structure."""
    
    # Start with existing or default
    if existing_mapping:
        mapping = existing_mapping.copy()
    else:
        mapping = {
            str(ch): {
                'name': name,
                'variations': [],
                'required': name in ['PTV70', 'Rectum', 'Bladder'],
                'notes': ''
            }
            for name, ch in CHANNEL_MAPPING.items()
        }
    
    # Track what we found
    classified = defaultdict(list)
    unclassified = []
    
    for name in all_names:
        category = classify_structure(name)
        if category != 'Unknown':
            classified[category].append(name)
        else:
            unclassified.append(name)
    
    # Update mapping with discovered names
    for category, names in classified.items():
        ch = str(CHANNEL_MAPPING[category])
        if ch in mapping:
            existing_variations = set(mapping[ch].get('variations', []))
            # Add new variations
            for name in names:
                if name not in existing_variations:
                    mapping[ch]['variations'].append(name)
                    existing_variations.add(name)
    
    # Add metadata
    mapping['_metadata'] = {
        'version': '1.0',
        'description': 'Structure name mapping for VMAT prostate preprocessing',
        'generated_by': 'generate_oar_mapping.py',
        'unclassified_structures': unclassified,
        'instructions': [
            "Review 'unclassified_structures' and manually add to appropriate channels",
            "Structure names are matched case-insensitively",
            "Date suffixes (e.g., '_Oct11') are handled by normalization"
        ]
    }
    
    return mapping


def print_scan_results(names_by_patient: Dict[str, Set[str]], all_names: List[str]):
    """Print scan results in a readable format."""
    
    print("\n" + "="*70)
    print("STRUCTURE NAME SCAN RESULTS")
    print("="*70)
    
    print(f"\nTotal unique structure names: {len(all_names)}")
    print(f"Patients scanned: {len(names_by_patient)}")
    
    # Classify and group
    classified = defaultdict(list)
    unclassified = []
    
    for name in all_names:
        category = classify_structure(name)
        if category != 'Unknown':
            classified[category].append(name)
        else:
            unclassified.append(name)
    
    print("\n" + "-"*70)
    print("CLASSIFIED STRUCTURES (auto-detected)")
    print("-"*70)
    
    for category in CHANNEL_MAPPING.keys():
        names = classified.get(category, [])
        ch = CHANNEL_MAPPING[category]
        print(f"\nChannel {ch} - {category}:")
        if names:
            for name in sorted(names):
                print(f"    ✓ {name}")
        else:
            print(f"    (none found)")
    
    print("\n" + "-"*70)
    print("UNCLASSIFIED STRUCTURES (review manually)")
    print("-"*70)
    
    if unclassified:
        for name in sorted(unclassified):
            print(f"    ? {name}")
        print(f"\n  → Review these and add to appropriate channel in oar_mapping.json")
    else:
        print("    (none - all structures classified)")
    
    # Show per-patient breakdown
    print("\n" + "-"*70)
    print("PER-PATIENT STRUCTURE COUNTS")
    print("-"*70)
    
    for patient_id, names in sorted(names_by_patient.items()):
        print(f"  {patient_id}: {len(names)} structures")


def get_default_input_dir():
    """
    Auto-detect the best default input directory for DICOM data.

    Priority order:
    1. Local ./data/raw with case folders (if data exists)
    2. External drive fallback (if mounted)

    Returns:
        str: Path to input directory, or None if not found
    """
    # Get script directory to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Local path (clean workstation)
    local_input = os.path.join(project_root, "data", "raw")

    # External drive path (WSL)
    external_input = "/mnt/i/anonymized_dicom"

    # Check local path first (must have actual case folders, not just symlinks)
    if os.path.isdir(local_input):
        case_dirs = [d for d in os.listdir(local_input)
                     if d.startswith("case_") and os.path.isdir(os.path.join(local_input, d))]
        if case_dirs and not os.path.islink(os.path.join(local_input, case_dirs[0])):
            return local_input

    # Check external drive
    if os.path.isdir(external_input):
        return external_input

    return None


def main():
    # Get environment-appropriate default
    default_input = get_default_input_dir()

    parser = argparse.ArgumentParser(
        description="Generate OAR mapping JSON from DICOM structure sets"
    )

    parser.add_argument('--input_dir', type=str, default=default_input,
                       help='Directory containing DICOM data (auto-detected if not specified)')
    parser.add_argument('--scan', action='store_true',
                       help='Scan and print structure names without generating file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for new oar_mapping.json')
    parser.add_argument('--update', type=str, default=None,
                       help='Path to existing oar_mapping.json to update')
    
    args = parser.parse_args()

    # Validate input_dir
    if args.input_dir is None:
        parser.error("--input_dir is required. No DICOM data directory was auto-detected.\n"
                     "Specify the path to your DICOM data manually.")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Scan for structure names
    print(f"Scanning {input_dir} for structure sets...")
    names_by_patient, all_names = scan_structure_names(input_dir)
    
    if not all_names:
        print("No structures found. Check input directory.")
        sys.exit(1)
    
    # Print results
    print_scan_results(names_by_patient, all_names)
    
    if args.scan:
        # Just scanning, don't generate file
        return
    
    # Load existing mapping if updating
    existing_mapping = None
    if args.update:
        update_path = Path(args.update)
        if update_path.exists():
            with open(update_path) as f:
                existing_mapping = json.load(f)
            print(f"\nLoaded existing mapping from {update_path}")
    
    # Generate mapping
    mapping = generate_mapping(all_names, existing_mapping)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.update:
        output_path = Path(args.update)
    else:
        output_path = Path('oar_mapping.json')
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n{'Updated' if args.update else 'Generated'} mapping: {output_path}")
    
    # Show what needs review
    unclassified = mapping.get('_metadata', {}).get('unclassified_structures', [])
    if unclassified:
        print(f"\n⚠ {len(unclassified)} unclassified structures need manual review!")
        print("  Edit the mapping file and add them to appropriate channels.")


if __name__ == '__main__':
    main()
