import os
import json
import pydicom
import argparse

def collect_unique_roi_names(input_base_dir):
    unique_names = set()
    plan_dirs = [os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    for plan_dir in sorted(plan_dirs):
        rtstruct_files = [f for f in os.listdir(plan_dir) if f.startswith('RS')]
        if not rtstruct_files:
            print(f"No RTSTRUCT in {plan_dir}; skipping")
            continue
        rtstruct_path = os.path.join(plan_dir, rtstruct_files[0])  # Assume one RS per case
        try:
            rtstruct = pydicom.dcmread(rtstruct_path)
            for roi in rtstruct.StructureSetROISequence:
                name = roi.ROIName.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '')
                unique_names.add(roi.ROIName)  # Store original for reference
                unique_names.add(name)
        except Exception as e:
            print(f"Error reading {rtstruct_path}: {e}")
    return sorted(list(unique_names))

def generate_initial_mapping(unique_names):
    # Suggested channels for PTV70/PTV56 focus (prostate+SV); edit manually
    structure_map = {
        "0": {"name": "ptv70", "variations": [n for n in unique_names if '70' in n.lower() or 'high' in n.lower()]},
        "1": {"name": "ptv56", "variations": [n for n in unique_names if '56' in n.lower() or 'sv' in n.lower()]},
        "2": {"name": "prostate", "variations": [n for n in unique_names if 'prostate' in n.lower()]},
        "3": {"name": "rectum", "variations": [n for n in unique_names if 'rectum' in n.lower()]},
        "4": {"name": "bladder", "variations": [n for n in unique_names if 'bladder' in n.lower()]},
        "5": {"name": "femur_l", "variations": [n for n in unique_names if 'femur' in n.lower() and ('left' in n.lower() or 'l' in n.lower())]},
        "6": {"name": "femur_r", "variations": [n for n in unique_names if 'femur' in n.lower() and ('right' in n.lower() or 'r' in n.lower())]},
        "7": {"name": "bowel", "variations": [n for n in unique_names if 'bowel' in n.lower()]}
        # Add more as needed (e.g., penile bulb)
    }
    return {k: v for k, v in structure_map.items() if v["variations"]}  # Filter empty

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial oar_mapping.json from DICOM RTSTRUCTs")
    parser.add_argument("--input_dir", default="~/vmat-diffusion-project/data/raw", help="Base dir with case_xxxx subdirs")
    parser.add_argument("--output_file", default="oar_mapping.json", help="Output JSON path")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    print(f"Scanning {input_dir} for unique ROI names...")
    unique_names = collect_unique_roi_names(input_dir)
    print(f"Found {len(unique_names)} unique names: {unique_names}")

    mapping = generate_initial_mapping(unique_names)
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Initial {args.output_file} generated. Edit variations manually using DICOM viewers for accuracy!")
    print("Focus on PTV70/PTV56 presence for filtering later.")