#usage python ./src/pinnacle_data_curation.py ../tar/ ../dicom_collection/ processed_log.csv
import os
import tarfile
import tempfile
import logging
import csv
import subprocess
import sys
import re
import pydicom
from pymedphys.cli import pymedphys_cli

pydicom.config.convert_wrong_length_to_UN = True

logging.basicConfig(level=logging.DEBUG, filename='process.log', filemode='w')
console = logging.StreamHandler()  # Print to terminal
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def load_processed(log_file):
    processed = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    processed.add((row[0], row[1], row[2]))
    return processed


def append_log(log_file, patient_id, plan_name, trial_name, rois):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([patient_id, plan_name, trial_name, ','.join(rois)])


def parse_plan_name(plan_info_path):
    with open(plan_info_path, 'r', encoding='cp1252', errors='replace') as f:
        content = f.read()
        match = re.search(r'PlanName\s*=\s*"([^";]+)"', content, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"


def parse_trial_names(trial_path):
    trial_names = []
    with open(trial_path, 'r', encoding='cp1252', errors='replace') as f:
        content = f.read()
        matches = re.findall(
            r'Trial\s*=\s*\{[^}]*?Name\s*=\s*"([^";]+)"',
            content, re.DOTALL | re.IGNORECASE)
        trial_names.extend(matches)
    return trial_names


def parse_roi_names(roi_path):
    roi_names = []
    with open(roi_path, 'r', encoding='cp1252', errors='replace') as f:
        content = f.read()
        matches = re.findall(r'name\s*:\s*(\S+)', content, re.IGNORECASE)
        roi_names.extend(matches)
    return roi_names


def main(tar_input_dir, output_base_dir, log_file):
    # Check if output_base_dir exists; exit if not
    if not os.path.exists(output_base_dir):
        logging.error(f"Output directory does not exist: {output_base_dir}")
        sys.exit(1)

    # Initialise CSV log (header only)
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'plan_name', 'trial_name', 'rois'])

    processed = load_processed(log_file)

    # Determine next case number
    existing_cases = [
        d for d in os.listdir(output_base_dir)
        if d.startswith('case_') and d[5:].isdigit()
    ]
    case_counter = max([int(d[5:]) for d in existing_cases], default=0) + 1

    # Process each .tar file
    for tar_name in os.listdir(tar_input_dir):
        if not tar_name.endswith('.tar'):
            continue
        tar_path = os.path.join(tar_input_dir, tar_name)
        logging.info(f"Processing tar: {tar_name}")

        with tempfile.TemporaryDirectory() as tmp_extract:
            try:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(tmp_extract, filter='data')
            except Exception as e:
                logging.error(f"Failed to extract {tar_name}: {e}")
                continue

            # Find institution directories (allow multiple Institution_*)
            top_dirs = [
                d for d in os.listdir(tmp_extract)
                if d.startswith('Institution_') and os.path.isdir(os.path.join(tmp_extract, d))
            ]
            if not top_dirs:
                logging.error(f"No Institution_ dirs found in {tar_name}")
                continue
            logging.debug(f"Found {len(top_dirs)} Institution_ dirs in {tar_name}: {top_dirs}")

            for inst_dir_name in top_dirs:
                institution_dir = os.path.join(tmp_extract, inst_dir_name)

                # Find Mount_ directories (allow multiple, but take first for simplicity)
                mount_dirs = [
                    d for d in os.listdir(institution_dir)
                    if d.startswith('Mount_') and os.path.isdir(os.path.join(institution_dir, d))
                ]
                if not mount_dirs:
                    logging.warning(f"No Mount_ dirs in {inst_dir_name} of {tar_name}")
                    continue
                # Take the first Mount_ (or loop if needed)
                patient_root = os.path.join(institution_dir, mount_dirs[0])
                logging.debug(f"Using Mount_ dir: {mount_dirs[0]}")

                # Define patient_dirs
                patient_dirs = [
                    d for d in os.listdir(patient_root)
                    if d.startswith('Patient_')
                ]
                logging.info(f"Found {len(patient_dirs)} patients in {tar_name}/{inst_dir_name}/{mount_dirs[0]}")

                # Process each patient
                for patient_dir in patient_dirs:
                    patient_id = patient_dir
                    patient_path = os.path.join(patient_root, patient_dir)
                    plan_subdirs = [
                        d for d in os.listdir(patient_path)
                        if d.startswith('Plan_')
                    ]
                    logging.debug(f"Patient {patient_id} has {len(plan_subdirs)} plans")

                    for plan_subdir in plan_subdirs:
                        plan_dir = os.path.join(patient_path, plan_subdir)
                        plan_info_file = os.path.join(plan_dir, 'plan.PlanInfo')
                        trial_file = os.path.join(plan_dir, 'plan.Trial')
                        roi_file = os.path.join(plan_dir, 'plan.roi')

                        if not all(os.path.exists(f) for f in [plan_info_file, trial_file, roi_file]):
                            logging.warning(f"Missing files in {plan_subdir}; skipping.")
                            continue

                        plan_name = parse_plan_name(plan_info_file)

                        # Skip QA/test/phantom plans in name
                        qa_keywords = ['qa', 'phantom', 'test', 'qaplan', 'verification']
                        matched_keyword = next((k for k in qa_keywords if k in plan_name.lower()), None)
                        if matched_keyword:
                            logging.info(f"Skipping QA-type plan '{plan_name}' in {plan_subdir} (matched: {matched_keyword})")
                            continue

                        # Parse trials
                        trial_names = parse_trial_names(trial_file)
                        logging.debug(f"Trials in {plan_subdir}: {trial_names}")

                        if len(trial_names) == 1:
                            selected_trial = trial_names[0]
                            if 'dnu' in selected_trial.lower():
                                logging.info(f"Skipping trial '{selected_trial}' in {plan_subdir} due to 'dnu' in name")
                                continue
                        else:
                            selected_trial = next((n for n in trial_names if 'poc' in n.lower()), None)
                            if not selected_trial:
                                logging.warning(f"No POC trial in {plan_subdir}; skipping.")
                                continue
                            if 'dnu' in selected_trial.lower():
                                logging.info(f"Skipping trial '{selected_trial}' in {plan_subdir} due to 'dnu' in name")
                                continue

                        # ROI check (strict for prostate; flexible for PTV70 variants) and phantom check in ROIs
                        roi_names = parse_roi_names(roi_file)
                        lower_rois = [r.lower() for r in roi_names]
                        logging.debug(f"ROIs in {plan_subdir}: {lower_rois}")
                        has_phantom_in_rois = any('phantom' in r for r in lower_rois)
                        if has_phantom_in_rois:
                            logging.info(f"Skipping plan due to 'phantom' in ROIs for {plan_subdir}")
                            continue
                        has_prostate = any('prostate' in r for r in lower_rois)  # Strict: only "prostate" (case-insensitive)
                        logging.debug(f"Has prostate ROI: {has_prostate} in {plan_subdir}")

                        # Check for TargetPrescription = 7000 anywhere in plan.Trial
                        with open(trial_file, 'r', encoding='cp1252', errors='replace') as f:
                            trial_content = f.read()
                        has_7000_prescription = bool(re.search(r'TargetPrescription\s*=\s*7000', trial_content, re.IGNORECASE))
                        logging.debug(f"Has TargetPrescription=7000: {has_7000_prescription} in {plan_subdir}")

                        if not (has_prostate and has_7000_prescription):
                            logging.debug(f"Missing required ROI or prescription (prostate: {has_prostate}, TargetPrescription=7000: {has_7000_prescription}) in {plan_subdir}; skipping.")
                            continue

                        # Unique trial key
                        unique_key = (patient_id, plan_name, selected_trial)
                        if unique_key in processed:
                            logging.debug(f"Already exported {unique_key}; skipping.")
                            continue

                        # EXPORT (repeated -m flags for modalities; limits to primary CT + RT, ignores MRI/PET)
                        with tempfile.TemporaryDirectory() as temp_export:
                            cmd = [
                                'pymedphys', 'pinnacle', 'export', patient_path,
                                '--plan', plan_name,
                                '--trial', selected_trial,
                                '-m', 'CT',
                                '-m', 'RTSTRUCT',
                                '-m', 'RTDOSE',
                                '-m', 'RTPLAN',
                                '-o', temp_export
                            ]
                            try:
                                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                                logging.debug(f"Export stdout: {result.stdout}")
                            except subprocess.CalledProcessError as e:
                                logging.error(f"Export failed {patient_id}/{plan_name}: {e.stderr}")
                                continue

                            if not os.listdir(temp_export):
                                logging.error(f"Export produced no files for {patient_id}/{plan_name}")
                                continue

                            # ANONYMISE
                            with tempfile.TemporaryDirectory() as anon_temp:
                                original_argv = sys.argv
                                sys.argv = [sys.argv[0], 'dicom', 'anonymise', '-o', anon_temp, temp_export]
                                try:
                                    pymedphys_cli()
                                except SystemExit:
                                    pass
                                finally:
                                    sys.argv = original_argv

                                final_dir = os.path.join(output_base_dir, f'case_{case_counter:04d}')
                                os.makedirs(final_dir, exist_ok=True)
                                for f in os.listdir(anon_temp):
                                    os.rename(os.path.join(anon_temp, f), os.path.join(final_dir, f))

                            # LOG SUCCESS
                            append_log(log_file, patient_id, plan_name, selected_trial, roi_names)
                            processed.add(unique_key)
                            logging.info(f"SUCCESS: {patient_id} | {plan_name} | {selected_trial}")
                            case_counter += 1

    logging.info(f"Finished processing. Exported {case_counter - 1} cases.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pinnacle_data_curation.py /path/to/tars/ ./dicom_collection/ ./processed_log.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
