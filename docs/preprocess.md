# Preprocessing Guide: preprocess_dicom_rt.py

## Overview
This script preprocesses anonymized prostate VMAT DICOM-RT plans (CT, RS, RD, RP files) into standardized .npz files for conditional DDPM training. It handles variable original shapes (e.g., 500x500 vs. 530x530 XY, 208-263 Z slices), spacings (1-2mm), and grids by resampling/aligning to a fixed reference (512x512x256 at 1x1x2 mm) using SimpleITK (affine-aware). Outputs include CT (normalized [0,1]), multi-channel masks (PTV70/PTV56/prostate/rectum/bladder/femur_L/R/bowel), dose (normalized /70 Gy), and AAPM constraints vector.

**Key Features**:
- Physical centering on prostate/PTV70 centroid.
- Robust contour mapping via oar_mapping.json (unions variants like "_AP").
- Debug: Logs (Z ranges, PTV sums), PNGs (mid-slice CT/dose/PTV70).
- Feasibility: Processes ~100 cases on NVIDIA 3090; scales to Argon.

**Rationale**: Diffusion models require fixed inputs; this ensures alignment for conditioning on anatomy/constraints, preserving clinical fidelity (e.g., ~5% PTV voxel loss post-resample due to nearest-neighbor).

## Assumptions
- **Data Source**: Pinnacle exports; prostate VMAT with PTV70/56 focus. Anonymized; no PHI.
- **Contours**: oar_mapping.json defines channels (0: PTV70, 1: PTV56, 2: prostate, etc.); assumes per-slice IPP/PS for mapping.
- **Dose Grid**: May differ from CT (e.g., coarser Z); assumes pydicom loads as (z,y,x); handles non-uniform/decreasing Z with sorting/mean spacing.
- **Normalization**: CT clipped [-1000,3000 HU] to [0,1]; dose /70 Gy (PTV70 prescription).
- **Output Shape**: 512x512x256 covers pelvic FOV; pads/crops centered (no info loss if target large).
- **Edge Cases**: Zero PTV sums skipped unless --relax_filter; out-of-bounds contours clamped.
- **Dependencies**: Pre-installed (pydicom, SimpleITK, numpy, etc.); no pip installs at runtime.
- **Safety**: For research; validate outputs with gamma/DVH before training.

## Input
- **Directory Structure**: `data/raw/case_xxxx/` with files:
  - CT.*.dcm (slices)
  - RS.*.dcm (structures)
  - RD.*.dcm (dose)
  - RP.*.dcm (plan, unused currently)
- **oar_mapping.json**: Root-level JSON with structure:
  ```json
  {
    "0": {"name": "PTV70", "variations": ["PTV70", "PTV_70Gy_AP"]},
    "1": {"name": "PTV56", "variations": ["PTV56", "PTV_56Gy"]},
    ...
  }
  ```
- **Flags**: Via argparse (see table below).

## Output
- **.npz Files**: In `processed_npz/case_xxxx.npz`:
  - `ct`: (512,512,256) float32 [0,1] (resampled HU).
  - `masks`: (N_channels,512,512,256) uint8 binary (e.g., channel 0: PTV70).
  - `dose`: (512,512,256) float32 [/70 Gy].
  - `constraints`: 1D float32 (AAPM normalized + PTV type one-hot [1,0,0]).
- **Debug PNGs**: `debug_case_xxxx.png` (mid-slice CT, dose in Gy, PTV70; titles/margins adjusted).
- **Logs**: Console output (Z ranges, PTV sums original/resampled, errors).

## Flags
| Flag              | Description                                      | Type    | Default                          | Required |
|-------------------|--------------------------------------------------|---------|----------------------------------|----------|
| --input_dir      | Path to raw DICOM directories                   | str     | ~/vmat-diffusion-project/data/raw | No      |
| --output_dir     | Path to save .npz and PNGs                      | str     | ~/vmat-diffusion-project/processed_npz | No      |
| --mapping_file   | Path to oar_mapping.json                        | str     | oar_mapping.json                 | No      |
| --use_gamma      | Compute self-gamma (placeholder for validation) | bool    | False                            | No      |
| --relax_filter   | Process cases even with zero PTV70/56 sums      | bool    | False                            | No      |
| --skip_plots     | Skip saving debug PNGs (for headless/Argon)     | bool    | False                            | No      |

## Usage Examples
- Basic batch processing:
  ```bash
  python scripts/preprocess_dicom_rt.py
  ```
- Relax filter for incomplete cases:
  ```bash
  python scripts/preprocess_dicom_rt.py --relax_filter
  ```
- Headless on Argon (no plots):
  ```bash
  python scripts/preprocess_dicom_rt.py --skip_plots
  ```
- SLURM example (for scaling):
  ```bash
  #!/bin/bash
  #SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=24:00:00
  module load python/3.10 cuda/11.8
  python scripts/preprocess_dicom_rt.py --input_dir /path/to/raw --output_dir /path/to/processed --skip_plots
  ```

## Workflow Details
1. **Load CT**: Stack slices, sort by Z, compute spacing/position.
2. **Contours to Masks**: Per-channel binary masks via skimage.polygon; clamp out-of-bounds.
3. **Dose Alignment**: Load/scaling, handle GridFrameOffsetVector, sort Z if needed.
4. **Centering**: Compute physical centroid (prostate fallback to PTV70).
5. **Resampling**: SimpleITK to fixed reference (linear for CT/dose, nearest for masks).
6. **Normalization/Save**: CT [0,1], dose /70, constraints concat.
7. **Debug**: PNGs with titles (adjusted margins to avoid clipping).

**Runtime**: ~1-2 min/case on 3090 (SimpleITK dominant); batch for 100 cases ~2-3 hours.

## To-Do for Preprocessing
- Integrate pymedphys gamma/DVH post-save for auto-validation.
- Dynamic target_shape (e.g., scan dataset for max extents).
- Add --target_spacing flag for flexibility.
- Support more sites (e.g., lung; dynamic PTV type).
- Pytest unit tests (e.g., mock DICOM, assert shapes/sums).

## Troubleshooting
- **Broadcast Errors**: Fixed via SimpleITK; if persists, check dose/CT shapes in logs.
- **Zero Sums**: Use --relax_filter; edit oar_mapping.json for missing contours.
- **Wayland/Qt Issues**: Use --skip_plots on headless.
- **Memory**: Reduce target_shape Z if OOM (e.g., to 128 for feasibility).

For code details, see inline docstrings in preprocess_dicom_rt.py. Contact for issues.