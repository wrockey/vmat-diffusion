# VMAT Diffusion Project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](CHANGELOG.md)

## Project Overview

This repository implements a generative AI model using diffusion techniques to create deliverable Volumetric Modulated Arc Therapy (VMAT) plans for radiation therapy. The core idea frames VMAT planning as a generative task analogous to AI image or video generation: Given a contoured CT dataset and organs-at-risk (OAR) dose constraints, the model samples a deliverable plan (dose distribution, arcs, MLC positions, dose rates) that meets clinical objectives while respecting linac physics.

### Current Phase: Dose Prediction (Phase 1)

The project is currently in **Phase 1**, focused on:
- Generating anatomically-plausible dose distributions from CT + contours
- Validating the diffusion model architecture on prostate VMAT cases
- Establishing feasibility on ~100 cases locally before HPC scaling

**Phase 2+** will add beam geometry extraction, MLC sequences, and full VMAT plan generation.

### Key Goals

1. **Define the Problem and Objectives**  
   Frame as a Generative Task: In VMAT, the goal is to generate a plan that delivers prescribed doses to PTVs while minimizing OAR exposure, respecting linac physics (e.g., beam modulation limits). A diffusion model would learn the distribution of high-quality plans from data, then sample new ones conditionally.  
   
   Key Outputs: 3D dose volume (Phase 1), arc angles, gantry speeds, MLC sequences, and fluence maps (Phase 2+).  
   
   Metrics for Success: Dose-volume histograms (DVHs), gamma pass rates (>95% at 3%/3mm), conformity index, and clinical acceptability.

2. **Gather and Prepare Data**  
   Dataset Requirements: Paired CT scans, segmented contours (PTV/OAR masks), dose objectives, and ground-truth dose distributions.  
   
   Current Focus: Prostate VMAT with SIB (simultaneous integrated boost):
   - PTV70: 70 Gy in 28 fractions (prostate)
   - PTV56: 56 Gy in 28 fractions (seminal vesicles)
   - PTV50.4: 50.4 Gy in 28 fractions (pelvic nodes, when present)

3. **Design the Model Architecture**  
   Base on Denoising Diffusion Probabilistic Models (DDPM) with conditional 3D U-Net.

4. **Train the Model**  
   Feasibility on NVIDIA 3090 (~100 cases), then scale to UIowa Argon HPC (1000+ cases).

5. **Inference and Post-Processing**  
   Fast sampling (DDIM), TPS refinement for deliverability.

6. **Validate and Deploy**  
   Retrospective testing, clinician review, publications.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/wrockey/vmat-diffusion.git
cd vmat-diffusion
```

Install dependencies (Python 3.8+):

```bash
pip install -r requirements.txt
```

**Requirements include:** pydicom, SimpleITK, numpy, scipy, scikit-image, matplotlib, pymedphys (for gamma/DVH), torch (for diffusion models).

**On UIowa Argon HPC:**
```bash
module load python/3.10
module load cuda/11.8
# Use virtualenv for dependencies
```

---

## Quick Start

### 1. Prepare Data

Place anonymized DICOM-RT files in `data/raw/`:

```
data/raw/
├── case_0001/
│   ├── CT.*.dcm       # CT slices
│   ├── RS.*.dcm       # Structure set
│   ├── RD.*.dcm       # Dose grid
│   └── RP.*.dcm       # RT Plan (recommended)
├── case_0002/
└── ...
```

### 2. Configure Structure Mapping

Edit `oar_mapping.json` to add any institution-specific contour name variations.

### 3. Preprocess

```bash
# Standard preprocessing (requires PTV70 + PTV56)
python scripts/preprocess_dicom_rt_v2.py \
    --input_dir ./data/raw \
    --output_dir ./processed_npz

# Include single-prescription cases
python scripts/preprocess_dicom_rt_v2.py --relax_filter

# Strict QA mode (recommended before HPC scaling)
python scripts/preprocess_dicom_rt_v2.py --strict_validation
```

### 4. Verify Outputs

```bash
jupyter notebook notebooks/verify_npz.ipynb
```

The verification notebook provides:
- Visual inspection of CT, dose, and contour alignment
- Automated validation checks
- DVH analysis
- Batch validation across all cases

### 5. Review Batch Summary

```bash
cat ./processed_npz/batch_summary.json
```

### 6. Train DDPM (Coming Soon)

```bash
# Forthcoming
python scripts/train_ddpm.py --data_dir ./processed_npz
```

---

## Directory Structure

```
vmat-diffusion/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── requirements.txt             # Python dependencies
├── oar_mapping.json             # Contour name variations
│
├── scripts/
│   ├── preprocess_dicom_rt.py   # v1.0 preprocessing (deprecated)
│   └── preprocess_dicom_rt_v2.py # v2.0 preprocessing (current)
│
├── notebooks/
│   └── verify_npz.ipynb         # Data verification notebook
│
├── docs/
│   ├── preprocessing_guide.md   # Preprocessing usage guide
│   ├── preprocessing_assumptions.md # Known limitations
│   └── verify_npz.md            # Notebook documentation
│
├── data/                        # Data directories (gitignored)
│   ├── raw/                     # Anonymized DICOM-RT
│   └── processed_npz/           # Output .npz files
│
├── models/                      # Future: DDPM checkpoints
│
└── .gitignore
```

---

## Preprocessing Output

Each `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `ct` | (512, 512, 256) | CT volume, normalized [0, 1] |
| `dose` | (512, 512, 256) | Dose, normalized to prescription |
| `masks` | (8, 512, 512, 256) | Binary structure masks |
| `constraints` | (13,) | Prescription targets + OAR constraints |
| `metadata` | dict | Case info, validation results |

### Mask Channels

| Channel | Structure |
|---------|-----------|
| 0 | PTV70 |
| 1 | PTV56 |
| 2 | Prostate |
| 3 | Rectum |
| 4 | Bladder |
| 5 | Femur_L |
| 6 | Femur_R |
| 7 | Bowel |

### SIB Case Types

The preprocessing automatically classifies cases:
- `single_rx`: PTV70 only (~10% of cases)
- `sib_2level`: PTV70 + PTV56 (~70% of cases)
- `sib_3level`: PTV70 + PTV56 + PTV50.4 (~20% of cases)

---

## Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Version history and migration guides |
| [docs/preprocessing_guide.md](docs/preprocessing_guide.md) | Preprocessing script usage |
| [docs/preprocessing_assumptions.md](docs/preprocessing_assumptions.md) | Known limitations and assumptions |
| [docs/verify_npz.md](docs/verify_npz.md) | Verification notebook guide |

---

## Scripts

### preprocess_dicom_rt_v2.py

Converts DICOM-RT to `.npz` format with:
- Automatic prescription extraction from RP file
- SIB case type classification
- Comprehensive validation checks
- Rich metadata storage

See [docs/preprocessing_guide.md](docs/preprocessing_guide.md) for details.

**Key options:**
```bash
--relax_filter      # Process cases without PTV56
--strict_validation # Fail on validation issues
--skip_plots        # Skip debug PNG generation
```

### verify_npz.ipynb

Interactive verification notebook with:
- Visual inspection (axial, coronal, sagittal views)
- Automated validation checks
- DVH plotting
- Batch validation

See [docs/verify_npz.md](docs/verify_npz.md) for details.

---

## Validation Workflow

### Before Training (~100 cases)

1. Run preprocessing with defaults
2. Review `batch_summary.json` for failures
3. Run `verify_npz.ipynb` on 5-10 random cases
4. Investigate any validation warnings

### Before HPC Scaling (1000+ cases)

1. Run with `--strict_validation`
2. Batch validate all cases in notebook
3. Document excluded cases with reasons
4. Verify case type distribution

---

## Contributing

- Use GitHub issues for bugs/features (label "enhancement" for features)
- Pull requests: Follow PEP8; include tests
- For Argon scaling: See `docs/argon_slurm.md` (forthcoming)

---

## Roadmap

### Phase 1: Dose Prediction (Current)
- [x] DICOM-RT preprocessing with validation
- [x] Verification notebook
- [x] SIB case support
- [ ] DDPM trainer implementation
- [ ] Feasibility validation on ~100 cases

### Phase 2: Deliverability
- [ ] Beam geometry extraction from RP
- [ ] MLC sequence extraction
- [ ] Physics-informed loss functions
- [ ] Hybrid Monte Carlo refinement

### Phase 3: Scale and Validate
- [ ] Scale to 1000+ cases on Argon HPC
- [ ] Retrospective clinical validation
- [ ] Publication (target: Medical Physics)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Citation

If you use this work, please cite:

```bibtex
@software{vmat_diffusion,
  title = {VMAT Diffusion: Conditional Diffusion Models for Radiation Therapy Planning},
  author = {[Author]},
  year = {2025},
  url = {https://github.com/wrockey/vmat-diffusion}
}
```

---

## License

MIT License. For research purposes; clinical deployment requires FDA/IRB validation.

---

## Acknowledgments

- Inspired by DoseDiff, DoseGAN, and related dose prediction literature
- Tools: pydicom, SimpleITK, pymedphys
- Data sources: TCIA, AAPM (cite appropriately)
- Compute: University of Iowa Argon HPC
