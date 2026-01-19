# Claude Code Instructions for VMAT Diffusion Project

## Project Context
This is a medical physics research project for VMAT dose prediction using diffusion models.
The goal is publishable, reproducible scientific work.

## Key Documentation (READ THESE)
Before starting work, familiarize yourself with:
- `docs/SCIENTIFIC_BEST_PRACTICES.md` - Reproducibility and publication guidelines
- `notebooks/EXPERIMENTS_INDEX.md` - Experiment tracking and naming conventions
- `notebooks/TEMPLATE_experiment.ipynb` - Template for new experiments

## Best Practices Summary

### For Every Experiment:
1. Record git commit hash before running
2. Use consistent random seeds (default: 42)
3. Create dated notebook: `YYYY-MM-DD_<experiment>_experiment.ipynb`
4. Update `notebooks/EXPERIMENTS_INDEX.md` with results
5. Save figures at 300 DPI for publication

### Reproducibility Requirements:
- Git hash, Python version, PyTorch version, CUDA version, GPU
- Random seeds for all stochastic operations
- Exact command to reproduce

### Figure Standards:
- 300 DPI minimum
- Colorblind-friendly palettes
- Save as PNG and PDF
- Font size 12pt minimum

### Medical Physics Metrics:
- MAE in Gy (target: < 3 Gy)
- Gamma pass rate 3%/3mm (target: > 95%)
- DVH comparison for all structures

## Data Locations
- Raw DICOM: `/mnt/i/anonymized_dicom/` (or `./data/raw/`)
- Processed NPZ: `/mnt/i/processed_npz/` (or `./processed/`)
- Training runs: `./runs/`
- Predictions: `./predictions/`

## Conda Environment
```bash
conda activate vmat-diffusion
```

## Current Status
- Preprocessing: v2.2.0, 23 cases ready
- Baseline U-Net: Complete, 3.73 Gy MAE
- DDPM: Not yet trained
