# Experiments Index

This document tracks all experiments conducted in this project for reproducibility and publication purposes.

---

## Experiment Log

| Date | Experiment ID | Notebook | Model | Best Metric | Status |
|------|---------------|----------|-------|-------------|--------|
| 2026-01-19 | baseline_unet_run1 | [2026-01-19_baseline_unet_experiment.ipynb](2026-01-19_baseline_unet_experiment.ipynb) | BaselineUNet3D | 3.73 Gy MAE (val) | Complete |
| 2026-01-19 | baseline_unet_test_eval | [2026-01-19_baseline_unet_test_evaluation.ipynb](2026-01-19_baseline_unet_test_evaluation.ipynb) | BaselineUNet3D | 1.43 Gy MAE, 14.2% Gamma (test) | Complete |

---

## Naming Conventions

### Notebooks
```
YYYY-MM-DD_<experiment_type>_<optional_description>.ipynb
```

Examples:
- `2026-01-19_baseline_unet_experiment.ipynb`
- `2026-01-20_ddpm_v1_experiment.ipynb`
- `2026-01-25_ablation_no_sdf.ipynb`

### Run Directories
```
./runs/<model_type>_<run_name>/
```

Examples:
- `./runs/baseline_unet_run1/`
- `./runs/ddpm_dose_v1/`
- `./runs/ablation_no_sdf/`

---

## Experiment Categories

### 1. Baseline Models
- [ ] ~~baseline_unet_run1~~ (Complete - 2026-01-19)
- [ ] baseline_unet_larger (Planned - increased capacity)

### 2. Diffusion Models (DDPM)
- [ ] ddpm_dose_v1 (Planned)
- [ ] ddpm_dose_v2_conditioned (Planned)

### 3. Ablation Studies
- [ ] ablation_no_sdf (Planned - binary masks only)
- [ ] ablation_no_constraints (Planned - no FiLM conditioning)
- [ ] ablation_patch_size (Planned - 64 vs 128 vs 160)

### 4. Hyperparameter Tuning
- [ ] lr_sweep (Planned - learning rate search)
- [ ] batch_size_sweep (Planned)

---

## Key Metrics to Track

For each experiment, record:

| Metric | Description | Target |
|--------|-------------|--------|
| val_mae_gy | Mean Absolute Error in Gy | < 3 Gy |
| val_loss | Validation MSE loss | Minimize |
| gamma_3mm3pct | Gamma pass rate (3%/3mm) | > 95% |
| training_time | Wall clock time | - |
| gpu_memory | Peak GPU memory usage | < 24 GB |

---

## Data Versions

| Version | Date | Cases | Notes |
|---------|------|-------|-------|
| v2.2.0 | 2026-01-18 | 23 | SDF fix, full MLC extraction |

---

## Publication Readiness Checklist

For each experiment to be publication-ready:

- [ ] Reproducibility info recorded (git hash, seeds, environment)
- [ ] Training curves saved as high-res figures
- [ ] Metrics CSV exported
- [ ] Best checkpoint saved
- [ ] Test set evaluation completed
- [ ] DVH analysis completed
- [ ] Gamma analysis completed
- [ ] Sample visualizations generated
- [ ] Statistical significance tested (if comparing models)

---

*Last updated: 2026-01-19 (Added test evaluation)*
