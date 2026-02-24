# Experiments Index

**MASTER LIST** - All experiments tracked here for reproducibility and publication.

For project strategy and next steps: see `.claude/instructions.md`.
For code conventions and experiment protocol: see `CLAUDE.md`.

---

## Experiment Log

> **WARNING: All experiments below are from the pilot phase (v2.2.0 preprocessing, 23 cases, n=2 test set). The v2.2.0 pipeline had a critical D95 artifact (#4) that corrupted PTV boundary dose values. All absolute metrics are invalid and cannot be cited. Relative comparisons between methods are questionable. These experiments established the methodology and loss function implementations — the metrics must be re-established on v2.3 data.**

| Date | Experiment ID | Git Hash | Notebook | Model | Best Metric | Status |
|------|---------------|----------|----------|-------|-------------|--------|
| 2026-01-19 | baseline_unet_run1 | `b3f0c08` | [2026-01-19_baseline_unet_experiment.ipynb](2026-01-19_baseline_unet_experiment.ipynb) | BaselineUNet3D | 3.73 Gy MAE (val) | Pilot (v2.2.0) |
| 2026-01-19 | baseline_unet_test_eval | `b3f0c08` | [2026-01-19_baseline_unet_test_evaluation.ipynb](2026-01-19_baseline_unet_test_evaluation.ipynb) | BaselineUNet3D | 1.43 Gy MAE, 14.2% Gamma (test) | Pilot (v2.2.0) |
| 2026-01-20 | ddpm_dose_v1 | `3efbea0` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | SimpleUNet3D+DDPM | 12.19->3.78 Gy MAE (optimized) | Pilot (v2.2.0) |
| 2026-01-20 | phase1_sampling | `206f84c` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | 3.80 Gy MAE (50 steps) | Pilot (v2.2.0) |
| 2026-01-20 | phase1_ensemble | `206f84c` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | 3.78 Gy MAE (n=1) | Pilot (v2.2.0) |
| 2026-01-20 | strategic_assessment | `206f84c` | [2026-01-20_strategic_assessment.ipynb](2026-01-20_strategic_assessment.ipynb) | Analysis | DDPM not recommended | Pilot (v2.2.0) |
| 2026-01-20 | grad_loss_0.1 | `5d111a0` | [2026-01-20_grad_loss_experiment.ipynb](2026-01-20_grad_loss_experiment.ipynb) | BaselineUNet3D+GradLoss | 3.67 Gy MAE (val), 27.9% Gamma (test) | Pilot (v2.2.0) |
| 2026-01-21 | grad_vgg_combined | `dca8446` | [2026-01-21_grad_vgg_combined.ipynb](2026-01-21_grad_vgg_combined.ipynb) | BaselineUNet3D+Grad+VGG | 2.27 Gy MAE (val), ~28% Gamma (test) | Pilot (v2.2.0) |
| 2026-01-21 | semi_multi_modal_hypothesis | — | [2026-01-21_semi_multi_modal_hypothesis.ipynb](2026-01-21_semi_multi_modal_hypothesis.ipynb) | Analysis | Dose prediction is semi-multi-modal | Pilot (v2.2.0) |
| 2026-01-22 | dvh_aware_loss | `1188d72` | [2026-01-22_dvh_aware_loss.ipynb](2026-01-22_dvh_aware_loss.ipynb) | BaselineUNet3D+Grad+DVH | 0.95 Gy MAE (test), 27.7% Gamma | Pilot (v2.2.0) |
| 2026-01-22 | structure_weighted_loss | `8b08506` | [2026-01-22_structure_weighted_loss.ipynb](2026-01-22_structure_weighted_loss.ipynb) | BaselineUNet3D+Grad+StructWeighted | 2.91 Gy MAE (val), 31.2% Gamma (test) | Pilot (v2.2.0) |
| 2026-01-23 | gamma_metric_analysis | `e0a0274` | [2026-01-23_gamma_metric_analysis.ipynb](2026-01-23_gamma_metric_analysis.ipynb) | Analysis | PTV underdose identified: -7 to -8 Gy | Pilot (v2.2.0) |
| 2026-01-23 | asymmetric_ptv_loss | `a88247b` | [2026-01-23_asymmetric_ptv_loss_experiment.ipynb](2026-01-23_asymmetric_ptv_loss_experiment.ipynb) | BaselineUNet3D+Grad+AsymPTV | 3.36 Gy MAE (val), D95 gap: -5.95 Gy | Pilot (v2.2.0) |

---

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Notebooks | `YYYY-MM-DD_<experiment>.ipynb` | `2026-01-20_grad_loss_experiment.ipynb` |
| Run directories | `runs/<experiment_id>/` | `runs/dvh_aware_loss/` |
| Figure scripts | `scripts/generate_<exp>_figures.py` | `scripts/generate_grad_loss_figures.py` |

---

## Data Versions

| Version | Date | Cases | Notes |
|---------|------|-------|-------|
| v2.2.0 | 2026-01-18 | 23 | SDF fix, full MLC extraction. **D95 artifact (#4) — all metrics invalid.** |
| v2.3.0 | 2026-02-23 | 74 | Crop instead of resample, B-spline dose interpolation, D95 verified >= 66.3 Gy. |

---

**For project strategy, next steps, and planning: see `.claude/instructions.md`**

*Last updated: 2026-02-23*
