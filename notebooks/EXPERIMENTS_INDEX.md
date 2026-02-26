# Experiments Index

**MASTER LIST** - All experiments tracked here for reproducibility and publication.

For project strategy and next steps: see `.claude/instructions.md`.
For code conventions and experiment protocol: see `CLAUDE.md`.

---

## Experiment Log

### v2.3 Experiments (current — valid metrics)

| Date | ID | Git Hash | Notebook | Model | MAE (Gy) | Gamma 3%/3mm | PTV70 D95 Gap | Status |
|------|-----|----------|----------|-------|----------|--------------|---------------|--------|
| 2026-02-24 | baseline_v23 (seed42) | `82bddc5` | [notebook](2026-02-24_baseline_v23_preliminary.ipynb) | BaselineUNet3D | 4.80 ± 2.45 (test, n=7) | 28.1 ± 12.6% (global), 85.5 ± 10.9% (PTV) | -0.86 ± 0.92 Gy | Preliminary |
| 2026-02-25 | baseline_v23 (seed123) | `c2454b8` | — | BaselineUNet3D | 4.33 ± 2.47 (test, n=7) | 34.3 ± 9.3% (global), 74.6 ± 10.7% (PTV) | -1.99 ± 1.33 Gy | Preliminary |
| 2026-02-26 | baseline_v23 (seed456) | `11afb2f` | — | BaselineUNet3D | 3.51 ± 3.04 (test, n=7) | 39.2 ± 8.7% (global), 78.8 ± 14.0% (PTV) | -2.46 ± 1.60 Gy | Preliminary |
| 2026-02-26 | **baseline_v23 (3-seed aggregate)** | `11afb2f` | — | BaselineUNet3D | **4.22 ± 0.53** (seed mean±std) | **33.8 ± 4.6%** (global), **80.2 ± 5.3%** (PTV) | **-1.76 ± 0.69 Gy** | Complete |

### v2.3 Experiments — In Progress

| ID | Seeds | Status | Tracking |
|----|-------|--------|----------|
| augmentation_ablation | 42 | Queued | #45 |
| combined_loss_pilot | 42 | Queued | #57 |
| architecture_scouts (C11,C13,C15) | 42 | Queued | #53 |

### Pilot Experiments (v2.2.0 — metrics invalid)

> **WARNING:** All experiments below used v2.2.0 preprocessing with the D95 artifact (#4). Absolute metrics are invalid. Relative comparisons are questionable. These experiments established methodology and loss implementations only — metrics must be re-established on v2.3 data (#37).

| Date | ID | Git Hash | Notebook | Model | MAE (Gy) | Gamma 3%/3mm | PTV70 D95 Gap | Status |
|------|-----|----------|----------|-------|----------|--------------|---------------|--------|
| 2026-01-19 | baseline_unet_run1 | `b3f0c08` | [notebook](2026-01-19_baseline_unet_experiment.ipynb) | BaselineUNet3D | 3.73 (val) | — | — | Pilot |
| 2026-01-19 | baseline_unet_test_eval | `b3f0c08` | [notebook](2026-01-19_baseline_unet_test_evaluation.ipynb) | BaselineUNet3D | 1.43 (test) | 14.2% | ~-20 Gy | Pilot |
| 2026-01-20 | ddpm_dose_v1 | `3efbea0` | [notebook](2026-01-20_ddpm_optimization.ipynb) | DDPM | 3.78 (val) | — | — | Pilot |
| 2026-01-20 | phase1_sampling | `206f84c` | [notebook](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | 3.80 (test) | — | — | Pilot |
| 2026-01-20 | phase1_ensemble | `206f84c` | [notebook](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | 3.78 (test) | — | — | Pilot |
| 2026-01-20 | strategic_assessment | `206f84c` | [notebook](2026-01-20_strategic_assessment.ipynb) | Analysis | — | — | — | Pilot |
| 2026-01-20 | grad_loss_0.1 | `5d111a0` | [notebook](2026-01-20_grad_loss_experiment.ipynb) | UNet+GradLoss | 3.67 (val) | 27.9% | ~-7 Gy | Pilot |
| 2026-01-21 | grad_vgg_combined | `dca8446` | [notebook](2026-01-21_grad_vgg_combined.ipynb) | UNet+Grad+VGG | 2.27 (val) | ~28% | — | Pilot |
| 2026-01-21 | semi_multi_modal | — | [notebook](2026-01-21_semi_multi_modal_hypothesis.ipynb) | Analysis | — | — | — | Pilot |
| 2026-01-22 | dvh_aware_loss | `1188d72` | [notebook](2026-01-22_dvh_aware_loss.ipynb) | UNet+Grad+DVH | 0.95 (test) | 27.7% | — | Pilot |
| 2026-01-22 | structure_weighted | `8b08506` | [notebook](2026-01-22_structure_weighted_loss.ipynb) | UNet+Grad+Struct | 2.91 (val) | 31.2% | ~-7 Gy | Pilot |
| 2026-01-23 | gamma_analysis | `e0a0274` | [notebook](2026-01-23_gamma_metric_analysis.ipynb) | Analysis | — | — | -7 to -8 Gy | Pilot |
| 2026-01-23 | asymmetric_ptv | `a88247b` | [notebook](2026-01-23_asymmetric_ptv_loss_experiment.ipynb) | UNet+Grad+AsymPTV | 3.36 (val) | — | -5.95 Gy | Pilot |

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

*Last updated: 2026-02-26*
