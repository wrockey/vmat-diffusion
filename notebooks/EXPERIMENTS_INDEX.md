# Experiments Index

**MASTER LIST** - All experiments tracked here for reproducibility and publication.

See `docs/EXPERIMENT_STRUCTURE.md` for organization guidelines.

---

## Experiment Log

| Date | Experiment ID | Git Hash | Notebook | Model | Best Metric | Status |
|------|---------------|----------|----------|-------|-------------|--------|
| 2026-01-19 | baseline_unet_run1 | `b3f0c08` | [2026-01-19_baseline_unet_experiment.ipynb](2026-01-19_baseline_unet_experiment.ipynb) | BaselineUNet3D | 3.73 Gy MAE (val) | Complete |
| 2026-01-19 | baseline_unet_test_eval | `b3f0c08` | [2026-01-19_baseline_unet_test_evaluation.ipynb](2026-01-19_baseline_unet_test_evaluation.ipynb) | BaselineUNet3D | 1.43 Gy MAE, 14.2% Gamma (test) | Complete |
| 2026-01-20 | ddpm_dose_v1 | `3efbea0` | **⚠️ NEEDS CREATION** | SimpleUNet3D+DDPM | 12.19 Gy MAE (val) | Complete (underperformed) |
| 2026-01-20 | phase1_sampling | TBD | - | DDPM inference | **3.80 Gy MAE** (50 steps) | Complete |
| 2026-01-20 | phase1_ensemble | TBD | - | DDPM inference | **3.78 Gy MAE** (n=1) | Complete |

### Phase 1 Optimization Results
**Root cause identified:** Training validation used high DDIM step counts, inflating MAE to 12.19 Gy.

| Exp | Finding | Best MAE (Gy) |
|-----|---------|---------------|
| 1.1 Sampling Steps | 50 steps optimal; more steps = worse | **3.80** |
| 1.2 Ensemble | n=1 optimal; averaging doesn't help | **3.78** |

**Conclusion:** DDPM now matches baseline (3.78 vs 3.73 Gy) with optimal inference settings.

### Strategic Assessment (Post-Phase 1)

**Key Finding: DDPM is NOT recommended for this task.**

| Red Flag | Implication |
|----------|-------------|
| More steps = worse results | Structural issue; model denoises away dose signal |
| Near-zero sample variability | Model is deterministic, not generative |
| DDPM = Baseline accuracy | Added complexity provides no benefit |
| 50/1000 steps optimal | Essentially one-shot prediction, not iterative refinement |

**Fundamental Mismatch:**
- Dose prediction is **deterministic** (one correct answer per patient)
- Diffusion models excel at **multi-modal generation** (many valid outputs)
- More data (n=100+) unlikely to change DDPM vs baseline comparison

**Recommended Path Forward:**
1. ✅ **Improve baseline** with perceptual/adversarial loss (recommended)
2. ✅ **Try Flow Matching** if generative approach desired
3. ❌ **Don't continue DDPM tuning** (diminishing returns)

See `docs/DDPM_OPTIMIZATION_PLAN.md` for detailed analysis.

### Notebooks Needing Creation
- [ ] `2026-01-20_ddpm_v1_experiment.ipynb` - Document ddpm_dose_v1 results
- [ ] `2026-01-20_phase1_optimization.ipynb` - Document Phase 1 optimization results

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

### 2. Diffusion Models (DDPM) - **NOT RECOMMENDED**
- [x] ~~ddpm_dose_v1~~ (Complete - 2026-01-20, git: 3efbea0)
  - Run directory: `runs/vmat_dose_ddpm/`
  - Platform: Native Windows/Pinokio (stable, GPU 44-58°C)
  - **Training Results: 12.19 Gy MAE (val)** - appeared to underperform baseline
  - **Optimized Results: 3.78 Gy MAE (val)** - matches baseline with 50 DDIM steps
  - Training: 37 epochs (early stopped), 1.94 hours
  - Best checkpoint: `checkpoints/best-epoch=015-val/mae_gy=12.19.ckpt`
  - **Conclusion:** DDPM provides no benefit over baseline; not recommended for this task
- [x] ~~phase1_optimization~~ (Complete - 2026-01-20)
  - Sampling steps ablation: 50 steps optimal (3.80 Gy)
  - Ensemble averaging: n=1 optimal (3.78 Gy)
  - **Finding:** More steps = worse results (structural issue)
- [ ] ~~ddpm_dose_v2~~ (Cancelled - DDPM not recommended)
- [ ] ~~ddpm_dose_v2_conditioned~~ (Cancelled - DDPM not recommended)

### 2b. Alternative Generative Approaches (Recommended)
- [ ] flow_matching_v1 (Planned - simpler than diffusion, better for regression)
- [ ] baseline_perceptual_loss (Planned - add perceptual/adversarial loss to baseline)

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

*Last updated: 2026-01-20 (Strategic assessment complete - DDPM not recommended, pivot to baseline improvements)*
