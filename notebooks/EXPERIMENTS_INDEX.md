# Experiments Index

**MASTER LIST** - All experiments tracked here for reproducibility and publication.

See `docs/EXPERIMENT_STRUCTURE.md` for organization guidelines.

---

## Experiment Log

| Date | Experiment ID | Git Hash | Notebook | Model | Best Metric | Status |
|------|---------------|----------|----------|-------|-------------|--------|
| 2026-01-19 | baseline_unet_run1 | `b3f0c08` | [2026-01-19_baseline_unet_experiment.ipynb](2026-01-19_baseline_unet_experiment.ipynb) | BaselineUNet3D | 3.73 Gy MAE (val) | Complete |
| 2026-01-19 | baseline_unet_test_eval | `b3f0c08` | [2026-01-19_baseline_unet_test_evaluation.ipynb](2026-01-19_baseline_unet_test_evaluation.ipynb) | BaselineUNet3D | 1.43 Gy MAE, 14.2% Gamma (test) | Complete |
| 2026-01-20 | ddpm_dose_v1 | `3efbea0` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | SimpleUNet3D+DDPM | 12.19→3.78 Gy MAE (optimized) | ✅ Complete |
| 2026-01-20 | phase1_sampling | `206f84c` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | **3.80 Gy MAE** (50 steps) | ✅ Complete |
| 2026-01-20 | phase1_ensemble | `206f84c` | [2026-01-20_ddpm_optimization.ipynb](2026-01-20_ddpm_optimization.ipynb) | DDPM inference | **3.78 Gy MAE** (n=1) | ✅ Complete |
| 2026-01-20 | strategic_assessment | `206f84c` | [2026-01-20_strategic_assessment.ipynb](2026-01-20_strategic_assessment.ipynb) | Analysis | - | Complete |
| 2026-01-20 | grad_loss_0.1 | `5d111a0` | [2026-01-20_grad_loss_experiment.ipynb](2026-01-20_grad_loss_experiment.ipynb) | BaselineUNet3D+GradLoss | **3.67 Gy MAE (val), 1.44 Gy MAE, 27.9% Gamma (test)** | ✅ Complete |
| 2026-01-21 | grad_vgg_combined | `dca8446` | [2026-01-21_grad_vgg_combined.ipynb](2026-01-21_grad_vgg_combined.ipynb) | BaselineUNet3D+Grad+VGG | **2.27 Gy MAE (val), 1.44 Gy MAE, ~28% Gamma (test)** | ✅ Complete |
| 2026-01-22 | dvh_aware_loss | `1188d72` | [2026-01-22_dvh_aware_loss.ipynb](2026-01-22_dvh_aware_loss.ipynb) | BaselineUNet3D+Grad+DVH | **3.61 Gy MAE (val), 0.95 Gy MAE, 27.7% Gamma (test)** | ✅ Complete |
| 2026-01-22 | structure_weighted_loss | `8b08506` | [2026-01-22_structure_weighted_loss.ipynb](2026-01-22_structure_weighted_loss.ipynb) | BaselineUNet3D+Grad+StructWeighted | **2.91 Gy MAE (val), 1.40 Gy MAE, 31.2% Gamma (test)** | ✅ Complete |
| 2026-01-23 | gamma_metric_analysis | `e0a0274` | [2026-01-23_gamma_metric_analysis.ipynb](2026-01-23_gamma_metric_analysis.ipynb) | Analysis | PTV underdose identified: -7 to -8 Gy | ✅ Complete |
| 2026-01-23 | asymmetric_ptv_loss | `a88247b` | [2026-01-23_asymmetric_ptv_loss_experiment.ipynb](2026-01-23_asymmetric_ptv_loss_experiment.ipynb) | BaselineUNet3D+Grad+AsymPTV | **3.36 Gy MAE (val), 1.89 Gy MAE, D95 gap: -5.95 Gy (test)** | ✅ Complete |

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

### Gradient Loss Experiment Results (2026-01-20)

**Key Finding: Gradient loss significantly improves Gamma pass rate!**

| Metric | Baseline | Gradient Loss | Change |
|--------|----------|---------------|--------|
| Val MAE | 3.73 Gy | **3.67 Gy** | -0.06 Gy ✅ |
| Test MAE | 1.43 Gy | **1.44 Gy** | +0.01 Gy (same) |
| Gamma (3%/3mm) | 14.2% | **27.9%** | **+13.7%** 🎯 |

**Analysis:**
- Gradient loss (3D Sobel) nearly doubled Gamma pass rate while maintaining MAE
- Best checkpoint at epoch 12 (same as baseline - consistent convergence)
- Training time 1.85 hours (baseline was 2.55 hours)

**Recommendation:** Proceed with Phase B (gradient + VGG combined) to push Gamma higher.

### Gradient + VGG Combined Experiment Results (2026-01-21) ← NEW

**Key Finding: VGG improves MAE but NOT Gamma!**

| Metric | Baseline | Grad Loss | Grad+VGG | Change (Grad→Grad+VGG) |
|--------|----------|-----------|----------|------------------------|
| Val MAE | 3.73 Gy | 3.67 Gy | **2.27 Gy** | **-38%** ✅ |
| Test MAE | 1.43 Gy | 1.44 Gy | **1.44 Gy** | 0% |
| Gamma (3%/3mm) | 14.2% | 27.9% | **~28%** | **0%** ❌ |
| Training Time | 2.55h | 1.85h | **9.74h** | +5x |

**Analysis:**
- VGG perceptual loss significantly improves validation MAE (2.27 vs 3.67 Gy, -38%)
- BUT: Gamma pass rate unchanged (~28%), indicating VGG doesn't help spatial accuracy
- VGG features (ImageNet-derived) optimize for semantic similarity, not dose gradients
- Training time increased 5x due to VGG feature extraction overhead

**Conclusion: Skip VGG in future experiments. VGG helps global accuracy but not Gamma.**

**Next steps per decision tree:**
1. Try adversarial loss (PatchGAN) for edge sharpness
2. Try structure-weighted loss for PTV/OAR accuracy
3. ~~Try DVH-aware loss for clinical metrics~~ ✅ DONE
4. Consider data augmentation to address n=23 limitation

### DVH-Aware Loss Experiment Results (2026-01-22) ← UPDATED WITH TEST RESULTS

**Key Finding: DVH-aware loss achieves BEST test MAE (0.95 Gy)!**

| Metric | Baseline | Grad Loss | Grad+VGG | **DVH-Aware** | Change |
|--------|----------|-----------|----------|---------------|--------|
| Val MAE | 3.73 Gy | 3.67 Gy | **2.27 Gy** | **3.61 Gy** | -3% vs baseline |
| **Test MAE** | 1.43 Gy | 1.44 Gy | 1.44 Gy | **0.95 Gy** | **-34% vs baseline** ✅ |
| **Gamma (3%/3mm)** | 14.2% | 27.9% | ~28% | **27.7%** | +95% vs baseline ✅ |
| Training Time | 2.55h | 1.85h | 9.74h | **11.2h** | |

**Test Set Results (2 held-out cases):**
| Case | MAE (Gy) | Gamma (3%/3mm) |
|------|----------|----------------|
| case_0007 | 1.25 | 26.5% |
| case_0021 | 0.65 | 29.0% |
| **Mean** | **0.95 ± 0.30** | **27.7 ± 1.2%** |

**Analysis:**
- **Best test MAE (0.95 Gy)** - 34% improvement over baseline!
- Gamma matches gradient loss (~28%), nearly doubles baseline (14.2%)
- DVH metrics (D95, V70) converge during training - model learns constraints
- Training takes longer (11.2h) due to DVH metric computation
- Best checkpoint at epoch 86 (late convergence - DVH needs more training time)

**Key insight:** DVH-aware loss achieves best overall results - both MAE and Gamma improvements. It provides explicit clinical constraint optimization that translates to better dose accuracy.

### Structure-Weighted Loss Experiment Results (2026-01-22) ← NEW

**Key Finding: Structure-weighted loss achieves BEST Gamma pass rate (31.2%)!**

| Metric | Baseline | Grad Loss | DVH-Aware | **Struct-Weighted** | Change |
|--------|----------|-----------|-----------|---------------------|--------|
| Val MAE | 3.73 Gy | 3.67 Gy | 3.61 Gy | **2.91 Gy** | **-22% vs baseline** ✅ |
| Test MAE | 1.43 Gy | 1.44 Gy | **0.95 Gy** | 1.40 Gy | -2% vs baseline |
| **Gamma (3%/3mm)** | 14.2% | 27.9% | 27.7% | **31.2%** | **+120% vs baseline** ✅ |
| Training Time | 2.55h | 1.85h | 11.2h | **2.62h** | Efficient |

**Test Set Results (2 held-out cases):**
| Case | MAE (Gy) | Gamma (3%/3mm) |
|------|----------|----------------|
| case_0007 | 1.63 | 33.4% |
| case_0021 | 1.16 | 29.0% |
| **Mean** | **1.40 ± 0.23** | **31.2 ± 2.2%** |

**Analysis:**
- **Best Gamma (31.2%)** - 3.3% improvement over gradient loss alone!
- **Best val MAE (2.91 Gy)** - 22% improvement over baseline
- Training is efficient (2.62h vs 11.2h for DVH)
- Weight configuration: 2x PTV, 1.5x OAR boundary, 0.5x background

**Key insight:** Weighting errors by clinical importance helps the model focus on critical regions. Structure-weighted loss achieves best Gamma while DVH achieves best test MAE - they may be complementary.

### Asymmetric PTV Loss Experiment Results (2026-01-23) ← NEW

**Key Finding: Asymmetric PTV loss improves D95 but doesn't fully solve underdosing!**

| Metric | Baseline | Grad Loss | DVH-Aware | Struct-Weight | **Asym PTV** | Change |
|--------|----------|-----------|-----------|---------------|--------------|--------|
| Val MAE | 3.73 Gy | 3.67 Gy | 3.61 Gy | 2.91 Gy | **3.36 Gy** | -10% vs baseline |
| Test MAE | 1.43 Gy | 1.44 Gy | 0.95 Gy | 1.40 Gy | **1.89 Gy** | +32% vs baseline |
| **D95 Gap** | ~-20 Gy | ~-7 Gy | ~-7 Gy | ~-7 Gy | **-5.95 Gy** | **Best D95** ✅ |
| Training Time | 2.55h | 1.85h | 11.2h | 2.62h | **2.6h** | Efficient |

**Hypothesis:** Penalizing underdosing 3x more than overdosing should improve PTV D95.

**Result:** Partial success:
- D95 gap improved from ~-7 Gy to -5.95 Gy (15% improvement)
- Validation showed epochs with **overdosing** (negative D95 gap = model learned to prefer overdose)
- Underdose fraction dropped from 80-90% to 40-50%

**Critical Insight:** The ground truth D95 for PTV70 is **55 Gy**, which **fails the 66.5 Gy clinical threshold**. This suggests the threshold may be too strict for this dataset.

**Conclusions:**
- DVH-aware loss is the best model for test MAE (0.95 Gy)
- Structure-weighted loss is best for Gamma (31.2%)
- Asymmetric PTV loss is best for D95 gap (-5.95 Gy)
- Gamma ~28% ceiling suggests loss function alone cannot reach 95% target
- Next steps: data augmentation, combined losses, or architecture changes

### Notebooks Needing Creation
- [x] `2026-01-20_ddpm_optimization.ipynb` - Document DDPM training + Phase 1 optimization ✅
- [x] `2026-01-20_strategic_assessment.ipynb` - Scientific value & path forward analysis ✅
- [x] `2026-01-20_grad_loss_experiment.ipynb` - Document gradient loss experiment results ✅
- [x] `2026-01-21_grad_vgg_combined.ipynb` - Document Grad+VGG experiment results ✅
- [x] `2026-01-22_dvh_aware_loss.ipynb` - Document DVH-aware loss experiment results ✅
- [x] `2026-01-23_gamma_metric_analysis.ipynb` - Gamma metric hypothesis testing ✅ NEW
- [x] `2026-01-23_asymmetric_ptv_loss_experiment.ipynb` - Asymmetric PTV loss results ✅ NEW
- [ ] `2026-01-21_semi_multi_modal_hypothesis.ipynb` - Semi-multi-modal hypothesis analysis

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
| v2.2.0 | 2026-01-18 | 23 | SDF fix, full MLC extraction |

---

**For project strategy, next steps, and planning: see `.claude/instructions.md`**

*Last updated: 2026-02-13 (Consolidated: strategy/planning moved to .claude/instructions.md)*
