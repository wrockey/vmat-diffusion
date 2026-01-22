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
| 2026-01-22 | dvh_aware_loss | `1188d72` | [2026-01-22_dvh_aware_loss.ipynb](2026-01-22_dvh_aware_loss.ipynb) | BaselineUNet3D+Grad+DVH | **3.61 Gy MAE (val, epoch 86)** | ✅ Complete |

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

### DVH-Aware Loss Experiment Results (2026-01-22) ← NEW

**Key Finding: DVH-aware loss achieves best MAE among clinically-focused losses!**

| Metric | Baseline | Grad Loss | Grad+VGG | **DVH-Aware** | Change |
|--------|----------|-----------|----------|---------------|--------|
| Val MAE | 3.73 Gy | 3.67 Gy | **2.27 Gy** | **3.61 Gy** | -3% vs baseline |
| Training Time | 2.55h | 1.85h | 9.74h | **11.2h** | |

**Analysis:**
- DVH-aware loss beats baseline (3.73 Gy) by 3%
- DVH metrics (D95, V70) converge during training - model learns constraints
- Training takes longer (11.2h) due to DVH metric computation
- High volatility in validation MAE (n=2 validation cases)
- Best epoch at 86 (late convergence - DVH needs more training time)

**Key insight:** DVH-aware loss provides explicit clinical constraint optimization while maintaining competitive MAE. Unlike VGG which has better MAE but doesn't help Gamma, DVH directly optimizes what clinicians care about.

**Next steps:**
- Run test set evaluation to compute Gamma pass rate
- If Gamma ≥ 35%: DVH approach working, tune weights
- If Gamma ≈ 28%: Add structure-weighted loss or adversarial loss

### Notebooks Needing Creation
- [x] `2026-01-20_ddpm_optimization.ipynb` - Document DDPM training + Phase 1 optimization ✅
- [x] `2026-01-20_strategic_assessment.ipynb` - Scientific value & path forward analysis ✅
- [x] `2026-01-20_grad_loss_experiment.ipynb` - Document gradient loss experiment results ✅
- [x] `2026-01-21_grad_vgg_combined.ipynb` - Document Grad+VGG experiment results ✅
- [x] `2026-01-22_dvh_aware_loss.ipynb` - Document DVH-aware loss experiment results ✅ NEW
- [ ] `2026-01-21_semi_multi_modal_hypothesis.ipynb` - Semi-multi-modal hypothesis analysis

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
- [x] ~~baseline_unet_run1~~ (Complete - 2026-01-19)
- [ ] baseline_unet_larger (Planned - increased capacity)

### 1a. Loss Function Experiments (Tier 1 Priority)

**🎯 GOAL: Achieve 95% Gamma (3%/3mm) for clinical deployment.**

Current: ~28% Gamma (Phase A & B) → Target: 95% Gamma

**Key insight (Phase B):** VGG perceptual loss does NOT improve Gamma. Need DVH-aware or structure-weighted losses.

**🔬 NEW: Semi-Multi-Modal Hypothesis (2026-01-21)**

Dose prediction may be semi-multi-modal: PTV/OAR regions are deterministic, but low/intermediate dose "spray" in no-man's land is flexible. Multiple valid solutions exist if DVH constraints are met. This reframes DDPM as potentially viable with proper metrics. See `notebooks/2026-01-21_semi_multi_modal_hypothesis.ipynb` for analysis.

#### Phase A: Gradient Loss ✅ COMPLETE
- [x] ~~grad_loss_0.1~~ ✅ **COMPLETE** (gradient loss only, weight=0.1)
  - Command: `python scripts\train_baseline_unet.py --exp_name grad_loss_0.1 --data_dir I:\processed_npz --use_gradient_loss --gradient_loss_weight 0.1 --epochs 100`
  - **Results:** Val MAE 3.67 Gy (epoch 12), Test MAE 1.44 Gy, **Gamma 27.9%** (nearly doubled from 14.2%!)
  - Training: 1.85 hours, early stopped at epoch 62
  - Best checkpoint: `runs/grad_loss_0.1/checkpoints/best-epoch=012-val/mae_gy=3.670.ckpt`
  - **Conclusion: Gradient loss significantly improves Gamma pass rate while maintaining MAE** ✅

#### Phase B: Combined Perceptual Losses ✅ COMPLETE
- [x] ~~grad_vgg_combined~~ ✅ **COMPLETE** (gradient + VGG perceptual loss)
  - Command: `python scripts\train_baseline_unet.py --exp_name grad_vgg_combined --data_dir I:\processed_npz --use_gradient_loss --gradient_loss_weight 0.1 --use_vgg_loss --vgg_loss_weight 0.001 --epochs 100`
  - **Results:** Val MAE **2.27 Gy** (epoch 32), Test MAE 1.44 Gy, **Gamma ~28%** (unchanged from Phase A!)
  - Training: 9.74 hours, early stopped at epoch 82
  - Best checkpoint: `runs/grad_vgg_combined/checkpoints/best-epoch=032-val/mae_gy=2.267.ckpt`
  - **Conclusion: VGG improves MAE but NOT Gamma. Skip VGG in future experiments.** ❌

#### Phase C: DVH-Aware & Structure-Weighted Losses
- [x] ~~**dvh_aware_loss**~~ ✅ **COMPLETE** (differentiable DVH metrics)
  - Penalize if PTV D95 < prescription dose
  - Penalize if OAR Dmean > constraint
  - **Results:** Val MAE **3.61 Gy** (epoch 86), beats baseline (3.73 Gy) by 3%
  - Training time: 11.2 hours (DVH computation adds overhead)
  - DVH metrics converge - model learns to respect D95 and V70 constraints
  - **Conclusion:** DVH-aware loss achieves best MAE among clinically-focused losses ✅
  - Best checkpoint: `runs/dvh_aware_loss/checkpoints/best-epoch=086-val/mae_gy=3.609.ckpt`
- [ ] **structure_weighted_loss** ⭐ HIGH PRIORITY (weight MSE by clinical importance)
  - 2x weight for PTV70/PTV56 regions
  - 1.5x weight for OAR boundaries (via SDF gradients)
  - 0.5x weight in "no-man's land" (flexible regions)
  - Rationale: Addresses D95 underdosing (-10 to -22 Gy error seen in baseline)
  - Implementation: Multiply MSE by mask weights before reduction
- [ ] grad_loss_sweep (tune gradient_loss_weight: 0.05, 0.1, 0.2) - lower priority
- [ ] ~~vgg_loss_sweep~~ - SKIP (VGG doesn't help Gamma)

#### Phase D: Adversarial Training
- [ ] adversarial_loss (PatchGAN discriminator for sharper edges)
  - Use 3D PatchGAN to critique local patches
  - Start with small λ=0.1 for stability
  - Expected: Gamma toward 80-95% if losses above stall
- [ ] combined_optimal (best losses combined after individual testing)

#### Phase E: Semi-Multi-Modal Validation & DDPM Revisit (if needed)
- [ ] **low_dose_variability_analysis** 🔬 DIAGNOSTIC (analyze ground-truth variation)
  - Compute variance of doses in no-man's land across cases
  - Define "acceptable bounds" from clinical data
  - Validate semi-multi-modal hypothesis
- [ ] **region_specific_gamma** 🔬 DIAGNOSTIC (understand error distribution)
  - Compute Gamma separately for PTV, OAR, and flexible regions
  - Understand where prediction errors concentrate
  - May reveal that Gamma failures are in flexible regions (acceptable) vs PTV (not acceptable)
- [ ] **physics_bounded_ddpm** (only if baseline + DVH plateaus at ~30% Gamma)
  - Region-aware noise schedules
  - Physics-informed regularizers (falloff, homogeneity constraints)
  - Bounded multi-modality sampling
  - Hot-spot prevention (max dose outside PTV < 105% Rx)

**CLI options in `train_baseline_unet.py`:**
- `--use_gradient_loss` - Enable 3D Sobel gradient loss
- `--gradient_loss_weight 0.1` - Weight for gradient loss (default: 0.1)
- `--use_vgg_loss` - Enable 2D VGG perceptual loss (slice-wise)
- `--vgg_loss_weight 0.001` - Weight for VGG loss (default: 0.001)
- `--vgg_slice_stride 8` - Process every Nth slice for VGG (default: 8)
- *TODO: Add `--use_structure_weighted`, `--use_dvh_loss`, `--use_adversarial`*

### 1b. Data & Augmentation (Tier 2 Priority - Critical with n=23)

- [ ] **augmentation_v1** ⭐ NEW (torchio augmentations)
  - Random affine transforms (rotations ±10°, shears)
  - Intensity shifts on CT (HU ±50)
  - Gaussian noise on doses (σ=0.01)
  - Use `torchio` or `SimpleITK` for implementation
  - Rationale: With only 23 cases, models overfit to averages (causing blurring)
- [ ] collect_100_cases (more training data when available)
- [ ] full_3d_gamma (compute proper 3D Gamma, not just central slice)
  - Current evaluation underestimates true performance
  - Use pymedphys with subsample=1 for accuracy

### 2. Diffusion Models (DDPM) - **STATUS REVISED**

**Previous assessment:** NOT RECOMMENDED (dose is deterministic, DDPM provides no benefit)

**Revised assessment (2026-01-21):** ⚠️ **MAY BE VIABLE WITH DVH METRICS**
- Semi-multi-modal hypothesis suggests low-dose regions are flexible
- DDPM's "blurring" may be averaging valid solutions, not failing
- Revisit DDPM *after* DVH-aware loss tested on baseline
- If baseline + DVH plateaus, physics-bounded DDPM worth exploring
- See: `notebooks/2026-01-21_semi_multi_modal_hypothesis.ipynb`

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

### 3. Architecture Improvements (Tier 3 Priority)

- [ ] attention_unet (add attention gates to U-Net)
- [ ] deeper_unet (increase to 96 base channels, add dropout 0.1-0.2)
- [ ] nnunet_baseline (try nnU-Net architecture)
- [ ] swin_unetr (Transformer-based architecture)

### 4. Alternative Approaches (Tier 4 Priority)

- [ ] flow_matching_v1 (simpler than diffusion, ODE-based, faster inference)
  - Use `torchdiffeq` for implementation
  - Condition on anatomy similar to current approach
- [ ] ensemble_models (combine multiple models)

### 5. Ablation Studies (Lower Priority)

- [ ] ablation_no_sdf (binary masks only)
- [ ] ablation_no_constraints (no FiLM conditioning)
- [ ] ablation_patch_size (64 vs 128 vs 160)

### 6. Post-95% Gamma (Future - Tier 5)

- [ ] physics_constraints (Monte Carlo surrogate loss for deliverability)
- [ ] mlc_prediction (Phase 2: MLC/arc sequence prediction from dose)

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

*Last updated: 2026-01-22 (Added DVH-aware loss results - 3.61 Gy val MAE, beats baseline by 3%)*
