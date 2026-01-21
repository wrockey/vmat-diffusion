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

### Notebooks Needing Creation
- [x] `2026-01-20_ddpm_optimization.ipynb` - Document DDPM training + Phase 1 optimization ✅
- [x] `2026-01-20_strategic_assessment.ipynb` - Scientific value & path forward analysis ✅
- [x] `2026-01-20_grad_loss_experiment.ipynb` - Document gradient loss experiment results ✅

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

Current: 27.9% Gamma (Phase A) → Target: 95% Gamma

#### Phase A: Gradient Loss ✅ COMPLETE
- [x] ~~grad_loss_0.1~~ ✅ **COMPLETE** (gradient loss only, weight=0.1)
  - Command: `python scripts\train_baseline_unet.py --exp_name grad_loss_0.1 --data_dir I:\processed_npz --use_gradient_loss --gradient_loss_weight 0.1 --epochs 100`
  - **Results:** Val MAE 3.67 Gy (epoch 12), Test MAE 1.44 Gy, **Gamma 27.9%** (nearly doubled from 14.2%!)
  - Training: 1.85 hours, early stopped at epoch 62
  - Best checkpoint: `runs/grad_loss_0.1/checkpoints/best-epoch=012-val/mae_gy=3.670.ckpt`
  - **Conclusion: Gradient loss significantly improves Gamma pass rate while maintaining MAE** ✅

#### Phase B: Combined Perceptual Losses ← CURRENT
- [ ] **grad_vgg_combined** 🔄 **NEXT UP** (gradient + VGG perceptual loss)
  - Command: `python scripts\train_baseline_unet.py --exp_name grad_vgg_combined --data_dir I:\processed_npz --use_gradient_loss --gradient_loss_weight 0.1 --use_vgg_loss --vgg_loss_weight 0.001 --epochs 100`
  - Expected: Gamma +10-20% over gradient-only (target: 40-50%)
  - Decision tree after results - see `.claude/instructions.md`

#### Phase C: Loss Tuning & Clinical Losses
- [ ] grad_loss_sweep (tune gradient_loss_weight: 0.05, 0.1, 0.2)
- [ ] vgg_loss_sweep (tune vgg_loss_weight: 0.0005, 0.001, 0.002)
- [ ] **structure_weighted_loss** ⭐ NEW (weight MSE by clinical importance)
  - 2x weight for PTV70/PTV56 regions
  - 1.5x weight for OAR boundaries (via SDF gradients)
  - Rationale: Addresses D95 underdosing (-10 to -22 Gy error seen in baseline)
  - Implementation: Multiply MSE by mask weights before reduction
- [ ] **dvh_aware_loss** ⭐ NEW (differentiable DVH metrics)
  - Penalize if PTV D95 < prescription dose
  - Penalize if OAR Dmean > constraint
  - Rationale: Directly optimizes what clinicians care about
  - Implementation: Approximate DVH via sorted dose histograms per structure (differentiable)

#### Phase D: Adversarial Training
- [ ] adversarial_loss (PatchGAN discriminator for sharper edges)
  - Use 3D PatchGAN to critique local patches
  - Start with small λ=0.1 for stability
  - Expected: Gamma toward 80-95% if losses above stall
- [ ] combined_optimal (best losses combined after individual testing)

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

*Last updated: 2026-01-21 (Added structure-weighted, DVH-aware, augmentation experiments; reorganized priority tiers for 95% Gamma goal)*
