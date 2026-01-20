# DDPM Optimization Plan

**Created:** 2026-01-20
**Status:** Active
**Goal:** Systematically investigate why DDPM underperforms baseline and identify improvements

---

## Problem Statement

DDPM v1 achieved 12.19 Gy MAE vs baseline's 3.73 Gy MAE (3x worse). Key observation:
- **val_loss decreased steadily** (0.108 → 0.004) ✓
- **val_mae was extremely volatile** (12-64 Gy range) ✗

The model learns to denoise but produces unstable dose predictions.

---

## Hypotheses

| ID | Hypothesis | Test | Priority |
|----|------------|------|----------|
| H1 | Too few sampling steps during inference | Ablation: 50, 100, 250, 500, 1000 steps | High |
| H2 | Cosine schedule suboptimal for dose prediction | Compare: linear, cosine, sqrt schedules | High |
| H3 | Conditioning (SDF/constraints) not effective | Ablation: remove SDF, remove constraints | Medium |
| H4 | Loss function doesn't capture dose quality | Add MAE/DVH loss terms | Medium |
| H5 | Small validation set (n=2) causes metric noise | Use k-fold or larger val split | Low |
| H6 | Architecture needs more capacity | Increase base_channels, add attention | Low |

---

## Experiment Plan

### Phase 1: Quick Wins (No Retraining)

These experiments use the existing checkpoint and only modify inference.

#### Experiment 1.1: Sampling Steps Ablation
- **Hypothesis:** H1
- **Method:** Load best checkpoint, generate predictions with varying sampling steps
- **Steps to test:** [50, 100, 250, 500, 1000]
- **Metrics:** MAE, inference time
- **Expected outcome:** More steps → lower MAE but slower inference
- **Notebook:** `2026-01-XX_ddpm_sampling_steps_ablation.ipynb`

#### Experiment 1.2: Multiple Samples Averaging
- **Hypothesis:** H1 (related)
- **Method:** Generate N predictions per case, average them
- **N to test:** [1, 3, 5, 10]
- **Metrics:** MAE, std across samples
- **Expected outcome:** Averaging reduces variance
- **Notebook:** `2026-01-XX_ddpm_ensemble_averaging.ipynb`

### Phase 2: Schedule & Loss Modifications (Requires Retraining)

#### Experiment 2.1: Diffusion Schedule Comparison
- **Hypothesis:** H2
- **Method:** Retrain with different schedules
- **Schedules:** linear, cosine (current), sqrt, sigmoid
- **Training:** 50 epochs each (quick comparison)
- **Metrics:** val_mae trajectory, final MAE
- **Notebook:** `2026-01-XX_ddpm_schedule_comparison.ipynb`

#### Experiment 2.2: Hybrid Loss Function
- **Hypothesis:** H4
- **Method:** Add dose-specific loss terms
- **Loss variants:**
  - Current: noise prediction MSE only
  - Variant A: noise MSE + 0.1 * dose MAE
  - Variant B: noise MSE + 0.1 * PTV dose MSE
- **Training:** 50 epochs each
- **Notebook:** `2026-01-XX_ddpm_hybrid_loss.ipynb`

### Phase 3: Architecture & Conditioning (Requires Retraining)

#### Experiment 3.1: Conditioning Ablation
- **Hypothesis:** H3
- **Method:** Train variants with reduced conditioning
- **Variants:**
  - Full (current): CT + SDF + constraints
  - No SDF: CT + binary masks + constraints
  - No constraints: CT + SDF only
  - Minimal: CT only
- **Training:** 50 epochs each
- **Notebook:** `2026-01-XX_ddpm_conditioning_ablation.ipynb`

#### Experiment 3.2: Architecture Scaling
- **Hypothesis:** H6
- **Method:** Increase model capacity
- **Variants:**
  - Current: base_channels=48 (23.7M params)
  - Larger: base_channels=64 (~42M params)
  - With attention: add self-attention at bottleneck
- **Training:** 50 epochs each
- **Notebook:** `2026-01-XX_ddpm_architecture_scaling.ipynb`

### Phase 4: Alternative Approaches (If DDPM Continues to Underperform)

#### Experiment 4.1: Direct Regression Improvements
- **Method:** Improve baseline U-Net instead of DDPM
- **Ideas:**
  - Perceptual/adversarial loss for sharper gradients
  - Multi-scale supervision
  - Attention mechanisms
- **Notebook:** `2026-01-XX_baseline_improvements.ipynb`

#### Experiment 4.2: Conditional Flow Matching
- **Method:** Replace DDPM with flow matching (simpler, often better for regression)
- **Rationale:** Flow matching has more direct path from noise to target
- **Notebook:** `2026-01-XX_flow_matching_comparison.ipynb`

---

## Execution Protocol

### Before Each Experiment:
1. `git status` - ensure clean working directory
2. `git add -A && git commit -m "Pre-experiment: <name>"`
3. Record commit hash
4. Create notebook from template

### During Each Experiment:
1. Log all hyperparameters
2. Save training curves
3. Save checkpoints
4. Monitor GPU temps (target < 80°C)
5. Use TodoWrite to show visible progress

### After Each Experiment:
1. Update `notebooks/EXPERIMENTS_INDEX.md` (single source of truth)
2. Update Results Tracking table in this file
4. Save figures (300 DPI, PNG + PDF) to `experiments/<phase>/figures/`
5. Write conclusions in notebook
6. `git commit -m "Results: <name> - <key finding>"`

### Output Directories:
```
experiments/
├── phase1_sampling/     # Exp 1.1: sampling steps ablation
├── phase1_ensemble/     # Exp 1.2: ensemble averaging
├── phase2_schedules/    # Exp 2.1: schedule comparison
└── phase2_loss/         # Exp 2.2: hybrid loss
```

Each directory should contain:
- `results.json` - structured metrics
- `figures/` - plots
- `predictions/` - sample outputs (optional)

---

## Results Tracking

**Full experiment log:** `notebooks/EXPERIMENTS_INDEX.md` (single source of truth)

**Optimization-specific quick reference:**

| Exp ID | Hypothesis | Key Result | MAE (Gy) | Conclusion |
|--------|------------|------------|----------|------------|
| baseline | - | Reference | 3.73 (val) | - |
| ddpm_v1 | - | Underperformed | 12.19 (val) | Loss/MAE disconnect |
| 1.1 sampling | H1 | TBD | TBD | TBD |
| 1.2 ensemble | H1 | TBD | TBD | TBD |
| 2.1 schedule | H2 | TBD | TBD | TBD |
| 2.2 loss | H4 | TBD | TBD | TBD |

**After each experiment:** Update both this table AND `notebooks/EXPERIMENTS_INDEX.md`

---

## Decision Points

### After Phase 1:
- If sampling steps help significantly → focus on inference optimization
- If no improvement → proceed to Phase 2

### After Phase 2:
- If schedule/loss changes help → tune further
- If no improvement → proceed to Phase 3

### After Phase 3:
- If DDPM still underperforms baseline → consider Phase 4 alternatives
- Document findings for publication regardless of outcome

### After Phase 4:
- Select best approach for scaling to 100+ cases
- Write up methodology for publication

---

## Resource Estimates

| Phase | Experiments | Training Time | Priority |
|-------|-------------|---------------|----------|
| 1 | 2 | ~1 hour (inference only) | Do first |
| 2 | 2 | ~4 hours (2x 50 epochs) | High |
| 3 | 2 | ~8 hours (4x 50 epochs) | Medium |
| 4 | 2 | ~8 hours | If needed |

**Total estimated time:** 10-20 hours depending on which phases are needed

---

## Notes

- All experiments use the same train/val/test split for fair comparison
- Results are preliminary (24 cases) - will re-evaluate at 100+ cases
- Focus on relative improvements, not absolute numbers
- Document negative results - they're valuable for the paper

---

*Last updated: 2026-01-20*
