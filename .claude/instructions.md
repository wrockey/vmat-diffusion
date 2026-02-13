# VMAT Diffusion — Project State

**This file is the single source of truth for project strategy, current state, and next steps.**
**It is automatically loaded every session. Keep it current.**

| Document | Role | Update when |
|----------|------|-------------|
| **This file** (`.claude/instructions.md`) | Living project state: strategy, decisions, next steps | Every session |
| `CLAUDE.md` | Static reference: code conventions, architecture, experiment protocol | Rarely |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log (table of all experiments) | After every experiment |

**Do not create new documentation files.** If it's living state, it goes here. If it's static reference, it goes in `CLAUDE.md`. If it's an experiment record, it goes in `EXPERIMENTS_INDEX.md`.

---

## PRIME DIRECTIVES

1. **Every experiment follows the full protocol in CLAUDE.md "Experiment Documentation Requirements" — automatically, every time, no reminders needed.** This means:
   - Git commit before training (record hash)
   - Publication-ready notebook with all 10 sections, captions, and written assessments on every figure
   - Figure generation script (`scripts/generate_<exp_name>_figures.py`) saving PNG (300 DPI) + PDF
   - `notebooks/EXPERIMENTS_INDEX.md` updated with date, git hash, metrics, notebook link
   - Results committed to git
   - **An experiment without full documentation is an experiment that never happened.**

2. **This file is updated at the end of every work session.** Move completed work, update the performance table, record new decisions.

3. **Figures are publication-ready from the start.** Serif font, 12pt minimum, 300 DPI, colorblind-friendly, labeled axes with units, legends, captions with clinical interpretation. No exceptions.

---

## STRATEGIC DIRECTION (Updated 2026-02-13)

### Primary Goal: Clinical Acceptability (NOT Global Gamma)

**Global Gamma (3%/3mm) is NOT the primary optimization target.** It penalizes clinically irrelevant differences in low-dose regions where multiple valid dose distributions are acceptable.

**Optimize for (in priority order):**

| Priority | Metric | Target | Rationale |
|----------|--------|--------|-----------|
| CRITICAL | PTV70 D95 | >= 66.5 Gy (95% of 70 Gy) | Target coverage drives patient outcome |
| CRITICAL | PTV56 D95 | >= 53.2 Gy (95% of 56 Gy) | Elective target coverage |
| CRITICAL | OAR DVH compliance | Per QUANTEC limits | Organ sparing |
| HIGH | PTV-region Gamma (3%/3mm) | > 95% | Accuracy where it matters clinically |
| HIGH | Dose gradient realism | Monotonic falloff from PTV, ~6mm penumbra | Proxy for physical deliverability |
| DIAGNOSTIC | Overall Gamma (3%/3mm) | Track only | Expected to be low due to valid low-dose diversity |
| DIAGNOSTIC | MAE (Gy) | Track only | Useful for training convergence, not a clinical endpoint |

### Why This Shift

Three insights from the gamma metric analysis (2026-01-23) and subsequent discussion:

1. **Global Gamma penalizes valid diversity.** The low-dose "spray" between PTVs and OARs varies across clinically acceptable plans. The model averages these valid solutions, producing blur and low Gamma — but the resulting dose may still be clinically acceptable.

2. **Low-dose region is clinically unconstrained.** It's physics-bounded (beam penumbra, inverse square law, MLC limits) but not DVH-constrained. Many valid solutions exist. Pixel-wise comparison there is measuring the wrong thing.

3. **Physical realism matters more than pixel accuracy in low-dose regions.** A dose with smooth, monotonic falloff from PTV is more clinically useful than one with correct pixel values but unphysical gradients.

### Loss Function Design Principle

```
Loss = w_ptv * L_ptv_asymmetric  +  w_oar * L_oar_dvh  +  w_gradient * L_gradient  +  w_bg * L_background
```

- **PTV:** Asymmetric (penalize underdose >> overdose) + DVH D95 term. Must be accurate.
- **OAR:** DVH compliance (Dmean, Vx). Must respect limits.
- **Gradient:** Enforce physically realistic falloff from PTV boundaries.
- **Background:** Very low weight (0.1x) or zero. Just needs physical plausibility.

All components already implemented in prior experiments — need to combine with appropriate weights.

### What NOT to Pursue

- Global Gamma as an optimization target (track it, don't chase it)
- DDPM tuning (structural mismatch — see Decisions Log)
- VGG perceptual loss (no Gamma improvement, 5x training overhead)
- Pure MSE/MAE optimization (causes PTV underdosing)

---

## CURRENT STATE (as of 2026-01-23)

### Model Performance

| Model | Val MAE | Test MAE | Gamma | PTV Gamma | D95 Gap | Key Strength |
|-------|---------|----------|-------|-----------|---------|--------------|
| Baseline U-Net | 3.73 Gy | 1.43 Gy | 14.2% | — | ~-20 Gy | Starting point |
| Gradient Loss 0.1 | 3.67 Gy | 1.44 Gy | 27.9% | — | ~-7 Gy | Doubled Gamma |
| DVH-Aware | 3.61 Gy | **0.95 Gy** | 27.7% | — | ~-7 Gy | **Best test MAE** |
| Structure-Weighted | **2.91 Gy** | 1.40 Gy | **31.2%** | **41.5%** | ~-7 Gy | **Best Gamma** |
| Asymmetric PTV | 3.36 Gy | 1.89 Gy | — | — | **-5.95 Gy** | **Best D95** |

### Key Numbers

- **Dataset:** 23 usable cases (24 total, case_0013 skipped — non-SIB). Expecting 100-150 near-term.
- **PTV-region Gamma:** 41.5% vs 31.2% overall — confirms model is more accurate where it matters.
- **Ground truth PTV70 D95:** 55 Gy — fails 66.5 Gy clinical threshold by 11.5 Gy. Threshold may be too strict for this dataset. Re-evaluate with 100+ cases.
- **All models pass OAR constraints** (conservative on OAR sparing).
- **All models systematically underdose PTVs** (MSE loss treats underdose/overdose equally; asymmetric loss partially addresses this).

### Key Checkpoints and Artifacts

| What | Path |
|------|------|
| Best test MAE model | `runs/dvh_aware_loss/checkpoints/best-epoch=086-val/mae_gy=3.609.ckpt` |
| Best val MAE model | `runs/grad_vgg_combined/checkpoints/best-epoch=032-val/mae_gy=2.267.ckpt` |
| Best Gamma model | `runs/structure_weighted_loss/checkpoints/` |
| Best D95 model | `runs/asymmetric_ptv_loss/checkpoints/` |
| Test predictions | `predictions/dvh_aware_loss_test/evaluation_results.json` |

---

## NEXT STEPS (Prioritized)

### Phase 1: Clinical Evaluation Framework (NOW)

Build a new evaluation script/module that replaces global Gamma as the primary metric:

- Per-structure DVH compliance (pass/fail per QUANTEC constraint)
- PTV-only Gamma (3%/3mm) — already shown to be 41.5% vs 31.2% overall
- Dose gradient/falloff analysis: monotonicity, penumbra width
- Single "clinical acceptability" report per case

**Then:** Re-evaluate ALL five existing models with the new framework. The DVH-aware or structure-weighted model may already be much closer to clinical acceptability than global Gamma suggests.

### Phase 2: Combined Loss Function (NOW)

Combine the three best loss components (all already implemented):

- Asymmetric PTV loss (underdose penalty >> overdose)
- DVH-aware loss (D95, Dmean/Vx compliance)
- Structure-weighted loss (2x PTV, 1.5x OAR boundary, 0.1x background)
- Gradient loss (3D Sobel for edge preservation)

Train and evaluate with the new clinical framework from Phase 1.

### Phase 3: Retrain on 100+ Cases (WHEN DATA ARRIVES)

1. Preprocess new cases with `preprocess_dicom_rt_v2.2.py`
2. Update train/val/test splits
3. Retrain combined-loss model on larger dataset
4. Evaluate with clinical framework
5. Re-assess ground truth D95 threshold with more cases

### Parking Lot (revisit only if above plateaus)

- Adversarial loss (PatchGAN) for edge sharpness
- Attention U-Net or deeper network (96 base channels)
- Flow Matching (generative: sample single plausible solutions instead of averaging)
- Physics-bounded DDPM (region-aware noise schedules)
- nnU-Net, Swin-UNETR (architecture alternatives)
- Data augmentation (torchio: on top of larger dataset, not instead of)
- Ensemble of existing models (quick experiment: average predictions)

---

## DECISIONS LOG

Key decisions with rationale. Do not revisit without new evidence.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-13 | **Shift primary metric from global Gamma to DVH compliance + PTV Gamma + gradient realism** | Global Gamma penalizes valid low-dose diversity; PTV Gamma (41.5%) is much higher than overall (31.2%); DVH compliance + physical realism are what clinicians actually evaluate |
| 2026-01-23 | Ground truth itself fails clinical D95 threshold | GT PTV70 D95 = 55 Gy vs 66.5 Gy threshold; dataset may have non-standard planning; re-evaluate with more data |
| 2026-01-21 | Dose prediction is semi-multi-modal | Low-dose regions are flexible; multiple valid solutions exist; pure pixel-wise metrics penalize valid diversity |
| 2026-01-21 | VGG perceptual loss not useful | Improves MAE but NOT Gamma; adds 5x training time |
| 2026-01-20 | DDPM not recommended | Matches baseline but doesn't beat it; structural mismatch (more steps = worse); near-zero sample variability means it's not generative; added complexity with no benefit. May revisit with physics-bounded approach if simpler methods plateau. |

---

## PLATFORM REFERENCE

| Setting | Value |
|---------|-------|
| Platform | **Native Windows** (NOT WSL for training) |
| Project | `C:\Users\Bill\vmat-diffusion-project` |
| Data | `I:\processed_npz` (23 cases) |
| Conda env | `vmat-win` (via Pinokio miniconda) |
| GPU | NVIDIA RTX 3090 (24 GB) |

### Activate Environment

```cmd
call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win
cd C:\Users\Bill\vmat-diffusion-project
```

### Claude Code Runs in WSL

- File operations (read, edit, git): Use WSL paths `/mnt/c/Users/Bill/vmat-diffusion-project`
- Python scripts (training, inference): Use `cmd.exe /c` passthrough for GPU access:
  ```bash
  cmd.exe /c "call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win && cd C:\Users\Bill\vmat-diffusion-project && python scripts\train_baseline_unet.py --exp_name test --data_dir I:\processed_npz --epochs 10"
  ```

### DataLoader Settings (avoid deadlocks)

- WSL: `num_workers=2`, `persistent_workers=False`
- Native Windows: `num_workers=0`

### Troubleshooting

Detailed troubleshooting for GPU stability, watchdog, training hangs: see `docs/training_guide.md`.

---

*Last updated: 2026-02-13 (Strategic pivot: clinical acceptability over global Gamma; documentation consolidation)*
