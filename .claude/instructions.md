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
| CRITICAL | PTV70 D95 | >= 66.5 Gy (95% of 70 Gy) | Prostate coverage — drives patient outcome |
| CRITICAL | PTV56 D95 | >= 53.2 Gy (95% of 56 Gy) | Seminal vesicle coverage (dose-painted SIB) |
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

## CURRENT STATE (as of 2026-02-13)

### Transition: Home (Pilot) → Work (Production)

The home phase (23 cases, RTX 3090) is complete. It was a **pilot study** that validated methodology and loss function design. The trained weights are not worth porting — the code and documented findings are the deliverables.

**What transfers from the pilot:**
- All loss function implementations (gradient, DVH-aware, structure-weighted, asymmetric PTV)
- The strategic direction (clinical acceptability > global Gamma)
- The decisions log (DDPM dead end, VGG useless, etc.)
- The preprocessing pipeline and evaluation infrastructure

**What does NOT transfer:**
- Trained checkpoints (retrain from scratch on 100+ cases)
- The 23-case test set results (n=2 test set is not statistically meaningful)

### Pilot Study Results (n=23, Home Machine — For Reference Only)

| Model | Val MAE | Test MAE | Gamma | PTV Gamma | D95 Gap | Key Strength |
|-------|---------|----------|-------|-----------|---------|--------------|
| Baseline U-Net | 3.73 Gy | 1.43 Gy | 14.2% | — | ~-20 Gy | Starting point |
| Gradient Loss 0.1 | 3.67 Gy | 1.44 Gy | 27.9% | — | ~-7 Gy | Doubled Gamma |
| DVH-Aware | 3.61 Gy | **0.95 Gy** | 27.7% | — | ~-7 Gy | **Best test MAE** |
| Structure-Weighted | **2.91 Gy** | 1.40 Gy | **31.2%** | **41.5%** | ~-7 Gy | **Best Gamma** |
| Asymmetric PTV | 3.36 Gy | 1.89 Gy | — | — | **-5.95 Gy** | **Best D95** |

### Key Findings from Pilot

- **PTV-region Gamma** (41.5%) much higher than overall (31.2%) — confirms model is more accurate where it matters clinically.
- **Ground truth PTV70 D95** was 55 Gy — failed 66.5 Gy clinical threshold by 11.5 Gy. Re-evaluate with 100+ cases.
- **All models pass OAR constraints** but systematically underdose PTVs.
- **Gradient loss is essential** — nearly doubled Gamma for free.

---

## NEXT STEPS (Prioritized)

### Phase 0: Work Machine Setup (NOW)

1. Clone repo to work machine, install conda environment (`environment.yml`)
2. Verify GPU access: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
3. Collect and anonymize 100+ DICOM-RT cases
4. Preprocess all cases: `python scripts/preprocess_dicom_rt_v2.2.py --skip_plots`
5. Verify preprocessing: spot-check 3-5 cases with `notebooks/verify_npz.ipynb`
6. **Fix D95 pipeline artifact** — GT PTV70 D95 reads 55 Gy but clinical plans guarantee >= 66.5 Gy. Root cause: PTV mask/dose grid boundary mismatch (see 2026-02-17 decision). Verify by: (a) eroding PTV mask by 1-2mm and recomputing D95, (b) checking dose grid coverage vs PTV extent, (c) comparing binary mask vs SDF-derived mask boundary. Must resolve before any DVH-based evaluation is meaningful.
7. Update PLATFORM REFERENCE section below with work machine paths

### Phase 1: Clinical Evaluation Framework

Build before training anything — defines what "good" means on the new dataset:

- Per-structure DVH compliance (pass/fail per QUANTEC constraint)
- PTV-only Gamma (3%/3mm)
- Dose gradient/falloff analysis: monotonicity, penumbra width
- Single "clinical acceptability" report per case
- Validate ground truth D95 thresholds on the larger dataset
- **Physician preference ranking** (blind side-by-side: predicted vs ground truth vs alternative plans). Strengthens publication and captures clinical quality beyond automated metrics. (Suggested by external review, 2026-02-17.)

### Phase 2: Combined Loss — First Real Experiment

Skip individual loss ablations (already done in pilot). Go straight to the combined loss:

- Gradient loss (3D Sobel, weight 0.1)
- Structure-weighted loss (2x PTV, 1.5x OAR boundary, 0.1x background)
- Asymmetric PTV loss (underdose penalty >> overdose)
- DVH-aware loss (D95, Dmean/Vx compliance)

Evaluate with the clinical framework from Phase 1. With 100+ cases, expect:
- 10-15 test cases (statistically meaningful)
- Significantly better absolute metrics than pilot
- Publishable results

### Phase 3: Iterate Based on Results

Depending on Phase 2 outcomes:
- If DVH compliance is close but not there → tune loss weights
- If architecture is the bottleneck → try attention U-Net or deeper network
- If data diversity is still limiting → add augmentation:
  - Standard geometric (torchio: random affine, elastic deformation, flip)
  - **OAR contour perturbations** — small random shifts to structure boundaries to simulate inter-observer contouring variability. Clinically motivated augmentation. (Suggested by external review, 2026-02-17.)
  - Random constraint vector perturbations (small noise on OAR limits)

### Parking Lot (revisit only if above plateaus)

- Adversarial loss (PatchGAN) for edge sharpness
- Flow Matching / Consistency Models (generative: sample single plausible solutions instead of averaging)
- Physics-bounded DDPM (region-aware noise schedules)
- nnU-Net, Swin-UNETR (architecture alternatives)
- Lightweight cross-attention or Swin blocks in bottleneck only (VRAM-conscious)
- Ensemble of existing models (quick experiment: average predictions)

---

## DECISIONS LOG

Key decisions with rationale. Do not revisit without new evidence.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-17 | **Paper framing: "Loss-function engineering for clinically acceptable prostate VMAT dose prediction"** | External review (Grok, 2026-02-17) independently validated project direction and suggested this framing. Pilot has 5 loss variants with clean ablation data — with 100+ cases and combined loss results, this is a strong Medical Physics submission. |
| 2026-02-17 | **Add physician preference ranking to Phase 1 eval framework** | Blind side-by-side comparison (predicted vs ground truth) captures clinical quality beyond automated metrics. Gamma alone is misleading (already known). Strengthens publication. Suggested by external review. |
| 2026-02-17 | **Add OAR contour perturbation to Phase 3 augmentation** | Small random shifts to structure boundaries simulate inter-observer contouring variability — a real clinical source of diversity. More clinically motivated than generic elastic deformation alone. Suggested by external review. |
| 2026-02-13 | **Start clean on work machine with 100+ cases** | 23-case pilot validated methodology and loss design; trained weights are throwaway; code + docs are the deliverable; n=2 test set not statistically meaningful; 100+ cases needed for publishable results |
| 2026-02-13 | **Shift primary metric from global Gamma to DVH compliance + PTV Gamma + gradient realism** | Global Gamma penalizes valid low-dose diversity; PTV Gamma (41.5%) is much higher than overall (31.2%); DVH compliance + physical realism are what clinicians actually evaluate |
| 2026-02-17 | **GT D95 = 55 Gy is a pipeline artifact, NOT a clinical finding** | Clinical confirmation: all delivered prostate VMAT plans have PTV70 D95 >= 66.5 Gy (95% of 70 Gy) — plans failing this are rejected and re-optimized. The 55 Gy reading is caused by PTV mask/dose grid boundary mismatch: linear dose interpolation smooths steep falloff at PTV edge + mask may extend 1-2 voxels beyond TPS boundary into falloff zone. ~5% of PTV voxels in falloff → D95 drops to 55 Gy. Fix: erode PTV mask 1-2mm before D95 evaluation, or verify dose grid fully covers PTV. Priority fix for Phase 0. Replaces 2026-01-23 entry. |
| 2026-01-21 | Dose prediction is semi-multi-modal | Low-dose regions are flexible; multiple valid solutions exist; pure pixel-wise metrics penalize valid diversity |
| 2026-01-21 | VGG perceptual loss not useful | Improves MAE but NOT Gamma; adds 5x training time |
| 2026-01-20 | DDPM not recommended | Matches baseline but doesn't beat it; structural mismatch (more steps = worse); near-zero sample variability means it's not generative; added complexity with no benefit. May revisit with physics-bounded approach if simpler methods plateau. |

---

## PLATFORM REFERENCE

### Work Machine (Active — update paths after setup)

| Setting | Value |
|---------|-------|
| Platform | TBD (update after setup) |
| Project | TBD |
| Data | TBD (100+ cases) |
| Conda env | `vmat-diffusion` (`environment.yml`) |
| GPU | NVIDIA RTX 3090 (24 GB) |

### Home Machine (Pilot — archived)

| Setting | Value |
|---------|-------|
| Platform | Native Windows (NOT WSL for training) |
| Project | `C:\Users\Bill\vmat-diffusion-project` |
| Data | `I:\processed_npz` (23 cases) |
| Conda env | `vmat-win` (via Pinokio miniconda) |
| GPU | NVIDIA RTX 3090 (24 GB) |

### DataLoader Settings (avoid deadlocks)

- Linux: `num_workers=2`, `persistent_workers=False`
- WSL: `num_workers=2`, `persistent_workers=False`
- Native Windows: `num_workers=0`

### Troubleshooting

Detailed troubleshooting for GPU stability, watchdog, training hangs: see `docs/training_guide.md`.

---

*Last updated: 2026-02-17 (Incorporated external review feedback: physician ranking, contour perturbation augmentation, paper framing, parking lot additions)*
