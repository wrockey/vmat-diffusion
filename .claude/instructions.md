# VMAT Diffusion — Project Plan & State

**This file is the SINGLE AUTHORITATIVE PLAN for the project.**
**It contains strategy, current state, the phased roadmap, and all decisions.**
**It is automatically loaded every session. Keep it current.**

| Document | Role | Update when |
|----------|------|-------------|
| **This file** (`.claude/instructions.md`) | **THE PLAN:** living project state, strategy, phased roadmap, decisions log | Every session |
| `CLAUDE.md` | Static reference: code conventions, architecture, experiment protocol | Rarely |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log (table of all experiments) | After every experiment |

### Documentation Rules

- **Do not create separate plan files.** All planning, strategy, roadmap, and decision content lives HERE.
- If a sub-plan is absolutely necessary (e.g., a complex multi-step investigation), it MUST be explicitly referenced from this file with a clear link and status.
- Currently there is one archived sub-plan: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED 2026-01-21, DDPM abandoned).
- **Do not create new documentation files.** If it's living state/planning, it goes here. If it's static reference, it goes in `CLAUDE.md`. If it's an experiment record, it goes in `EXPERIMENTS_INDEX.md`.

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

## STRATEGIC DIRECTION (Updated 2026-02-17)

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

## CURRENT STATE (as of 2026-02-17)

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
- **Ground truth PTV70 D95 reads 55 Gy** — now identified as a **pipeline artifact** (PTV mask/dose grid boundary mismatch), NOT a clinical finding. All delivered plans have D95 >= 66.5 Gy. Priority fix for Phase 0. See decisions log 2026-02-17.
- **Pilot Gamma (28-31% global)** is in line with the field at n=23 — literature benchmarks show 75-85% at n=50-100. Not a model failure. Expect 75-88% global / 90-95% PTV-region with 100+ cases.
- **All models pass OAR constraints** but D95 gap appeared to show systematic PTV underdosing (may be partly or fully explained by the D95 artifact above).
- **Gradient loss is essential** — nearly doubled Gamma for free.

### New Phase 2 Utilities (added 2026-02-17)

- `scripts/uncertainty_loss.py` — UncertaintyWeightedLoss module (Kendall et al. 2018). Ready to import; replaces manual loss weight tuning.
- `scripts/calibrate_loss_normalization.py` — Loss calibration script. Loads NPZ files, computes average raw loss values, recommends `initial_log_sigma` per component. Has stub loss functions — replace with real implementations during Phase 2 setup.

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
8. **Data provenance & ethics (required for Medical Physics submission):**
   - IRB approval status and protocol number (TBD)
   - Anonymization method (DICOM de-identification procedure)
   - Informed consent / waiver documentation
   - Case mapping: which raw DICOMs map to which NPZ case IDs
   - Data availability statement for publication (will data be shared? TCIA deposit?)
   - Record exclusion criteria and any cases excluded with reasons

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

**Loss weight strategy** (external review consensus, 2026-02-17):
1. **Normalize first:** Run each loss individually for 10-20 epochs on validation set, record mean value, divide each term by its mean so all start ~1.0 magnitude. Removes 80% of weight-tuning difficulty.
2. **Uncertainty Weighting (Kendall et al. 2018):** Learn one scalar σ per loss during training; weights become 1/(2σ²) automatically. Stable when all losses act on same output (dose volume). One line of code in Lightning. This is the recommended approach — NOT grid search.
3. **Fallback:** If any loss dominates (monitor per-component curves in TensorBoard), apply GradNorm for last 20-30% of training, or do cheap sequential ±20% weight ablation for 5 epochs each.

**Component ablation sweep** (cheap validation, ~5% extra compute):
- At 50% training, run 5-epoch ablation: turn each loss on/off one at a time
- Verifies whether pilot loss rankings hold on the larger dataset
- Pilot rankings have ~40-60% chance of shifting at n=100+ (structure-weighted and DVH losses most likely to stay strong; asymmetric PTV may weaken or strengthen)

**Multi-seed runs (required for publication):**
- Minimum 3 seeds: 42, 123, 789
- Report mean +/- std for all primary metrics (MAE, Gamma, D95, DVH compliance)
- Paired Wilcoxon signed-rank test for key comparisons (combined loss vs best individual loss)
- Per-case results table in notebook (enables paired analysis)

Evaluate with the clinical framework from Phase 1. With 100+ cases, expect:
- 10-15 test cases (statistically meaningful)
- Significantly better absolute metrics than pilot
- **Realistic Gamma targets** (from literature benchmarks, 2026-02-17):
  - Global 3%/3mm: 75-88% (comparable papers report 75-85% at n=50-100 with U-Net)
  - PTV-region 3%/3mm: 90-95%+ (where clinical accuracy matters)
  - Note: pilot 28-31% global is in line with field at n=23 — not a red flag
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
| 2026-02-17 | **Publication target: Medical Physics, single comprehensive paper** | External review confirms Med Phys is the ideal venue for a ~100-case, 5-ablation, combined-results loss-engineering paper. PMB if emphasizing physics/novelty. JACMP for a follow-up clinical implementation study. Don't split prematurely — one strong paper first, second "clinical validation" paper (with physician rankings + deliverability) later if warranted. |
| 2026-02-17 | **Use Uncertainty Weighting for combined loss, NOT grid search** | Kendall et al. 2018 approach: learn σ per loss, weights = 1/(2σ²). Normalize losses first (run each 10-20 epochs, divide by mean). External review consensus — stable, cheap, proven in medical MTL. GradNorm as fallback if any loss dominates. |
| 2026-02-17 | **Pilot 28-31% global Gamma is expected at n=23, not a failure** | Literature benchmarks: pure global 3%/3mm on <60 cases = 75-85% typical; our lower number explained by (a) n=23, (b) true 3D patch training + sliding-window blending artifacts, (c) no dose thresholding (most papers ignore <10% dose voxels). With 100+ cases + combined losses: expect 75-88% global, 90-95%+ PTV-region. Strategic pivot to DVH + PTV Gamma confirmed as correct by external review. |
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

Detailed troubleshooting for GPU stability, watchdog, training hangs: see `docs/training_guide.md` (partially superseded — DDPM architecture sections are historical, but troubleshooting/GPU/monitoring sections remain valid).

---

*Last updated: 2026-02-17 (Incorporated external review feedback: loss weight strategy, publication venue, Gamma benchmarks, physician ranking, contour perturbation, paper framing)*
