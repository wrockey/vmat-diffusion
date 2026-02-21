# VMAT Diffusion — Project Plan & State

**This file is the SINGLE AUTHORITATIVE PLAN for the project.**
**It contains strategy, current state, the phased roadmap, and all decisions.**
**It is automatically loaded every session. Keep it current.**

## Documentation & Tracking Hierarchy

| Document | Role | Update when |
|----------|------|-------------|
| **This file** (`.claude/instructions.md`) | **THE PLAN:** strategy, current state, phased roadmap overview, decisions summary | Every session |
| `CLAUDE.md` | Static reference: code conventions, architecture, experiment protocol, GitHub workflow | Rarely |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log (table of all experiments) | After every experiment |
| **GitHub Issues** | Individual tasks, bugs, backburner ideas, decision records, experiment plans | As work progresses |
| **GitHub Milestones** | Phase-level progress tracking (Phase 0–3) | When issues are closed |

### Documentation Rules

- **Do not create separate plan files.** All planning, strategy, roadmap, and decision content lives HERE.
- **Individual tasks and TODOs go in GitHub Issues**, not in this file. This file contains the *overview*; issues contain the *details*.
- **Decision records** live as GitHub Issues with the `type/decision` label. This file has a summary table; issues have the full rationale.
- If a sub-plan is absolutely necessary, it MUST be explicitly referenced from this file with a clear link and status.
- Currently there is one archived sub-plan: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED 2026-01-21, DDPM abandoned).
- **Do not create new documentation files.** If it's living state/planning, it goes here. If it's static reference, it goes in `CLAUDE.md`. If it's an experiment record, it goes in `EXPERIMENTS_INDEX.md`. If it's a task, it's a GitHub Issue.

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

4. **Tasks are tracked in GitHub Issues.** Before starting work, check open issues. When work completes, close the issue with a reference to the commit. When new tasks emerge, create issues.

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

## CURRENT STATE (as of 2026-02-21)

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

### Phase 2 Utilities (added 2026-02-17)

- `scripts/uncertainty_loss.py` — UncertaintyWeightedLoss module (Kendall et al. 2018). Ready to import; replaces manual loss weight tuning.
- `scripts/calibrate_loss_normalization.py` — Loss calibration script. Has stub loss functions — see GitHub issue for replacing with real implementations.

---

## ROADMAP OVERVIEW

Detailed tasks for each phase are tracked as **GitHub Issues** with phase labels and milestones. This section provides the strategic overview only.

### Phase 0: Work Machine Setup — Milestone: `Phase 0: Setup`

**Goal:** Production environment ready with 100+ cases and fixed evaluation pipeline.

Key items (see GitHub Issues with `phase/0-setup` label):
- WSL environment setup on work machine
- Collect and anonymize 100+ DICOM-RT cases
- Preprocess all cases
- **Fix D95 pipeline artifact** (CRITICAL — blocks all DVH evaluation)
- Document data provenance and ethics for publication

### Phase 1: Clinical Evaluation Framework — Milestone: `Phase 1: Evaluation Framework`

**Goal:** Define what "good" means before training anything on the new dataset.

Key items (see GitHub Issues with `phase/1-eval` label):
- Per-structure DVH compliance evaluation
- PTV-region Gamma at multiple dose thresholds
- Dose gradient/falloff analysis
- Single-case clinical acceptability report
- Physician preference ranking study (plan early, execute after Phase 2)

### Phase 2: Combined Loss Experiment — Milestone: `Phase 2: Combined Loss`

**Goal:** First publishable experiment — combined 5-component loss with uncertainty weighting.

Key items (see GitHub Issues with `phase/2-combined` label):
- Replace calibration script stubs with real loss functions
- Extend UncertaintyWeightedLoss for per-component initialization
- Integrate combined loss into training script
- Run 3-seed experiment on 100+ cases with mid-training ablation

**Loss weight strategy** (decided 2026-02-17):
1. **Normalize first:** Calibrate with real losses → `loss_normalization_calib.json`
2. **Uncertainty Weighting (Kendall 2018):** Learn σ per loss, weights = 1/(2σ²)
3. **Fallback:** GradNorm if any loss dominates

**Expected results with 100+ cases:**
- Global 3%/3mm: 75-88% (literature benchmark)
- PTV-region 3%/3mm: 90-95%+
- Publishable results with multi-seed statistics

### Phase 3: Iterate & Publish — Milestone: `Phase 3: Iterate & Publish`

**Goal:** Result-driven iteration, failure analysis, and manuscript submission.

Key items (see GitHub Issues with `phase/3-iterate` label):
- Analyze Phase 2 results → determine next direction
- Failure case report (bottom 10%)
- Code release preparation (anonymize, DOI)
- Medical Physics manuscript

### Parking Lot — GitHub Issues with `type/backburner` label

Ideas to revisit only if the above plateaus:
- Adversarial loss (PatchGAN)
- Flow Matching / Consistency Models
- Architecture alternatives (nnU-Net, Swin-UNETR)
- Ensemble of existing models
- OAR contour perturbation augmentation

---

## DECISIONS LOG (Summary)

Key decisions with rationale. Full decision records are GitHub Issues with the `type/decision` label.

| Date | Decision | GitHub Issue |
|------|----------|-------------|
| 2026-02-17 | Paper framing: "Loss-function engineering for clinically acceptable prostate VMAT dose prediction" | `type/decision` |
| 2026-02-17 | Add physician preference ranking to Phase 1 eval framework | `type/decision` |
| 2026-02-17 | Publication target: Medical Physics, single comprehensive paper | `type/decision` |
| 2026-02-17 | Use Uncertainty Weighting for combined loss, NOT grid search | `type/decision` |
| 2026-02-17 | Pilot 28-31% global Gamma is expected at n=23, not a failure | See Strategic Direction |
| 2026-02-17 | Add OAR contour perturbation to Phase 3 augmentation | `type/backburner` |
| 2026-02-17 | GT D95 = 55 Gy is a pipeline artifact, NOT a clinical finding | `phase/0-setup` + `bug` |
| 2026-02-13 | Start clean on work machine with 100+ cases | See Current State |
| 2026-02-13 | Shift primary metric from global Gamma to DVH + PTV Gamma + gradient realism | `type/decision` |
| 2026-01-21 | Dose prediction is semi-multi-modal (low-dose regions flexible) | See Strategic Direction |
| 2026-01-21 | VGG perceptual loss not useful (improves MAE, NOT Gamma, 5x slower) | See What NOT to Pursue |
| 2026-01-20 | DDPM not recommended (structural mismatch, no benefit) | `type/decision` |

---

## PLATFORM REFERENCE

### Work Machine (Active — update paths after setup)

| Setting | Value |
|---------|-------|
| Platform | WSL2 (Ubuntu) on Windows — recommended for both Claude Code and training |
| Project | TBD (update after setup) |
| Data | TBD — store on Linux filesystem, NOT /mnt/c/ (100+ cases) |
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

- Linux/WSL: `num_workers=2`, `persistent_workers=False`
- Native Windows: `num_workers=0`

### Troubleshooting

Detailed troubleshooting for GPU stability, watchdog, training hangs: see `docs/training_guide.md` (partially superseded — DDPM architecture sections are historical, but troubleshooting/GPU/monitoring sections remain valid).

---

*Last updated: 2026-02-21 (Migrated task tracking to GitHub Issues + Milestones. Slimmed roadmap to overview-only — details in issues.)*
