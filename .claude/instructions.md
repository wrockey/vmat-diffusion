# VMAT Diffusion — Project Plan & State

**This file is the SINGLE AUTHORITATIVE PLAN for the project.**
**It contains strategy, current state, the phased roadmap, and all decisions.**
**It is automatically loaded every session. Keep it current.**

## Documentation & Tracking Hierarchy

| Document | Role | Update when |
|----------|------|-------------|
| **This file** (`.claude/instructions.md`) | **THE PLAN:** strategy, current state, phased roadmap overview, decisions summary, session log | Every session (start + end) |
| `CLAUDE.md` | Static reference: code conventions, architecture, experiment protocol, GitHub workflow | Rarely |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log (table of all experiments) | After every experiment |
| **GitHub Issues** | Individual tasks, bugs, backburner ideas, decision records, experiment plans | As work progresses |
| **GitHub Milestones** | Phase-level progress tracking (Phase 0–3) | When issues are closed |
| **GitHub Project Board** | "VMAT Diffusion Roadmap" — kanban view with Phase field. **Private** (owner-only workflow view) | When issue statuses change |
| **GitHub Discussions** | AI review feedback channel (Grok, Claude, etc.). See [guidelines](#ai-review-workflow) | When external reviews arrive |

### Documentation Rules

- **GitHub is the source of truth for project progress.** Detailed findings, verification results, bug descriptions, and work logs go in GitHub issue comments and commit messages — NOT in this file. If you're writing more than 2-3 lines about something in this file, it probably belongs in a GitHub issue comment instead.
- **This file stays lean.** It provides orientation (current state, blockers, strategy) and pointers (issue numbers, commit hashes). Target: under 300 lines. If it grows past that, trim session log and move details to GitHub.
- **Do not create separate plan files.** All planning, strategy, roadmap, and decision content lives HERE.
- **Individual tasks and TODOs go in GitHub Issues**, not in this file. This file contains the *overview*; issues contain the *details*.
- **Decision records** live as GitHub Issues with the `type/decision` label (closed when decided). This file has a summary table with issue numbers; issues have the full rationale.
- If a sub-plan is absolutely necessary, it MUST be explicitly referenced from this file with a clear link and status.
- Currently there is one archived sub-plan: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED 2026-01-21, DDPM abandoned).
- **Do not create new documentation files.** If it's living state/planning, it goes here. If it's static reference, it goes in `CLAUDE.md`. If it's an experiment record, it goes in `EXPERIMENTS_INDEX.md`. If it's a task, it's a GitHub Issue.

---

## SESSION PROTOCOLS

### Start-of-Session Checklist (DO THIS FIRST, EVERY SESSION)

Before doing any work, run through this checklist silently (do not narrate unless something is wrong):

1. **Read this file** — already auto-loaded, but verify the "Last updated" date at the bottom. If stale (>1 session old), flag to user.
2. **Check GitHub Issues** — `gh issue list --state open --limit 50` — scan for new issues, status changes, or AI reviews since last session.
3. **Check GitHub Discussions** — `gh api graphql` query for recent discussions — look for new Grok/AI reviews to triage.
4. **Check git log** — `git log --oneline -10` — understand what changed since last session.
5. **Check project board** — `gh project item-list 2 --owner wrockey` — verify board reflects reality.
6. **Orient** — Based on the above, identify: (a) what phase we're in, (b) what's blocking progress, (c) what the next action is.

If any of the above reveals a discrepancy (e.g., issues closed but board not updated, new AI reviews not triaged), fix it before starting work.

### GitHub Project Board = Primary Status Tracker

**The project board is the single source of truth for what's done, in progress, and planned.** Update it immediately — not at end of session — whenever issue status changes:

- **Starting work on an issue?** Move it to "In Progress" on the board NOW.
- **Finished an issue?** Close it AND move it to "Done" on the board NOW.
- **Created a new issue?** Add it to the board with correct Phase and Status NOW.
- **Discovered something is blocked?** Add the `status/blocked` label NOW.

Do NOT batch board updates to end-of-session. Stale boards hide project state and create confusion.

```bash
# Board update reference (Status field: PVTSSF_lAHOAkj6uc4BP7oyzg-MoCc)
# Todo=f75ad846, In Progress=47fc9ee4, Done=98236657
# Phase field: PVTSSF_lAHOAkj6uc4BP7oyzg-Mtf0
# Phase 0=e73b984f, Phase 1=9dcbf090, Phase 2=eacee063, Phase 3=fddd52ce, Backburner=c4c2f48d, Decision=7f44e30c
gh project item-edit --project-id PVT_kwHOAkj6uc4BP7oy --id "<ITEM_ID>" --field-id "<FIELD_ID>" --single-select-option-id "<OPTION>"
```

### End-of-Session Checklist (DO THIS LAST, EVERY SESSION)

Before the session ends, complete ALL of the following:

1. **Verify GitHub is up to date:**
   - All completed issues are closed (with commit references) and "Done" on board
   - Progress comments added to open issues with specific findings/results
   - New issues created for any tasks discovered during the session (added to board)
   - Labels/milestones current
   - **Board matches reality** — no item should be in a stale status
2. **Update this file — BRIEFLY:**
   - Update "CURRENT STATE" date and blockers (2-3 lines max)
   - Add a 1-line session log entry pointing to commits/issues (see SESSION LOG format)
   - Update decisions table or parking lot ONLY if those sections changed
   - **DO NOT duplicate information already in GitHub issues or commit messages**
3. **Triage AI reviews:**
   - If new Discussions arrived, triage per workflow below
4. **Commit this file:**
   ```bash
   git add .claude/instructions.md
   git commit -m "docs: Update project state — <brief summary of session>"
   ```

### GitHub Sync Commands (Reference)

```bash
# Check open issues
gh issue list --state open --limit 50

# Check recent discussions
gh api graphql -f query='{ repository(owner:"wrockey",name:"vmat-diffusion") { discussions(first:10,orderBy:{field:CREATED_AT,direction:DESC}) { nodes { number title category{name} createdAt } } } }'

# Check project board state
gh project item-list 2 --owner wrockey --format json --jq '.items[] | "\(.content.number // "draft") | \(.status) | \(.phase) | \(.title)"'
```

---

## PRIME DIRECTIVES

0. **PROTECT HEALTH INFORMATION — PARAMOUNT, OVERRIDES ALL OTHER DIRECTIVES.** Protected health information (PHI) must NEVER be committed to git, pushed to GitHub, written to any tracked file, or included in any issue, discussion, or comment. This includes but is not limited to:
   - Patient names, MRNs (medical record numbers), dates of birth, any HIPAA identifiers
   - ProKnow API private keys or credentials (JSON key files, tokens)
   - Any file containing PHI must be in `.gitignore` BEFORE it touches the filesystem
   - Before ANY `git add` or `git commit`, verify no PHI or credentials are staged: `git diff --cached --name-only` and inspect suspicious files
   - If PHI is discovered in the repo at any point, **stop all work immediately** and alert the user
   - This directive is unconditional. No experiment, no deadline, no convenience justifies an exception.

1. **Every experiment follows the full protocol in CLAUDE.md "Experiment Documentation Requirements" — automatically, every time, no reminders needed.** This means:
   - Git commit before training (record hash)
   - Publication-ready notebook with all 10 sections, captions, and written assessments on every figure
   - Figure generation script (`scripts/generate_<exp_name>_figures.py`) saving PNG (300 DPI) + PDF
   - `notebooks/EXPERIMENTS_INDEX.md` updated with date, git hash, metrics, notebook link
   - Results committed to git
   - **An experiment without full documentation is an experiment that never happened.**

2. **This file is updated at the end of every work session** following the End-of-Session Checklist above. No exceptions.

3. **Figures are publication-ready from the start.** Serif font, 12pt minimum, 300 DPI, colorblind-friendly, labeled axes with units, legends, captions with clinical interpretation. No exceptions.

4. **Tasks are tracked in GitHub Issues.** Before starting work, check open issues. When work completes, close the issue with a reference to the commit. When new tasks emerge, create issues.

5. **GitHub state stays in sync.** Project board, milestones, and issue statuses must reflect reality. If they don't, fix them before starting new work.

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

### What NOT to Pursue (Currently)

- Global Gamma as an optimization target (track it, don't chase it)
- DDPM tuning — deprioritized, not abandoned. Revisit if Phase 2 U-Net results plateau (#27)
- VGG perceptual loss (no Gamma improvement, 5x training overhead — pilot observation, may change with correct data)
- Pure MSE/MAE optimization (causes PTV underdosing — pilot observation)

---

## CURRENT STATE (as of 2026-02-23)

### Current Phase: Phase 0 (Setup) — Data collection in progress

**Dataset:**
- Institution A: 76 cases collected, 74 processed (11 SIB, 63 single-Rx). Multiple planners.
- Institution B: ~150 cases expected (all SIB with PTV70 + PTV56). Single planner.
- **Projected usable: ~161 SIB cases** from 2 institutions (after applying inclusion criteria)
- **IRB: Approved**
- **Compute: RTX 3090 (local), Argon HPC (available for Phase 2 scaling)**

**Completed:**
- #1 WSL dev environment (CLOSED)
- #4 D95 pipeline artifact fix (CLOSED — verified on 74 cases, all D95 >= 66.3 Gy)

**In progress:**
- #2 Collect and anonymize remaining ~150 Institution B cases
- #3 Preprocess all cases (74/76 Institution A done)

**Blocking next phase:**
- #38 Locked stratified test set (needs data collection complete)
- #39 Formal inclusion/exclusion criteria (needs to be defined)
- #40 Data lock and versioning (needs data collection complete)

### Transition: Pilot → Production

The pilot phase (23 cases, single institution, v2.2.0 data) is complete. It validated methodology and loss function design. The code transfers; the metrics do not.

**What transfers:** Loss implementations, strategic direction, pipeline code, decisions log.
**What does NOT transfer:** Any trained weights or specific metrics (all invalidated by D95 artifact).

### Pilot Study Results (n=23, Home Machine — METRICS INVALID)

> **WARNING:** All pilot metrics below were computed on v2.2.0 data with the D95 artifact (#4). Absolute values are invalid and cannot be cited. The pilot established methodology and loss implementations — not reliable metrics. See EXPERIMENTS_INDEX.md for details.

| Model | Val MAE | Test MAE | Gamma | PTV Gamma | D95 Gap | Key Strength |
|-------|---------|----------|-------|-----------|---------|--------------|
| Baseline U-Net | 3.73 Gy | 1.43 Gy | 14.2% | — | ~-20 Gy | Starting point |
| Gradient Loss 0.1 | 3.67 Gy | 1.44 Gy | 27.9% | — | ~-7 Gy | Doubled Gamma |
| DVH-Aware | 3.61 Gy | **0.95 Gy** | 27.7% | — | ~-7 Gy | **Best test MAE** |
| Structure-Weighted | **2.91 Gy** | 1.40 Gy | **31.2%** | **41.5%** | ~-7 Gy | **Best Gamma** |
| Asymmetric PTV | 3.36 Gy | 1.89 Gy | — | — | **-5.95 Gy** | **Best D95** |

### What Transfers from the Pilot (and What Doesn't)

**Valid (methodology, not metrics):**
- Loss function implementations (gradient, DVH-aware, structure-weighted, asymmetric PTV)
- The qualitative observation that PTV-region Gamma > global Gamma
- The semi-multi-modal hypothesis (low-dose diversity is expected)
- Strategic direction: DVH + PTV Gamma > global Gamma (clinically motivated)

**Invalid (must re-establish on v2.3 data):**
- All absolute metrics (MAE, Gamma %, D95 gap) — corrupted by PTV boundary artifact
- Relative rankings between methods — artifact may affect architectures differently
- DDPM conclusion (#27) — downgraded to PROVISIONAL (n=23, corrupted data)
- The "D95 gap" numbers — dominated by the preprocessing artifact, not model error

### Phase 2 Utilities (added 2026-02-17)

- `scripts/uncertainty_loss.py` — UncertaintyWeightedLoss module (Kendall et al. 2018). Ready to import; replaces manual loss weight tuning.
- `scripts/calibrate_loss_normalization.py` — Loss calibration script. Has stub loss functions — see GitHub issue #11 for replacing with real implementations.

---

## ROADMAP OVERVIEW

Detailed tasks for each phase are tracked as **GitHub Issues** with phase labels and milestones. This section provides the strategic overview only.

### Phase 0: Work Machine Setup — Milestone: `Phase 0: Setup`

**Goal:** Dataset collected, processed, locked, and ready for experiments.

Key items (see GitHub Issues with `phase/0-setup` label):
- ~~WSL environment setup~~ (#1 DONE)
- ~~Fix D95 pipeline artifact~~ (#4 DONE)
- Collect and anonymize ~220 DICOM-RT cases (#2 — in progress)
- Preprocess all cases with v2.3 pipeline (#3 — in progress, 74/76 done)
- Define inclusion/exclusion criteria (#39)
- Lock dataset and version NPZ files (#40)
- Define locked stratified test set (#38)
- Document IRB approval (#42)

### Phase 1: Clinical Evaluation Framework — Milestone: `Phase 1: Evaluation Framework`

**Goal:** Define what "good" means, establish baseline, validate pipeline end-to-end.

Key items (see GitHub Issues with `phase/1-eval` label):
- **Literature review** (#41) — establish benchmarks from published prostate dose prediction papers
- **Run baseline U-Net on v2.3 data** (#37) — validates pipeline, establishes new baseline
- Implement enhanced data augmentation (#44)
- Per-structure DVH compliance evaluation (#6)
- PTV-region Gamma at multiple thresholds (#7)
- Dose gradient/falloff analysis (#8)
- Single-case clinical acceptability report (#9)
- Physician preference ranking study (#10 — plan early, execute after Phase 2)

### Phase 2: Ablation Study — Milestone: `Phase 2: Combined Loss`

**Goal:** Systematic ablation study of loss-function engineering. This IS the paper.

**See Pre-Registered Analysis Plan below for full experimental design.**

Key items (see GitHub Issues with `phase/2-combined` label):
- Loss calibration (#11) and uncertainty weighting (#12, #13)
- Ablation study design (#43) — 10 conditions × 3 seeds = 30 runs
- Run experiments (#14) — RTX 3090 or Argon HPC

### Phase 3: Iterate & Publish — Milestone: `Phase 3: Iterate & Publish`

**Goal:** Result-driven iteration, failure analysis, and manuscript submission.

Key items (see GitHub Issues with `phase/3-iterate` label):
- Analyze Phase 2 results → determine next direction (#15)
- Failure case report (bottom 10%) (#16)
- Cross-institutional validation analysis
- Code release preparation (#17)
- Medical Physics manuscript (#18)

### Parking Lot — GitHub Issues with `type/backburner` label

Ideas to revisit only if the above plateaus. See GitHub for full list.

---

## PRE-REGISTERED ANALYSIS PLAN

**Commit this plan before seeing any v2.3 experiment results.** Changes after results are seen must be documented as deviations with justification.

### Study Design

**Title:** "Loss-function engineering for clinically acceptable prostate VMAT dose prediction"
**Target journal:** Medical Physics

### Dataset

| Source | Cases | Plan Type | Planners |
|--------|-------|-----------|----------|
| Institution A | ~11 SIB (of ~70 total) | SIB (70/56 Gy) | Multiple |
| Institution B | ~150 | SIB (70/56 Gy) | Single |
| **Total** | **~161 SIB** | | |

**Inclusion criteria (#39):**
- Prostate VMAT with SIB protocol (PTV70 + PTV56)
- All critical structures contoured (PTV70, PTV56, Rectum, Bladder minimum)
- PTV70 D95 ≥ 64 Gy in ground truth
- Complete DICOM-RT (CT + RTSTRUCT + RTDOSE + RTPLAN)

**Exclusion:** Non-SIB plans, missing PTV70/PTV56, corrupt contours

### Data Split (#38)

**Locked stratified split** — defined once before any training, frozen for all experiments.

- **~80% train** (~129 cases), **~10% val** (~16), **~10% test** (~16)
- Stratified by: institution (both represented in test) and PTV70 volume tertile
- Saved to `data/splits/` and committed to git
- Each seed uses the SAME test/val sets; only training order and augmentation differ

### Experimental Conditions (#43)

| # | Condition | Loss Components |
|---|-----------|----------------|
| 1 | Baseline (MSE) | MSE only |
| 2 | +Gradient | MSE + Gradient |
| 3 | +DVH | MSE + DVH-aware |
| 4 | +Structure | MSE + Structure-weighted |
| 5 | +AsymPTV | MSE + Asymmetric PTV |
| 6 | **Full combined** | **All 5 + uncertainty weighting** |
| 7 | Full − Gradient | Ablation |
| 8 | Full − DVH | Ablation |
| 9 | Full − Structure | Ablation |
| 10 | Full − AsymPTV | Ablation |

**Total: 10 conditions × 3 seeds = 30 training runs**

### Metrics (pre-registered)

**Primary outcomes** (what determines if a method is "better"):

| Metric | Definition | Clinically meaningful threshold |
|--------|-----------|-------------------------------|
| PTV70 D95 error | \|predicted D95 − GT D95\| | < 2 Gy |
| PTV56 D95 error | \|predicted D95 − GT D95\| | < 2 Gy |
| OAR DVH compliance | % of OAR constraints met | > 90% |

**Secondary outcomes** (reported, support primary):

| Metric | Definition |
|--------|-----------|
| PTV-region Gamma 3%/3mm | Gamma pass rate within PTV70 + 5mm margin |
| MAE (Gy) | Mean absolute error across all voxels |
| Per-structure Dmean error | \|predicted Dmean − GT Dmean\| per OAR |

**Diagnostic** (tracked, not used for decisions):

| Metric | Definition |
|--------|-----------|
| Global Gamma 3%/3mm | Full-volume gamma pass rate |
| Global Gamma 2%/2mm | For literature comparison |

### Statistical Analysis

> **Amendment (2026-02-23, pre-results):** Corrected seed-case independence violation and upgraded multiple comparison method. Rationale: seed×case pairs are NOT independent (same case across seeds shares anatomy); pooling them inflates sample size. Holm-Bonferroni is uniformly more powerful than Bonferroni with the same FWER guarantee. These are methodological corrections made before any v2.3 results were observed.

1. **Per-condition summary:** mean ± std across 3 seeds (averaged per-case first), with 95% bootstrap CI (case-level resampling, n=~16)
2. **Pairwise comparison vs baseline:** Wilcoxon signed-rank test on per-case means (average across seeds per case first → n=~16 independent paired observations per comparison). NOT pooled seed×case pairs.
3. **Multiple comparison correction:** Holm-Bonferroni step-down (9 comparisons vs baseline). More powerful than Bonferroni, same FWER guarantee.
4. **Effect size:** Cohen's d (paired) for each comparison
5. **Ablation analysis:** Full combined vs each remove-one variant (4 paired tests, Holm-Bonferroni-corrected)
6. **Power note:** With n=~16 and Holm-Bonferroni, the study has limited power for small effects. If effect size < 2× std, consider 5-seed runs (seeds 42, 123, 456, 789, 1024).

### Cross-Institutional Validation

Secondary analysis (reported in paper, not primary outcome):
- Train on Institution B only → test on Institution A SIB cases
- Train on Institution A + B → test (standard)
- Report performance gap: same-institution vs cross-institution

### Pre-Specified Decision Rules

These decisions are made NOW, before seeing results:

| If... | Then... |
|-------|---------|
| Full combined beats baseline on ALL primary metrics (p < 0.005) | Report as primary finding |
| Full combined beats baseline on SOME but not all primary metrics | Report both improvements and regressions transparently |
| No condition beats baseline significantly | Report as negative result — still publishable as "loss engineering does not help beyond MSE for this dataset" |
| One individual loss component (e.g., +Gradient alone) matches or beats full combined | Report as finding — simpler is better |
| Cross-institutional performance drops > 20% relative | Report as limitation, discuss generalizability |

### Deviations from Plan

Any change to the analysis plan after seeing v2.3 results MUST be:
1. Documented as a deviation in the experiment notebook
2. Justified with a non-results-based rationale
3. Reported transparently in the paper (e.g., "we additionally performed X, which was not pre-specified")

---

## DECISIONS LOG (Summary)

Key decisions with rationale. Full decision records are closed GitHub Issues with the `type/decision` label. Use `gh issue list --state closed --label type/decision` to find them.

| Date | Decision | GitHub Issue |
|------|----------|-------------|
| 2026-02-23 | SIB-only dataset: exclude single-Rx cases, train on ~161 SIB plans from 2 institutions | #39 |
| 2026-02-23 | Pre-registered analysis plan: 10 ablation conditions × 3 seeds, locked stratified split | See Analysis Plan |
| 2026-02-17 | Paper framing: "Loss-function engineering for clinically acceptable prostate VMAT dose prediction" | #24 (closed) |
| 2026-02-17 | Add physician preference ranking to Phase 1 eval framework | #10 |
| 2026-02-17 | Publication target: Medical Physics, single comprehensive paper | #28 (closed) |
| 2026-02-17 | Use Uncertainty Weighting for combined loss, NOT grid search | #25 (closed) |
| 2026-02-17 | Pilot 28-31% global Gamma is expected at n=23, not a failure | See Strategic Direction |
| 2026-02-17 | Add OAR contour perturbation to Phase 3 augmentation | #23 |
| 2026-02-17 | GT D95 = 55 Gy is a pipeline artifact, NOT a clinical finding | #4 |
| 2026-02-13 | Start clean on work machine with 100+ cases | See Current State |
| 2026-02-13 | Shift primary metric from global Gamma to DVH + PTV Gamma + gradient realism | #26 (closed) |
| 2026-01-21 | Dose prediction is semi-multi-modal (low-dose regions flexible) | See Strategic Direction |
| 2026-01-21 | VGG perceptual loss not useful (improves MAE, NOT Gamma, 5x slower) | See What NOT to Pursue |
| 2026-01-20 | DDPM deprioritized — PROVISIONAL (based on corrupted pilot data, revisit if Phase 2 plateaus) | #27 (reopened) |

---

## AI REVIEW WORKFLOW

External AI assistants (Grok, Claude, etc.) review the public repo and provide feedback via GitHub Discussions and Issues.

- **Discussions** (https://github.com/wrockey/vmat-diffusion/discussions) — open-ended feedback, architecture reviews, experiment ideas
- **Issues with `source/ai-review` label** — actionable suggestions promoted from discussions
- **Guidelines post:** Discussion #29 — format and category guide for AI reviewers
- **Project board is private** — AI reviewers see issues/discussions/code but not the kanban board

**Triage workflow when new AI reviews arrive:**
1. Read the discussion/issues
2. Close duplicates of existing issues
3. Relabel speculative suggestions as `type/backburner`
4. Promote genuinely actionable items to Issues with proper phase/type labels and add to project board
5. Log the review in the Session Log below

---

## SESSION LOG

Reverse chronological. **One line per session** — just enough to orient the next session. Details live in git commits and GitHub issue comments.

Format: `YYYY-MM-DD — <summary>. Commits: <hashes>. Issues: <numbers>.`

- **2026-02-23** — Centralized evaluation framework: 4 new modules (`eval_core`, `eval_clinical`, `eval_metrics`, `eval_statistics`), migrated 7 scripts, fixed D95/gamma/Bowel-V45/Dmax bugs, corrected stats methodology (per-case Wilcoxon, Holm-Bonferroni). Commit: `722551b`. Issues: #6 #7 closed, #9 #37 updated. Pre-registered analysis plan amended (stats corrections).
- **2026-02-23** — Project foundation overhaul: pre-registered analysis plan, 2-institution study design (~161 SIB cases), augmentation (rotation+noise), experiment protocol rewrite (multi-seed, standard figures, stats). Flagged all pilot metrics as invalid. Commits: `1f64172`..`75dde9a`. Issues: #27 reopened (DDPM provisional), #37-47 created, #42 #44 closed.
- **2026-02-23** — Processed 74/76 cases, expanded OAR mapping, board cleanup. Commits: `fa81a3a`..`1f64172`. Issues: #4 closed+verified, #3 updated, #30 dup, #35 #36 created.
- **2026-02-23** — GitHub project board setup, AI review workflow, triaged Grok review. Issues: #29-34.

**Board views still needed (manual, web UI):**
- "By Phase" table view grouped by Phase field
- "Active Work" board view filtered to Phase 0-3
- "Backburner" table view filtered to Backburner phase

---

## PLATFORM REFERENCE

### Work Machine (Active)

| Setting | Value |
|---------|-------|
| Platform | WSL2 Ubuntu 24.04 LTS |
| Project | `/home/wrockey/projects/vmat-diffusion` |
| Data (anonymized DICOM) | `~/data/anonymized_dicom/` (Linux filesystem, NOT /mnt/c/) |
| Data (processed NPZ) | `~/data/processed_npz/` (created by preprocessing script) |
| Conda | Miniforge3 (`~/miniforge3`), env: `vmat-diffusion` |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu126 |
| PyTorch Lightning | 2.6.1 |
| CUDA | 12.6 (driver 560.94) |
| GPU | NVIDIA RTX 3090 (24 GB) |

### Argon HPC (Available for Phase 2 scaling)

| Setting | Value |
|---------|-------|
| Cluster | Argon HPC |
| Use case | Phase 2 parallel training if >2 weeks on local RTX 3090 (#46) |
| Status | Available, environment not yet set up |

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

*Last updated: 2026-02-23 (Centralized evaluation framework — 4 modules, 7 migrations, critical bug fixes, stats methodology corrected, pre-registered plan amended.)*
