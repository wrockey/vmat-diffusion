# VMAT Diffusion — Project Plan & State

**This file provides strategic orientation for every session.**
**GitHub (issues, board, milestones) is the source of truth for project status and task tracking.**
**Keep this file under 300 lines. If you're writing task details, they belong in a GitHub issue.**

## Documentation Hierarchy

| Document | Role |
|----------|------|
| **This file** | Strategy, directives, pre-registered plan, platform reference |
| `CLAUDE.md` | Static reference: code conventions, architecture, experiment protocol |
| `notebooks/EXPERIMENTS_INDEX.md` | Master experiment log |
| **GitHub Issues** | All tasks, bugs, decisions, experiment plans |
| **GitHub Project Board** | Kanban status tracker (project #2, `gh project item-list 2 --owner wrockey --limit 100`) |
| **GitHub Milestones** | Phase-level progress (Phase 0–3) |
| **GitHub Discussions** | AI review feedback channel |

### Rules

- **GitHub is the source of truth for progress.** Do NOT duplicate issue lists or status in this file.
- **When you update this file, also update GitHub** (issues, board) if the change affects task state. See memory file for the sync reminder.
- **Do not create separate plan files.** One archived sub-plan exists: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED).

---

## SESSION PROTOCOLS

### Start-of-Session Checklist (DO THIS FIRST, EVERY SESSION)

Run silently (do not narrate unless something is wrong):

1. **Check GitHub Issues** — `gh issue list --state open --limit 50`
2. **Check project board** — `gh project item-list 2 --owner wrockey --limit 100`
3. **Check git log** — `git log --oneline -10`
4. **Check discussions** — `gh api graphql -f query='{ repository(owner:"wrockey",name:"vmat-diffusion") { discussions(first:5,orderBy:{field:CREATED_AT,direction:DESC}) { nodes { number title createdAt } } } }'`
5. **Orient** — Identify: (a) current phase, (b) blockers, (c) next action.

Fix discrepancies (stale board, untriaged reviews) before starting work.

### Board Update Reference

```bash
# Project ID: PVT_kwHOAkj6uc4BP7oy
# Status field: PVTSSF_lAHOAkj6uc4BP7oyzg-MoCc  (Todo=f75ad846, In Progress=47fc9ee4, Done=98236657)
# Phase field: PVTSSF_lAHOAkj6uc4BP7oyzg-Mtf0  (P0=e73b984f, P1=9dcbf090, P2=eacee063, P3=fddd52ce, Backburner=c4c2f48d, Decision=7f44e30c)
gh project item-edit --project-id PVT_kwHOAkj6uc4BP7oy --id "<ITEM_ID>" --field-id "<FIELD_ID>" --single-select-option-id "<OPTION>"
```

Update the board **immediately** when issue status changes — not at end of session.

### End-of-Session Checklist

1. **GitHub up to date:** closed issues → "Done" on board, progress comments on open issues, new issues created and added to board
2. **Commit any code changes**
3. **Triage new AI reviews** if any arrived

---

## PRIME DIRECTIVES

0. **PROTECT HEALTH INFORMATION — PARAMOUNT.** No PHI in git, GitHub, or any tracked file. No patient names, MRNs, DOBs, credentials. Before any `git add`, verify: `git diff --cached --name-only`. If PHI found, **stop all work immediately**.

1. **Every experiment follows the full protocol in CLAUDE.md** — automatically, no reminders. Multi-seed (42, 123, 456), publication-ready notebook, figure script, EXPERIMENTS_INDEX.md updated. An undocumented experiment never happened.

2. **Figures are publication-ready from the start.** Serif font, 12pt+, 300 DPI, colorblind-friendly (Wong 2011), labeled axes, captions.

3. **Tasks are tracked in GitHub Issues.** Check before starting, close with commit refs when done, create issues for new tasks.

4. **GitHub state stays in sync.** Board, milestones, issue statuses reflect reality.

---

## STRATEGIC DIRECTION

### Primary Goal: Clinical Acceptability (NOT Global Gamma)

| Priority | Metric | Target | Rationale |
|----------|--------|--------|-----------|
| CRITICAL | PTV70 D95 | >= 66.5 Gy | Prostate coverage |
| CRITICAL | PTV56 D95 | >= 53.2 Gy | Seminal vesicle coverage |
| CRITICAL | OAR DVH compliance | Per QUANTEC | Organ sparing |
| HIGH | PTV-region Gamma 3%/3mm | > 95% | Accuracy where it matters |
| HIGH | Dose gradient realism | Monotonic falloff, ~6mm penumbra | Deliverability proxy |
| DIAGNOSTIC | Overall Gamma 3%/3mm | Track only | Low due to valid low-dose diversity |
| DIAGNOSTIC | MAE (Gy) | Track only | Convergence metric, not clinical |

### Loss Design Principle

```
Loss = w_ptv * L_ptv_asymmetric + w_oar * L_oar_dvh + w_gradient * L_gradient + w_bg * L_background
```

All components implemented. Phase 2 combines them with uncertainty weighting (Kendall 2018).

### What NOT to Pursue Currently

- Global Gamma as optimization target (track, don't chase)
- DDPM tuning — deprioritized, revisit if Phase 2 plateaus (#27)
- VGG perceptual loss (no Gamma improvement, 5x overhead)
- Pure MSE/MAE optimization (causes PTV underdosing)

---

## CURRENT STATE

**Phase 0 (Setup) — data collection in progress.** See board for full status.

- ~161 SIB cases expected (Institution A: 11 SIB of 74 processed; Institution B: ~150 pending)
- **Blockers:** #38 (locked test set), #39 (inclusion criteria), #40 (data lock) — all need data collection
- IRB approved (#42). Compute: RTX 3090 local, Argon HPC available for Phase 2.

---

## PRE-REGISTERED ANALYSIS PLAN

**Committed before seeing v2.3 results. Changes after results require documented deviations.**

### Study Design

**Title:** "Loss-function engineering for clinically acceptable prostate VMAT dose prediction"
**Target journal:** Medical Physics
**Dataset:** ~161 SIB cases from 2 institutions. Inclusion: SIB protocol, PTV70+PTV56 contoured, D95 >= 64 Gy, complete DICOM-RT (#39).
**Split (#38):** ~80/10/10 train/val/test, stratified by institution + PTV70 volume tertile, locked before training.

### Experimental Conditions (#43)

**Loss ablation (C1-C10):**

| # | Condition | Loss Components |
|---|-----------|----------------|
| 1 | Baseline (MSE) | MSE only |
| 2 | +Gradient | MSE + Gradient |
| 3 | +DVH | MSE + DVH-aware |
| 4 | +Structure | MSE + Structure-weighted |
| 5 | +AsymPTV | MSE + Asymmetric PTV |
| 6 | **Full combined** | **All 5 + uncertainty weighting** |
| 7 | Full - Gradient | Ablation |
| 8 | Full - DVH | Ablation |
| 9 | Full - Structure | Ablation |
| 10 | Full - AsymPTV | Ablation |

**Architecture comparison (C11-C16, Amendment 1, 2026-02-24 pre-results):**

> Rationale: Architecture is a confound. Added before any v2.3 results observed.

| # | Condition | Architecture | Loss |
|---|-----------|-------------|------|
| 11 | AttentionUNet + MSE | AttentionUNet3D | MSE |
| 12 | AttentionUNet + Full | AttentionUNet3D | Full + UW |
| 13 | BottleneckAttn + MSE | BottleneckAttnUNet3D | MSE |
| 14 | BottleneckAttn + Full | BottleneckAttnUNet3D | Full + UW |
| 15 | WiderBaseline + MSE | BaselineUNet3D (bc=50) | MSE |
| 16 | WiderBaseline + Full | BaselineUNet3D (bc=50) | Full + UW |

**Total: 16 conditions x 3 seeds = 48 runs (~624 GPU-hours). Issues: #14, #53.**

### Metrics

**Primary:** PTV70 D95 error (<2 Gy), PTV56 D95 error (<2 Gy), OAR DVH compliance (>90%).
**Secondary:** PTV-region Gamma 3%/3mm, MAE (Gy), per-structure Dmean error.
**Diagnostic:** Global Gamma 3%/3mm, Global Gamma 2%/2mm.

### Statistical Analysis

> Amendment (2026-02-23, pre-results): Corrected seed-case independence violation, upgraded to Holm-Bonferroni.

1. **Per-condition:** mean +/- std across 3 seeds (per-case averages first), 95% bootstrap CI
2. **Family 1 (loss):** Wilcoxon signed-rank on per-case means (n=~16) vs baseline. 9 tests, Holm-Bonferroni.
3. **Ablation:** Full vs each remove-one (4 tests, Holm-Bonferroni).
4. **Family 2 (architecture):** 6 vs-baseline + 4 attention-vs-control = 10 tests, Holm-Bonferroni (separate family).
5. **Effect size:** Cohen's d (paired). If < 2x std, expand to 5 seeds.
6. **Exploratory:** 2-way interaction (architecture x loss type).

### Decision Rules (pre-specified)

| If... | Then... |
|-------|---------|
| Full combined beats baseline on ALL primary (p<0.005) | Primary finding |
| Beats on SOME but not all | Report both transparently |
| Nothing beats baseline | Negative result — still publishable |
| Single component matches full combined | Simpler is better |
| Cross-institutional drop >20% | Limitation |
| Attention beats baseline AND wider control | Attention mechanism helps |
| Attention beats baseline but NOT wider control | Capacity, not attention |
| No architecture variant beats baseline | Architecture not the bottleneck |

### Cross-Institutional Validation

Secondary: Train on B only -> test on A; train on A+B -> test (standard). Report gap.

---

## AI REVIEW WORKFLOW

- **Discussions:** https://github.com/wrockey/vmat-diffusion/discussions (guidelines: #29)
- **Triage:** Close duplicates, relabel speculative as `type/backburner`, promote actionable to Issues with proper labels, add to board.

---

## PLATFORM REFERENCE

### Work Machine

| Setting | Value |
|---------|-------|
| Platform | WSL2 Ubuntu 24.04 LTS |
| Project | `/home/wrockey/projects/vmat-diffusion` |
| Data (DICOM) | `~/data/anonymized_dicom/` |
| Data (NPZ) | `~/data/processed_npz/` |
| Conda | `vmat-diffusion` env, Python 3.12, PyTorch 2.10+cu126, PL 2.6.1 |
| GPU | RTX 3090 (24 GB), CUDA 12.6 |
| DataLoader | `num_workers=2`, `persistent_workers=False` (WSL deadlock avoidance) |

### Argon HPC

Available for Phase 2 scaling. Environment not yet set up (#46).

---

*Last updated: 2026-02-24*
