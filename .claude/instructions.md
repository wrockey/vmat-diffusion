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
| `paper/` | Manuscript preparation: outline, figures inventory, notes, references |
| **GitHub Issues** | All tasks, bugs, decisions, experiment plans |
| **GitHub Project Board** | Kanban status tracker (project #2, `gh project item-list 2 --owner wrockey --limit 100`) |
| **GitHub Milestones** | Phase-level progress (Phase 0–3) |
| **GitHub Discussions** | AI review feedback channel |

### Rules

- **GitHub is the single source of truth for project status.** Do NOT duplicate results, task lists, or status in this file or MEMORY.md. If it belongs on GitHub, put it there — not here.
- **This file stores only:** strategy, directives, pre-registered analysis plan, platform config. It should rarely change.
- **MEMORY.md stores only:** session workflow rules, local environment gotchas, CLI traps. NOT status, NOT results, NOT task lists.
- **When something changes,** update the GitHub issue (or pinned #63), not these files.
- **Do not create separate plan files.** One archived sub-plan exists: `docs/DDPM_OPTIMIZATION_PLAN.md` (ARCHIVED).

---

## SESSION PROTOCOLS

### Start-of-Session Checklist (DO THIS FIRST, EVERY SESSION)

Run silently (do not narrate unless something is wrong):

1. **Check GitHub Issues** — `gh issue list --state open --limit 100`
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
| CRITICAL | PTV50.4 D95 | >= 47.88 Gy | Nodal coverage (when present) |
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
- DDPM — definitively rejected on v2.3 data (#49 closed, #27 confirmed). Not viable.
- VGG perceptual loss (no Gamma improvement, 5x overhead)
- Pure MSE/MAE optimization (causes PTV underdosing)

---

## CURRENT STATE

**Do NOT duplicate project status here. Check GitHub instead:**
- **Pinned issue #63** — current results, critical path, ablation plan, blocker
- **`gh issue list --state open`** — all open tasks
- **`gh api repos/wrockey/vmat-diffusion/milestones`** — phase-level progress

---

## PRE-REGISTERED ANALYSIS PLAN

**Committed before seeing v2.3 results. Changes after results require documented deviations.**

### Study Design

**Title:** "Multi-protocol prostate VMAT dose prediction via clinical loss engineering and constraint conditioning"
**Target journal:** Medical Physics
**Dataset:** ~200 cases from 2 institutions, 3 plan types (Amendment 4, 2026-03-07, #39, #66).

> **Amendment 4 (2026-03-07):** Reversed PTV50.4 exclusion. Adding PTV50.4 as 9th SDF input channel (#66) and fixing absent-structure SDF convention to 0.0 (#67) enables training on all ~200 cases. The dose hallucination that motivated the original exclusion was caused by (a) missing spatial input for PTV50.4 and (b) SDF=+1.0 for absent structures being indistinguishable from "far away." Both are now fixed. See #66, #67, #68 for implementation.

| Plan Type | N | Targets |
|-----------|---|---------|
| 2-level SIB | ~91 | PTV70 + PTV56 |
| 3-level SIB | ~40 | PTV70 + PTV56 + PTV50.4 |
| Nodal-only | ~62 | PTV70 + PTV50.4 |

**Split (#69):** ~80/10/10 train/val/test, stratified by plan type + institution + PTV70 volume tertile, locked before training.

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
| 11 | Full combined, FiLM disabled | FiLM ablation (#71) |

**Architecture comparison: DESCOPED (Amendment 2, 2026-03-06)**

> Single-seed scouts showed no improvement over baseline on 70 cases. Architecture is not the bottleneck. Moved to backburner (#53). Run only if reviewer requests.

**Final plan: 11 conditions x 3 seeds = 33 runs (~150 GPU-hours). Argon scripts: #62.**

### Metrics

**Primary:** PTV70 D95 error (<2 Gy), PTV56 D95 error (<2 Gy), PTV50.4 D95 error (<2 Gy, when present), OAR DVH compliance (>90%).
**Secondary:** PTV-region Gamma 3%/3mm, MAE (Gy), per-structure Dmean error.
**Diagnostic:** Global Gamma 3%/3mm, Global Gamma 2%/2mm.

### Statistical Analysis

> Amendment (2026-02-23, pre-results): Corrected seed-case independence violation, upgraded to Holm-Bonferroni.

1. **Per-condition:** mean +/- std across 3 seeds (per-case averages first), 95% bootstrap CI
2. **Family 1 (loss):** Wilcoxon signed-rank on per-case means (n=~16) vs baseline. 9 tests, Holm-Bonferroni.
3. **Ablation:** Full vs each remove-one (4 tests, Holm-Bonferroni).
4. ~~**Family 2 (architecture):** descoped, see Amendment 2 above.~~
5. **Effect size:** Cohen's d (paired). If < 2x std, expand to 5 seeds.

### Decision Rules (pre-specified)

| If... | Then... |
|-------|---------|
| Full combined beats baseline on ALL primary (p<0.005) | Primary finding |
| Beats on SOME but not all | Report both transparently |
| Nothing beats baseline | Negative result — still publishable |
| Single component matches full combined | Simpler is better |
| Cross-institutional drop >20% | Limitation |
| ~~Attention beats baseline AND wider control~~ | ~~descoped~~ |
| ~~Attention beats baseline but NOT wider control~~ | ~~descoped~~ |
| ~~No architecture variant beats baseline~~ | Confirmed on 70-case scouts — architecture not the bottleneck |

### Cross-Institutional Validation

~~Secondary: Train on B only -> test on A; train on A+B -> test (standard). Report gap.~~

**Amendment 3 (2026-03-07):** Revised to single-institution (B) with external validation (U).
**Amendment 4 (2026-03-07):** With multi-protocol expansion, Institution U contributes ~70 cases across plan types. Cross-institutional split may now be viable — reassess after QA review (#64) confirms per-institution counts by plan type. If U has sufficient cases per type, restore cross-institutional training arm.

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

Available for Phase 2 ablation. Batch scripts drafted in `scripts/argon/` (#62).
Full reference: `memory/argon_cluster.md`. SGE scheduler (not SLURM), UI-GPU queue, A100/H100 GPUs.

---

*Last updated: 2026-03-07*
