# Experiment Organization Structure

**This document defines the standard structure for all experiments.**

---

## Directory Structure

```
vmat-diffusion-project/
├── notebooks/
│   ├── EXPERIMENTS_INDEX.md          # MASTER LIST - all experiments with git hashes
│   ├── TEMPLATE_experiment.ipynb     # Copy this for new experiments
│   ├── 2026-01-19_baseline_unet_experiment.ipynb
│   ├── 2026-01-19_baseline_unet_test_evaluation.ipynb
│   ├── 2026-01-20_ddpm_v1_experiment.ipynb           # TBD - needs creation
│   └── ...future experiment notebooks...
│
├── runs/                              # Training outputs (checkpoints, metrics)
│   ├── baseline_unet_run1/
│   │   ├── checkpoints/
│   │   ├── training_config.json
│   │   ├── epoch_metrics.csv
│   │   └── training_summary.json
│   ├── vmat_dose_ddpm/               # DDPM v1 run
│   │   └── ...
│   └── ...future runs...
│
├── experiments/                       # Analysis outputs (from optimization experiments)
│   ├── phase1_sampling/              # Quick inference experiments
│   ├── phase1_ensemble/
│   ├── phase2_schedules/
│   └── phase2_loss/
│
├── predictions/                       # Inference outputs for evaluation
│   ├── baseline_unet_test/
│   └── ...
│
└── docs/
    ├── DDPM_OPTIMIZATION_PLAN.md     # Current optimization focus
    └── EXPERIMENT_STRUCTURE.md       # This file
```

---

## Single Source of Truth

### `notebooks/EXPERIMENTS_INDEX.md`
**This is the MASTER experiment list.** Every experiment MUST be logged here with:
- Date
- Experiment ID
- Notebook link
- Model
- Git commit hash
- Best metric
- Status

### Relationship to Other Files

| File | Purpose | Updates |
|------|---------|---------|
| `notebooks/EXPERIMENTS_INDEX.md` | Master list of ALL experiments | After every experiment |
| `docs/DDPM_OPTIMIZATION_PLAN.md` | Current optimization focus & hypotheses | During optimization work |
| `runs/<experiment>/training_summary.json` | Raw training outputs | Automatic during training |
| `notebooks/<date>_<name>.ipynb` | Detailed analysis & figures | After each experiment |

---

## Required for Each Experiment

### 1. Before Running
- [ ] Clean git working directory
- [ ] Commit: `git commit -m "Pre-experiment: <name>"`
- [ ] Record commit hash

### 2. During Running
- [ ] Training outputs to `runs/<experiment_name>/`
- [ ] Use consistent hyperparameter logging

### 3. After Running
- [ ] Create notebook: `notebooks/YYYY-MM-DD_<experiment>_experiment.ipynb`
- [ ] Update `notebooks/EXPERIMENTS_INDEX.md` with:
  - Git hash
  - Best metric
  - Status
  - Link to notebook
- [ ] Commit: `git commit -m "Results: <name> - <key finding>"`

---

## Notebook Template Sections

Every experiment notebook should include (see TEMPLATE_experiment.ipynb):

1. **Experiment Overview** - Goal, hypothesis
2. **Reproducibility Info** - Git hash, seed, environment
3. **Data** - Train/val/test split
4. **Model & Config** - Architecture, hyperparameters
5. **Training** - Curves, metrics
6. **Evaluation** - Test set results
7. **Analysis** - What did we learn?
8. **Conclusions** - Next steps
9. **Artifacts** - Links to checkpoints, figures

---

## Git Commit Hash Recording

**Critical for reproducibility.** Record in both:
1. `notebooks/EXPERIMENTS_INDEX.md` - in the table
2. `notebooks/<experiment>.ipynb` - Section 2 (Reproducibility Info)

Format: `git: <short-hash>` (e.g., `git: 3efbea0`)

---

## Naming Conventions

### Experiments
```
<model>_<variant>_<version>
```
Examples: `baseline_unet_run1`, `ddpm_dose_v1`, `ddpm_sampling_1000steps`

### Notebooks
```
YYYY-MM-DD_<experiment_id>_experiment.ipynb
```

### Run Directories
```
runs/<experiment_id>/
```

---

## Outstanding Items to Fix

1. [ ] Create notebook for ddpm_dose_v1 (currently TBD in index)
2. [ ] Ensure all future experiments follow this structure

---

*Created: 2026-01-20*
