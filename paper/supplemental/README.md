# Supplemental Material

## Planned Sections

| ID | Title | Content | Status |
|----|-------|---------|--------|
| S1 | DDPM Comparison | DVH figure + metrics table showing DDPM failure | Data ready |
| S2 | Architecture Ablation | Table comparing Attention, BottleneckAttn, Wider variants | Data ready |
| S3 | Augmentation Ablation | With/without augmentation comparison | Data ready |
| S4 | Loss Tuning Details | 3:1 → 2:1 → 2.5:1 progression table + figures | Data ready |
| S5 | Per-Case Results | Full per-case metrics for all 7 test cases × 3 seeds | Data ready |
| S6 | Anatomical Variability | Bowel involvement driving outlier cases | Notebook exists |
| S7 | Training Configuration | Full hyperparameter table, environment details | Auto-captured |

## Data Sources

All supplemental data is derived from experiment results in:
- `predictions/*/evaluation_results.json`
- `runs/*/training_config.json`
- `notebooks/EXPERIMENTS_INDEX.md`
