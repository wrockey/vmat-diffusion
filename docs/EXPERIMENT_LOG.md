# Experiment Progress Log

**Auto-updated during experiments. Check this file for quick status.**

---

## Latest Status

**Current Phase:** Not started
**Current Experiment:** None
**Last Updated:** 2026-01-20 (initial creation)

---

## Completed Experiments

| Timestamp | Experiment | Result | MAE (Gy) | Key Finding |
|-----------|------------|--------|----------|-------------|
| 2026-01-20 | ddpm_v1 (baseline) | Complete | 12.19 | Loss/MAE disconnect |

---

## Running Notes

_Append findings here as experiments complete_

### [Template Entry]
**Experiment:** [name]
**Time:** [timestamp]
**Hypothesis:** [H1/H2/etc]
**Result:** [brief outcome]
**MAE:** [value] Gy
**Conclusion:** [what we learned]
**Next:** [what this suggests we try next]

---

## Quick Findings Summary

_Key takeaways so far:_

1. Baseline U-Net achieves 3.73 Gy MAE (val), 1.43 Gy MAE (test)
2. DDPM v1 achieved 12.19 Gy MAE - 3x worse than baseline
3. DDPM shows loss/MAE disconnect: loss decreases but MAE volatile
4. TBD - more findings as experiments complete

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | Start with Phase 1 | No retraining needed, quick feedback |

---

*This file is updated automatically during experiments. Last human review: 2026-01-20*
