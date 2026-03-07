# Project Assessment: Publishability, Novelty, and Recommendations

**Project:** Deep Learning VMAT Dose Prediction for Prostate Cancer
**Architecture:** 3D U-Net with FiLM Conditioning on Dose Constraints
**Key Innovation:** Systematic Loss Function Ablation (5 components + uncertainty weighting)
**Dataset:** Expanding from ~91 single-protocol to ~200 multi-protocol cases
**Target Journal:** Medical Physics
**Date:** 2026-03-07

---

## 1. Overall Verdict

**Publishable as a Medical Physics paper.** The strongest framing is "multi-protocol dose prediction with systematic loss engineering." The multi-protocol expansion (N approximately 200) with FiLM conditioning combined with a pre-registered 10-condition loss ablation is a genuine contribution to the prostate dose prediction literature.

The project has three publishable elements that together form a cohesive story:

1. **Multi-protocol FiLM conditioning** -- a single model serving three plan types (PTV70+PTV56, PTV70+PTV56+PTV50.4, PTV70+PTV50.4) via learned constraint embedding. No published dose prediction work uses FiLM for protocol conditioning.
2. **Systematic loss engineering** -- 5-component loss with Kendall uncertainty weighting, ablated across 10 conditions with 3-seed replication. This exceeds the methodological rigor of any published loss engineering study in the dose prediction literature.
3. **Clinically motivated asymmetric PTV penalty** -- moves PTV70 D95 error from -1.76 Gy (systematic underdosing) to +0.06 Gy (essentially zero bias), directly encoding the clinical asymmetry between underdose and overdose risk.

The current results on 70 cases already demonstrate competitive D95 accuracy (0.06 +/- 0.26 Gy vs published 0.2-1.0 Gy) and near-clinical PTV gamma (94.3 +/- 2.2% vs 95% target). Dataset expansion to approximately 200 cases with three plan types would strengthen the contribution substantially and place the dataset size among the largest prostate-specific cohorts in the literature.

---

## 2. Novelty Assessment

| Aspect | Novelty Level | Evidence | Literature Gap |
|--------|--------------|----------|---------------|
| Multi-protocol FiLM conditioning | HIGH | No published dose prediction paper uses FiLM for protocol conditioning. Enables a single model across plan types. Norouzi Kandalan 2020 handled multi-protocol via transfer learning, not conditioning. | No prior work; FiLM ablation (C11) needed to demonstrate utility. |
| Asymmetric PTV loss | HIGH | Clinically motivated, unreported in literature. D95 gap improved from -1.76 to +0.06 Gy, directly eliminating systematic underdosing. Section 4.1 of the literature review confirms: "not reported in published literature." | Unique contribution; no comparable published mechanism. |
| 5-component loss + Kendall weighting | HIGH | Individual components (gradient, DVH, structure-weighted) exist in the literature but have never been combined with learned uncertainty weighting for dose prediction. Kendall 2018 is from multi-task learning; its application to dose prediction loss balancing is novel. | Jhanwar 2022 used moment-based DVH loss; Bai 2021 used gradient-aware loss; Zimmermann 2021 used feature loss. None combined more than 2 components with learned weighting. |
| Pre-registered ablation design | HIGH | 10 conditions pre-specified before seeing results. Statistical plan includes Holm-Bonferroni correction, pre-specified decision rules, and seed protocol. Extremely rare in medical physics ML. | Almost no published dose prediction paper pre-registers an analysis plan. This is a methodological contribution in itself. |
| SDF inputs | MODERATE | Aligns with DoseDiff 2024 (Zhang et al., IEEE TMI) emerging practice. Not novel in isolation but part of a strong input representation pipeline. | Emerging best practice, not a standalone contribution. |
| Absent-structure SDF convention (0.0) | MODERATE | Simple but important for multi-protocol training. When a structure (e.g., PTV50.4) is absent from a plan, representing it as SDF = 0.0 (on the boundary) rather than +1.0 (far outside) prevents the model from hallucinating dose to non-existent targets. Not discussed in any published multi-protocol work. | Practical insight with no literature precedent for dose prediction. |
| DDPM negative result on prostate | MODERATE | First prostate-specific DDPM evaluation. Catastrophic failure: D95 of 4-6 Gy vs 70 Gy target. Four diffusion papers exist (DiffDP 2023, DoseDiff 2024, MD-Dose 2024, DMTP 2025) but none target prostate VMAT. | Valuable negative result for the community; prevents others from pursuing this dead end for prostate dose prediction. |
| 3D U-Net architecture | LOW | Well-established architecture (Ronneberger 2015, adapted to 3D). The architecture scout experiments (C11 Attention, C13 BottleneckAttn, C15 Wider) confirmed it is not the bottleneck. | Not a contribution. Correctly positioned as a baseline enabling the loss engineering study. |

---

## 3. Strengths

1. **Systematic ablation design (C1-C10)** -- Most published papers try one loss function and report that it works. This project tests 10 conditions (5 additive, 1 combined, 4 ablation) with 3 seeds each, providing statistically grounded evidence for each component's contribution. This is rare rigor for the field.

2. **Multi-protocol dataset (N approximately 200)** -- Among the largest prostate-specific datasets in the literature. Three plan types in a single model is matched only by Norouzi Kandalan 2020 (who used transfer learning, not a unified model). Most comparable studies use 60-160 single-protocol cases.

3. **FiLM conditioning tested with real protocol variation** -- This is not just a mechanism bolted onto a standard pipeline. The multi-protocol expansion creates a genuine test case where constraint conditioning must work to avoid cross-protocol interference. If FiLM succeeds here, it demonstrates practical utility beyond architectural elegance.

4. **Pre-registered analysis plan** -- The analysis plan (in `.claude/instructions.md`) was committed before seeing v2.3 results, with amendments documented and justified. Decision rules are pre-specified (e.g., "if full combined beats baseline on ALL primary metrics at p<0.005, primary finding"). This is unusual rigor for medical physics ML and will be recognized by reviewers.

5. **Clinical metric focus** -- The evaluation framework prioritizes D95, DVH compliance, and PTV-region gamma over surrogate metrics like body-wide MAE. This aligns with clinical decision-making and makes results directly interpretable by clinicians.

6. **Honest negative results** -- The DDPM failure (D95 of 4-6 Gy vs 70 Gy target), global gamma of approximately 30%, and architecture scout null results are documented transparently. This builds credibility and provides useful information to the community.

7. **Asymmetric PTV loss encodes clinical knowledge** -- The insight that symmetric losses produce symmetric errors while clinical consequences are asymmetric (underdosing tumor is categorically worse than mild overdose) is simple, compelling, and demonstrably effective. The D95 improvement from -1.76 to +0.06 Gy is the project's most striking single result.

---

## 4. Weaknesses and Reviewer Concerns

### 4.1 Critical (must address before submission)

**W1: Per-structure MAE not yet in main results** (GitHub #61 -- CLOSED, implemented; report on new runs)

Body-wide MAE of 4.07 Gy is not comparable to published per-structure MAE values of 0.9-2.1 Gy. Reviewers will immediately flag this as an unfair comparison. The evaluation infrastructure to compute per-structure MAE exists (implemented in #61), but it has not been run on the current combined loss results or reported in any experiment notebook.

- **Action:** Report MAE within PTV70, PTV56, PTV50.4, Rectum, Bladder, Femur L, Femur R, and Bowel for all ablation conditions. This is likely to show competitive numbers (structure-specific MAE should be much lower than body-wide) and is essential for the comparison table in the Discussion.

**W2: Soft DVH loss not validated against exact DVH computation** (GitHub #70)

The DVH-aware loss uses a differentiable soft histogram approximation (`soft_d95_histogram` in `train_baseline_unet.py`, lines 569-628) to compute D95 during training. This approximation uses Gaussian kernel soft assignments with a temperature parameter. However, this soft D95 has never been compared to the exact D95 computed during evaluation.

If the soft D95 does not correlate well with exact D95, the DVH loss component is not optimizing what we claim it optimizes, and any ablation result involving DVH loss (C3, C6, C8) is undermined.

- **Action:** Scatter plot of soft D95 (training) vs exact D95 (evaluation) across all training cases and structures. Report Pearson/Spearman correlation. If correlation is below 0.9, investigate alternative differentiable DVH formulations before running the full ablation.

**W3: No aggregate clinical acceptability metric** (GitHub #72)

Hou et al. 2025 reports 62.6% of predictions as "clinically acceptable" across 7 tumor types. Reviewers will expect a comparable metric. Currently, per-case QUANTEC compliance is tracked but no aggregate "X% of plans are clinically acceptable" figure is reported.

- **Action:** Define acceptability criteria (all QUANTEC constraints met, PTV D95 within tolerance, gamma > threshold). Compute per-case pass/fail. Report aggregate acceptability percentage with 95% CI. Target: above 75% for the combined loss condition.

**W4: Early stopping monitors val MAE, not clinical metrics**

Model selection uses validation MAE for early stopping (patience=50). However, D95 and gamma may still be improving when MAE plateaus, or vice versa. This means the "best" checkpoint by MAE may not be the best checkpoint by the primary clinical metrics.

- **Action (low implementation priority but must be documented):** Acknowledge in Methods as a limitation. Note that monitoring a composite clinical metric (e.g., weighted combination of D95 error and gamma pass rate) could improve model selection but was not implemented in this study. Report the correlation between val MAE and primary clinical metrics to quantify the impact.

### 4.2 Important (strengthens paper significantly)

**W5: FiLM conditioning never ablated** (GitHub #71)

FiLM conditioning is listed as a novel contribution and is central to the multi-protocol narrative, but its effect has never been directly tested. On the current single-protocol data (70 cases, all PTV70+PTV56), FiLM may or may not be contributing because the constraint vector has limited variation.

The multi-protocol expansion creates the perfect ablation opportunity: train two models with identical loss (C6 full combined), one with FiLM enabled and one with constraints zeroed out or removed. If FiLM helps on multi-protocol data, it validates the mechanism. If it does not help, the paper must be reframed.

- **Action:** Add C11 condition (full combined loss, FiLM disabled) to the ablation plan. This is a single 3-seed experiment that directly tests the paper's headline contribution.

**W6: Gamma subsample consistency and approximation error** (GitHub #73)

Training validation and inference both use `gamma_subsample=4` for speed. The approximation error introduced by subsampling has never been quantified against a full-resolution (subsample=1) computation. If the approximation inflates or deflates gamma by several percentage points, all reported gamma values carry an unknown systematic bias.

- **Action:** Run subsample=1 gamma on a small subset (3-5 cases) and compare to subsample=4. Report the mean absolute difference. If the difference is above 1 percentage point, consider reporting full-resolution gamma or documenting the bias.

**W7: Missing 2%/2mm gamma and D98/D2 metrics** (GitHub #73)

Many Medical Physics papers report both 3%/3mm and 2%/2mm gamma criteria. Additionally, D98 (near-minimum dose) and D2 (near-maximum dose) complement D95 for PTV assessment and are standard in DVH reporting (ICRU 83).

- **Action:** Add 2%/2mm gamma computation and D98/D2 reporting to the evaluation pipeline. These are straightforward additions to the existing infrastructure.

**W8: Failure case characterization** (GitHub #16 -- open)

Bowel involvement drives outlier cases (cases 0027, 0079 identified in the anatomical variability analysis). Systematic characterization of the bottom 10% by gamma or DVH metrics would strengthen the paper by (a) demonstrating awareness of limitations, (b) identifying anatomical predictors of failure, and (c) suggesting when the model should not be trusted.

- **Action:** Complete #16 as planned. Report: which cases fail, what anatomical features predict failure, and what the model's performance is when outliers are excluded vs included.

### 4.3 Minor (nice to have)

**W9: No physician evaluation** (GitHub #10 -- open, low priority)

Blinded clinician review would strengthen clinical claims. For Medical Physics, this is not strictly required but would be expected for IJROBP or a clinical acceptability claim. Church et al. 2024 included plan quality review; Hou et al. 2025 included clinical evaluation.

- **Impact:** Low for Medical Physics submission, high for IJROBP. Consider for revision or follow-up study.

**W10: VGG perceptual loss implementation flaw**

The VGG perceptual loss (`VGGPerceptualLoss2D`, line 391 of `train_baseline_unet.py`) repeats single-channel dose slices to 3-channel RGB input for a pretrained ImageNet VGG. This breaks the feature representations that VGG learned, as it was trained on natural color images. The loss was tested and rejected (no gamma improvement, 5x overhead), which is correct, but if cited as "tried and failed" in the paper, a caveat is needed explaining that the implementation was not well-suited to the task.

- **Action:** If mentioning VGG loss in the paper, note the grayscale-to-RGB mismatch as a limitation of the implementation, not necessarily of perceptual losses in general.

**W11: Negative dose penalty outside uncertainty weighting**

A 0.1x relu(-pred) penalty is always applied to enforce non-negative dose predictions, regardless of the uncertainty weighting state. This is not subject to Kendall weighting and could interact with the learned sigma values in unpredictable ways.

- **Action:** Document in Methods. This is a minor implementation detail but should be mentioned for reproducibility.

**W12: Learned uncertainty weights (sigma) not reported** (GitHub #74)

The Kendall uncertainty weighting learns one sigma per loss component during training. The evolution of these sigma values reveals which losses the model prioritizes and how the balance shifts during training. This is easy to extract from TensorBoard logs and provides publishable insight into loss dynamics.

- **Action:** Extract sigma trajectories from TensorBoard logs for the combined loss runs. Create a figure showing sigma evolution over training epochs. Report final sigma values in a table. This is low-effort, high-insight analysis.

---

## 5. Recommendations (Priority Order)

### 5.1 Critical Path (blocks submission)

1. **Implement PTV50.4 channel + fix SDF absent-structure convention** (GitHub #66, #67) -- Add 9th SDF channel for PTV50.4; use 0.0 (not +1.0) for absent structures. This is the foundation of the multi-protocol expansion.
2. **Reprocess all approximately 200 cases** (GitHub #68) -- New preprocessing run with 9-channel SDF output, covering all three plan types.
3. **Plan-type stratified data split** (GitHub #69) -- Ensure train/val/test split is stratified by plan type, institution, and PTV70 volume. Lock before any training.
4. **Run full ablation C1-C10 on approximately 200 cases** (GitHub #43, #14) -- 10 conditions x 3 seeds = 30 runs. This is the core experiment. Use Argon HPC scripts (#62).
5. **Report per-structure MAE** (GitHub #61) -- Computation exists; needs to run on all ablation results and be included in the results tables.
6. **Validate soft DVH loss** (GitHub #70) -- Scatter plot of soft D95 vs exact D95. Must be done before the ablation runs to ensure DVH loss is functioning as claimed.
7. **Clinical acceptability analysis** (GitHub #72) -- Define criteria, compute per-case pass/fail, report aggregate percentage.

### 5.2 Strengthens Paper (do if time permits)

8. **FiLM ablation C11** (GitHub #71) -- Full combined loss with FiLM disabled. Directly tests the headline contribution. One 3-seed experiment.
9. **Per-plan-type evaluation breakdown** (GitHub #75) -- Report metrics separately for each of the three plan types. Essential for understanding whether multi-protocol training helps or hurts.
10. **2%/2mm gamma + D98/D2 reporting** (GitHub #73) -- Standard metrics expected by Medical Physics reviewers.
11. **Failure case characterization** (GitHub #16) -- Systematic analysis of outlier cases, anatomical predictors of failure.
12. **Report learned sigma values** (GitHub #74) -- Extract from TensorBoard logs. Figure + table of uncertainty weight evolution.
13. **Dose profile figure** -- 1D dose along a line through PTV center (predicted vs ground truth). Shows penumbra accuracy and gradient realism.

### 5.3 Framing Recommendations

- **Title:** Lead with multi-protocol or loss engineering, not "U-Net for prostate." A title like "Multi-protocol prostate VMAT dose prediction via clinical loss engineering and constraint conditioning" signals both contributions and positions the work above generic architecture papers.

- **Abstract:** Lead with N approximately 200, three plan types, FiLM conditioning, and the ablation result. The hook should be: "We present a single model that predicts dose across three prostate VMAT plan types using FiLM-based constraint conditioning and a 5-component loss function with learned uncertainty weighting, validated through a pre-registered 10-condition ablation study."

- **Introduction:** Frame as two interleaved contributions:
  1. FiLM-based multi-protocol prediction (architectural contribution)
  2. Systematic loss engineering with pre-registered ablation (methodological contribution)
  Avoid framing as "small-dataset challenge" -- the expanded dataset is among the largest. Instead, frame as "multi-protocol challenge."

- **Discussion:** Compare per-structure MAE (not body-wide) to published values of 0.9-2.1 Gy. Explain global gamma honestly as a diagnostic metric dominated by low-dose regions. Position the asymmetric PTV loss result as the headline finding.

- **Supplemental material:** DDPM failure report (supplemental figure + table), architecture scout results (supplemental table), augmentation ablation (supplemental table), sigma evolution figure.

---

## 6. Comparison to Published Work

### 6.1 Dataset Size

| Study | N | Protocols | This Project's Advantage |
|-------|---|-----------|--------------------------|
| Norouzi Kandalan 2020 | 248 | Multi-protocol (4 plan styles) | Comparable N; we add systematic loss ablation and FiLM conditioning |
| Lempart 2021 | 160 | Single | Larger N with multi-protocol diversity |
| Kearney 2018 | 151 | Single (SBRT) | Larger N, conventional fractionation (more common clinically) |
| Church 2024 | 140 | SIB (2-level) | Larger N, more plan types, systematic loss ablation |
| Kontaxis 2020 | 101 | Single | Larger N, multi-protocol, no physics-based input requirement |
| Nguyen 2019 | 88 | Single | Much larger N, 3D vs 2D architecture |
| Kadoya 2023 | 68 | Single | Much larger N, multi-protocol |
| **This project** | **~200** | **3 plan types** | -- |

### 6.2 Methodology

| Feature | This Project | Best Published |
|---------|-------------|---------------|
| Architecture | 3D U-Net (standard) | 3D U-Net variants (standard) |
| Input features | CT + 9 SDFs + FiLM constraints | CT + masks/distance maps (most papers); CT + Rx (Norouzi Kandalan 2020) |
| Loss function | 5-component + Kendall uncertainty weighting | MSE or MSE + 1 additional component (typically) |
| Ablation rigor | 10 conditions x 3 seeds, pre-registered | Rarely more than 3 conditions; no pre-registration |
| Multi-protocol | 3 plan types, single model | Max 4 styles via transfer learning (Norouzi Kandalan 2020) |
| Pre-registered analysis | Yes, with statistical corrections | Very rare in the field |
| Asymmetric loss | Yes (underdose/overdose weighting) | Not reported |
| External validation | 6 held-out cases from second institution | Rare; Norouzi Kandalan 2020 is the benchmark |

### 6.3 Expected Performance Targets

| Metric | Published SOTA | Current (70 cases) | Expected (200 cases) | Assessment |
|--------|---------------|--------------------|--------------------|------------|
| Per-structure MAE | 0.9-2.1 Gy | Not yet reported (body-wide: 4.07 Gy) | Likely competitive when computed per-structure | Must compute to compare fairly |
| PTV D95 error | 0.2-1.0 Gy | 0.06 +/- 0.26 Gy | Likely maintained or improved | **Already matches or exceeds SOTA** |
| PTV Gamma 3%/3mm | 95-100% | 94.3 +/- 2.2% | Likely >95% with more training data | Near-SOTA, expected to cross threshold |
| PTV D98 error | 0.7-2.0% of Rx | Not yet computed | TBD | Must add to evaluation pipeline |
| Clinical acceptability | 62.6% (Hou 2025, multi-tumor) | Not yet computed | Target >75% | Must implement acceptability criteria |
| Global Gamma 3%/3mm | 90-97% | 30.4 +/- 3.6% | Unlikely to reach published values | Document honestly; dominated by low-dose regions. Position PTV gamma as the relevant metric. |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Multi-protocol training degrades single-protocol performance | Medium | High | Report per-plan-type metrics; include single-protocol comparison arm if degradation is observed |
| FiLM does not help (SDF channels are sufficient for protocol discrimination) | Medium | Low | Still publishable -- report honestly, note that the mechanism enables future extension to arbitrary protocols. Reframe paper around loss engineering as primary contribution. |
| Soft DVH loss does not correlate with exact DVH | Low | High | Validate before running the full ablation; replace with moment-based formulation (Jhanwar 2022) if correlation is poor |
| N approximately 200 is insufficient for 3-way plan type split | Low | Medium | At minimum approximately 40 per plan type based on current distribution; use stratified sampling and report per-type sample sizes. If one type has fewer than 30 cases, use stratified k-fold. |
| Reviewer demands end-to-end plan comparison (dose prediction -> deliverable plan) | Medium | Medium | Acknowledge as limitation; cite Church 2024 as complementary work that includes automated plan generation from predicted dose. Position this work as the dose prediction component of the pipeline. |
| Global gamma of approximately 30% becomes a distraction for reviewers | High | Medium | Preemptively explain in Methods/Discussion: global gamma is dominated by low-dose voxels far from PTV where small absolute errors produce large relative failures. Report PTV-region gamma as the primary metric. Consider showing a gamma-vs-threshold sensitivity analysis. |
| Architecture scout null results weaken the "loss > architecture" narrative | Low | Low | Null results strengthen the narrative: three architectures tested, none improved over baseline, confirming that loss engineering is the higher-leverage intervention. Report in supplemental. |
| Kendall uncertainty weighting does not improve over manual weights | Medium | Medium | The ablation design (C6 full combined vs individually tuned components) will reveal this. If manual weights match or beat Kendall, report honestly and recommend the simpler approach. |

---

## 8. Summary of Required Pre-Submission Actions

| # | Action | Blocking? | Estimated Effort | Depends On |
|---|--------|-----------|-----------------|------------|
| 1 | PTV50.4 channel + SDF convention fix | Yes | 1-2 days | -- |
| 2 | Reprocess approximately 200 cases | Yes | 1 day (automated) | #1 |
| 3 | Stratified data split | Yes | 0.5 days | #2 |
| 4 | Validate soft DVH loss | Yes | 0.5 days | -- |
| 5 | Full ablation C1-C10 (30 runs) | Yes | 5-7 days GPU time (Argon) | #2, #3, #4 |
| 6 | Per-structure MAE reporting | Yes | 0.5 days | #5 |
| 7 | Clinical acceptability metric | Yes | 0.5 days | #5 |
| 8 | FiLM ablation C11 | No (but strongly recommended) | 2 days GPU time | #5 |
| 9 | Per-plan-type evaluation | No (but strongly recommended) | 0.5 days | #5 |
| 10 | 2%/2mm gamma + D98/D2 | No | 0.5 days | #5 |
| 11 | Failure case characterization | No | 1 day | #5 |
| 12 | Sigma value reporting | No | 0.25 days | #5 |
| 13 | Manuscript draft | Yes | 2-3 weeks | #5-#7 |

**Total estimated timeline:** 3-4 weeks from data reprocessing to submission-ready manuscript, assuming Argon GPU availability for the ablation runs.

---

*Created: 2026-03-07*

## GitHub Issue Cross-Reference

| Issue | Title | Role in Assessment |
|-------|-------|-------------------|
| #14 | Run combined loss on 200 cases | Critical path #4 — core ablation experiment |
| #16 | Failure case report | Recommendation #11 — outlier characterization |
| #39 | Inclusion criteria | Updated — multi-protocol expansion |
| #43 | Ablation study design | Updated — added C11 FiLM condition, 33 runs |
| #61 | Per-structure MAE | Critical path #5 — W1 |
| #62 | Argon HPC scripts | Infrastructure for ablation runs |
| #63 | Project status | Updated — strategic pivot to multi-protocol |
| #66 | Add PTV50.4 as input channel | Critical path #1 — multi-protocol foundation |
| #67 | Fix SDF absent-structure convention | Critical path #1 — 0.0 instead of +1.0 |
| #68 | Reprocess ~200 cases (v2.4) | Critical path #2 — data preparation |
| #69 | Plan-type stratified split | Critical path #3 — balanced train/val/test |
| #70 | Validate soft DVH loss | Critical path #6 — W2 |
| #71 | FiLM conditioning ablation C11 | Recommendation #8 — W5 |
| #72 | Clinical acceptability analysis | Critical path #7 — W3 |
| #73 | 2%/2mm gamma + D98/D2 + subsample audit | Recommendation #10 — W6, W7 |
| #74 | Report learned sigma values | Recommendation #12 — W12 |
| #75 | Per-plan-type evaluation | Recommendation #9 |
