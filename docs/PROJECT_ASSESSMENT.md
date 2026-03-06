# VMAT Diffusion Project Assessment

**Date:** 2026-03-06
**Prepared for:** Publication readiness review, strategic planning
**Scope:** Literature positioning, publishability, journal targeting, codebase readiness, path to vmat-planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Landscape](#2-literature-landscape)
3. [Novelty & Differentiation Assessment](#3-novelty--differentiation-assessment)
4. [Publication Readiness](#4-publication-readiness)
5. [Gap Analysis: What Remains](#5-gap-analysis-what-remains)
6. [Journal Recommendations](#6-journal-recommendations)
7. [Strategic Path to vmat-planning](#7-strategic-path-to-vmat-planning)
8. [Recommended Actions](#8-recommended-actions)

---

## 1. Executive Summary

**Bottom line: This project is publishable in Medical Physics, with work remaining.**

The project occupies a well-defined niche: systematic loss-function engineering for prostate VMAT dose prediction, with multi-institutional validation. While 3D U-Net dose prediction for prostate is a crowded subfield (20+ papers since 2019), the specific contribution -- a pre-registered ablation study of 5 clinically-motivated loss components with uncertainty weighting across 16 conditions -- has not been done. The pre-registered design, multi-seed protocol, and clinical metric focus (DVH/QUANTEC compliance rather than just MAE) elevate this above incremental work.

**Current results (70 cases, 2.5:1 combined loss):**
- PTV Gamma 3%/3mm: 94.3 +/- 2.2% (target: >95%)
- PTV70 D95 error: +0.06 +/- 0.26 Gy (essentially zero bias)
- MAE: 4.07 +/- 0.64 Gy
- Massive improvement over MSE baseline (80.2% -> 94.3% PTV gamma, -1.76 -> +0.06 Gy D95)

**Key risks:** Dataset size (~200 cases) is moderate but competitive with literature. The 4.07 Gy MAE is higher than some reports (~1-2 Gy for simpler protocols), but the SIB complexity and honest 3D evaluation explain this. The contribution is engineering/empirical, not architectural -- this is a strength for Medical Physics but a weakness for methods-focused journals.

### Verdict by Question

| Question | Answer |
|----------|--------|
| Is it publishable? | **Yes**, given completion of remaining experiments on the full ~200-case dataset |
| Are GitHub goals sufficient? | **Mostly yes** -- the 16-condition ablation is the paper; a few issues need updates (see Section 5) |
| Is it too derivative? | **No** -- the ablation study design and clinical metric focus differentiate it |
| Is it useful? | **Yes** -- provides practical guidance on which loss components matter for clinical dose prediction |
| Best journal? | **Medical Physics** (primary), Physics in Medicine and Biology (backup) |

---

## 2. Literature Landscape

### 2.1 Key Papers: Prostate Dose Prediction (2018-2026)

| Year | Authors | Journal | n | Architecture | Inputs | Key Results | Loss |
|------|---------|---------|---|-------------|--------|-------------|------|
| 2019 | Nguyen et al. | Sci. Reports | ~80 | 2D U-Net (slicewise) | CT + contours | Dice 0.91 on isodose volumes | L2 |
| 2019 | Kearney et al. (DoseNet) | Phys Med Biol | 151 | 3D fully-conv | CT + distance maps | MAE ~2% of Rx | L2 |
| 2020 | Kontaxis et al. (DeepDose) | Phys Med Biol | 101 | 3D CNN | Physics-based inputs + MLC shapes | Clinical QA gamma pass | L2 |
| 2020 | Gronberg et al. | Med Phys | 248 | HD-UNet | CT + contours | Transfer learning across institutions | L2 |
| 2021 | Babier et al. (OpenKBP) | Med Phys | 340 (H&N) | Various (challenge) | CT + contours + distance maps | Benchmark: dose score 2.58, DVH score 1.49 | Various |
| 2021 | Zimmermann et al. | Med Phys | 200 (H&N) | ResNet-UNet | CT + contours | DVH diffs <1%, feature loss | L1 + feature |
| 2021 | Lee et al. | Front AI | 100 | DenseNet | CT + contours | V100 0.51%, output initialization | MSE |
| 2022 | Chen et al. | J Radiation Res | 68 | HD-UNet (AIVOT) | VMAT contours | DVH diff 1.32+/-1.35% | L2 |
| 2023 | Xiao et al. | Physica Medica | 104 | 3D U-Net | CT + contours | Structure loss improves DVH accuracy | L2 + Lstruct |
| 2024 | Fransson et al. | Med Phys | ~50 | U-Net | MR + contours (MR-Linac) | MR-guided prostate | L2 |
| 2024 | Church et al. | PHIRO | 140 | ResUNet | CT + contours | Fully automated TPS pipeline, V100 <1% diff | L2 |
| 2025 | Arberet et al. | Med Phys | -- | 3D BEV-to-fluence | Dose maps -> fluence | 180 fluence maps in <20ms | L2 |
| 2026 | Johansson et al. | Adv Biomed Res | 45 | Swin-UNETR | CT + contours | Recent VMAT model | L2 |

### 2.2 Benchmark Metrics from Literature

| Metric | Best Reported | Typical Range | This Project |
|--------|---------------|---------------|-------------|
| MAE (Gy) | 0.5-1.5 Gy (simpler protocols) | 1.5-4.0 Gy | 4.07 Gy (SIB, honest 3D) |
| Gamma 3%/3mm (global) | 85-97% | 70-90% | 30.4% (low-dose diversity) |
| Gamma 3%/3mm (PTV-region) | 90-99% | 85-95% | 94.3% |
| PTV D95 error | <1 Gy | 1-3 Gy | +0.06 Gy |
| DVH parameter accuracy | <2% | 1-5% | Per-structure varies |
| Dataset size (prostate) | 248 (Gronberg) | 40-150 | ~200 (with Institution B) |

**Note on MAE:** Your 4.07 Gy MAE looks high compared to some reports of <1.5 Gy, but those papers often (a) use 2D slice evaluation, (b) evaluate simpler single-PTV protocols (not SIB), (c) report validation MAE (not test), or (d) use generous dose cutoffs. Your honest 3D volumetric MAE on a held-out test set with SIB is not directly comparable. **This should be explicitly discussed in the paper.**

### 2.3 Loss Function Engineering in the Literature

This is where the novelty claim is strongest. Most dose prediction papers use **MSE or L1 only**:

| Paper | Loss Engineering | Components | Ablation? |
|-------|-----------------|------------|-----------|
| Zimmermann 2021 | Feature loss + L1 | 2 components | No formal ablation |
| Xiao 2023 | Structure loss (Lstruct) | 2 components (L2 + Lstruct) | L2 vs L2+Lstruct only |
| OpenKBP winners | Various (GAN, feature, weighted) | Challenge-optimized | No systematic study |
| **This project** | **5 components + uncertainty weighting** | **MSE + gradient + DVH + structure + asymmetric PTV** | **16-condition ablation** |

**No published paper has performed a systematic ablation study of 5+ clinically-motivated loss components for dose prediction.** The Xiao 2023 paper in Physica Medica is the closest, but it tested only 2 conditions (L2 vs L2+structure) on 104 cases. Your 16-condition design with pre-registration and 3-seed protocol is substantially more rigorous.

### 2.4 Diffusion Models for Dose Prediction

Few published results exist:
- Sohl-Dickstein/Ho-style DDPM has been explored for medical image synthesis but not extensively for dose prediction
- Your DDPM experiments showed no advantage over U-Net (deprioritized, Issue #27)
- The project name "vmat-diffusion" is somewhat misleading given the pivot to baseline U-Net; consider whether the DDPM negative result merits a brief mention in the paper as supporting evidence for the U-Net choice

### 2.5 Multi-Institutional Validation

Rare in dose prediction literature:
- Gronberg et al. 2020: 248 cases, 2 institutions, transfer learning focus
- Most papers: single institution
- **Your ~200 cases from 2 institutions is competitive** and addresses a common reviewer concern

---

## 3. Novelty & Differentiation Assessment

### Is it too derivative?

**No.** While 3D U-Net dose prediction for prostate is well-established, the specific contribution is differentiated:

| Aspect | Derivative? | Assessment |
|--------|------------|------------|
| Architecture (3D U-Net) | Yes | Standard, but this is deliberate -- architecture is controlled so loss effects are isolated |
| Loss function engineering | **Novel** | No 5-component ablation exists in literature |
| Pre-registered design | **Novel** | Essentially unprecedented in dose prediction |
| Uncertainty weighting (Kendall 2018) | Applied novelty | Used in multi-task learning, not in dose prediction loss balancing |
| SDF inputs | Moderate novelty | Distance maps used in OpenKBP; SDFs less common for prostate |
| FiLM conditioning on constraints | Moderate novelty | Some papers condition on DVH; FiLM is uncommon |
| Prostate SIB (2-PTV) | Niche value | Most prostate papers are single-PTV |
| Multi-institutional (2 sites) | Competitive | Few dose prediction papers validate across sites |

### What makes it useful?

The practical value is a **recipe**: "If you want clinically acceptable prostate VMAT dose predictions, use this combination of loss components." Most papers optimize one loss function and report results. Your ablation tells practitioners:
- Which components matter most (asymmetric PTV? gradient? DVH?)
- Whether uncertainty weighting beats manual tuning
- What the interaction effects are
- Clear decision rules (pre-specified)

This is directly actionable for anyone building a clinical dose prediction pipeline.

---

## 4. Publication Readiness

### 4.1 Codebase Assessment: 7.5/10

| Component | Score | Status |
|-----------|-------|--------|
| Training script | 9/10 | Publication-ready. Multi-seed, comprehensive logging, all losses |
| Inference script | 9/10 | Clean API, proper evaluation framework |
| Preprocessing | 8/10 | Well-documented, versioned (v2.3), backwards-compatible |
| Loss functions (all 5) | 9/10 | Scientifically sound, well-documented, differentiable |
| Evaluation pipeline | 9/10 | Gamma (AAPM TG-218), DVH, QUANTEC constraints all correct |
| Statistical framework | 9/10 | Proper multi-seed aggregation, Wilcoxon, Holm-Bonferroni |
| Architecture variants | 8/10 | 3 variants implemented and tested |
| Documentation | 8/10 | CLAUDE.md, docstrings comprehensive; API docs missing |
| Code cleanliness | 5/10 | **3 hardcoded username paths must be removed before release** |
| Experiment tracking | 9/10 | GitHub Issues + notebooks + EXPERIMENTS_INDEX |

**Critical pre-release items (Issue #17):**
1. Remove `/home/wrockey/` hardcoded paths from 3 files (generate_baseline_v23_figures.py, analyze_anatomical_variability.py, run_baseline_v23.sh)
2. Add installation section to README
3. Document backwards compatibility with v2.2 NPZ files

### 4.2 Experiment Status

| Milestone | Status | Completion |
|-----------|--------|------------|
| Phase 0: Setup | 7/10 closed | **Blocked on data** (#2, #5, #39) |
| Phase 1: Evaluation | 10/14 closed | Core framework done |
| Phase 2: Combined Loss | 4/7 closed | Combined loss validated; full ablation pending data |
| Phase 3: Publication | 0/6 closed | Not started (blocked on Phase 0/2) |

### 4.3 What the Paper Needs That Doesn't Exist Yet

| Item | Status | Blocking? |
|------|--------|-----------|
| ~130 Institution B cases preprocessed | Not started | **YES** -- paper requires ~200 cases |
| Locked stratified test set (#38) | Provisional | **YES** -- all results on provisional split |
| Inclusion/exclusion criteria (#39) | Not documented | **YES** -- reviewers will ask |
| Full C1-C10 loss ablation on final dataset | Not started | **YES** -- core contribution |
| C7-C10 "remove-one" ablation runs | Not started | **YES** -- ablation is the paper |
| Literature review (#41) | This document | Partial |
| Data provenance & ethics (#5) | Not started | **YES** -- Medical Physics requires |
| Failure case analysis (#16) | Not started | HIGH priority |
| Manuscript draft (#18) | Not started | -- |
| Code release cleanup (#17) | Not started | Before submission |

---

## 5. Gap Analysis: What Remains

### 5.1 Critical Path (must complete before submission)

```
Institution B data (~130 cases)
        |
        v
Preprocess with v2.3 pipeline (#3 reopened)
        |
        v
Define inclusion/exclusion criteria (#39)
        |
        v
Lock dataset + stratified split (#38, #40)
        |
        v
Re-calibrate loss normalization (calibrate_loss_normalization.py)
        |
        v
Run full ablation: C1-C10 x 3 seeds = 30 runs (~390 GPU-hours on RTX 3090)
  [Parallelizable on Argon: ~130 GPU-hours wall-clock with 3 simultaneous jobs]
        |
        v
Run architecture comparison: C11-C16 x 3 seeds = 18 runs (~234 GPU-hours)
  [Can skip if C11-C15 scouts on 70 cases show no signal -- already shown]
        |
        v
Statistical analysis + cross-condition comparison
        |
        v
Failure case analysis (#16)
        |
        v
Manuscript draft (#18) + figure generation
        |
        v
Code release cleanup (#17)
        |
        v
Submit to Medical Physics
```

### 5.2 GPU Time Estimates

| Task | Runs | Hours/Run | Total GPU-hrs | Parallelism |
|------|------|-----------|---------------|-------------|
| C1-C10 (loss ablation) x 3 seeds | 30 | ~13 | ~390 | 3 seeds parallel on Argon |
| C11-C16 (architecture) x 3 seeds | 18 | ~13 | ~234 | 3 seeds parallel on Argon |
| Inference (all) | 48 | ~0.5 | ~24 | Sequential per seed |
| **Total** | **48+48** | | **~648** | |

**With Argon HPC (3 A100s parallel):** ~216 hours wall-clock for training = ~9 days
**With RTX 3090 only (sequential):** ~648 hours = ~27 days

**Recommendation on architecture experiments (C11-C16):** Your scout results on 70 cases show no architecture beats baseline (AttentionUNet PTV gamma 81.1%, BottleneckAttn 84.0%, Wider baseline 86.2%, vs baseline 80.2%). This is a clear negative result. Running full 3-seed x 3 architectures x 2 losses = 18 runs may not be the best use of GPU time. Consider:
- **Option A:** Run only C11 (AttentionUNet+MSE) and C12 (AttentionUNet+Full) at 3 seeds (6 runs) as representative architecture control. Report scout results for others.
- **Option B:** Drop architecture comparison entirely, report scouts as "we verified architecture is not the bottleneck" in 1 paragraph. This saves ~234 GPU-hours and 10 days.
- **Option C:** Run all 18 for completeness (strongest paper, most GPU time).

Option A or B is recommended given the clear negative signal from scouts and compute constraints. The paper's story is about loss engineering, not architecture search.

### 5.3 Updated GitHub Issues Needed

| New/Updated Issue | Purpose |
|-------------------|---------|
| Reopen #3 | Preprocess Institution B cases |
| Update #14 | Full combined loss on ~200 cases (currently says "100+") |
| Update #43 | Add decision on C11-C16 scope reduction |
| New issue | Batch launcher scripts for Argon HPC (all 48 runs) |
| New issue | Cross-condition statistical analysis script |
| Update #53 | Architecture comparison -- document scout negative result, decide scope |

---

## 6. Journal Recommendations

### Ranked Options

| Rank | Journal | IF | Fit | Acceptance | Recommendation |
|------|---------|-----|-----|------------|----------------|
| **1** | **Medical Physics** | 3.2-3.5 | Excellent | ~30-40% | **Primary target.** Natural home for loss engineering + clinical metrics. AAPM readership. |
| **2** | Physics in Medicine & Biology | 3.4-3.5 | Very good | Unknown | Strong backup. Faster review (~35 days to first decision). More European readership. |
| **3** | JACMP | 2.5 | Good | ~56% | High acceptance. Good if Medical Physics rejects. Less prestige but solid. |
| **4** | Frontiers in Oncology | 3.1-3.3 | Moderate | ~50% | Open access. Quick turnaround. Less domain-specific prestige. |
| **5** | Radiotherapy & Oncology | 4.4 | Moderate | Unknown | Higher IF, but clinical readership may not appreciate engineering contribution. Reach option. |
| **6** | Scientific Reports | 3.9 | Moderate | ~50% | Broad scope. Good fallback. Less domain credibility. |
| -- | Red Journal (IJROBP) | 6.4 | Low | ~33% | Not recommended -- too clinical, no patient outcomes |
| -- | Medical Image Analysis | 10.7+ | Low | ~15-20% | Not recommended -- needs novel methodology, not engineering ablation |

### Primary Strategy: Medical Physics

**Why Medical Physics is the right choice:**
1. The journal regularly publishes dose prediction work (Fransson 2024, Zimmermann 2021, multiple OpenKBP papers)
2. Your contribution (systematic loss ablation, clinical metrics, pre-registered design) fits the journal's emphasis on methodological rigor and clinical relevance
3. The comparable Xiao 2023 structure-loss paper went to the lower-tier Physica Medica with only 2 conditions and 104 cases; your 16-condition study with ~200 multi-institutional cases exceeds that scope
4. AAPM readership = exactly the people who build clinical auto-planning pipelines
5. Open access option available ($3,040)

**Potential reviewer pushback and mitigations:**
- "Architecture is standard" --> Deliberately controlled. The paper IS about loss engineering.
- "Single disease site" --> SIB adds complexity. Multi-institutional addresses generalizability.
- "Only ~200 cases" --> Competitive with Gronberg (248), larger than most prostate studies (40-150)
- "MAE is high" --> Address head-on: SIB protocol, honest 3D evaluation, clinical metrics show accuracy where it matters

---

## 7. Strategic Path to vmat-planning

### 7.1 The Two-Paper Strategy

**Paper 1 (vmat-diffusion, current):** "Loss-function engineering for clinically acceptable prostate VMAT dose prediction"
- Establishes domain credibility
- Provides the upstream dose prediction model
- Publication target: Medical Physics

**Paper 2 (vmat-planning, future):** "End-to-end deliverable VMAT plan generation for prostate SIB using deep learning"
- Takes Paper 1's dose predictor as input
- Adds dose-to-deliverable-plan conversion
- Significantly higher impact potential
- Publication target: Medical Physics or Red Journal (if clinical validation is strong)

### 7.2 Current State of Automated Plan Generation

The field has converged on a pipeline:
```
Patient anatomy --> Dose prediction (DL) --> Plan optimization --> Deliverable VMAT plan
```

Recent milestones:
- **Church et al. 2024 (PHIRO):** Fully automated prostate VMAT via DL dose prediction + scripted Eclipse optimization. 140 patients. First fully-automated pipeline in commercial TPS.
- **Hrinivich et al. 2024 (Med Phys):** RL-based VMAT machine parameter optimization. Deliverable plans in 3.3 seconds. Prostate cancer.
- **Arberet et al. 2025 (Med Phys):** Predicts 180 VMAT fluence maps in <20ms via 3D network.
- **ECHO-VMAT (PortPy, open-source):** DL dose prediction + sequential convex programming for VMAT optimization.
- **Shaffer et al. 2026 (Med Phys):** Tandem RL framework for planning + machine parameter optimization.

### 7.3 What Would Make vmat-planning Novel

| Differentiator | Why It Matters |
|----------------|---------------|
| Prostate SIB (2-PTV) | Most plan generation papers target single-PTV. SIB adds optimization complexity |
| Diffusion-model dose backbone | If gradient loss improves dose realism -> better deliverable plans |
| Multi-institutional validation | Almost no automated planning papers validate across institutions |
| End-to-end without commercial TPS | True independence from vendor-specific scripting |
| Open-source release | PortPy exists but is MSKCC-specific; an independent open implementation has value |

### 7.4 Recommended Approach for vmat-planning

**Option A (Recommended): Dose prediction + TPS scripting (Church et al. approach)**
- Use Paper 1's dose predictor
- Extract DVH objectives from predicted dose
- Feed into Eclipse/RayStation scripting API
- Simplest path, most clinically translatable
- Risk: depends on commercial TPS access

**Option B (Higher impact): Direct MLC sequence prediction**
- Train encoder-decoder to predict MLC positions per control point
- Requires beam parameter extraction from DICOM-RT Plan (already in preprocessing pipeline)
- Heilemann et al. 2023 demonstrated this with 619 plans -- your ~200 may need augmentation
- Independent of commercial TPS

**Option C (Highest risk/reward): RL-based machine parameter optimization**
- Hrinivich approach: policy network generates control points sequentially
- Most data-efficient (learns policy, not direct mapping)
- Fast inference (3.3 seconds)
- Most complex to implement

### 7.5 Data Already Available

Your preprocessing pipeline (v2.3) already extracts:
- MLC leaf positions (beam0_mlc_a, beam0_mlc_b)
- Gantry angles per control point
- Cumulative meterset weights (MU fractions)
- Jaw positions
- Beam energy, dose rates

This data is stored in NPZ files and ready for training a plan generation model. Issue #60 (add beam/RT Plan data to preprocessing) appears to already be partially complete based on the v2.2 CHANGELOG entries.

---

## 8. Recommended Actions

### Immediate (this week)

1. **Update .claude/instructions.md** with corrected case counts (~200 total, not 161)
2. **Update GitHub Issue #2** with Institution B status
3. **Create new issue:** Argon HPC batch launcher for full ablation study
4. **Update Issue #53:** Document architecture scout negative results, recommend scope reduction (Option A or B from Section 5.2)

### Before training starts (when Institution B data arrives)

5. **Close Issue #39:** Document inclusion/exclusion criteria
6. **Preprocess Institution B** with v2.3 pipeline
7. **Lock dataset (#38, #40):** Stratified split by institution + PTV70 volume tertile
8. **Re-run loss calibration** on full ~200-case dataset
9. **Prepare Argon job scripts** for full ablation (array jobs, dependency chains)

### During training (~2-4 weeks)

10. **Run C1-C10 x 3 seeds** (30 training + 30 inference runs)
11. **Run C11-C12 x 3 seeds** (6 runs, if architecture comparison retained)
12. **Begin manuscript outline** and literature review section

### After training completes

13. **Statistical analysis:** Cross-condition comparison, Wilcoxon tests, effect sizes
14. **Failure case analysis (#16):** Bottom 10% by PTV Gamma and DVH
15. **Generate all figures** (standard 8-figure set per condition + cross-experiment comparison)
16. **Create aggregate notebooks** for each condition family
17. **Draft manuscript (#18)**
18. **Code cleanup (#17):** Remove hardcoded paths, prepare for public release
19. **Submit to Medical Physics**

### Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Institution B preprocessing | 1-2 days | Depends on data format consistency |
| Dataset lock + calibration | 1 day | |
| Full ablation training (Argon) | 9-14 days | 30-48 runs, 3 parallel A100s |
| Inference + evaluation | 2-3 days | |
| Statistical analysis + figures | 3-5 days | |
| Manuscript draft | 2-3 weeks | |
| Internal review + revision | 1-2 weeks | |
| Code cleanup + submission | 1 week | |
| **Total from data arrival** | **~6-8 weeks** | |

---

## Appendix A: Key References

### Dose Prediction (Prostate)
- Nguyen et al. (2019). Sci Reports. DOI: 10.1038/s41598-018-37741-x
- Kearney et al. (2018). DoseNet. Phys Med Biol. PMID: 30511663
- Gronberg et al. (2020). Model adaptation. Med Phys. DOI: 10.1002/mp.14537
- Xiao et al. (2023). Structure loss. Physica Medica 107:102550
- Fransson et al. (2024). MR-guided. Med Phys. PMID: 39106418
- Church et al. (2024). Automated plan generation. PHIRO. DOI: 10.1016/j.phro.2024.100632

### Loss Engineering
- Kendall et al. (2018). Multi-task uncertainty weighting. CVPR. arXiv: 1705.07115
- Zimmermann et al. (2021). Feature loss + One Cycle. Med Phys 48(9):5562-5570
- Babier et al. (2021). OpenKBP challenge. Med Phys 48(9):5549-5561

### Automated Plan Generation
- Heilemann et al. (2023). MLC sequence prediction. Med Phys. PMID: 37314944
- Hrinivich et al. (2024). RL for VMAT machine parameters. Med Phys. DOI: 10.1002/mp.17100
- Arberet et al. (2025). BEV-to-fluence 3D network. Med Phys. DOI: 10.1002/mp.17673
- Shaffer et al. (2026). Tandem RL framework. Med Phys. DOI: 10.1002/mp.70306

### Open-Source Tools
- PortPy: https://github.com/PortPy-Project/PortPy (129 prostate cases benchmark)
- ECHO-VMAT: https://github.com/PortPy-Project/ECHO-VMAT

---

## Appendix B: Codebase Readiness Details

### Loss Function Correctness Summary

| Loss | Implementation | Differentiable | Scientifically Sound | Score |
|------|---------------|----------------|---------------------|-------|
| GradientLoss3D | 3D Sobel kernels, L1 | Yes | Yes | 9/10 |
| DVHAwareLoss | Soft histogram, sigmoid Vx | Yes | Yes (soft != hard D95 -- documented) | 9/10 |
| StructureWeightedLoss | SDF-based region weighting | Yes | Yes | 9/10 |
| AsymmetricPTVLoss | 3x underdose penalty | Yes | Yes (clinically motivated) | 9/10 |
| UncertaintyWeightedLoss | Kendall 2018 implementation | Yes | Yes | 9/10 |

### Evaluation Pipeline Correctness

| Component | Standard | Status |
|-----------|----------|--------|
| Gamma analysis | AAPM TG-218, pymedphys | Correct |
| DVH metrics | np.percentile(method='lower'), matches TPS nearest-rank | Correct |
| Clinical constraints | QUANTEC-based, prostate SIB | Correct |
| Statistical tests | Wilcoxon signed-rank, Holm-Bonferroni, bootstrap CI | Correct |
| Unit of observation | Per-case means (not seed x case pairs) | Correct |

### Pre-Release Cleanup Items

| Priority | Item | Files Affected |
|----------|------|---------------|
| CRITICAL | Remove `/home/wrockey/` paths | generate_baseline_v23_figures.py, analyze_anatomical_variability.py, run_baseline_v23.sh |
| HIGH | Add installation docs to README | README.md |
| MEDIUM | Document v2.2 backwards compatibility | preprocessing_guide.md |
| LOW | Sphinx/pdoc API documentation | New setup |
