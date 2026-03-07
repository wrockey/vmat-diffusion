# Deep Learning-Based VMAT Dose Prediction for Prostate Cancer: A U-Net Approach with Clinical Loss Engineering

> Working title — refine as the narrative solidifies.

## Target Journal

TBD. Candidates:
- **Medical Physics** (AAPM) — strong ML + dose prediction track record
- **Physics in Medicine & Biology** (IPEM) — computational methods focus
- **IJROBP** (ASTRO) — clinical audience, higher impact, shorter methods tolerance

---

## Abstract

[Draft when results are final. Structure: Background, Purpose, Methods, Results, Conclusions.]

---

## 1. Introduction

### Key points to make:
- VMAT is standard of care for prostate; treatment planning is time-consuming
- Deep learning dose prediction can accelerate planning (knowledge-based planning, autoplanning)
- Prior work: U-Net architectures dominate (cite Nguyen 2019, Kearney 2018, etc.)
- Gap: Most work uses large institutional datasets (200-500+); few address smaller datasets (~70 cases) common at many centers
- Gap: Loss function design for clinical metric optimization is underexplored
- Our contribution: Systematic loss engineering to optimize clinical metrics (PTV D95, gamma) with a modest dataset

### Framing decisions:
- [ ] Position as "loss engineering" paper vs "dose prediction" paper?
- [ ] Emphasize small-dataset challenge or treat it as a limitation?
- [ ] Include diffusion model comparison in intro or save for discussion?

---

## 2. Methods

### 2.1 Dataset
- N=74 prostate VMAT cases (SIB: 70 Gy / 56 Gy in 28 fractions)
- Institutional IRB approval (get protocol number)
- Preprocessing: native resolution CT, B-spline dose resampling, crop to 300x300xZ
- 8 structures: PTV70, PTV56, Prostate, Rectum, Bladder, Femur L/R, Bowel
- SDF representation for structure channels
- 80/10/10 train/val/test split, fixed across all experiments

### 2.2 Model Architecture
- 3D U-Net with FiLM conditioning on dose constraints
- 5 resolution levels, base_channels=48 (~23.7M parameters)
- Input: 9 channels (CT + 8 SDFs), Output: 1 channel (normalized dose)
- Patch-based training (128^3), sliding window inference (overlap=64)
- [ ] Architecture figure needed

### 2.3 Loss Functions
- **Baseline:** MSE only
- **Combined loss:** MSE + Gradient + DVH + Structure-weighted + Asymmetric PTV
  - Uncertainty weighting (Kendall 2018) for automatic balancing
  - Key innovation: asymmetric PTV penalty (underdose penalized 2.5x more than overdose)
- Loss calibration procedure for initial uncertainty weights
- [ ] Loss function equations (LaTeX)
- [ ] Ablation showing contribution of each component?

### 2.4 Training
- PyTorch Lightning, mixed precision (fp16)
- Adam optimizer, lr=1e-4, weight_decay=1e-2
- Early stopping (patience=50 on val MAE)
- 3-seed protocol (42, 123, 456) for publishable results
- Data augmentation: left-right flip only (anatomically valid for prostate)

### 2.5 Evaluation Metrics
- MAE (Gy) — overall dose accuracy
- Gamma analysis (3%/3mm) — global and PTV-region
- DVH metrics: D95, D50, Dmean per structure
- QUANTEC compliance check
- Statistical: mean +/- std across seeds, paired Wilcoxon for comparisons

---

## 3. Results

### 3.1 Baseline Performance
- Table: baseline 3-seed aggregate metrics
- MAE 4.22 +/- 0.53 Gy, PTV gamma 80.2 +/- 5.3%, D95 gap -1.76 +/- 0.69 Gy
- Key finding: baseline systematically underdoses PTV (negative D95 gap)

### 3.2 Architecture Exploration (brief)
- Tested AttentionUNet, BottleneckAttn, wider baseline — all comparable to baseline
- Conclusion: architecture is not the bottleneck, loss function is

### 3.3 Combined Loss Results
- Progressive tuning: 3:1 (overdoses) -> 2:1 (gamma drops) -> 2.5:1 (sweet spot)
- **3-seed aggregate:** MAE 4.07 +/- 0.64, PTV gamma 94.3 +/- 2.2%, D95 +0.06 +/- 0.26
- Key finding: asymmetric PTV penalty eliminates systematic underdosing
- Key finding: PTV gamma improved 80.2% -> 94.3% (near 95% clinical target)

### 3.4 Per-Structure Analysis
- Which structures are well-predicted? Which are challenging?
- Bowel involvement drives outlier cases (0027, 0079)

### Figures for main text:
1. Architecture diagram
2. Dose colorwash (best case) — predicted vs ground truth
3. DVH comparison (representative case)
4. Box plots: baseline vs combined loss on MAE, PTV gamma, D95
5. Cross-experiment comparison table

---

## 4. Discussion

### Key discussion points:
- Loss engineering > architecture changes for clinical metric optimization
- Asymmetric PTV penalty: clinical rationale (underdosing tumor is worse than mild overdose)
- Global gamma remains ~30% — dominated by low-dose regions far from PTV
  - [ ] Is this a fair metric? Discuss clinical relevance
- Small dataset (n=74): results promising but need validation on larger cohort
- Comparison to literature: how do our metrics compare? (careful — different datasets/sites)

### DDPM comparison (discussion or supplemental):
- Tested conditional DDPM with equivalent architecture — catastrophic failure
- DDPM predicts ~10% of target PTV dose (D95 4-6 Gy vs 70 Gy target)
- Interpretation: dose prediction is well-conditioned regression, not generative modeling
- Justifies direct regression approach

### Limitations:
- Single institution, single disease site (prostate SIB)
- n=74 cases — larger validation needed
- No plan quality comparison (our predictions vs clinical plans)
- No integration with treatment planning system (standalone prediction only)
- Fixed patch size may miss long-range dose dependencies

---

## 5. Conclusion

- Combined loss with asymmetric PTV penalty significantly improves PTV dose accuracy
- PTV gamma 94.3% approaches clinical 95% target with only 74 training cases
- D95 gap reduced from -1.76 Gy (underdose) to +0.06 Gy (essentially zero bias)
- Loss engineering is more effective than architecture changes for this task
- Dataset expansion to 200+ cases expected to push past 95% PTV gamma

---

## Acknowledgments

[Funding, data access, compute resources]

---

## References

See `references.bib`
