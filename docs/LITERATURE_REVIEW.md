# Literature Review: Deep Learning for Prostate VMAT Dose Prediction

**Project:** VMAT Diffusion (GitHub Issue #41)
**Date:** 2026-03-06
**Scope:** Deep learning-based dose prediction for prostate radiation therapy (2017-2026)

---

## 1. Executive Summary

Deep learning dose prediction for prostate radiation therapy has progressed rapidly from 2D feasibility studies (2017-2019) to fully 3D, clinically deployable pipelines (2023-2026). The dominant architecture remains the U-Net and its variants, though GANs, transformers, and diffusion models have recently entered the field. Dataset sizes are typically 60-160 patients from single institutions; multi-institutional validation remains rare. Loss function engineering is an active area of innovation, with DVH-aware, moment-based, and gradient-penalty losses all showing improvements over standard MSE/MAE losses.

**State-of-the-art benchmarks for prostate dose prediction:**
- **MAE:** 0.9-2.1 Gy (structure-specific); structure errors within 1-3% of prescription dose
- **Gamma pass rate (3%/3mm):** 95-100% in PTV regions; 90-97% globally
- **DVH accuracy:** D95 errors typically 0.2-1.0 Gy; Dmean errors within 1-3% of prescription

**This project's current results** (combined loss 2.5:1, 3-seed aggregate on 70 cases):
- MAE: 4.07 +/- 0.64 Gy (body-wide, including low-dose regions)
- PTV Gamma (3%/3mm): 94.3 +/- 2.2%
- PTV70 D95 error: +0.06 +/- 0.26 Gy

The D95 accuracy is competitive with or superior to published results. The PTV gamma rate approaches published benchmarks. The body-wide MAE is higher than best-reported values but reflects a different evaluation protocol (full body vs. within-PTV or within-contour evaluation).

---

## 2. Key Papers and Comparison Table

### 2.1 Prostate-Specific Dose Prediction Papers

| # | Authors | Year | Journal | N | Architecture | Input Features | Loss | Key Metrics | Multi-Inst? |
|---|---------|------|---------|---|-------------|---------------|------|-------------|-------------|
| 1 | Nguyen et al. | 2019 | Sci Reports | 88 | Modified 2D U-Net | Contour masks (6 structures) | MSE | DSC 0.91 (isodose); max/mean dose error <5% Rx | No |
| 2 | Kearney et al. | 2018 | Phys Med Biol | 151 | DoseNet (3D FCN) | CT + contours + distance maps | MAE + perceptual | Superior to U-Net and FC methods for SBRT prostate | No |
| 3 | Norouzi Kandalan et al. | 2020 | Radiother Oncol | 248 | 3D Dense Dilated U-Net | CT + contours + Rx | MSE | DSC 0.87-0.99 (after TL); generalized across 4 styles | Yes (2) |
| 4 | Lempart et al. | 2021 | PHIRO | 160 | Densely Connected U-Net | CT + contours (triplets) | MSE | D95 error 1.0%, D98 1.9%; 100% gamma 3%/3mm pass | No |
| 5 | Adabi et al. | 2022 | Proc SPIE | 140 | SA-Net (Scale Attention) | CT + distance maps | MSE | Avg dose diff 0.94 Gy (2.1% of Rx=45 Gy) | No |
| 6 | Kadoya et al. | 2023 | J Radiat Res | 68 | HD-UNet (AIVOT) | CT + contours | MSE | DVH diff: target 1.32+/-1.35%, OAR 2.08+/-2.79% | No |
| 7 | Fransson et al. | 2024 | Med Phys | 35 (212 fx) | DL pipeline | MR + contours | MSE | D95 error 0.7% Rx, D98 0.7% Rx, D2 1.7% Rx for CTV | No |
| 8 | Church et al. | 2024 | PHIRO | 140 | ResUNet + scripted opt | CT + contours + Rx (SIB) | MSE | D95 0.27+/-0.37%; first fully automated TPS workflow | No |
| 9 | Kontaxis et al. | 2020 | Phys Med Biol | 101 | DeepDose (3D CNN) | Physics-based MLC inputs | MSE | Segment-level dose; passes clinical QA gamma | No |

### 2.2 Multi-Site, Challenge, and Cross-Site Papers

| # | Authors | Year | Journal | N | Architecture | Key Metrics | Site |
|---|---------|------|---------|---|-------------|-------------|------|
| 10 | Babier et al. (OpenKBP) | 2021 | Med Phys | 340 | Challenge (multiple) | Dose 2.26, DVH 1.27 (best) | H&N |
| 11 | Zimmermann et al. | 2021 | Med Phys | 340 | U-Net + ResNet | Dose 2.62, DVH 1.52; DVH diff <1% | H&N |
| 12 | Mashayekhi et al. | 2022 | Med Phys | 70+58 | 3D U-Net (site-agnostic) | DVH loss improves coverage; transfer learning | Multi |
| 13 | nnDoseNet | 2025 | medRxiv | 80+OpenKBP | nnU-Net adapted | D95 met 11/35 vs 3/35 clinical | Prostate+H&N |
| 14 | Hou et al. | 2025 | Radiat Oncol | 622 | U-Net variants | Multi-tumor eval; 62.6% clinically acceptable | 7 types |

### 2.3 Diffusion Models for Dose Prediction

| # | Authors | Year | Venue | Architecture | Site |
|---|---------|------|-------|-------------|------|
| 15 | DiffDP | 2023 | MICCAI | Conditional DDPM | Rectal (130) |
| 16 | DoseDiff (Zhang) | 2024 | IEEE TMI | Distance-aware DDPM + MMFNet | Multi-site |
| 17 | MD-Dose (Fu) | 2024 | arXiv | Mamba-based diffusion | Thoracic (300) |
| 18 | Yu et al. (DMTP) | 2025 | Front Oncol | Score-based diffusion | H&N, chest, abdomen |

### 2.4 Loss Function Engineering Papers

| # | Authors | Year | Loss Innovation | Key Result |
|---|---------|------|----------------|------------|
| 19 | Jhanwar et al. | 2022 | Moment-based loss (DVH surrogate) | DVH score +11% over MAE (p<0.01) |
| 20 | Bai et al. (Sharp loss) | 2021 | Gradient-aware "sharp loss" | Better in high-gradient regions |
| 21 | Zimmermann | 2021 | Feature loss (video classifier) | 2nd in OpenKBP DVH stream |
| 22 | Zhan et al. (Mc-GAN) | 2022 | Locality-constrained + self-supervised | Improved local accuracy |

---

## 3. State-of-the-Art Benchmarks

### 3.1 MAE (Mean Absolute Error)

| Evaluation Region | Best Reported MAE | Source |
|-------------------|-------------------|--------|
| Within PTV only | 0.7-1.0 Gy (1-2% of Rx) | Lempart 2021, Fransson 2024 |
| Within contoured structures | 1.0-2.1 Gy | Adabi 2022 (SA-Net), Kadoya 2023 |
| Full body/volume | 2-5 Gy | Varies with low-dose inclusion |

**Note on comparability:** Most published prostate studies report MAE within specific structures (PTV, OARs) or within dose-relevant regions, not as a whole-body voxel average. This project's MAE of 4.07 Gy is computed over the full volume including low-dose background, making it not directly comparable to structure-specific MAE values. **Reporting structure-specific MAE is essential for fair comparison.**

### 3.2 Gamma Pass Rates

| Criteria | Best Reported | Typical Range |
|----------|--------------|---------------|
| 3%/3mm (PTV region) | 97-100% | 90-99% |
| 3%/3mm (global) | 90-97% | 75-95% |
| 2%/2mm (PTV region) | 85-95% | 80-95% |

### 3.3 DVH Metric Accuracy

| Metric | Best Reported Error | Typical Range |
|--------|-------------------|---------------|
| PTV D95 | 0.27% of Rx (~0.2 Gy) | 0.2-1.0 Gy |
| PTV D98 | 0.7-0.84% of Rx | 0.7-2.0% |
| OAR Dmean | Within 1-3% of Rx | 1-5% |
| OAR DVH params | Within 2.08+/-2.79% | 1-5% |

---

## 4. Key Methodological Trends

### 4.1 Loss Function Engineering

Most dose prediction papers use **MSE or L1 only**. Emerging innovations:

1. **Gradient/sharp losses:** Bai et al. (2021) -- penalizes errors in high-gradient regions
2. **DVH-aware losses:** Moment-based surrogate (Jhanwar 2022, +11%), soft sorting (nnDoseNet 2025)
3. **Feature losses:** Zimmermann (2021) -- pretrained video classifier, 2nd in OpenKBP
4. **Adversarial losses:** GAN-based (DoseGAN, Mc-GAN, DiffDP)
5. **Asymmetric/directional losses:** Penalize PTV under-dosing more than over-dosing -- **not reported in published literature**
6. **Uncertainty-weighted multi-task losses:** Kendall (2018) -- **not applied to dose prediction loss balancing in any published work**

### 4.2 Input Features

| Feature Type | Usage | Status |
|--------------|-------|--------|
| Binary masks | Most common | Standard |
| Distance maps | Growing (SA-Net, DoseDiff) | Emerging |
| **SDFs** | **DoseDiff 2024, this project** | **Emerging best practice** |
| CT images | Nearly universal | Standard |
| **Dose constraints / Rx** | **Rare (Norouzi Kandalan 2020, this project)** | **Novel for FiLM conditioning** |

### 4.3 Multi-Institutional Validation

Maximum: 2 institutions (Norouzi Kandalan 2020). **This project's 2-institution design matches the best published.**

### 4.4 Diffusion Models

No published diffusion model targets prostate VMAT specifically. This project's DDPM attempt is among the first.

---

## 5. Gaps This Project Fills

1. **SIB-specific dose prediction (70/56 Gy)** -- only Church (60/54 Gy) addresses SIB
2. **FiLM constraint conditioning** -- not reported in dose prediction literature
3. **5-component loss + Kendall uncertainty weighting** -- unique combination
4. **Asymmetric PTV loss** -- clinically motivated but unreported
5. **SDF inputs for prostate U-Net** -- aligns with emerging best practice
6. **Prostate DDPM attempt** -- first for this disease site

---

## 6. Dataset Size Comparison

| Study | N | Protocol |
|-------|---|----------|
| Norouzi Kandalan 2020 | 248 | Prostate VMAT (transfer learning) |
| Lempart 2021 | 160 | Prostate VMAT |
| Kearney 2018 | 151 | Prostate SBRT |
| Church 2024 | 140 | Prostate VMAT SIB |
| **This project** | **~70 (expanding to 200)** | **Prostate VMAT SIB** |

200 cases would rank among the largest prostate-specific datasets.

---

## 7. Recommendations for Publication

### Metrics to Report for Comparability
- MAE within each structure (PTV70, PTV56, rectum, bladder, femurs, bowel)
- Gamma at 3%/3mm (primary) and 2%/2mm (secondary), with protocol specified
- DVH: D95, D98, D2, Dmean for PTVs; Dmean, V-at-dose for OARs
- Consider DSC for isodose volumes (Nguyen comparison)

### Additional Experiments to Consider
- 2%/2mm gamma evaluation
- Blinded physician evaluation (Gronberg 2023 protocol)
- Cross-protocol generalization via FiLM conditioning

---

## 8. References

### Prostate-Specific
1. Nguyen D, et al. Sci Reports. 2019;9:1076. DOI: 10.1038/s41598-018-37741-x. PMID: 30705354
2. Kearney V, et al. Phys Med Biol. 2018;63(23):235022. PMID: 30511663
3. Norouzi Kandalan R, Nguyen D, et al. Radiother Oncol. 2020;155:105-113. PMC7908143
4. Lempart M, et al. PHIRO. 2021;19:112-119. PMC8353474
5. Adabi S, Tsen TC, Yuan Y. Proc SPIE. 2022. PMID: 36147747
6. Kadoya N, et al. J Radiat Res. 2023;64(5):842-851. PMID: 37607667
7. Fransson S, et al. Med Phys. 2024. PMID: 39106418
8. Church C, et al. PHIRO. 2024;32:100645. PMID: 39310221
9. Kontaxis C, et al. Phys Med Biol. 2020;65(7):075002. PMID: 32053803

### Challenge and Multi-Site
10. Babier A, et al. Med Phys. 2021;48(9):5549-5561 (OpenKBP). PMID: 34156719
11. Zimmermann L, et al. Med Phys. 2021;48(9):5562-5566. PMID: 34156727
12. Mashayekhi M, et al. Med Phys. 2022;49(3):1391-1406. PMID: 35037276
13. nnDoseNet (Chang HH, et al.). medRxiv. 2025. DOI: 10.1101/2025.03.21.25324413
14. Hou Z, et al. Radiat Oncol. 2025;20:80. DOI: 10.1186/s13014-025-02634-7. PMID: 40390053

### Diffusion Models
15. DiffDP (Feng Z, et al.). MICCAI. 2023. arXiv: 2307.09794
16. Zhang Y, et al. IEEE TMI. 2024;43(10):3621-3633. PMID: 38564344
17. Fu L, et al. arXiv. 2024. arXiv: 2403.08479
18. Yu X, et al. Front Oncol. 2025;14:1473050. DOI: 10.3389/fonc.2024.1473050. PMID: 39830643

### Loss Engineering
19. Jhanwar G, et al. Phys Med Biol. 2022;67(18):185012. PMC9490215
20. Bai X, et al. BioMed Eng OnLine. 2021;20:101. PMC8501531
21. Zhan B, et al. Med Image Anal. 2022;77:102339. DOI: 10.1016/j.media.2021.102339. PMID: 34990905
22. Kendall A, et al. CVPR. 2018. arXiv: 1705.07115

### Automated Plan Generation
23. Heilemann G, et al. Med Phys. 2023. PMID: 37314944
24. Hrinivich WT, et al. Med Phys. 2024. DOI: 10.1002/mp.17100
25. Arberet S, et al. Med Phys. 2025. DOI: 10.1002/mp.17673
26. Shaffer R, et al. Med Phys. 2026. DOI: 10.1002/mp.70306

### Open-Source
27. PortPy: https://github.com/PortPy-Project/PortPy
28. ECHO-VMAT: https://github.com/PortPy-Project/ECHO-VMAT
