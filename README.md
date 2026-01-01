# VMAT Diffusion Project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Project Overview

This repository implements a generative AI model using diffusion techniques to create deliverable Volumetric Modulated Arc Therapy (VMAT) plans for radiation therapy. The core idea frames VMAT planning as a generative task analogous to AI image or video generation: Given a contoured CT dataset and organs-at-risk (OAR) dose constraints, the model samples a deliverable plan (dose distribution, arcs, MLC positions, dose rates) that meets clinical objectives while respecting linac physics.

### Key Goals
1. **Define the Problem and Objectives**  
   Frame as a Generative Task: In VMAT, the goal is to generate a plan that delivers prescribed doses to PTVs while minimizing OAR exposure, respecting linac physics (e.g., beam modulation limits). A diffusion model would learn the distribution of high-quality plans from data, then sample new ones conditionally.  
   Analogous to image generation: CT scan + contours + dose constraints = "prompt"; output = dose distribution + beam parameters (like a generated image).  
   Key Outputs: 3D dose volume (e.g., in DICOM RT format), arc angles, gantry speeds, MLC sequences, and fluence maps.  
   Incorporate Physics: Integrate a beam model (e.g., from Pinnacle) to ensure deliverability, perhaps via a hybrid approach where the diffusion model predicts initial doses, then refines with Monte Carlo simulation approximations.  
   Metrics for Success: Dose-volume histograms (DVHs), gamma pass rates (>95% at 3%/3mm), conformity index, and clinical acceptability (e.g., OAR doses below constraints like <20 Gy mean to lungs).

2. **Gather and Prepare Data**  
   Dataset Requirements: Need paired data: CT scans (HU values), segmented contours (PTV/OAR masks), dose objectives (e.g., 70 Gy to PTV, <45 Gy max to spinal cord), ground-truth dose distributions, and VMAT parameters (from optimized plans).  
   Size: Aim for 1,000–10,000 cases initially; diffusion models thrive on large datasets.  
   Sources: Public repositories like TCIA (The Cancer Imaging Archive), AAPM Grand Challenges (e.g., for head-and-neck or prostate VMAT), or institutional IRB-approved data. Augment with synthetic data generated via TPS simulations.  
   Preprocessing:  
   - Resample CTs to uniform voxel size (e.g., 1x1x3 mm).  
   - Create multi-channel inputs: Stack CT slices, binary masks for PTVs/OARs, and signed distance maps (SDFs) to encode anatomical distances—useful for conditioning, as seen in distance-aware models.  
   - Normalize doses (e.g., to [0,1] range) and handle class imbalances (e.g., high doses rare).  
   Split: 80% train, 10% validation, 10% test, stratified by cancer site (e.g., prostate, lung) for generalizability.

3. **Design the Model Architecture**  
   Base on Denoising Diffusion Probabilistic Models (DDPM): Use a conditional diffusion framework, like in DiffDP or DoseDiff, where the model learns to reverse a noise-adding process.  
   Forward Process: Gradually add Gaussian noise to ground-truth dose distributions over T timesteps (e.g., T=1000), turning them into pure noise. This is fixed and doesn't require training.  
   Reverse Process: Train a neural network (e.g., U-Net variant) to predict and remove noise at each step, conditioned on inputs. The network outputs denoised dose maps.  
   Conditioning Mechanism: Embed CT, contours, and constraints into the model.  
   - Use a multi-scale fusion encoder (e.g., CNN or Transformer) to process anatomy.  
   - For VMAT specifics: Condition on beam fields or initial arc setups, as in beam-guided models. Optionally, extend to predict MLC sequences by treating them as additional channels.  
   Advanced: Incorporate Mamba architecture for efficient long-range dependencies in 3D volumes. Or use score-based diffusion for therapeutic dose prediction.  
   Hybrid Extensions: To ensure physical deliverability, couple with a differentiable TPS simulator (e.g., approximate Monte Carlo via neural surrogates) during reverse steps, penalizing undeliverable plans.  
   Efficiency Tweaks: Use latent diffusion (compress data to lower dimensions) to reduce compute, similar to image gen optimizations.

4. **Train the Model**  
   Hardware: Multi-GPU setup (e.g., 4x A100s) for 3D volumes; training could take days to weeks. Initially, establish feasibility on NVIDIA 3090 with ~100 cases, then scale to UIowa Argon HPC for 1000+.  
   Loss Function: Standard diffusion loss (e.g., MSE between predicted and actual noise), plus radiotherapy-specific terms like DVH penalties or conformity losses to enforce clinical constraints.  
   Regularization: Add physics-informed losses (e.g., beam energy conservation) to avoid hallucinations.  
   Hyperparameters: Learning rate 1e-4, batch size 8–16 (memory-limited), scheduler like cosine annealing.  
   Training Loop: For each batch, sample a timestep t, add noise to doses, feed conditioned inputs to the network, and optimize.  
   Augmentations: Random rotations/flips of CTs to improve robustness.  
   Monitor: Validate on held-out data with metrics like mean absolute error (MAE) on doses (<2 Gy).  
   Iterative Refinement: Fine-tune on site-specific data (e.g., prostate VMAT) after general training.

5. **Inference and Post-Processing**  
   Generation: Start from noise, iteratively denoise over T steps (or fewer with sampling tricks like DDIM for speed). Condition on new patient CT/contours/constraints to output a dose and plan.  
   Time: Aim for <5 minutes per plan, vs. hours for manual Monte Carlo.  
   Make Deliverable: Post-optimize the predicted plan in a TPS (e.g., Pinnacle) to refine MLC/gantry parameters, ensuring linac compatibility.  
   Safety Checks: Run Monte Carlo validation on generated plans; reject if gamma fails.

6. **Validate and Deploy**  
   Evaluation: Retrospective testing on unseen patients; compare to expert plans via blind clinician reviews. Use metrics like OAR sparing improvement (e.g., 10–20% dose reduction).  
   Challenges to Address:  
   - Generalization: Handle anatomical variations; use diverse data.  
   - Compute: Diffusion inference is slow—optimize with distillation or fewer steps.  
   - Regulatory: FDA clearance needed; start with research prototypes, validate against standards like TG-119.  
   - Ethics: Ensure no bias in datasets; prioritize patient safety.  
   Deployment: Integrate into TPS software as a "one-click" tool, with human oversight. Ultimate aim: Scientific publications (e.g., MedPhys) and clinically useful product.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/vmat-diffusion-project.git
cd vmat-diffusion-project
```

Install dependencies (Python 3.8+):
```bash
pip install -r requirements.txt
```
Requirements include: pydicom, SimpleITK, numpy, scipy, scikit-image, matplotlib, pymedphys (for gamma/DVH), torch (for diffusion models).

On UIowa Argon HPC: Use modules (e.g., `module load python/3.10`, `module load cuda/11.8`) and virtualenv. No internet access needed post-install; runs offline.

## Directory Structure
```
vmat-diffusion-project/
├── README.md               # This file
├── requirements.txt        # Dependencies
├── scripts/                # Core scripts
│   └── preprocess_dicom_rt.py  # DICOM-RT to .npz preprocessing
├── data/                   # Data directories (gitignore large files)
│   ├── raw/                # Anonymized DICOM-RT (case_xxxx/CT/RS/RD/RP.dcm)
│   └── processed_npz/      # Output .npz + debug PNGs
├── oar_mapping.json        # Contour variations mapping
├── docs/                   # Detailed documentation
│   └── preprocess.md       # Preprocessing guide
├── notebooks/              # Jupyter for verification/exploration
│   └── verify_npz.ipynb    # Sample verification script
├── models/                 # Future: DDPM models/checkpoints
└── .gitignore              # Ignore data, .npz, etc.
```

## Quick Start
1. Prepare data: Place anonymized DICOM-RT in `data/raw/case_xxxx`.
2. Edit `oar_mapping.json` for contour variations (e.g., PTV70, rectum).
3. Preprocess:
   ```bash
   python scripts/preprocess_dicom_rt.py --relax_filter
   ```
4. Verify outputs in `data/processed_npz` (e.g., via notebooks/verify_npz.ipynb).
5. Train DDPM (forthcoming): Use processed .npz for conditional training on 3090, scale to Argon.

## Scripts
### preprocess_dicom_rt.py
See [docs/preprocess.md](docs/preprocess.md) for details. In brief: Batch-converts DICOM-RT to fixed-shape .npz for DDPM input, handling variable grids via SimpleITK.

## Contributing
- Use GitHub issues for bugs/to-dos (label "enhancement" for features).
- Pull requests: Follow PEP8; include tests.
- For scaling to Argon: Submit SLURM jobs (example script in docs/argon_slurm.md, forthcoming).

## To-Do / Roadmap
- Implement DDPM trainer (Goal 4 feasibility on 3090).
- Add pymedphys integration for DVH/gamma in validation.
- Hybrid refinement script (Monte Carlo surrogates, Goal 3).
- SLURM templates for Argon (1000+ cases).
- Unit tests (pytest) for preprocessing.
- Publications: Retrospective validation paper (e.g., MedPhys submission).

See [issues](https://github.com/yourusername/vmat-diffusion-project/issues) for full roadmap.

## License
MIT License. For research purposes; clinical deployment requires FDA/IRB validation. Contact: your.email@example.com.

## Acknowledgments
Inspired by DoseDiff/SegDiff papers. Tools: pydicom, SimpleITK, pymedphys. Data sources: TCIA/AAPM (cite appropriately).