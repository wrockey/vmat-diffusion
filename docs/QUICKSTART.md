> **SUPERSEDED (2026-02-17)** — This quickstart guide is from the initial DDPM-focused phase.
> The primary model is now the baseline U-Net with loss-function engineering (not DDPM).
> For current commands and workflow: see `CLAUDE.md` (section "Key Commands").
> For project state: see `.claude/instructions.md`.

# VMAT Dose Prediction: Quickstart Guide

Complete workflow from raw DICOM data to trained model evaluation.

**Time Estimates:**
- Preprocessing: ~5-10 min per case
- Training (diffusion): ~5-10 hours for 25 cases
- Training (baseline): ~3-5 hours for 25 cases
- Inference: ~5-10 min per case (diffusion), ~30 sec per case (baseline)

---

## 1. Prerequisites

### 1.1 Environment Setup

```bash
# Create conda environment
conda create -n vmat python=3.10
conda activate vmat

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install pytorch-lightning>=2.0
pip install numpy scipy scikit-image
pip install pydicom rt-utils
pip install pymedphys  # For gamma evaluation
pip install tensorboard rich
pip install jupyter matplotlib

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 1.2 Directory Structure

```
vmat-diffusion-project/
├── scripts/                      # Python scripts
│   ├── preprocess_dicom_rt_v2.2.py
│   ├── generate_oar_mapping.py
│   ├── train_dose_ddpm_v2.py
│   ├── inference_dose_ddpm.py
│   ├── train_baseline_unet.py
│   └── inference_baseline_unet.py
│
├── oar_mapping.json              # Structure name variations (site-specific)
├── data/
│   └── raw/                      # Raw DICOM-RT data
│       ├── case_0001/
│       │   ├── CT*.dcm
│       │   ├── RS*.dcm           # Structure set
│       │   ├── RD*.dcm           # Dose
│       │   └── RP*.dcm           # Plan
│       ├── case_0002/
│       └── ...
│
├── processed/                    # Preprocessed .npz files
│   ├── case_0001.npz
│   ├── case_0002.npz
│   └── ...
│
├── runs/                         # Training outputs (auto-created)
├── predictions/                  # Inference outputs (auto-created)
├── figures/                      # Generated figures (auto-created)
└── notebooks/                    # Jupyter notebooks
    ├── verify_npz.ipynb
    └── analyze_results.ipynb
```

```bash
# Create directories
mkdir -p data/raw processed runs predictions figures notebooks
```

**Note:** Scripts auto-detect data locations. You can also use symlinks or specify custom paths via command-line arguments if your data is stored elsewhere.

---

## 2. Preprocessing

### 2.1 Setup Structure Name Mapping

Before preprocessing, create the structure name mapping file. This maps your site's structure naming conventions to the standard channels.

```bash
cd vmat-project

# Option 1: Scan your data to discover structure names
python scripts/generate_oar_mapping.py --input_dir ./data/raw --scan

# Option 2: Generate mapping file from your data
python scripts/generate_oar_mapping.py --input_dir ./data/raw --output oar_mapping.json

# Option 3: Use the provided template and edit manually
cp scripts/oar_mapping.json ./oar_mapping.json
# Then edit to add your site's naming variations
```

The script will classify structures automatically and flag unclassified ones for manual review. Common variations like `PTV70_Oct11` or `Rectum_final` are handled by normalization.

**oar_mapping.json structure:**
```json
{
  "0": {"name": "PTV70", "variations": ["PTV70", "PTV_70", "PTV70Gy", ...]},
  "1": {"name": "PTV56", "variations": ["PTV56", "PTV_56", ...]},
  "3": {"name": "Rectum", "variations": ["Rectum", "RECTUM", "Rectal_Wall", ...]},
  ...
}
```

### 2.2 Run Preprocessing

```bash
# Process all cases (auto-detects ./data/raw and ./processed)
python scripts/preprocess_dicom_rt_v2.2.py \
    --mapping_file ./oar_mapping.json

# Or specify paths explicitly
python scripts/preprocess_dicom_rt_v2.2.py \
    --input_dir ./data/raw \
    --output_dir ./processed \
    --mapping_file ./oar_mapping.json
```

### 2.3 Expected Output

```
./processed/
├── case_0001.npz      # ~400-500 MB each (compressed)
├── case_0002.npz
├── ...
└── batch_summary.json
```

### 2.4 Sanity Check: Verify Preprocessing

```bash
# Quick check of one file
python -c "
import numpy as np
data = np.load('./processed/case_0001.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('CT shape:', data['ct'].shape)           # Should be (512, 512, 256)
print('Dose shape:', data['dose'].shape)       # Should be (512, 512, 256)
print('Masks shape:', data['masks'].shape)     # Should be (8, 512, 512, 256)
print('Masks SDF shape:', data['masks_sdf'].shape)  # Should be (8, 512, 512, 256)
print('Constraints shape:', data['constraints'].shape)  # Should be (13,)
print('Dose range:', data['dose'].min(), '-', data['dose'].max())  # Should be ~0-1.1
print('CT range:', data['ct'].min(), '-', data['ct'].max())  # Should be ~0-1
"
```

**Expected output:**
```
Keys: ['ct', 'dose', 'masks', 'masks_sdf', 'constraints', 'metadata', ...]
CT shape: (512, 512, 256)
Dose shape: (512, 512, 256)
Masks shape: (8, 512, 512, 256)
Masks SDF shape: (8, 512, 512, 256)
Constraints shape: (13,)
Dose range: 0.0 - 1.05
CT range: 0.0 - 1.0
```

### 2.5 Visual Verification (Jupyter)

Open the verification notebook:

```bash
jupyter notebook notebooks/verify_npz.ipynb
```

Or use the quick verification script:

```python
# Quick inline check (or use verify_npz.ipynb for full visualization)
import numpy as np
import matplotlib.pyplot as plt

data = np.load('./processed/case_0001.npz', allow_pickle=True)
z = data['ct'].shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(data['ct'][:,:,z], cmap='gray')
axes[0].set_title('CT')
axes[1].imshow(data['dose'][:,:,z], cmap='jet')
axes[1].set_title('Dose')
axes[2].imshow(data['masks'][0,:,:,z], cmap='Reds')
axes[2].set_title('PTV70')
plt.tight_layout()
plt.show()
```

**What to check:**
- CT shows anatomy (prostate visible)
- Dose is highest in PTV, falls off around it
- PTV mask covers prostate region

**For detailed verification:** See `verify_npz.ipynb` which includes:
- All structure visualization
- SDF verification
- Beam geometry display
- Constraint vector inspection

---

## 3. Training

### 3.1 Start TensorBoard (Recommended)

Open a separate terminal:

```bash
cd vmat-project
tensorboard --logdir ./runs --port 6006

# Open in browser: http://localhost:6006
```

Keep this running during training to monitor progress.

### 3.2 Train Diffusion Model

Training scripts auto-detect the data directory (same logic as preprocessing).

```bash
# Basic training (RTX 3090, 24GB) - auto-detects data path
python scripts/train_dose_ddpm_v2.py \
    --epochs 200 \
    --batch_size 2 \
    --patch_size 128 \
    --base_channels 48 \
    --exp_name vmat_ddpm_v1

# Or specify data path explicitly
python scripts/train_dose_ddpm_v2.py \
    --data_dir /path/to/processed \
    --epochs 200

# For A100 (40GB+) - can use larger batch
python scripts/train_dose_ddpm_v2.py \
    --epochs 200 \
    --batch_size 4 \
    --base_channels 64 \
    --exp_name vmat_ddpm_v1
```

### 3.3 Train Baseline Model (For Comparison)

```bash
# Same data, same seed = same split (auto-detects data path)
python scripts/train_baseline_unet.py \
    --epochs 200 \
    --batch_size 2 \
    --exp_name baseline_v1
```

### 3.4 Expected Training Output

```
runs/
├── vmat_ddpm_v1/
│   ├── checkpoints/
│   │   ├── best-epoch=150-val_mae_gy=1.850.ckpt
│   │   ├── last.ckpt
│   │   └── ...
│   ├── training_config.json
│   ├── training_summary.json
│   ├── epoch_metrics.csv
│   ├── test_cases.json          # Held-out test cases
│   └── version_0/               # TensorBoard logs
│       └── events.out.tfevents.*
│
└── baseline_v1/
    ├── checkpoints/
    └── ...
```

### 3.5 TensorBoard Monitoring

In browser at `http://localhost:6006`, watch these metrics:

| Metric | Good Sign | Warning Sign |
|--------|-----------|--------------|
| `train/loss` | Decreasing smoothly | Stuck, NaN, or spiky |
| `val/loss` | Decreasing, close to train | Much higher than train (overfitting) |
| `val/mae_gy` | Decreasing toward <2.0 | Stuck above 5.0 |
| `val/gamma_3mm3pct` | Increasing toward >95% | Stuck below 80% |

**Expected Learning Curve:**

```
Epoch   train/loss   val/mae_gy   val/gamma   Status
─────────────────────────────────────────────────────
  1       0.15         8.5         60%        Starting
 25       0.06         4.5         75%        Learning
 50       0.04         3.0         85%        Improving  
100       0.025        2.2         92%        Good
150       0.020        1.8         95%        Target met ✓
200       0.018        1.7         96%        Converged
```

### 3.6 Sanity Check: During Training

```bash
# Check GPU memory usage
nvidia-smi

# Expected: 18-22 GB used for batch_size=2, patch_size=128
```

```bash
# Check training is progressing (look at recent logs)
tail -f runs/vmat_ddpm_v1/version_0/metrics.csv
```

---

## 4. Inference & Evaluation

### 4.1 Run Inference on Test Set

First, identify test cases:

```bash
cat runs/vmat_ddpm_v1/test_cases.json
# Shows which cases were held out for testing
```

```bash
# Create test directory with only test cases
mkdir -p test_npz
# Copy test cases (example - adjust based on test_cases.json)
cp processed/case_0010.npz test_npz/
cp processed/case_0025.npz test_npz/
```

Run inference:

```bash
# Diffusion model
python scripts/inference_dose_ddpm.py \
    --checkpoint ./runs/vmat_ddpm_v1/checkpoints/best*.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions/ddpm \
    --compute_metrics \
    --gamma_subsample 2

# Baseline model
python scripts/inference_baseline_unet.py \
    --checkpoint ./runs/baseline_v1/checkpoints/best*.ckpt \
    --input_dir ./test_npz \
    --output_dir ./predictions/baseline \
    --compute_metrics
```

### 4.2 Expected Inference Output

```
predictions/
├── ddpm/
│   ├── case_0010_pred.npz
│   ├── case_0025_pred.npz
│   └── evaluation_results.json    # All metrics
│
└── baseline/
    ├── case_0010_pred.npz
    ├── case_0025_pred.npz
    └── baseline_evaluation_results.json
```

### 4.3 Review Results

```bash
# View summary
cat predictions/ddpm/evaluation_results.json | python -m json.tool | head -50
```

**Expected output:**
```json
{
  "n_cases": 2,
  "aggregate_metrics": {
    "mae_gy_mean": 1.85,
    "mae_gy_std": 0.32,
    "gamma_pass_rate_mean": 95.2,
    "gamma_pass_rate_std": 1.8
  },
  "clinical_constraints": {
    "cases_all_passed": 2,
    "total_violations": 0
  },
  "goal_assessment": {
    "mae_goal_met": true,
    "gamma_goal_met": true
  }
}
```

### 4.4 Compare Models

For detailed model comparison, use the analysis notebook (Section 5).

Quick command-line comparison:

```bash
python -c "
import json
with open('predictions/ddpm/evaluation_results.json') as f: d = json.load(f)
with open('predictions/baseline/baseline_evaluation_results.json') as f: b = json.load(f)
print(f'DDPM MAE: {d[\"aggregate_metrics\"][\"mae_gy_mean\"]:.2f} ± {d[\"aggregate_metrics\"][\"mae_gy_std\"]:.2f} Gy')
print(f'Base MAE: {b[\"aggregate_metrics\"][\"mae_gy_mean\"]:.2f} ± {b[\"aggregate_metrics\"][\"mae_gy_std\"]:.2f} Gy')
"
```

---

## 5. Visualization & Analysis

### 5.1 Open Analysis Notebook

The `analyze_results.ipynb` notebook provides comprehensive visualization:

```bash
jupyter notebook notebooks/analyze_results.ipynb
```

### 5.2 What the Notebook Includes

| Section | Description |
|---------|-------------|
| Model Comparison | Side-by-side metrics table (DDPM vs Baseline) |
| Per-Case Metrics | Bar charts of MAE and Gamma by case |
| Dose Visualization | GT vs Pred vs Difference maps (axial, sagittal, coronal) |
| DVH Analysis | DVH curves and metrics tables per structure |
| Clinical Constraints | Pass/fail summary and violation breakdown |
| Error Analysis | Error histograms and statistics |
| Publication Figures | Combined multi-panel figures (300 DPI) |
| Batch Export | Generate all figures for all cases |

### 5.3 Quick Command-Line Check

```bash
# View aggregate results
cat predictions/ddpm/evaluation_results.json | python -m json.tool | head -30

# Compare models
python -c "
import json
with open('predictions/ddpm/evaluation_results.json') as f: ddpm = json.load(f)
with open('predictions/baseline/baseline_evaluation_results.json') as f: base = json.load(f)
print(f'DDPM MAE: {ddpm[\"aggregate_metrics\"][\"mae_gy_mean\"]:.2f} Gy')
print(f'Baseline MAE: {base[\"aggregate_metrics\"][\"mae_gy_mean\"]:.2f} Gy')
"
```

### 5.4 Generated Figures

After running the notebook, figures are saved to `./figures/`:

```
figures/
├── per_case_metrics.png        # Bar charts
├── clinical_constraints.png    # Pass/fail pie chart
├── case_0001_dose.png        # Dose comparison
├── case_0001_dvh.png         # DVH curves
├── case_0001_publication.png # Combined figure
├── case_0001_publication.pdf # Vector format
└── summary_report.md           # Markdown summary
```

---

## 6. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| OOM during training | GPU memory exceeded | Reduce `--batch_size` to 1 or `--base_channels` to 32 |
| NaN loss | Gradient explosion | Reduce `--lr` to 1e-5, check data for NaN |
| val_loss much higher than train_loss | Overfitting | Add more data, reduce epochs, increase augmentation |
| MAE stuck > 5 Gy | Model not learning | Check data loading, increase model capacity |
| Preprocessing fails | Missing DICOM files | Ensure CT, RS, RD, RP all present |
| No structures found | Naming mismatch | Check structure names match expected (PTV, Rectum, etc.) |

### Debug Commands

```bash
# Check CUDA errors
CUDA_LAUNCH_BLOCKING=1 python scripts/train_dose_ddpm_v2.py ...

# Verbose preprocessing
python scripts/preprocess_dicom_rt_v2.2.py --input_dir ... --verbose

# Check for NaN in data
python -c "
import numpy as np
from pathlib import Path
for f in Path('./processed').glob('*.npz'):
    data = np.load(f)
    for key in ['ct', 'dose', 'masks', 'masks_sdf']:
        if np.isnan(data[key]).any():
            print(f'{f.name}: NaN in {key}')
"
```

---

## 7. Quick Reference

### File Sizes

| File Type | Expected Size |
|-----------|---------------|
| Raw DICOM (per patient) | 500 MB - 2 GB |
| Processed .npz | 200 - 400 MB |
| Model checkpoint | 100 - 200 MB |
| Prediction .npz | 50 - 100 MB |

### Training Time Estimates

| GPU | Cases | Epochs | Time |
|-----|-------|--------|------|
| RTX 3090 | 25 | 200 | ~6 hours |
| RTX 3090 | 100 | 200 | ~20 hours |
| A100 | 25 | 200 | ~3 hours |
| A100 | 100 | 200 | ~10 hours |

### Key Metrics to Report

| Metric | Target | How to Get |
|--------|--------|------------|
| MAE (Gy) | < 2.0 | `evaluation_results.json` |
| Gamma 3%/3mm (%) | > 95% | `evaluation_results.json` |
| PTV D95 error (Gy) | < 1.0 | `dvh_metrics` in results |
| Clinical constraint pass rate | 100% | `clinical_constraints` in results |

---

## 8. Next Steps

After successful Phase 1:

1. **Analyze failures**: Which cases have high error? Why? (Use `analyze_results.ipynb`)
2. **Uncertainty estimation**: Run inference multiple times, compute std
3. **Phase 2 preparation**: MLC prediction from dose
4. **Multi-site validation**: Test on external data

---

## Summary Checklist

- [ ] Environment set up with PyTorch + CUDA
- [ ] Raw DICOM data organized by patient
- [ ] Preprocessing complete, .npz files verified (`verify_npz.ipynb`)
- [ ] TensorBoard running for monitoring
- [ ] Diffusion model trained, val_mae < 2 Gy
- [ ] Baseline model trained for comparison
- [ ] Inference run on held-out test set
- [ ] Results reviewed: MAE, Gamma, DVH, clinical constraints (`analyze_results.ipynb`)
- [ ] Visualizations generated for publication (`figures/` directory)
