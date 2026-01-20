# Scientific Best Practices for VMAT Diffusion Project

This document outlines best practices for maintaining scientific rigor, reproducibility, and publication-readiness throughout the project.

---

## 1. Reproducibility

### 1.1 Version Control
- **Always commit before experiments**: Record git hash in experiment notebooks
- **Tag significant milestones**: `git tag -a v1.0-baseline -m "Baseline U-Net complete"`
- **Never modify committed experiment code**: Create new versions instead

### 1.2 Random Seeds
```python
# Always set seeds for reproducibility
import random
import numpy as np
import torch
import pytorch_lightning as pl

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)
```

### 1.3 Environment Documentation
For each experiment, record:
- Python version
- PyTorch version
- CUDA version
- Key package versions (pytorch-lightning, numpy, scipy)
- GPU model and memory

Consider using `pip freeze > requirements_experiment.txt` for each major experiment.

### 1.4 Data Versioning
- Record preprocessing script version in metadata
- Keep preprocessing parameters consistent across experiments
- Document any data cleaning or filtering applied

---

## 2. Experiment Documentation

### 2.1 Notebook Structure
Each experiment notebook should include:

1. **Header**: Date, experiment ID, status, key results
2. **Objective**: What are we testing/learning?
3. **Reproducibility Info**: Git hash, environment, seeds
4. **Dataset**: Split, preprocessing, statistics
5. **Model**: Architecture, parameters, config
6. **Training**: Hyperparameters, loss functions
7. **Results**: Metrics, curves, analysis
8. **Conclusions**: What did we learn?
9. **Next Steps**: What should follow?

### 2.2 Logging During Training
Automatically log:
- Training/validation loss per epoch
- Validation metrics (MAE, gamma)
- Learning rate schedule
- Sample predictions periodically
- GPU memory usage

### 2.3 Checkpoint Management
- Save best model by validation metric
- Save periodic checkpoints for debugging
- Include hyperparameters in checkpoint
- Clear naming: `best-epoch=012-val_mae=3.73.ckpt`

---

## 3. Statistical Rigor

### 3.1 Dataset Splits
- Use consistent random seeds for splits
- Report split sizes and rationale
- Consider k-fold cross-validation for small datasets
- **Never tune on test set** - use validation only

### 3.2 Multiple Runs
For publication, consider:
- 3-5 runs with different seeds
- Report mean ± std of metrics
- Statistical significance tests (paired t-test, Wilcoxon)

### 3.3 Confidence Intervals
```python
import scipy.stats as stats

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h
```

---

## 4. Figures for Publication

### 4.1 Style Guidelines
```python
import matplotlib.pyplot as plt

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
```

### 4.2 Color Schemes
Use colorblind-friendly palettes:
```python
# Recommended colors
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
}

# Or use seaborn colorblind palette
import seaborn as sns
sns.set_palette('colorblind')
```

### 4.3 Figure Types Needed

1. **Training curves**: Loss and metrics vs epoch
2. **DVH comparison**: Predicted vs ground truth DVH
3. **Dose distribution**: Axial/coronal/sagittal slices
4. **Gamma maps**: Pass/fail visualization
5. **Box plots**: Metric distributions across test cases
6. **Architecture diagram**: Model structure

### 4.4 Saving Figures
```python
# Save in multiple formats
fig.savefig('figure.png', dpi=300, bbox_inches='tight')
fig.savefig('figure.pdf', bbox_inches='tight')  # Vector for publication
fig.savefig('figure.svg', bbox_inches='tight')  # Editable vector
```

---

## 5. Medical Physics Metrics

### 5.1 Dose Accuracy
- **MAE (Mean Absolute Error)**: Overall dose accuracy
- **Structure-specific MAE**: Per-OAR accuracy
- **Dmax error**: Maximum dose point accuracy

### 5.2 DVH Metrics
```python
def compute_dvh_metrics(dose, mask, rx_dose):
    """Compute clinically relevant DVH metrics."""
    struct_dose = dose[mask > 0]
    metrics = {
        'D2': np.percentile(struct_dose, 98),   # Near-max dose
        'D50': np.percentile(struct_dose, 50),  # Median dose
        'D95': np.percentile(struct_dose, 5),   # Coverage
        'D98': np.percentile(struct_dose, 2),   # Near-min dose
        'Dmean': np.mean(struct_dose),
        'Dmax': np.max(struct_dose),
        'V95': np.mean(struct_dose >= 0.95 * rx_dose) * 100,  # % receiving 95%
    }
    return metrics
```

### 5.3 Gamma Analysis
```python
# Using pymedphys
import pymedphys

gamma = pymedphys.gamma(
    axes_reference=(y_coords, x_coords, z_coords),
    dose_reference=dose_gt,
    axes_evaluation=(y_coords, x_coords, z_coords),
    dose_evaluation=dose_pred,
    dose_percent_threshold=3,  # 3% dose criterion
    distance_mm_threshold=3,   # 3mm DTA criterion
    lower_percent_dose_cutoff=10,  # Ignore < 10% of max dose
)

pass_rate = np.mean(gamma[np.isfinite(gamma)] <= 1) * 100
```

---

## 6. Negative Results

### 6.1 Document Failures
- Record experiments that didn't work
- Note why they failed
- Include in supplementary materials

### 6.2 Ablation Studies
Show what components matter:
- Model without SDF features
- Model without constraint conditioning
- Smaller model capacity
- Different loss functions

---

## 7. Publication Checklist

### Before Submission
- [ ] All experiments reproducible (git hash, seeds documented)
- [ ] Multiple runs with statistics
- [ ] Ablation studies complete
- [ ] Comparison with baselines
- [ ] Test set evaluation (never used for tuning)
- [ ] High-resolution figures (300 DPI minimum)
- [ ] Code repository clean and documented
- [ ] README with reproduction instructions

### Supplementary Materials
- [ ] Full hyperparameter tables
- [ ] Additional visualizations
- [ ] Per-case results
- [ ] Failure analysis
- [ ] Code availability statement

---

## 8. File Organization

**Primary Location:** `C:\Users\Bill\vmat-diffusion-project` (Windows)

See `docs/EXPERIMENT_STRUCTURE.md` for detailed structure.

```
vmat-diffusion-project/
├── notebooks/
│   ├── EXPERIMENTS_INDEX.md          # MASTER experiment log (single source of truth)
│   ├── TEMPLATE_experiment.ipynb     # Copy for new experiments
│   └── YYYY-MM-DD_<experiment>.ipynb # One per experiment
├── runs/                             # Training outputs
│   └── <experiment_name>/
│       ├── checkpoints/
│       ├── training_config.json
│       ├── training_summary.json
│       └── epoch_metrics.csv
├── experiments/                      # Optimization experiment outputs
│   └── phase<N>_<name>/
├── predictions/                      # Inference outputs
├── docs/
│   ├── DDPM_OPTIMIZATION_PLAN.md     # Current optimization focus
│   ├── EXPERIMENT_STRUCTURE.md       # Organization guidelines
│   └── SCIENTIFIC_BEST_PRACTICES.md  # This file
└── scripts/                          # Training/inference code
```

---

*Document created: 2026-01-19*
*Last updated: 2026-01-20 (Updated file organization to match EXPERIMENT_STRUCTURE.md)*
