"""
Baseline Direct Regression U-Net for VMAT Dose Prediction

This script trains a direct regression model (no diffusion) for comparison
with the DDPM model. Same architecture, but:
- No time embedding (no diffusion process)
- Directly predicts dose from CT + SDFs + constraints
- Single forward pass inference (~100x faster than diffusion)

Purpose: Answer "Is diffusion actually helping?" by providing a baseline.

Version: 1.0
Compatible with: preprocess_dicom_rt_v2.2.py output

Usage:
    python train_baseline_unet.py --data_dir ./processed_npz --epochs 200

After training, evaluate with:
    python inference_baseline_unet.py --checkpoint ./runs/baseline_unet/best.ckpt ...

Or use the shared evaluation functions from inference_dose_ddpm.py
"""

import os
import sys
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping,
    RichProgressBar, Callback
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Optional: pymedphys for gamma evaluation
try:
    from pymedphys import gamma as pymedphys_gamma
    HAS_PYMEDPHYS = True
except ImportError:
    HAS_PYMEDPHYS = False
    print("Warning: pymedphys not installed. Gamma evaluation disabled.")


# =============================================================================
# Constants — imported from centralized evaluation framework
# =============================================================================

from eval_core import STRUCTURE_CHANNELS, DEFAULT_SPACING_MM, get_spacing_from_metadata

SCRIPT_VERSION = "1.1.0"


# =============================================================================
# Dataset (reused from diffusion script)
# =============================================================================

class VMATDosePatchDataset(Dataset):
    """
    Dataset that extracts random 3D patches from VMAT dose data.
    Same as diffusion version but returns condition and dose only (no timestep needed).
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 128,
        patches_per_volume: int = 4,
        augment: bool = True,
        mode: str = 'train',
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment and (mode == 'train')
        self.mode = mode
        
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.files) * self.patches_per_volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.patches_per_volume

        # Load data
        data = np.load(self.files[file_idx], allow_pickle=True)

        ct = data['ct'].astype(np.float32)
        dose = data['dose'].astype(np.float32)
        masks_sdf = data['masks_sdf'].astype(np.float32)
        constraints = data['constraints'].astype(np.float32)

        # Read spacing from metadata (v2.3+) with fallback
        metadata = data['metadata'].item() if 'metadata' in data.files else {}
        spacing = get_spacing_from_metadata(metadata)

        # Extract patch
        center = self._sample_patch_center(dose)
        ct_patch, dose_patch, sdf_patch = self._extract_patch(ct, dose, masks_sdf, center)

        # Augment
        if self.augment:
            ct_patch, dose_patch, sdf_patch = self._augment(ct_patch, dose_patch, sdf_patch)

        # Build condition: CT + SDFs
        condition = np.concatenate([ct_patch[np.newaxis], sdf_patch], axis=0)

        return {
            'condition': torch.from_numpy(condition),
            'dose': torch.from_numpy(dose_patch[np.newaxis]),
            'constraints': torch.from_numpy(constraints),
            'spacing': torch.tensor(spacing, dtype=torch.float32),
        }
    
    def _sample_patch_center(self, dose: np.ndarray) -> Tuple[int, int, int]:
        """Sample patch center, biased toward high-dose regions."""
        ps = self.patch_size
        half = ps // 2

        # Valid range for center — clamp when volume dim <= patch_size
        y_range = (half, max(half + 1, dose.shape[0] - half))
        x_range = (half, max(half + 1, dose.shape[1] - half))
        z_range = (half, max(half + 1, dose.shape[2] - half))

        # 50% chance: sample from high-dose region
        if np.random.rand() < 0.5:
            high_dose_mask = dose > 0.1
            candidates = np.where(high_dose_mask)

            if len(candidates[0]) > 0:
                valid_mask = (
                    (candidates[0] >= y_range[0]) & (candidates[0] < y_range[1]) &
                    (candidates[1] >= x_range[0]) & (candidates[1] < x_range[1]) &
                    (candidates[2] >= z_range[0]) & (candidates[2] < z_range[1])
                )
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) > 0:
                    idx = np.random.choice(valid_indices)
                    return (candidates[0][idx], candidates[1][idx], candidates[2][idx])

        # Random sampling
        y = np.random.randint(y_range[0], y_range[1])
        x = np.random.randint(x_range[0], x_range[1])
        z = np.random.randint(z_range[0], z_range[1])

        return (y, x, z)
    
    def _extract_patch(
        self, ct: np.ndarray, dose: np.ndarray, sdf: np.ndarray, center: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract a patch centered at the given location."""
        ps = self.patch_size
        half = ps // 2
        y, x, z = center
        
        ct_patch = ct[y-half:y+half, x-half:x+half, z-half:z+half]
        dose_patch = dose[y-half:y+half, x-half:x+half, z-half:z+half]
        sdf_patch = sdf[:, y-half:y+half, x-half:x+half, z-half:z+half]
        
        return ct_patch, dose_patch, sdf_patch
    
    def _augment(
        self, ct: np.ndarray, dose: np.ndarray, sdf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentations.

        Augmentations applied:
        1. Left-right (X-axis) flip (50% probability)
           - Valid for prostate due to bilateral symmetry
           - Swaps Femur_L/Femur_R SDF channels

        2. Random translation ±16 voxels (50% probability)
           - Simulates patient positioning variation between fractions
           - ±16 voxels ≈ ±16-32mm at 1-2mm resolution

        3. Small Z-axis rotation ±5° (50% probability)
           - Simulates patient setup rotation error
           - All volumes rotated together (preserves dose-anatomy relationship)

        4. CT intensity noise (30% probability)
           - Gaussian noise σ=0.02 in normalized space (~80 HU)
           - Applied only to CT, not dose or SDFs
           - Improves robustness to scanner noise variation

        Augmentations NOT applied (physics violations):
        - Y-flip (anterior-posterior): Beam entry direction matters
        - Z-flip (superior-inferior): Anatomy not symmetric
        - 90° rotations: Beam angles are meaningful
        - Dose intensity shifts: Dose values are physical quantities
        """
        from scipy.ndimage import shift as ndimage_shift, rotate as ndimage_rotate

        # 1. Random flip along X-axis only (left-right)
        if np.random.rand() > 0.5:
            ct = np.flip(ct, axis=0)
            dose = np.flip(dose, axis=0)
            sdf = np.flip(sdf, axis=1)

            # Swap femur channels
            sdf_copy = sdf.copy()
            sdf[5] = sdf_copy[6]
            sdf[6] = sdf_copy[5]

        # 2. Random translation ±16 voxels
        if np.random.rand() > 0.5:
            max_shift = 16
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_z = np.random.randint(-max_shift, max_shift + 1)

            ct = ndimage_shift(ct, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            dose = ndimage_shift(dose, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            sdf = ndimage_shift(sdf, (0, shift_y, shift_x, shift_z), order=1, mode='nearest')

        # 3. Small Z-axis rotation ±5°
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-5.0, 5.0)
            # Rotate in Y-X plane (axes 0,1) = rotation around Z axis
            ct = ndimage_rotate(ct, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            dose = ndimage_rotate(dose, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            for ch in range(sdf.shape[0]):
                sdf[ch] = ndimage_rotate(sdf[ch], angle, axes=(0, 1), reshape=False, order=1, mode='nearest')

        # 4. CT intensity noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, ct.shape).astype(np.float32)
            ct = np.clip(ct + noise, 0.0, 1.26)

        return np.ascontiguousarray(ct), np.ascontiguousarray(dose), np.ascontiguousarray(sdf)


# =============================================================================
# Baseline U-Net Model (No Time Embedding)
# =============================================================================

def _group_norm_num_groups(channels: int, preferred: int = 8) -> int:
    """Find largest divisor of channels that is <= preferred, for GroupNorm."""
    for g in range(preferred, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    """3D convolution block with optional FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: Optional[int] = None):
        super().__init__()

        num_groups = _group_norm_num_groups(out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()
        
        # Conditioning projection (FiLM)
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, out_ch * 2)
            )
        else:
            self.cond_mlp = None
        
        # Residual connection
        self.residual = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Apply FiLM conditioning
        if self.cond_mlp is not None and cond is not None:
            cond_out = self.cond_mlp(cond)
            cond_out = cond_out.view(cond_out.shape[0], -1, 1, 1, 1)
            scale, shift = cond_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual(x)


# =============================================================================
# Perceptual Loss Components
# =============================================================================

class GradientLoss3D(nn.Module):
    """
    3D Sobel gradient loss for edge preservation.

    Computes spatial gradients using Sobel operators and penalizes
    differences between predicted and target gradients.

    Memory overhead: ~25 MB (negligible)
    """

    def __init__(self):
        super().__init__()

        # 3D Sobel kernels for gradient computation
        # Shape: (3, 1, 3, 3, 3) - one kernel per axis (Y, X, Z)
        sobel_y = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
            [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3, 3)

        sobel_x = torch.tensor([
            [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
            [[-2,  0,  2], [-4,  0,  4], [-2,  0,  2]],
            [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_z = torch.tensor([
            [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
            [[-2, -4, -2], [ 0,  0,  0], [ 2,  4,  2]],
            [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Normalize kernels
        sobel_y = sobel_y / sobel_y.abs().sum()
        sobel_x = sobel_x / sobel_x.abs().sum()
        sobel_z = sobel_z / sobel_z.abs().sum()

        # Register as buffers (not trainable, but move with model)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_z', sobel_z)

    def compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D spatial gradients using Sobel operators.

        Args:
            x: (B, 1, H, W, D) tensor

        Returns:
            (B, 3, H, W, D) tensor of gradient magnitudes per axis
        """
        grad_y = F.conv3d(x, self.sobel_y, padding=1)
        grad_x = F.conv3d(x, self.sobel_x, padding=1)
        grad_z = F.conv3d(x, self.sobel_z, padding=1)

        return torch.cat([grad_y, grad_x, grad_z], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss between prediction and target.

        Args:
            pred: (B, 1, H, W, D) predicted dose
            target: (B, 1, H, W, D) target dose

        Returns:
            Scalar gradient loss (L1 between gradients)
        """
        pred_grads = self.compute_gradients(pred)
        target_grads = self.compute_gradients(target)

        # L1 loss on gradients (more robust to outliers than L2)
        return F.l1_loss(pred_grads, target_grads)


class VGGPerceptualLoss2D(nn.Module):
    """
    2D VGG perceptual loss applied slice-by-slice.

    Uses pretrained VGG16 features to encourage perceptually similar
    dose distributions with preserved edges and structure.

    Memory overhead: ~400 MB for VGG16 features
    """

    def __init__(self, slice_stride: int = 8, feature_layers: list = None):
        """
        Args:
            slice_stride: Process every Nth slice to control memory/compute
            feature_layers: Which VGG layers to use (default: [4, 9, 16])
        """
        super().__init__()

        self.slice_stride = slice_stride
        self.feature_layers = feature_layers or [4, 9, 16]  # relu1_2, relu2_2, relu3_3

        # Load pretrained VGG16
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        # Extract feature layers
        self.features = nn.ModuleList()
        prev_layer = 0
        for layer_idx in self.feature_layers:
            self.features.append(nn.Sequential(*list(vgg.features.children())[prev_layer:layer_idx]))
            prev_layer = layer_idx

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (VGG expects this)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize for VGG (expects ImageNet-style input)."""
        return (x - self.mean) / self.std

    def extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale features from VGG."""
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between prediction and target.

        Args:
            pred: (B, 1, H, W, D) predicted dose
            target: (B, 1, H, W, D) target dose

        Returns:
            Scalar perceptual loss
        """
        B, C, H, W, D = pred.shape

        total_loss = 0.0
        n_slices = 0

        # Process slices along depth dimension
        for z in range(0, D, self.slice_stride):
            # Extract 2D slices: (B, 1, H, W)
            pred_slice = pred[:, :, :, :, z]
            target_slice = target[:, :, :, :, z]

            # Convert grayscale to RGB by repeating channels
            pred_rgb = pred_slice.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            target_rgb = target_slice.repeat(1, 3, 1, 1)

            # Normalize for VGG
            pred_rgb = self.normalize(pred_rgb)
            target_rgb = self.normalize(target_rgb)

            # Extract features
            pred_features = self.extract_features(pred_rgb)
            target_features = self.extract_features(target_rgb)

            # Compute feature-wise L1 loss
            for pf, tf in zip(pred_features, target_features):
                total_loss = total_loss + F.l1_loss(pf, tf)

            n_slices += 1

        # Average over slices and feature layers
        return total_loss / (n_slices * len(self.feature_layers))


class DVHAwareLoss(nn.Module):
    """
    Differentiable DVH-aware loss for VMAT dose prediction.

    Computes clinical metrics (D95, Dmean, Vx) during training to directly
    optimize what clinicians care about. Uses soft approximations to maintain
    differentiability:
    - D95: Histogram-based soft percentile (O(N×bins) memory)
    - Vx: Sigmoid approximation for volume at threshold
    - Dmean: Standard mean (already differentiable)

    Structure indices (from STRUCTURE_CHANNELS):
        0: PTV70, 1: PTV56, 2: Prostate, 3: Rectum, 4: Bladder,
        5: Femur_L, 6: Femur_R, 7: Bowel
    """

    def __init__(
        self,
        rx_dose_gy: float = 70.0,
        d95_temperature: float = 0.1,
        vx_temperature: float = 1.0,
        d95_weight: float = 10.0,
        vx_weight: float = 2.0,
        dmean_weight: float = 1.0,
        n_bins: int = 100,
        min_voxels: int = 100,
    ):
        """
        Args:
            rx_dose_gy: Prescription dose in Gy (for absolute dose calculations)
            d95_temperature: Temperature for soft D95 computation (lower = sharper)
            vx_temperature: Temperature for sigmoid Vx approximation
            d95_weight: Weight for PTV D95 penalties
            vx_weight: Weight for OAR Vx constraint penalties
            dmean_weight: Weight for OAR Dmean penalties
            n_bins: Number of histogram bins for soft D95
            min_voxels: Minimum voxels required for valid computation
        """
        super().__init__()
        self.rx_dose_gy = rx_dose_gy
        self.d95_temperature = d95_temperature
        self.vx_temperature = vx_temperature
        self.d95_weight = d95_weight
        self.vx_weight = vx_weight
        self.dmean_weight = dmean_weight
        self.n_bins = n_bins
        self.min_voxels = min_voxels

        # Clinical constraints (as fractions of rx_dose)
        # Rectum V70 < 15%, Bladder V70 < 25%
        self.rectum_v70_limit = 0.15
        self.bladder_v70_limit = 0.25

    def extract_masks(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract structure masks from condition tensor SDFs.

        The condition tensor has channels: [CT, SDF_0, SDF_1, ..., SDF_7]
        SDFs are signed distance fields where SDF < 0 means inside the structure.

        Args:
            condition: (B, 9, H, W, D) tensor with CT + 8 SDF channels

        Returns:
            Dict mapping structure names to boolean masks (B, H, W, D)
        """
        masks = {}
        structure_names = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder',
                          'Femur_L', 'Femur_R', 'Bowel']

        for idx, name in enumerate(structure_names):
            # SDF channels start at index 1 (index 0 is CT)
            sdf = condition[:, idx + 1, :, :, :]
            masks[name] = sdf < 0  # Inside structure where SDF is negative

        return masks

    def soft_d95_histogram(
        self,
        dose: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        Compute soft D95 using differentiable histogram approximation.

        D95 = dose received by at least 95% of the volume.
        Uses soft histogram bins with temperature-scaled assignment.

        Args:
            dose: (B, 1, H, W, D) dose tensor (normalized 0-1)
            mask: (B, H, W, D) boolean mask for structure
            temperature: Softness of histogram bins (default: self.d95_temperature)

        Returns:
            (B,) tensor of D95 values per batch element
        """
        if temperature is None:
            temperature = self.d95_temperature

        B = dose.shape[0]
        d95_values = []

        for b in range(B):
            # Get dose values within mask
            mask_b = mask[b]  # (H, W, D)
            dose_b = dose[b, 0]  # (H, W, D)

            if mask_b.sum() < self.min_voxels:
                # Not enough voxels - return 0 (will be skipped in loss)
                d95_values.append(torch.tensor(0.0, device=dose.device, dtype=dose.dtype))
                continue

            # Extract masked dose values
            masked_dose = dose_b[mask_b]  # (N,)
            n_voxels = masked_dose.shape[0]

            # Create histogram bins (0 to 1.5 to handle overdose)
            bin_edges = torch.linspace(0, 1.5, self.n_bins + 1, device=dose.device)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (n_bins,)

            # Soft histogram assignment using Gaussian kernels
            # Shape: (N, n_bins)
            bin_width = bin_edges[1] - bin_edges[0]
            sigma = bin_width / 2  # Overlap between bins

            # Compute soft assignments
            diff = masked_dose.unsqueeze(1) - bin_centers.unsqueeze(0)  # (N, n_bins)
            weights = torch.exp(-0.5 * (diff / sigma) ** 2)  # Gaussian weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize

            # Compute histogram (counts per bin)
            histogram = weights.sum(dim=0)  # (n_bins,)

            # Cumulative distribution (from high to low dose)
            # D95 means 95% of volume receives at least this dose
            histogram_reversed = histogram.flip(0)
            cdf_reversed = histogram_reversed.cumsum(0) / n_voxels

            # Find D95: where cumulative from high reaches 95%
            target_fraction = 0.95

            # Soft interpolation to find D95
            bin_centers_reversed = bin_centers.flip(0)

            # Weight bins by how close CDF is to target
            weights_d95 = torch.exp(-((cdf_reversed - target_fraction) ** 2) / (2 * temperature ** 2))
            weights_d95 = weights_d95 / (weights_d95.sum() + 1e-8)

            d95 = (weights_d95 * bin_centers_reversed).sum()
            d95_values.append(d95)

        return torch.stack(d95_values)

    def soft_vx(
        self,
        dose: torch.Tensor,
        mask: torch.Tensor,
        threshold: float,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        Compute soft Vx (volume receiving at least x dose) using sigmoid.

        Args:
            dose: (B, 1, H, W, D) dose tensor (normalized 0-1)
            mask: (B, H, W, D) boolean mask for structure
            threshold: Dose threshold (normalized, e.g., 1.0 for 100% rx)
            temperature: Sigmoid sharpness (default: self.vx_temperature)

        Returns:
            (B,) tensor of Vx fractions (0-1) per batch element
        """
        if temperature is None:
            temperature = self.vx_temperature

        B = dose.shape[0]
        vx_values = []

        for b in range(B):
            mask_b = mask[b]  # (H, W, D)
            dose_b = dose[b, 0]  # (H, W, D)

            n_voxels = mask_b.sum()
            if n_voxels < self.min_voxels:
                vx_values.append(torch.tensor(0.0, device=dose.device, dtype=dose.dtype))
                continue

            # Get masked dose values
            masked_dose = dose_b[mask_b]  # (N,)

            # Soft threshold using sigmoid
            # sigmoid((dose - threshold) / temperature)
            # High temperature = soft, low temperature = sharp
            above_threshold = torch.sigmoid((masked_dose - threshold) / temperature)

            # Fraction of volume above threshold
            vx = above_threshold.mean()
            vx_values.append(vx)

        return torch.stack(vx_values)

    def compute_dmean(
        self,
        dose: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean dose within structure (already differentiable).

        Args:
            dose: (B, 1, H, W, D) dose tensor
            mask: (B, H, W, D) boolean mask

        Returns:
            (B,) tensor of mean dose values
        """
        B = dose.shape[0]
        dmean_values = []

        for b in range(B):
            mask_b = mask[b]
            dose_b = dose[b, 0]

            if mask_b.sum() < self.min_voxels:
                dmean_values.append(torch.tensor(0.0, device=dose.device, dtype=dose.dtype))
                continue

            dmean = dose_b[mask_b].mean()
            dmean_values.append(dmean)

        return torch.stack(dmean_values)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DVH-aware loss.

        Args:
            pred: (B, 1, H, W, D) predicted dose (normalized 0-1)
            target: (B, 1, H, W, D) target dose (normalized 0-1)
            condition: (B, 9, H, W, D) condition tensor (CT + SDFs)

        Returns:
            Tuple of (total_loss, metrics_dict)
            metrics_dict contains individual loss components for logging
        """
        # Extract structure masks
        masks = self.extract_masks(condition)

        metrics = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # --- PTV D95 losses ---
        # Penalize if pred D95 < target D95 (underdosing is worse than overdosing)

        # PTV70 D95
        if masks['PTV70'].any():
            pred_d95_ptv70 = self.soft_d95_histogram(pred, masks['PTV70'])
            target_d95_ptv70 = self.soft_d95_histogram(target, masks['PTV70'])

            # Asymmetric loss: only penalize underdosing
            d95_deficit_ptv70 = F.relu(target_d95_ptv70 - pred_d95_ptv70)
            loss_d95_ptv70 = d95_deficit_ptv70.mean()

            total_loss = total_loss + self.d95_weight * loss_d95_ptv70
            metrics['dvh/ptv70_d95_pred'] = pred_d95_ptv70.mean()
            metrics['dvh/ptv70_d95_target'] = target_d95_ptv70.mean()
            metrics['dvh/ptv70_d95_loss'] = loss_d95_ptv70

        # PTV56 D95 (secondary target, lower weight)
        if masks['PTV56'].any():
            pred_d95_ptv56 = self.soft_d95_histogram(pred, masks['PTV56'])
            target_d95_ptv56 = self.soft_d95_histogram(target, masks['PTV56'])

            d95_deficit_ptv56 = F.relu(target_d95_ptv56 - pred_d95_ptv56)
            loss_d95_ptv56 = d95_deficit_ptv56.mean()

            # Lower weight for secondary PTV
            total_loss = total_loss + (self.d95_weight * 0.5) * loss_d95_ptv56
            metrics['dvh/ptv56_d95_pred'] = pred_d95_ptv56.mean()
            metrics['dvh/ptv56_d95_target'] = target_d95_ptv56.mean()
            metrics['dvh/ptv56_d95_loss'] = loss_d95_ptv56

        # --- OAR Vx constraint losses ---
        # Penalize if Vx exceeds clinical limits

        # Rectum V70 < 15%
        if masks['Rectum'].any():
            pred_v70_rectum = self.soft_vx(pred, masks['Rectum'], threshold=1.0)
            target_v70_rectum = self.soft_vx(target, masks['Rectum'], threshold=1.0)

            # Penalize exceeding limit OR being worse than target
            excess_rectum = F.relu(pred_v70_rectum - self.rectum_v70_limit)
            worse_than_target = F.relu(pred_v70_rectum - target_v70_rectum)
            loss_v70_rectum = (excess_rectum + 0.5 * worse_than_target).mean()

            total_loss = total_loss + self.vx_weight * loss_v70_rectum
            metrics['dvh/rectum_v70_pred'] = pred_v70_rectum.mean()
            metrics['dvh/rectum_v70_target'] = target_v70_rectum.mean()
            metrics['dvh/rectum_v70_loss'] = loss_v70_rectum

        # Bladder V70 < 25%
        if masks['Bladder'].any():
            pred_v70_bladder = self.soft_vx(pred, masks['Bladder'], threshold=1.0)
            target_v70_bladder = self.soft_vx(target, masks['Bladder'], threshold=1.0)

            excess_bladder = F.relu(pred_v70_bladder - self.bladder_v70_limit)
            worse_than_target = F.relu(pred_v70_bladder - target_v70_bladder)
            loss_v70_bladder = (excess_bladder + 0.5 * worse_than_target).mean()

            total_loss = total_loss + self.vx_weight * loss_v70_bladder
            metrics['dvh/bladder_v70_pred'] = pred_v70_bladder.mean()
            metrics['dvh/bladder_v70_target'] = target_v70_bladder.mean()
            metrics['dvh/bladder_v70_loss'] = loss_v70_bladder

        # --- OAR Dmean losses ---
        # Soft penalty if pred Dmean > target Dmean

        for oar_name in ['Rectum', 'Bladder', 'Bowel']:
            if masks[oar_name].any():
                pred_dmean = self.compute_dmean(pred, masks[oar_name])
                target_dmean = self.compute_dmean(target, masks[oar_name])

                # Only penalize if prediction is higher than target
                dmean_excess = F.relu(pred_dmean - target_dmean)
                loss_dmean = dmean_excess.mean()

                total_loss = total_loss + self.dmean_weight * loss_dmean
                metrics[f'dvh/{oar_name.lower()}_dmean_pred'] = pred_dmean.mean()
                metrics[f'dvh/{oar_name.lower()}_dmean_target'] = target_dmean.mean()
                metrics[f'dvh/{oar_name.lower()}_dmean_loss'] = loss_dmean

        metrics['dvh/total_loss'] = total_loss

        return total_loss, metrics


class StructureWeightedLoss(nn.Module):
    """
    Structure-weighted MSE loss for VMAT dose prediction.

    Weights errors by clinical importance:
    - High weight (2x) for PTV regions where accurate dosing is critical
    - Medium weight (1.5x) for OAR boundaries where sharp gradients matter
    - Low weight (0.5x) for "no-man's land" where dose is more flexible

    This focuses the model's learning capacity on clinically important regions
    while allowing more flexibility in areas between targets and OARs.
    """

    def __init__(
        self,
        ptv_weight: float = 2.0,
        oar_boundary_weight: float = 1.5,
        background_weight: float = 0.5,
        boundary_width_mm: float = 5.0,
        sdf_clip_mm: float = 50.0,
    ):
        """
        Args:
            ptv_weight: Weight multiplier for PTV regions (default: 2.0)
            oar_boundary_weight: Weight for OAR boundary regions (default: 1.5)
            background_weight: Weight for "no-man's land" (default: 0.5)
            boundary_width_mm: Width of OAR boundary region in mm (default: 5.0)
            sdf_clip_mm: SDF clip distance used in preprocessing (default: 50.0)
        """
        super().__init__()
        self.ptv_weight = ptv_weight
        self.oar_boundary_weight = oar_boundary_weight
        self.background_weight = background_weight
        # Convert boundary width from mm to normalized SDF units [-1, 1]
        # SDF is clipped at ±clip_mm then normalized by clip_mm, so
        # boundary_width_mm maps to boundary_width_mm / clip_mm
        self.boundary_threshold = boundary_width_mm / sdf_clip_mm

    def extract_masks(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract structure masks from condition tensor SDFs.

        Args:
            condition: (B, 9, H, W, D) tensor with CT + 8 SDF channels

        Returns:
            Dict mapping structure names to masks/SDFs
        """
        masks = {}
        sdfs = {}
        structure_names = ['PTV70', 'PTV56', 'Prostate', 'Rectum', 'Bladder',
                          'Femur_L', 'Femur_R', 'Bowel']

        for idx, name in enumerate(structure_names):
            # SDF channels start at index 1 (index 0 is CT)
            sdf = condition[:, idx + 1, :, :, :]
            masks[name] = sdf < 0  # Inside structure where SDF is negative
            sdfs[name] = sdf

        return masks, sdfs

    def compute_weight_map(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-voxel weight map based on structure locations.

        Priority (highest wins):
        1. PTV regions → ptv_weight
        2. OAR boundaries → oar_boundary_weight
        3. Everything else → background_weight

        Args:
            condition: (B, 9, H, W, D) tensor with CT + 8 SDF channels

        Returns:
            (B, 1, H, W, D) weight map
        """
        B, _, H, W, D = condition.shape
        device = condition.device
        dtype = condition.dtype

        # Start with background weight everywhere
        weight_map = torch.full((B, 1, H, W, D), self.background_weight,
                                device=device, dtype=dtype)

        masks, sdfs = self.extract_masks(condition)

        # OAR boundaries: |SDF| < threshold (near the surface)
        oar_names = ['Rectum', 'Bladder', 'Bowel', 'Femur_L', 'Femur_R']
        for oar_name in oar_names:
            sdf = sdfs[oar_name]
            # Near boundary: |SDF| < threshold
            near_boundary = torch.abs(sdf) < self.boundary_threshold
            weight_map[:, 0] = torch.where(
                near_boundary,
                torch.full_like(weight_map[:, 0], self.oar_boundary_weight),
                weight_map[:, 0]
            )

        # PTVs have highest priority (overwrite everything)
        for ptv_name in ['PTV70', 'PTV56']:
            ptv_mask = masks[ptv_name]
            weight_map[:, 0] = torch.where(
                ptv_mask,
                torch.full_like(weight_map[:, 0], self.ptv_weight),
                weight_map[:, 0]
            )

        return weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute structure-weighted MSE loss.

        Args:
            pred: (B, 1, H, W, D) predicted dose (normalized 0-1)
            target: (B, 1, H, W, D) target dose (normalized 0-1)
            condition: (B, 9, H, W, D) condition tensor (CT + SDFs)

        Returns:
            Tuple of (weighted_mse_loss, metrics_dict)
        """
        # Compute weight map
        weight_map = self.compute_weight_map(condition)

        # Squared error
        squared_error = (pred - target) ** 2

        # Weighted MSE
        weighted_mse = (weight_map * squared_error).mean()

        # Compute metrics for logging
        metrics = {}
        masks, _ = self.extract_masks(condition)

        # PTV region error
        ptv_mask = masks['PTV70'] | masks['PTV56']
        if ptv_mask.any():
            ptv_mse = squared_error[:, 0][ptv_mask].mean()
            metrics['struct_weight/ptv_mse'] = ptv_mse

        # OAR region error
        oar_mask = masks['Rectum'] | masks['Bladder'] | masks['Bowel']
        if oar_mask.any():
            oar_mse = squared_error[:, 0][oar_mask].mean()
            metrics['struct_weight/oar_mse'] = oar_mse

        # Background region error (everything not PTV or major OAR)
        covered_mask = ptv_mask | oar_mask | masks['Prostate']
        background_mask = ~covered_mask
        if background_mask.any():
            background_mse = squared_error[:, 0][background_mask].mean()
            metrics['struct_weight/background_mse'] = background_mse

        # Weight statistics
        metrics['struct_weight/mean_weight'] = weight_map.mean()
        metrics['struct_weight/weighted_loss'] = weighted_mse

        return weighted_mse, metrics


class AsymmetricPTVLoss(nn.Module):
    """
    Asymmetric loss that penalizes PTV underdosing more heavily than overdosing.

    Rationale:
    - PTV underdose = treatment failure (tumor recurrence) - CRITICAL
    - PTV slight overdose = acceptable (within tolerance)

    The loss applies asymmetric weights in PTV regions:
    - underdose (pred < target): weight = underdose_weight (default 3.0)
    - overdose (pred > target): weight = overdose_weight (default 1.0)

    This encourages the model to err on the side of slightly higher doses
    in PTVs rather than underdosing.
    """

    def __init__(
        self,
        underdose_weight: float = 3.0,
        overdose_weight: float = 1.0,
        rx_dose_gy: float = 70.0,
    ):
        """
        Args:
            underdose_weight: Weight multiplier for underdosing (pred < target)
            overdose_weight: Weight multiplier for overdosing (pred > target)
            rx_dose_gy: Prescription dose for Gy conversion in metrics
        """
        super().__init__()
        self.underdose_weight = underdose_weight
        self.overdose_weight = overdose_weight
        self.rx_dose_gy = rx_dose_gy

    def extract_ptv_mask(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Extract combined PTV mask from condition tensor SDFs.

        Args:
            condition: (B, 9, H, W, D) tensor with CT + 8 SDF channels

        Returns:
            (B, H, W, D) boolean mask for combined PTV70 + PTV56
        """
        # SDF channels: 1=PTV70, 2=PTV56 (0 is CT)
        ptv70_mask = condition[:, 1, :, :, :] < 0
        ptv56_mask = condition[:, 2, :, :, :] < 0
        return ptv70_mask | ptv56_mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute asymmetric PTV loss.

        Args:
            pred: (B, 1, H, W, D) predicted dose (normalized 0-1)
            target: (B, 1, H, W, D) target dose (normalized 0-1)
            condition: (B, 9, H, W, D) condition tensor (CT + SDFs)

        Returns:
            Tuple of (asymmetric_loss, metrics_dict)
        """
        # Extract PTV mask
        ptv_mask = self.extract_ptv_mask(condition)

        # Compute error (positive = overdose, negative = underdose)
        error = pred[:, 0] - target[:, 0]

        # Asymmetric weights: higher weight for underdosing (error < 0)
        weights = torch.where(
            error < 0,
            torch.full_like(error, self.underdose_weight),
            torch.full_like(error, self.overdose_weight)
        )

        # Apply PTV mask and compute weighted squared error
        squared_error = error ** 2
        ptv_weighted_error = weights * squared_error * ptv_mask.float()

        # Compute mean over PTV voxels only (avoid divide by zero)
        n_ptv_voxels = ptv_mask.sum()
        if n_ptv_voxels > 0:
            asymmetric_loss = ptv_weighted_error.sum() / n_ptv_voxels
        else:
            asymmetric_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Compute metrics for logging
        metrics = {}
        if n_ptv_voxels > 0:
            # Error statistics in Gy
            error_gy = error * self.rx_dose_gy
            ptv_error_gy = error_gy[ptv_mask]
            metrics['asym_ptv/mean_error_gy'] = ptv_error_gy.mean()
            metrics['asym_ptv/min_error_gy'] = ptv_error_gy.min()  # Most underdosed
            metrics['asym_ptv/max_error_gy'] = ptv_error_gy.max()  # Most overdosed

            # Underdose statistics
            underdose_mask = (error < 0) & ptv_mask
            if underdose_mask.any():
                metrics['asym_ptv/underdose_mean_gy'] = error_gy[underdose_mask].mean()
                metrics['asym_ptv/underdose_fraction'] = underdose_mask.sum().float() / n_ptv_voxels

            # D95 proxy (95th percentile of dose in PTV)
            pred_gy = pred[:, 0] * self.rx_dose_gy
            target_gy = target[:, 0] * self.rx_dose_gy
            if ptv_mask.any():
                # Compute D95 as the value at the 5th percentile when sorted descending
                pred_ptv_sorted = torch.sort(pred_gy[ptv_mask], descending=True)[0]
                target_ptv_sorted = torch.sort(target_gy[ptv_mask], descending=True)[0]
                d95_idx = int(0.95 * len(pred_ptv_sorted))
                if d95_idx < len(pred_ptv_sorted):
                    metrics['asym_ptv/pred_d95_gy'] = pred_ptv_sorted[d95_idx]
                    metrics['asym_ptv/target_d95_gy'] = target_ptv_sorted[d95_idx]
                    metrics['asym_ptv/d95_gap_gy'] = target_ptv_sorted[d95_idx] - pred_ptv_sorted[d95_idx]

        metrics['asym_ptv/loss'] = asymmetric_loss
        metrics['asym_ptv/n_ptv_voxels'] = n_ptv_voxels.float()

        return asymmetric_loss, metrics


class BaselineUNet3D(nn.Module):
    """
    3D U-Net for direct dose regression.
    
    Input: CT (1ch) + SDFs (8ch) = 9 channels
    Output: Dose (1ch)
    Conditioning: Constraints via FiLM (no time embedding)
    """
    
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_channels: int = 48,
        constraint_dim: int = 13,
    ):
        super().__init__()
        
        # Embedding dimension for constraints
        cond_dim = 256
        
        # Constraint embedding
        self.constraint_mlp = nn.Sequential(
            nn.Linear(constraint_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Channel progression
        ch = [base_channels * (2**i) for i in range(5)]  # e.g., [48, 96, 192, 384, 768]
        
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, ch[0], cond_dim)
        self.enc2 = ConvBlock3D(ch[0], ch[1], cond_dim)
        self.enc3 = ConvBlock3D(ch[1], ch[2], cond_dim)
        self.enc4 = ConvBlock3D(ch[2], ch[3], cond_dim)
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(ch[3], ch[3], cond_dim)
        
        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec4 = ConvBlock3D(ch[3] * 2, ch[2], cond_dim)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = ConvBlock3D(ch[2] * 2, ch[1], cond_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = ConvBlock3D(ch[1] * 2, ch[0], cond_dim)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = ConvBlock3D(ch[0] * 2, ch[0], cond_dim)
        
        # Output
        self.out_conv = nn.Conv3d(ch[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, H, W, D) - CT + SDFs
            constraints: (B, 13) - planning constraints
        
        Returns:
            (B, 1, H, W, D) - predicted dose
        """
        # Embed constraints
        cond = self.constraint_mlp(constraints)
        
        # Encoder
        e1 = self.enc1(x, cond)
        e2 = self.enc2(self.pool(e1), cond)
        e3 = self.enc3(self.pool(e2), cond)
        e4 = self.enc4(self.pool(e3), cond)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4), cond)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), cond)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), cond)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)
        
        # Output (no activation - dose can be any positive value)
        out = self.out_conv(d1)
        
        return out


# =============================================================================
# PyTorch Lightning Module
# =============================================================================

class BaselineDosePredictor(pl.LightningModule):
    """
    Direct regression model for dose prediction.
    
    Much simpler than diffusion:
    - Forward pass: condition -> dose
    - Loss: MSE on dose
    - Inference: single forward pass
    """
    
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_channels: int = 48,
        constraint_dim: int = 13,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        rx_dose_gy: float = 70.0,
        architecture: str = 'baseline',
        # Perceptual loss options
        use_gradient_loss: bool = False,
        gradient_loss_weight: float = 0.1,
        use_vgg_loss: bool = False,
        vgg_loss_weight: float = 0.001,
        vgg_slice_stride: int = 8,
        # DVH loss options
        use_dvh_loss: bool = False,
        dvh_loss_weight: float = 0.5,
        dvh_d95_weight: float = 10.0,
        dvh_vx_weight: float = 2.0,
        dvh_dmean_weight: float = 1.0,
        dvh_temperature: float = 0.1,
        # Structure-weighted loss options
        use_structure_weighted: bool = False,
        structure_weighted_weight: float = 1.0,
        structure_ptv_weight: float = 2.0,
        structure_oar_boundary_weight: float = 1.5,
        structure_background_weight: float = 0.5,
        structure_boundary_width_mm: float = 5.0,
        # Asymmetric PTV loss options
        use_asymmetric_ptv: bool = False,
        asymmetric_ptv_weight: float = 1.0,
        asymmetric_underdose_weight: float = 3.0,
        asymmetric_overdose_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if architecture == 'baseline':
            self.model = BaselineUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                constraint_dim=constraint_dim,
            )
        else:
            from architectures import build_model
            self.model = build_model(
                architecture=architecture,
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                constraint_dim=constraint_dim,
            )

        # Initialize perceptual loss modules
        self.gradient_loss = GradientLoss3D() if use_gradient_loss else None
        self.vgg_loss = VGGPerceptualLoss2D(slice_stride=vgg_slice_stride) if use_vgg_loss else None

        # Initialize DVH loss
        self.dvh_loss = DVHAwareLoss(
            rx_dose_gy=rx_dose_gy,
            d95_temperature=dvh_temperature,
            vx_temperature=dvh_temperature * 10,  # Vx needs softer temperature
            d95_weight=dvh_d95_weight,
            vx_weight=dvh_vx_weight,
            dmean_weight=dvh_dmean_weight,
        ) if use_dvh_loss else None

        # Initialize structure-weighted loss
        self.structure_weighted_loss = StructureWeightedLoss(
            ptv_weight=structure_ptv_weight,
            oar_boundary_weight=structure_oar_boundary_weight,
            background_weight=structure_background_weight,
            boundary_width_mm=structure_boundary_width_mm,
        ) if use_structure_weighted else None

        # Initialize asymmetric PTV loss
        self.asymmetric_ptv_loss = AsymmetricPTVLoss(
            underdose_weight=asymmetric_underdose_weight,
            overdose_weight=asymmetric_overdose_weight,
            rx_dose_gy=rx_dose_gy,
        ) if use_asymmetric_ptv else None

        self.validation_step_outputs = []
    
    def forward(self, condition: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        return self.model(condition, constraints)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        condition = batch['condition']
        dose = batch['dose']
        constraints = batch['constraints']

        # Forward pass
        pred_dose = self.model(condition, constraints)

        # Primary loss: MSE on dose
        mse_loss = F.mse_loss(pred_dose, dose)

        # Physics penalty (negative dose)
        negative_penalty = F.relu(-pred_dose).mean()

        # Combine losses
        total_loss = mse_loss + 0.1 * negative_penalty

        # Log base losses
        self.log('train/mse_loss', mse_loss, on_step=False, on_epoch=True)
        self.log('train/neg_penalty', negative_penalty, on_step=False, on_epoch=True)

        # Optional: Gradient loss
        if self.gradient_loss is not None:
            grad_loss = self.gradient_loss(pred_dose, dose)
            total_loss = total_loss + self.hparams.gradient_loss_weight * grad_loss
            self.log('train/grad_loss', grad_loss, on_step=False, on_epoch=True)

        # Optional: VGG perceptual loss
        if self.vgg_loss is not None:
            vgg_loss = self.vgg_loss(pred_dose, dose)
            total_loss = total_loss + self.hparams.vgg_loss_weight * vgg_loss
            self.log('train/vgg_loss', vgg_loss, on_step=False, on_epoch=True)

        # Optional: DVH-aware loss
        if self.dvh_loss is not None:
            dvh_loss, dvh_metrics = self.dvh_loss(pred_dose, dose, condition)
            total_loss = total_loss + self.hparams.dvh_loss_weight * dvh_loss
            self.log('train/dvh_loss', dvh_loss, on_step=False, on_epoch=True)
            # Log individual DVH metrics
            for metric_name, metric_value in dvh_metrics.items():
                self.log(f'train/{metric_name}', metric_value, on_step=False, on_epoch=True)

        # Optional: Structure-weighted loss
        if self.structure_weighted_loss is not None:
            struct_loss, struct_metrics = self.structure_weighted_loss(pred_dose, dose, condition)
            total_loss = total_loss + self.hparams.structure_weighted_weight * struct_loss
            self.log('train/struct_weighted_loss', struct_loss, on_step=False, on_epoch=True)
            # Log individual structure metrics
            for metric_name, metric_value in struct_metrics.items():
                self.log(f'train/{metric_name}', metric_value, on_step=False, on_epoch=True)

        # Optional: Asymmetric PTV loss (penalizes underdosing more)
        if self.asymmetric_ptv_loss is not None:
            asym_loss, asym_metrics = self.asymmetric_ptv_loss(pred_dose, dose, condition)
            total_loss = total_loss + self.hparams.asymmetric_ptv_weight * asym_loss
            self.log('train/asym_ptv_loss', asym_loss, on_step=False, on_epoch=True)
            # Log individual asymmetric PTV metrics
            for metric_name, metric_value in asym_metrics.items():
                self.log(f'train/{metric_name}', metric_value, on_step=False, on_epoch=True)

        self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        condition = batch['condition']
        dose = batch['dose']
        constraints = batch['constraints']
        
        # Forward pass
        pred_dose = self.model(condition, constraints)
        
        # Loss
        val_loss = F.mse_loss(pred_dose, dose)
        self.log('val/loss', val_loss, prog_bar=True, on_epoch=True)
        
        # MAE in Gy
        rx = self.hparams.rx_dose_gy
        mae_gy = F.l1_loss(pred_dose * rx, dose * rx)
        self.log('val/mae_gy', mae_gy, prog_bar=True, on_epoch=True)

        # DVH metrics for validation
        if self.dvh_loss is not None:
            with torch.no_grad():
                _, dvh_metrics = self.dvh_loss(pred_dose, dose, condition)
                for metric_name, metric_value in dvh_metrics.items():
                    self.log(f'val/{metric_name}', metric_value, on_epoch=True)

        # Asymmetric PTV metrics for validation
        if self.asymmetric_ptv_loss is not None:
            with torch.no_grad():
                _, asym_metrics = self.asymmetric_ptv_loss(pred_dose, dose, condition)
                for metric_name, metric_value in asym_metrics.items():
                    self.log(f'val/{metric_name}', metric_value, on_epoch=True)

        # Store for gamma computation
        if batch_idx == 0:
            spacing = batch.get('spacing', None)
            self.validation_step_outputs.append({
                'pred': pred_dose.cpu(),
                'target': dose.cpu(),
                'spacing': spacing[0].cpu().numpy() if spacing is not None else None,
            })
    
    def on_validation_epoch_end(self) -> None:
        """Compute gamma on first validation sample."""
        if not self.validation_step_outputs or not HAS_PYMEDPHYS:
            self.validation_step_outputs.clear()
            return
        
        output = self.validation_step_outputs[0]
        pred = output['pred'][0, 0].numpy() * self.hparams.rx_dose_gy
        target = output['target'][0, 0].numpy() * self.hparams.rx_dose_gy
        spacing = tuple(output['spacing']) if output.get('spacing') is not None else None

        try:
            from eval_metrics import compute_gamma as _compute_gamma
            if spacing is None:
                spacing = DEFAULT_SPACING_MM
            gamma_dict = _compute_gamma(
                pred, target, spacing_mm=spacing, subsample=4,
            )
            gamma_pass_rate = gamma_dict.get('gamma_pass_rate')
            if gamma_pass_rate is not None:
                self.log('val/gamma_3mm3pct', gamma_pass_rate, prog_bar=True)
        except Exception as e:
            print(f"Gamma computation failed: {e}")

        self.validation_step_outputs.clear()
    
    def predict_full_volume(
        self,
        condition: torch.Tensor,
        constraints: torch.Tensor,
        patch_size: int = 128,
        overlap: int = 64,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Predict full volume using sliding window with Gaussian weighting.
        
        Single forward pass per patch (no diffusion iteration!).
        Much faster than DDPM inference.
        """
        device = next(self.parameters()).device
        condition = condition.to(device)
        constraints = constraints.to(device)
        
        _, _, H, W, D = condition.shape
        
        # Output accumulator
        output = torch.zeros(1, 1, H, W, D, device=device)
        weights = torch.zeros(1, 1, H, W, D, device=device)
        
        # Gaussian weighting
        gaussian = self._create_gaussian_weights(patch_size, device)
        
        # Sliding window — generate positions that guarantee full coverage
        stride = patch_size - overlap

        def _axis_positions(length: int) -> list:
            """Generate start positions along one axis ensuring full coverage."""
            if length <= patch_size:
                return [0]
            starts = list(range(0, length - patch_size + 1, stride))
            # Ensure the last position covers the volume end
            if starts[-1] + patch_size < length:
                starts.append(length - patch_size)
            return starts

        y_positions = _axis_positions(H)
        x_positions = _axis_positions(W)
        z_positions = _axis_positions(D)

        positions = []
        for y in y_positions:
            for x in x_positions:
                for z in z_positions:
                    positions.append((y, x, z))
        
        if verbose:
            print(f"  Predicting {len(positions)} patches...")
        
        with torch.no_grad():
            for i, (y, x, z) in enumerate(positions):
                # Extract patch
                cond_patch = condition[:, :, y:y+patch_size, x:x+patch_size, z:z+patch_size]
                
                # Single forward pass (no diffusion!)
                pred_patch = self.model(cond_patch, constraints)
                
                # Accumulate with Gaussian weighting
                output[:, :, y:y+patch_size, x:x+patch_size, z:z+patch_size] += pred_patch * gaussian
                weights[:, :, y:y+patch_size, x:x+patch_size, z:z+patch_size] += gaussian
        
        # Normalize
        output = output / (weights + 1e-8)
        
        # Clip to valid range
        output = torch.clamp(output, 0, 1.5)
        
        return output
    
    def _create_gaussian_weights(self, size: int, device: torch.device) -> torch.Tensor:
        """Create 3D Gaussian weights for smooth blending."""
        sigma = size / 4
        x = torch.arange(size, device=device) - size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_3d = gaussian_1d[:, None, None] * gaussian_1d[None, :, None] * gaussian_1d[None, None, :]
        return gaussian_3d.unsqueeze(0).unsqueeze(0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 200,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# =============================================================================
# Logging Callback
# =============================================================================

class BaselineLoggingCallback(Callback):
    """Log training configuration and results."""
    
    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None
        self.epoch_metrics = []
    
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        
        # Save config
        arch_name = pl_module.hparams.get('architecture', 'baseline')
        config = {
            'script': 'train_baseline_unet.py',
            'version': SCRIPT_VERSION,
            'architecture': arch_name,
            'model': pl_module.model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'hparams': dict(pl_module.hparams),
            'model_params': sum(p.numel() for p in pl_module.parameters()),
        }
        
        with open(self.log_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            'epoch': trainer.current_epoch,
            'train_loss': float(trainer.callback_metrics.get('train/loss_epoch', 0)),
            'val_loss': float(trainer.callback_metrics.get('val/loss', 0)),
            'val_mae_gy': float(trainer.callback_metrics.get('val/mae_gy', 0)),
            'val_gamma': float(trainer.callback_metrics.get('val/gamma_3mm3pct', 0)),
        }
        self.epoch_metrics.append(metrics)
    
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        
        summary = {
            'total_time_sec': total_time,
            'total_time_hours': total_time / 3600,
            'final_metrics': self.epoch_metrics[-1] if self.epoch_metrics else {},
            'best_val_mae_gy': min(m['val_mae_gy'] for m in self.epoch_metrics) if self.epoch_metrics else None,
        }
        
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation MAE: {summary['best_val_mae_gy']:.2f} Gy")


# =============================================================================
# Main
# =============================================================================

def get_default_data_dir():
    """
    Auto-detect the best default data directory based on environment.

    Priority order:
    1. Local ./processed with .npz files (if data exists)
    2. External drive fallback (if mounted)

    Returns:
        str: Path to data directory, or None if not found
    """
    # Get script directory to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Local path (clean workstation) - could be real dir or symlink
    local_data = os.path.join(project_root, "processed")

    # External drive path (WSL)
    external_data = "/mnt/i/processed_npz"

    # Check local path first (works for both real dir and symlink)
    if os.path.isdir(local_data):
        npz_files = [f for f in os.listdir(local_data) if f.endswith('.npz')]
        if npz_files:
            print(f"[Auto-detect] Using local data path: {local_data} ({len(npz_files)} .npz files)")
            return local_data

    # Check external drive
    if os.path.isdir(external_data):
        npz_files = [f for f in os.listdir(external_data) if f.endswith('.npz')]
        if npz_files:
            print(f"[Auto-detect] Using external drive: {external_data} ({len(npz_files)} .npz files)")
            return external_data

    # No data found
    print("[Auto-detect] No preprocessed data found. Please specify --data_dir")
    return None


def main():
    # Get environment-appropriate default
    default_data_dir = get_default_data_dir()

    parser = argparse.ArgumentParser(
        description="Train Baseline U-Net for VMAT Dose Prediction"
    )

    # Data
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                       help='Path to preprocessed .npz files (auto-detected if not specified)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio (held out)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--patches_per_volume', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    
    # Model
    parser.add_argument('--base_channels', type=int, default=48,
                       help='Base filter count (same as diffusion model)')
    parser.add_argument('--architecture', type=str, default='baseline',
                       choices=['baseline', 'attention_unet', 'bottleneck_attn'],
                       help='Model architecture variant (default: baseline)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--exp_name', type=str, default='baseline_unet')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')

    # Perceptual loss options
    parser.add_argument('--use_gradient_loss', action='store_true',
                        help='Enable 3D Sobel gradient loss for edge preservation')
    parser.add_argument('--gradient_loss_weight', type=float, default=0.1,
                        help='Weight for gradient loss (default: 0.1)')
    parser.add_argument('--use_vgg_loss', action='store_true',
                        help='Enable 2D VGG perceptual loss (slice-wise)')
    parser.add_argument('--vgg_loss_weight', type=float, default=0.001,
                        help='Weight for VGG loss (default: 0.001)')
    parser.add_argument('--vgg_slice_stride', type=int, default=8,
                        help='Process every Nth slice for VGG loss (default: 8)')

    # DVH loss options
    parser.add_argument('--use_dvh_loss', action='store_true',
                        help='Enable DVH-aware loss for clinical metrics optimization')
    parser.add_argument('--dvh_loss_weight', type=float, default=0.5,
                        help='Overall weight for DVH loss (default: 0.5)')
    parser.add_argument('--dvh_d95_weight', type=float, default=10.0,
                        help='Weight for PTV D95 penalties within DVH loss (default: 10.0)')
    parser.add_argument('--dvh_vx_weight', type=float, default=2.0,
                        help='Weight for OAR Vx constraint penalties within DVH loss (default: 2.0)')
    parser.add_argument('--dvh_dmean_weight', type=float, default=1.0,
                        help='Weight for OAR Dmean penalties within DVH loss (default: 1.0)')
    parser.add_argument('--dvh_temperature', type=float, default=0.1,
                        help='Temperature for soft DVH approximations (default: 0.1)')

    # Structure-weighted loss options
    parser.add_argument('--use_structure_weighted', action='store_true',
                        help='Enable structure-weighted MSE loss (2x PTV, 1.5x OAR boundary)')
    parser.add_argument('--structure_weighted_weight', type=float, default=1.0,
                        help='Overall weight for structure-weighted loss (default: 1.0)')
    parser.add_argument('--structure_ptv_weight', type=float, default=2.0,
                        help='Weight multiplier for PTV regions (default: 2.0)')
    parser.add_argument('--structure_oar_boundary_weight', type=float, default=1.5,
                        help='Weight for OAR boundary regions (default: 1.5)')
    parser.add_argument('--structure_background_weight', type=float, default=0.5,
                        help='Weight for background/no-mans-land (default: 0.5)')
    parser.add_argument('--structure_boundary_width_mm', type=float, default=5.0,
                        help='Width of OAR boundary region in mm (default: 5.0)')

    # Asymmetric PTV loss options
    parser.add_argument('--use_asymmetric_ptv', action='store_true',
                        help='Enable asymmetric PTV loss (penalizes underdosing 3x more)')
    parser.add_argument('--asymmetric_ptv_weight', type=float, default=1.0,
                        help='Overall weight for asymmetric PTV loss (default: 1.0)')
    parser.add_argument('--asymmetric_underdose_weight', type=float, default=3.0,
                        help='Weight for PTV underdosing (pred < target) (default: 3.0)')
    parser.add_argument('--asymmetric_overdose_weight', type=float, default=1.0,
                        help='Weight for PTV overdosing (pred > target) (default: 1.0)')

    args = parser.parse_args()

    # Validate data_dir
    if args.data_dir is None:
        parser.error("--data_dir is required. No preprocessed data was auto-detected.\n"
                     "Run preprocessing first or specify the path manually.")

    # Seed
    pl.seed_everything(args.seed, workers=True)

    # Load files
    print(f"\nLoading data from {args.data_dir}")
    data_dir = Path(args.data_dir)
    all_files = sorted(list(data_dir.glob("*.npz")))
    
    if not all_files:
        raise ValueError(f"No .npz files found in {args.data_dir}")
    
    # Train/val/test split (same as diffusion)
    n_files = len(all_files)
    n_test = max(1, int(n_files * args.test_split)) if args.test_split > 0 else 0
    n_val = max(1, int(n_files * args.val_split))
    n_train = n_files - n_val - n_test
    
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_files)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:] if n_test > 0 else []
    
    train_files = [str(all_files[i]) for i in train_indices]
    val_files = [str(all_files[i]) for i in val_indices]
    test_files = [str(all_files[i]) for i in test_indices] if n_test > 0 else []
    
    print(f"Dataset split (seed={args.seed}):")
    print(f"  Train: {len(train_files)} cases ({100*len(train_files)/n_files:.0f}%)")
    print(f"  Val: {len(val_files)} cases ({100*len(val_files)/n_files:.0f}%)")
    print(f"  Test: {len(test_files)} cases ({100*len(test_files)/n_files:.0f}%) [HELD OUT]")
    
    # Save test cases
    if test_files:
        test_list_path = Path(args.log_dir) / args.exp_name / 'test_cases.json'
        test_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_list_path, 'w') as f:
            json.dump({
                'test_cases': [Path(f).stem for f in test_files],
                'test_files': test_files,
                'seed': args.seed,
            }, f, indent=2)
    
    # Create datasets
    train_dataset = VMATDosePatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        augment=True,
        mode='train',
    )
    train_dataset.files = [Path(f) for f in train_files]
    
    val_dataset = VMATDosePatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        augment=False,
        mode='val',
    )
    val_dataset.files = [Path(f) for f in val_files]
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Model
    model = BaselineDosePredictor(
        in_channels=9,
        out_channels=1,
        base_channels=args.base_channels,
        constraint_dim=13,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        rx_dose_gy=args.rx_dose_gy,
        architecture=args.architecture,
        # Perceptual loss options
        use_gradient_loss=args.use_gradient_loss,
        gradient_loss_weight=args.gradient_loss_weight,
        use_vgg_loss=args.use_vgg_loss,
        vgg_loss_weight=args.vgg_loss_weight,
        vgg_slice_stride=args.vgg_slice_stride,
        # DVH loss options
        use_dvh_loss=args.use_dvh_loss,
        dvh_loss_weight=args.dvh_loss_weight,
        dvh_d95_weight=args.dvh_d95_weight,
        dvh_vx_weight=args.dvh_vx_weight,
        dvh_dmean_weight=args.dvh_dmean_weight,
        dvh_temperature=args.dvh_temperature,
        # Structure-weighted loss options
        use_structure_weighted=args.use_structure_weighted,
        structure_weighted_weight=args.structure_weighted_weight,
        structure_ptv_weight=args.structure_ptv_weight,
        structure_oar_boundary_weight=args.structure_oar_boundary_weight,
        structure_background_weight=args.structure_background_weight,
        structure_boundary_width_mm=args.structure_boundary_width_mm,
        # Asymmetric PTV loss options
        use_asymmetric_ptv=args.use_asymmetric_ptv,
        asymmetric_ptv_weight=args.asymmetric_ptv_weight,
        asymmetric_underdose_weight=args.asymmetric_underdose_weight,
        asymmetric_overdose_weight=args.asymmetric_overdose_weight,
    )

    param_count = sum(p.numel() for p in model.parameters())
    arch_desc = model.model.__class__.__name__
    print(f"\nModel: {arch_desc} (architecture={args.architecture}, base_channels={args.base_channels})")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    # Print loss configuration
    print("\nLoss configuration:")
    print(f"  MSE loss: weight=1.0")
    print(f"  Negative penalty: weight=0.1")
    if args.use_gradient_loss:
        print(f"  Gradient loss (3D Sobel): weight={args.gradient_loss_weight}")
    if args.use_vgg_loss:
        print(f"  VGG perceptual loss: weight={args.vgg_loss_weight}, slice_stride={args.vgg_slice_stride}")
    if args.use_dvh_loss:
        print(f"  DVH-aware loss: weight={args.dvh_loss_weight}")
        print(f"    D95 weight: {args.dvh_d95_weight}, Vx weight: {args.dvh_vx_weight}, Dmean weight: {args.dvh_dmean_weight}")
        print(f"    Temperature: {args.dvh_temperature}")
    if args.use_structure_weighted:
        print(f"  Structure-weighted loss: weight={args.structure_weighted_weight}")
        print(f"    PTV weight: {args.structure_ptv_weight}, OAR boundary: {args.structure_oar_boundary_weight}, Background: {args.structure_background_weight}")
        print(f"    Boundary width: {args.structure_boundary_width_mm} mm")
    if args.use_asymmetric_ptv:
        print(f"  Asymmetric PTV loss: weight={args.asymmetric_ptv_weight}")
        print(f"    Underdose penalty: {args.asymmetric_underdose_weight}x, Overdose penalty: {args.asymmetric_overdose_weight}x")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(args.log_dir) / args.exp_name / 'checkpoints',
            filename='best-{epoch:03d}-{val/mae_gy:.3f}',
            monitor='val/mae_gy',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=Path(args.log_dir) / args.exp_name / 'checkpoints',
            filename='last',
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val/mae_gy',
            patience=50,
            mode='min',
        ),
        BaselineLoggingCallback(args.log_dir, args.exp_name),
        RichProgressBar(),
    ]
    
    # Loggers
    loggers = [
        TensorBoardLogger(args.log_dir, name=args.exp_name),
        CSVLogger(args.log_dir, name=args.exp_name),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=args.devices,
        strategy=args.strategy,
        precision='16-mixed',
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic="warn",  # True incompatible with trilinear upsample backward
        gradient_clip_val=1.0,
    )
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Logging to: {args.log_dir}/{args.exp_name}")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val MAE: {callbacks[0].best_model_score:.3f} Gy")
    print("\nTo evaluate:")
    print(f"  python inference_baseline_unet.py \\")
    print(f"      --checkpoint {callbacks[0].best_model_path} \\")
    print(f"      --input_dir ./test_npz --output_dir ./predictions")


if __name__ == '__main__':
    main()
