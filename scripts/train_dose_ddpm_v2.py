"""
Conditional DDPM Trainer for VMAT Dose Prediction (Phase 1 Feasibility)

Version: 2.2
Compatible with: preprocess_dicom_rt_v2.2.py output

Key Features:
- Patch-based training (128³) to fit on 3090 (24GB VRAM)
- Uses SDFs (masks_sdf) for smooth gradients during backprop
- Proper conditioning via FiLM (Feature-wise Linear Modulation)
- Cosine noise schedule (better than linear for images)
- DDIM sampling for fast inference
- Sliding window inference for full volumes
- Proper gamma evaluation via pymedphys
- Comprehensive logging for publication
- Physics-informed loss (negative dose penalty)
- Train/val/test split (80/10/10) for proper evaluation

Architecture:
- 3D U-Net with attention at bottleneck
- Time embedding via sinusoidal positional encoding
- Constraint conditioning via FiLM layers
- Input: CT (1ch) + SDFs (8ch) = 9 channels
- Output: Noise prediction (1ch)

Changes in v2.2:
- Increased default base_channels from 32 to 48 (~2x parameters)
- Fixed augmentation: removed Y-flip, Z-flip, and 90° rotations
  (only left-right X-flip preserved - physically valid for prostate)
- Added random translation augmentation (±16 voxels) for robustness
- Added --test_split for held-out test set (80/10/10 default)
- Added physics-informed loss term (penalizes negative dose predictions)

Usage:
    python train_dose_ddpm_v2.py --data_dir ./processed_npz --epochs 200

Hardware Requirements:
    - NVIDIA 3090 (24GB) with batch_size=2, patch_size=128
    - NVIDIA 4090 (24GB) with batch_size=4, patch_size=128
    - For A100 (40GB+): can increase patch_size to 160 or batch_size to 8
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
import platform
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping, 
    RichProgressBar, Callback, Timer
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
# Constants
# =============================================================================

STRUCTURE_CHANNELS = {
    0: 'PTV70',
    1: 'PTV56', 
    2: 'Prostate',
    3: 'Rectum',
    4: 'Bladder',
    5: 'Femur_L',
    6: 'Femur_R',
    7: 'Bowel'
}

DEFAULT_SPACING_MM = (1.0, 1.0, 2.0)  # From preprocessing

SCRIPT_VERSION = "2.2.0"


# =============================================================================
# Publication Logging Callback
# =============================================================================

class PublicationLoggingCallback(Callback):
    """
    Comprehensive logging for publication-ready results.
    
    Logs:
    - Training configuration and environment
    - Dataset splits (case IDs)
    - Per-epoch timing
    - Hardware info
    - Best metrics and convergence info
    """
    
    def __init__(self, log_dir: str, train_files: List[str], val_files: List[str]):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_files = train_files
        self.val_files = val_files
        
        self.epoch_times = []
        self.epoch_metrics = []
        self.start_time = None
        self.best_mae = float('inf')
        self.best_epoch = 0
        
    def on_fit_start(self, trainer, pl_module):
        """Log training configuration at start."""
        self.start_time = time.time()
        
        # Collect environment info
        config = {
            'script_version': SCRIPT_VERSION,
            'timestamp_start': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'python_version': sys.version,
            'platform': platform.platform(),
            
            # Hardware
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            
            # Model
            'model_class': type(pl_module.model).__name__,
            'total_parameters': sum(p.numel() for p in pl_module.parameters()),
            'trainable_parameters': sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            
            # Hyperparameters
            'hyperparameters': dict(pl_module.hparams),
            
            # Dataset
            'train_cases': len(self.train_files),
            'val_cases': len(self.val_files),
            'train_case_ids': [Path(f).stem for f in self.train_files],
            'val_case_ids': [Path(f).stem for f in self.val_files],
            
            # Training settings
            'max_epochs': trainer.max_epochs,
            'precision': str(trainer.precision),
            'accumulate_grad_batches': trainer.accumulate_grad_batches,
            'gradient_clip_val': trainer.gradient_clip_val,
        }
        
        # Save config
        config_path = self.log_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"\nTraining configuration saved to: {config_path}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Record epoch start time."""
        self._epoch_start = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record epoch timing and metrics."""
        epoch_time = time.time() - self._epoch_start
        self.epoch_times.append(epoch_time)
        
        # Get current metrics
        metrics = {
            'epoch': trainer.current_epoch,
            'epoch_time_sec': epoch_time,
            'train_loss': float(trainer.callback_metrics.get('train/loss_epoch', 0)),
            'val_loss': float(trainer.callback_metrics.get('val/loss', 0)),
            'val_mae_gy': float(trainer.callback_metrics.get('val/mae_gy', 0)),
            'learning_rate': float(trainer.callback_metrics.get('lr-AdamW', 0)),
        }
        
        if 'val/gamma_3mm3pct' in trainer.callback_metrics:
            metrics['val_gamma'] = float(trainer.callback_metrics['val/gamma_3mm3pct'])
        
        self.epoch_metrics.append(metrics)
        
        # Track best
        current_mae = metrics['val_mae_gy']
        if current_mae > 0 and current_mae < self.best_mae:
            self.best_mae = current_mae
            self.best_epoch = trainer.current_epoch
        
        # Log ETA
        if len(self.epoch_times) > 1:
            avg_time = np.mean(self.epoch_times[-10:])  # Last 10 epochs
            remaining = trainer.max_epochs - trainer.current_epoch - 1
            eta_hours = (avg_time * remaining) / 3600
            pl_module.log('misc/eta_hours', eta_hours)
            pl_module.log('misc/epoch_time_sec', epoch_time)
    
    def on_fit_end(self, trainer, pl_module):
        """Save final summary."""
        total_time = time.time() - self.start_time
        
        summary = {
            'timestamp_end': datetime.now().isoformat(),
            'total_training_time_hours': total_time / 3600,
            'total_epochs_completed': trainer.current_epoch + 1,
            'avg_epoch_time_sec': np.mean(self.epoch_times) if self.epoch_times else 0,
            
            # Best results
            'best_val_mae_gy': self.best_mae,
            'best_epoch': self.best_epoch,
            'converged_epoch': self._find_convergence_epoch(),
            
            # Final metrics
            'final_train_loss': self.epoch_metrics[-1]['train_loss'] if self.epoch_metrics else None,
            'final_val_loss': self.epoch_metrics[-1]['val_loss'] if self.epoch_metrics else None,
            'final_val_mae_gy': self.epoch_metrics[-1]['val_mae_gy'] if self.epoch_metrics else None,
            
            # Early stopping
            'early_stopped': trainer.current_epoch + 1 < trainer.max_epochs,
        }
        
        # Save summary
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed epoch log
        epochs_path = self.log_dir / 'epoch_metrics.csv'
        if self.epoch_metrics:
            keys = self.epoch_metrics[0].keys()
            with open(epochs_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.epoch_metrics)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - PUBLICATION SUMMARY")
        print('='*60)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Epochs completed: {trainer.current_epoch + 1}")
        print(f"Best MAE: {self.best_mae:.3f} Gy (epoch {self.best_epoch})")
        print(f"\nResults saved to:")
        print(f"  {summary_path}")
        print(f"  {epochs_path}")
    
    def _find_convergence_epoch(self, window: int = 10, threshold: float = 0.01) -> Optional[int]:
        """Find epoch where validation loss stabilized."""
        if len(self.epoch_metrics) < window * 2:
            return None
        
        val_losses = [m['val_loss'] for m in self.epoch_metrics]
        
        for i in range(window, len(val_losses) - window):
            recent = np.mean(val_losses[i:i+window])
            previous = np.mean(val_losses[i-window:i])
            
            if previous > 0 and abs(recent - previous) / previous < threshold:
                return i
        
        return None


class GPUMemoryCallback(Callback):
    """Log GPU memory usage."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            pl_module.log('misc/gpu_memory_gb', memory_gb)


class GPUCoolingCallback(Callback):
    """
    Add small pauses between batches to help prevent GPU overheating.

    This is useful for sustained training on consumer GPUs (like RTX 3090)
    that may throttle or crash under continuous maximum load.
    """

    def __init__(self, pause_every_n_batches: int = 10, pause_seconds: float = 0.5):
        super().__init__()
        self.pause_every_n_batches = pause_every_n_batches
        self.pause_seconds = pause_seconds
        self.total_pauses = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.pause_every_n_batches == 0:
            # Sync GPU to ensure all operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(self.pause_seconds)
            self.total_pauses += 1

    def on_train_epoch_end(self, trainer, pl_module):
        # Longer pause between epochs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(2.0)  # 2 second pause between epochs


# =============================================================================
# Dataset with Patch Extraction
# =============================================================================

class VMATDosePatchDataset(Dataset):
    """
    Loads .npz files and extracts random 3D patches for training.
    
    Uses SDFs (masks_sdf) instead of binary masks for smooth gradients.
    Patches are centered on regions with dose > threshold to focus learning.
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 128,
        patches_per_volume: int = 4,
        augment: bool = False,
        mode: str = 'train',  # 'train', 'val', 'test'
        dose_threshold: float = 0.1,  # Focus on regions with dose > 10% Rx
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment and (mode == 'train')
        self.dose_threshold = dose_threshold
        
        # Find all .npz files
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        print(f"Found {len(self.files)} cases for {mode}")
        
        # Validate first file structure
        self._validate_file_structure(self.files[0])
    
    def _validate_file_structure(self, filepath: Path):
        """Check that file has expected keys from v2.2 preprocessing."""
        data = np.load(filepath, allow_pickle=True)
        required_keys = ['ct', 'dose', 'masks_sdf', 'constraints', 'metadata']
        
        for key in required_keys:
            if key not in data.files:
                raise ValueError(f"Missing key '{key}' in {filepath}. "
                               f"Expected v2.2 preprocessing output.")
        
        # Check shapes
        ct_shape = data['ct'].shape
        sdf_shape = data['masks_sdf'].shape
        
        if len(ct_shape) != 3:
            raise ValueError(f"CT should be 3D, got shape {ct_shape}")
        if len(sdf_shape) != 4:
            raise ValueError(f"masks_sdf should be 4D (C,H,W,D), got shape {sdf_shape}")
        if sdf_shape[0] != 8:
            raise ValueError(f"Expected 8 SDF channels, got {sdf_shape[0]}")
        
        print(f"Validated file structure: CT {ct_shape}, SDF {sdf_shape}")
    
    def __len__(self):
        return len(self.files) * self.patches_per_volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine which file and which patch
        file_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        # Load data
        data = np.load(self.files[file_idx], allow_pickle=True)
        
        ct = data['ct'].astype(np.float32)              # (H, W, D)
        dose = data['dose'].astype(np.float32)          # (H, W, D)
        masks_sdf = data['masks_sdf'].astype(np.float32)  # (8, H, W, D)
        constraints = data['constraints'].astype(np.float32)  # (13,)
        
        # Extract metadata for potential use
        metadata = data['metadata'].item() if 'metadata' in data.files else {}
        
        # Get volume shape
        H, W, D = ct.shape
        P = self.patch_size
        
        # Find valid patch center (focus on dose region)
        center = self._sample_patch_center(dose, H, W, D, P, patch_idx)
        
        # Extract patches
        y0, y1 = center[0] - P//2, center[0] + P//2
        x0, x1 = center[1] - P//2, center[1] + P//2
        z0, z1 = center[2] - P//2, center[2] + P//2
        
        ct_patch = ct[y0:y1, x0:x1, z0:z1]
        dose_patch = dose[y0:y1, x0:x1, z0:z1]
        sdf_patch = masks_sdf[:, y0:y1, x0:x1, z0:z1]
        
        # Data augmentation
        if self.augment:
            ct_patch, dose_patch, sdf_patch = self._augment(
                ct_patch, dose_patch, sdf_patch
            )
        
        # Convert to tensors
        # Condition: CT (1ch) + SDFs (8ch) = 9 channels
        ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0)  # (1, P, P, P)
        sdf_tensor = torch.from_numpy(sdf_patch)              # (8, P, P, P)
        condition = torch.cat([ct_tensor, sdf_tensor], dim=0)  # (9, P, P, P)
        
        dose_tensor = torch.from_numpy(dose_patch).unsqueeze(0)  # (1, P, P, P)
        constraint_tensor = torch.from_numpy(constraints)         # (13,)
        
        return {
            'condition': condition,
            'dose': dose_tensor,
            'constraints': constraint_tensor,
            'file_idx': file_idx,
            'center': torch.tensor(center),
        }
    
    def _sample_patch_center(
        self, dose: np.ndarray, H: int, W: int, D: int, P: int, patch_idx: int
    ) -> Tuple[int, int, int]:
        """
        Sample a patch center, biased toward high-dose regions.
        
        For training diversity, uses patch_idx to vary sampling strategy.
        """
        half_p = P // 2
        
        # Valid range for patch centers (ensure patch stays in volume)
        y_range = (half_p, H - half_p)
        x_range = (half_p, W - half_p)
        z_range = (half_p, D - half_p)
        
        # Strategy: alternate between dose-focused and random sampling
        if patch_idx % 2 == 0:
            # Dose-focused: sample from high-dose region
            dose_mask = dose > self.dose_threshold
            
            # Restrict to valid center range
            valid_mask = np.zeros_like(dose_mask)
            valid_mask[y_range[0]:y_range[1], 
                      x_range[0]:x_range[1], 
                      z_range[0]:z_range[1]] = True
            
            candidates = np.where(dose_mask & valid_mask)
            
            if len(candidates[0]) > 0:
                idx = np.random.randint(len(candidates[0]))
                return (candidates[0][idx], candidates[1][idx], candidates[2][idx])
        
        # Random sampling (fallback or for diversity)
        y = np.random.randint(y_range[0], y_range[1])
        x = np.random.randint(x_range[0], x_range[1])
        z = np.random.randint(z_range[0], z_range[1])
        
        return (y, x, z)
    
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
           - Applied to CT, dose, and SDFs together (preserves spatial relationships)
        
        Augmentations NOT applied (physics violations):
        - Y-flip (anterior-posterior): Beam entry direction matters
        - Z-flip (superior-inferior): Anatomy not symmetric
        - 90° rotations: Beam angles are meaningful
        - Intensity shifts: Dose was computed from original CT values
        """
        
        # 1. Random flip along X-axis only (left-right)
        if np.random.rand() > 0.5:
            ct = np.flip(ct, axis=0)
            dose = np.flip(dose, axis=0)
            sdf = np.flip(sdf, axis=1)  # +1 because sdf has channel dim
            
            # Also swap left/right femur SDF channels (indices 5 and 6)
            sdf_copy = sdf.copy()
            sdf[5] = sdf_copy[6]  # Femur_L <- Femur_R
            sdf[6] = sdf_copy[5]  # Femur_R <- Femur_L
        
        # 2. Random translation ±16 voxels
        if np.random.rand() > 0.5:
            # Random shift in each dimension
            max_shift = 16
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_z = np.random.randint(-max_shift, max_shift + 1)
            
            # Apply shift using scipy.ndimage for proper edge handling
            from scipy.ndimage import shift as ndimage_shift
            
            # mode='nearest' fills edges with nearest values (no wrap-around artifacts)
            ct = ndimage_shift(ct, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            dose = ndimage_shift(dose, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            
            # For SDF, shift spatial dimensions (axes 1, 2, 3)
            sdf = ndimage_shift(sdf, (0, shift_y, shift_x, shift_z), order=1, mode='nearest')
        
        # Ensure contiguous arrays
        return np.ascontiguousarray(ct), np.ascontiguousarray(dose), np.ascontiguousarray(sdf)


class VMATDoseFullVolumeDataset(Dataset):
    """
    Dataset for validation/inference on full volumes.
    Returns the complete volume without patching.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.npz")))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        
        ct = torch.from_numpy(data['ct'].astype(np.float32)).unsqueeze(0)
        dose = torch.from_numpy(data['dose'].astype(np.float32)).unsqueeze(0)
        masks_sdf = torch.from_numpy(data['masks_sdf'].astype(np.float32))
        masks_binary = torch.from_numpy(data['masks'].astype(np.float32))
        constraints = torch.from_numpy(data['constraints'].astype(np.float32))
        
        condition = torch.cat([ct, masks_sdf], dim=0)
        
        metadata = data['metadata'].item() if 'metadata' in data.files else {}
        
        return {
            'condition': condition,
            'dose': dose,
            'masks_binary': masks_binary,  # For DVH calculation
            'constraints': constraints,
            'metadata': metadata,
            'filepath': str(self.files[idx]),
        }


# =============================================================================
# Noise Schedules
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
    (Nichol & Dhariwal, 2021). Better for images than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from original DDPM paper."""
    return torch.linspace(beta_start, beta_end, timesteps)


# =============================================================================
# Sinusoidal Time Embedding
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


# =============================================================================
# 3D U-Net with FiLM Conditioning
# =============================================================================

class ConvBlock3D(nn.Module):
    """3D convolution block with GroupNorm and SiLU activation."""
    
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        
        # Time embedding projection (FiLM: scale and shift)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch * 2)  # scale and shift
            )
        else:
            self.time_mlp = None
        
        # Residual connection
        self.residual = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Apply time conditioning via FiLM
        if self.time_mlp is not None and t_emb is not None:
            t_out = self.time_mlp(t_emb)
            t_out = t_out.view(t_out.shape[0], -1, 1, 1, 1)
            scale, shift = t_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual(x)


class SimpleUNet3D(nn.Module):
    """
    Simplified 3D U-Net for 3090 memory constraints.
    Fewer levels and channels than full UNet3D.
    """
    
    def __init__(
        self,
        in_channels: int = 10,  # 1 (noisy dose) + 9 (CT + 8 SDFs)
        out_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        constraint_dim: int = 13,
    ):
        super().__init__()
        
        # Time + constraint embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.constraint_embed = nn.Sequential(
            nn.Linear(constraint_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Combined embedding dimension
        emb_dim = time_emb_dim * 2
        
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_channels, emb_dim)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, emb_dim)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, emb_dim)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8, emb_dim)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 8, emb_dim)
        
        # Decoder
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 4, emb_dim)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 2, emb_dim)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels, emb_dim)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels, emb_dim)
        
        # Output
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        constraints: torch.Tensor
    ) -> torch.Tensor:
        # Embeddings
        t_emb = self.time_embed(t)
        c_emb = self.constraint_embed(constraints)
        emb = torch.cat([t_emb, c_emb], dim=-1)
        
        # Encoder
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.pool(e1), emb)
        e3 = self.enc3(self.pool(e2), emb)
        e4 = self.enc4(self.pool(e3), emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4), emb)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1), emb)
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1), emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), emb)
        
        return self.out_conv(d1)


# =============================================================================
# DDPM Lightning Module
# =============================================================================

class DoseDDPM(pl.LightningModule):
    """
    Conditional DDPM for dose prediction.
    
    Uses epsilon (noise) prediction with cosine schedule.
    Supports both full DDPM sampling and faster DDIM sampling.
    """
    
    def __init__(
        self,
        in_channels: int = 10,  # 1 (noisy dose) + 9 (CT + 8 SDFs)
        timesteps: int = 1000,
        schedule: str = 'cosine',
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        use_simple_unet: bool = True,  # True for 3090, False for larger GPUs
        base_channels: int = 32,
        rx_dose_gy: float = 70.0,  # For denormalization
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        if use_simple_unet:
            self.model = SimpleUNet3D(
                in_channels=in_channels,
                out_channels=1,
                base_channels=base_channels,
            )
        else:
            raise NotImplementedError("Full UNet3D not implemented in this version")
        
        # Noise schedule (registered as buffers so they move with model)
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # Tracking
        self.validation_step_outputs = []
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        constraints: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss with physics-informed penalty.
        
        Loss = MSE(predicted_noise, actual_noise) + λ * negative_dose_penalty
        
        The negative dose penalty encourages the model to avoid predicting
        doses that would be negative after denoising, which is physically
        impossible.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Concatenate noisy dose with condition for input
        x_input = torch.cat([x_noisy, condition], dim=1)  # (B, 1+9, H, W, D)
        
        # Predict noise
        predicted_noise = self.model(x_input, t, constraints)
        
        # MSE loss on noise (primary loss)
        mse_loss = F.mse_loss(predicted_noise, noise)
        
        # Physics-informed loss: penalize predictions that would lead to negative dose
        # Estimate what dose would be predicted (approximate x0 from predicted noise)
        # x0 ≈ (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        # Predicted x0 (dose)
        pred_x0 = (x_noisy - sqrt_one_minus_alpha * predicted_noise) / (sqrt_alpha + 1e-8)
        
        # Penalize negative predictions (ReLU of negative values)
        negative_penalty = F.relu(-pred_x0).mean()
        
        # Combined loss (lambda = 0.1 for physics penalty)
        physics_weight = 0.1
        loss = mse_loss + physics_weight * negative_penalty
        
        return loss
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        condition = batch['condition']  # (B, 9, P, P, P)
        dose = batch['dose']            # (B, 1, P, P, P)
        constraints = batch['constraints']  # (B, 13)
        
        # Sample random timesteps
        t = torch.randint(
            0, self.hparams.timesteps, (dose.shape[0],), 
            device=self.device, dtype=torch.long
        )
        
        loss = self.p_losses(dose, condition, constraints, t)
        
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Validate on patches."""
        condition = batch['condition']
        dose = batch['dose']
        constraints = batch['constraints']
        
        # Sample random timesteps for validation loss
        t = torch.randint(
            0, self.hparams.timesteps, (dose.shape[0],),
            device=self.device, dtype=torch.long
        )
        
        val_loss = self.p_losses(dose, condition, constraints, t)
        
        self.log('val/loss', val_loss, prog_bar=True, on_epoch=True)
        
        # For first batch: do full denoising and compute metrics
        if batch_idx == 0:
            with torch.no_grad():
                # DDIM sampling for speed
                pred_dose = self.ddim_sample(
                    condition, constraints, 
                    shape=dose.shape[2:],
                    steps=50
                )
                
                # Compute MAE in Gy
                rx = self.hparams.rx_dose_gy
                mae_gy = F.l1_loss(pred_dose * rx, dose * rx)
                self.log('val/mae_gy', mae_gy, prog_bar=True)
                
                # Store for epoch-end metrics
                self.validation_step_outputs.append({
                    'pred': pred_dose.cpu(),
                    'target': dose.cpu(),
                    'masks': batch.get('masks_binary', None),
                })
    
    def on_validation_epoch_end(self) -> None:
        """Compute gamma pass rate at end of validation epoch."""
        if not self.validation_step_outputs or not HAS_PYMEDPHYS:
            self.validation_step_outputs.clear()
            return
        
        # Get first validation sample
        output = self.validation_step_outputs[0]
        pred = output['pred'][0, 0].numpy() * self.hparams.rx_dose_gy
        target = output['target'][0, 0].numpy() * self.hparams.rx_dose_gy
        
        # Compute gamma (subsampled for speed)
        try:
            gamma_result = self._compute_gamma_subsampled(pred, target, subsample=4)
            self.log('val/gamma_3mm3pct', gamma_result, prog_bar=True)
        except Exception as e:
            print(f"Gamma computation failed: {e}")
        
        self.validation_step_outputs.clear()
    
    def _compute_gamma_subsampled(
        self, 
        pred: np.ndarray, 
        target: np.ndarray, 
        subsample: int = 4,
        dose_threshold_pct: float = 10.0,
    ) -> float:
        """Compute gamma pass rate on subsampled volume."""
        # Subsample
        pred_sub = pred[::subsample, ::subsample, ::subsample]
        target_sub = target[::subsample, ::subsample, ::subsample]
        
        # Spacing after subsampling
        spacing_sub = tuple(s * subsample for s in DEFAULT_SPACING_MM)
        
        # Create coordinate axes
        axes = tuple(
            np.arange(s) * sp for s, sp in zip(pred_sub.shape, spacing_sub)
        )
        
        # Compute gamma
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_sub,
            axes_evaluation=axes,
            dose_evaluation=pred_sub,
            dose_percent_threshold=3.0,
            distance_mm_threshold=3.0,
            lower_percent_dose_cutoff=dose_threshold_pct,
        )
        
        # Pass rate
        valid = np.isfinite(gamma_map)
        if not valid.any():
            return 0.0
        
        pass_rate = np.mean(gamma_map[valid] <= 1.0) * 100
        return pass_rate
    
    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        constraints: torch.Tensor,
        shape: Tuple[int, ...],
        steps: int = 50,
        eta: float = 0.0,  # eta=0 for deterministic DDIM
    ) -> torch.Tensor:
        """
        DDIM sampling for faster inference.
        
        Args:
            condition: Conditioning input (B, 9, H, W, D)
            constraints: Constraint vector (B, 13)
            shape: Spatial shape (H, W, D)
            steps: Number of denoising steps
            eta: Stochasticity (0 = deterministic)
        
        Returns:
            Denoised dose prediction (B, 1, H, W, D)
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # Start from pure noise
        x = torch.randn((batch_size, 1, *shape), device=device)
        
        # Timestep schedule for DDIM
        step_size = self.hparams.timesteps // steps
        timesteps = list(range(0, self.hparams.timesteps, step_size))
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Model input
            x_input = torch.cat([x, condition], dim=1)
            
            # Predict noise
            pred_noise = self.model(x_input, t_batch, constraints)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 2)  # Clip to reasonable range
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev)) * pred_noise
            
            # Random noise for stochastic sampling
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x) * eta * torch.sqrt(1 - alpha_cumprod_t_prev)
            else:
                noise = 0
            
            # Update x
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + noise
        
        return torch.clamp(x, 0, 1.5)  # Dose can slightly exceed 1.0 (prescription)
    
    @torch.no_grad()
    def predict_full_volume(
        self,
        condition: torch.Tensor,
        constraints: torch.Tensor,
        patch_size: int = 128,
        overlap: int = 32,
        ddim_steps: int = 50,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Predict dose for full volume using sliding window with overlap averaging.
        
        Uses Gaussian weighting in overlap regions for smooth blending.
        
        Args:
            condition: Full volume conditioning (1, 9, H, W, D)
            constraints: Constraint vector (1, 13)
            patch_size: Size of patches to process
            overlap: Overlap between adjacent patches (should be >= 16)
            ddim_steps: Number of DDIM sampling steps
            verbose: Print progress
        
        Returns:
            Full volume dose prediction (1, 1, H, W, D)
        """
        device = condition.device
        _, C, H, W, D = condition.shape
        
        # Stride between patches
        stride = patch_size - overlap
        
        # Calculate number of patches in each dimension
        n_patches_h = max(1, (H - patch_size) // stride + 1)
        n_patches_w = max(1, (W - patch_size) // stride + 1)
        n_patches_d = max(1, (D - patch_size) // stride + 1)
        
        # Handle edge cases where volume is smaller than patch
        if H <= patch_size:
            n_patches_h = 1
        if W <= patch_size:
            n_patches_w = 1
        if D <= patch_size:
            n_patches_d = 1
        
        total_patches = n_patches_h * n_patches_w * n_patches_d
        
        if verbose:
            print(f"Sliding window inference:")
            print(f"  Volume: {H}x{W}x{D}")
            print(f"  Patch size: {patch_size}, Overlap: {overlap}, Stride: {stride}")
            print(f"  Grid: {n_patches_h}x{n_patches_w}x{n_patches_d} = {total_patches} patches")
        
        # Create Gaussian weight kernel for blending
        weight_kernel = self._create_gaussian_kernel(patch_size, sigma=patch_size/4)
        weight_kernel = weight_kernel.to(device)
        
        # Accumulators for weighted averaging
        output_sum = torch.zeros((1, 1, H, W, D), device=device)
        weight_sum = torch.zeros((1, 1, H, W, D), device=device)
        
        patch_idx = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                for k in range(n_patches_d):
                    # Calculate patch coordinates
                    h_start = min(i * stride, H - patch_size)
                    w_start = min(j * stride, W - patch_size)
                    d_start = min(k * stride, D - patch_size)
                    
                    h_end = h_start + patch_size
                    w_end = w_start + patch_size
                    d_end = d_start + patch_size
                    
                    # Extract patch
                    condition_patch = condition[:, :, h_start:h_end, w_start:w_end, d_start:d_end]
                    
                    # Predict dose for this patch
                    pred_patch = self.ddim_sample(
                        condition_patch,
                        constraints,
                        shape=(patch_size, patch_size, patch_size),
                        steps=ddim_steps,
                    )
                    
                    # Add weighted prediction to accumulators
                    output_sum[:, :, h_start:h_end, w_start:w_end, d_start:d_end] += pred_patch * weight_kernel
                    weight_sum[:, :, h_start:h_end, w_start:w_end, d_start:d_end] += weight_kernel
                    
                    patch_idx += 1
                    if verbose and patch_idx % 10 == 0:
                        print(f"  Processed {patch_idx}/{total_patches} patches")
        
        # Normalize by weights
        output = output_sum / (weight_sum + 1e-8)
        
        if verbose:
            print(f"  Done! Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return output
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 3D Gaussian kernel for smooth blending."""
        coords = torch.arange(size).float() - size // 2
        
        # 1D Gaussian
        gauss_1d = torch.exp(-coords**2 / (2 * sigma**2))
        
        # 3D Gaussian via outer product
        gauss_3d = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
        
        # Normalize
        gauss_3d = gauss_3d / gauss_3d.max()
        
        # Add batch and channel dims
        return gauss_3d.unsqueeze(0).unsqueeze(0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 200,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# =============================================================================
# Training Script
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
        description="Train DDPM for VMAT Dose Prediction (Phase 1)"
    )

    # Data
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                       help='Path to preprocessed .npz files (auto-detected if not specified)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio (held out entirely, not used during training)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (use 1-2 for 3090)')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for training')
    parser.add_argument('--patches_per_volume', type=int, default=4,
                       help='Number of patches to sample per volume per epoch')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    
    # Model
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--schedule', type=str, default='cosine',
                       choices=['cosine', 'linear'])
    parser.add_argument('--base_channels', type=int, default=48,
                       help='Base filter count (default: 48, reduce to 32 if OOM)')
    parser.add_argument('--use_simple_unet', action='store_true', default=True,
                       help='Use simplified U-Net for memory efficiency')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--exp_name', type=str, default='vmat_dose_ddpm')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--rx_dose_gy', type=float, default=70.0,
                       help='Prescription dose for denormalization')
    
    # Multi-GPU (for HPC)
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--strategy', type=str, default='auto',
                       help='Distributed strategy (auto, ddp, etc.)')

    # GPU stability options
    parser.add_argument('--gpu_cooling', action='store_true', default=False,
                       help='Enable GPU cooling pauses to prevent overheating')
    parser.add_argument('--cooling_interval', type=int, default=10,
                       help='Pause every N batches (if --gpu_cooling enabled)')
    parser.add_argument('--cooling_pause', type=float, default=0.5,
                       help='Pause duration in seconds (if --gpu_cooling enabled)')
    
    args = parser.parse_args()

    # Validate data_dir
    if args.data_dir is None:
        parser.error("--data_dir is required. No preprocessed data was auto-detected.\n"
                     "Run preprocessing first or specify the path manually.")

    # Seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Get all files and create reproducible split
    print(f"\nLoading data from {args.data_dir}")
    data_dir = Path(args.data_dir)
    all_files = sorted(list(data_dir.glob("*.npz")))
    
    if not all_files:
        raise ValueError(f"No .npz files found in {args.data_dir}")
    
    # Reproducible train/val/test split at the FILE level (not patch level)
    n_files = len(all_files)
    n_test = max(1, int(n_files * args.test_split)) if args.test_split > 0 else 0
    n_val = max(1, int(n_files * args.val_split))
    n_train = n_files - n_val - n_test
    
    if n_train < 1:
        raise ValueError(f"Not enough files for split: {n_files} files, "
                        f"val_split={args.val_split}, test_split={args.test_split}")
    
    # Shuffle with seed for reproducibility
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
    print(f"  Train cases: {[Path(f).stem for f in train_files[:5]]}{'...' if len(train_files) > 5 else ''}")
    print(f"  Val cases: {[Path(f).stem for f in val_files]}")
    if test_files:
        print(f"  Test cases: {[Path(f).stem for f in test_files]} [NOT USED IN TRAINING]")
    
    # Save test set file list for later evaluation
    if test_files:
        test_list_path = Path(args.log_dir) / args.exp_name / 'test_cases.json'
        test_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_list_path, 'w') as f:
            json.dump({
                'test_cases': [Path(f).stem for f in test_files],
                'test_files': test_files,
                'seed': args.seed,
            }, f, indent=2)
        print(f"  Test case list saved to: {test_list_path}")
    
    # Create datasets
    train_dataset = VMATDosePatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        augment=True,
        mode='train',
    )
    # Filter to only use train files
    train_dataset.files = [Path(f) for f in train_files]
    
    val_dataset = VMATDosePatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        augment=False,
        mode='val',
    )
    # Filter to only use val files
    val_dataset.files = [Path(f) for f in val_files]
    
    print(f"Train patches per epoch: {len(train_dataset)}")
    print(f"Val patches per epoch: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,  # Disabled for WSL stability
        prefetch_factor=2 if args.num_workers > 0 else None,  # Limit queue size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Smaller batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Model
    model = DoseDDPM(
        in_channels=1 + 9,  # Noisy dose + (CT + 8 SDFs)
        timesteps=args.timesteps,
        schedule=args.schedule,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_simple_unet=args.use_simple_unet,
        base_channels=args.base_channels,
        rx_dose_gy=args.rx_dose_gy,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {'SimpleUNet3D' if args.use_simple_unet else 'UNet3D'}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create experiment directory
    exp_dir = Path(args.log_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / 'checkpoints',
            filename='best-{epoch:03d}-{val/mae_gy:.2f}',
            monitor='val/mae_gy',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val/mae_gy',
            patience=20,
            mode='min',
            min_delta=0.1,
        ),
        RichProgressBar(),
        Timer(),  # Track epoch timing
        PublicationLoggingCallback(
            log_dir=exp_dir,
            train_files=train_files,
            val_files=val_files,
        ),
        GPUMemoryCallback(),
    ]

    # Add GPU cooling callback if requested
    if args.gpu_cooling:
        print(f"GPU cooling enabled: pause {args.cooling_pause}s every {args.cooling_interval} batches")
        callbacks.append(GPUCoolingCallback(
            pause_every_n_batches=args.cooling_interval,
            pause_seconds=args.cooling_pause,
        ))
    
    # Loggers - both TensorBoard and CSV for easy access
    loggers = [
        TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.exp_name,
        ),
        CSVLogger(
            save_dir=args.log_dir,
            name=args.exp_name,
        ),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 else 'auto',
        precision='16-mixed',  # Mixed precision for memory efficiency
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=4 if args.batch_size == 1 else 1,  # Effective batch size
        deterministic="warn",  # True incompatible with trilinear upsample backward
    )
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patch size: {args.patch_size}³")
    print(f"  Patches/volume: {args.patches_per_volume}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Log dir: {exp_dir}")
    print(f"  Devices: {args.devices}")
    print("="*60 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {callbacks[0].best_model_path}")
    print(f"Best MAE: {callbacks[0].best_model_score:.2f} Gy")
    print(f"\nPublication files saved to: {exp_dir}")
    print(f"  - training_config.json: Full configuration")
    print(f"  - training_summary.json: Results summary")
    print(f"  - epoch_metrics.csv: Per-epoch metrics")


if __name__ == '__main__':
    main()
