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

DEFAULT_SPACING_MM = (1.0, 1.0, 2.0)

SCRIPT_VERSION = "1.0.0"


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
        }
    
    def _sample_patch_center(self, dose: np.ndarray) -> Tuple[int, int, int]:
        """Sample patch center, biased toward high-dose regions."""
        ps = self.patch_size
        half = ps // 2
        
        # Valid range for center
        y_range = (half, dose.shape[0] - half)
        x_range = (half, dose.shape[1] - half)
        z_range = (half, dose.shape[2] - half)
        
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
        """Apply augmentations (same as diffusion: X-flip + translation only)."""
        
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
            from scipy.ndimage import shift as ndimage_shift
            
            max_shift = 16
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_z = np.random.randint(-max_shift, max_shift + 1)
            
            ct = ndimage_shift(ct, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            dose = ndimage_shift(dose, (shift_y, shift_x, shift_z), order=1, mode='nearest')
            sdf = ndimage_shift(sdf, (0, shift_y, shift_x, shift_z), order=1, mode='nearest')
        
        return np.ascontiguousarray(ct), np.ascontiguousarray(dose), np.ascontiguousarray(sdf)


# =============================================================================
# Baseline U-Net Model (No Time Embedding)
# =============================================================================

class ConvBlock3D(nn.Module):
    """3D convolution block with optional FiLM conditioning."""
    
    def __init__(self, in_ch: int, out_ch: int, cond_dim: Optional[int] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
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
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = BaselineUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            constraint_dim=constraint_dim,
        )
        
        self.validation_step_outputs = []
    
    def forward(self, condition: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        return self.model(condition, constraints)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        condition = batch['condition']
        dose = batch['dose']
        constraints = batch['constraints']
        
        # Forward pass
        pred_dose = self.model(condition, constraints)
        
        # MSE loss on dose
        loss = F.mse_loss(pred_dose, dose)
        
        # Physics penalty (negative dose)
        negative_penalty = F.relu(-pred_dose).mean()
        loss = loss + 0.1 * negative_penalty
        
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
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
        
        # Store for gamma computation
        if batch_idx == 0:
            self.validation_step_outputs.append({
                'pred': pred_dose.cpu(),
                'target': dose.cpu(),
            })
    
    def on_validation_epoch_end(self) -> None:
        """Compute gamma on first validation sample."""
        if not self.validation_step_outputs or not HAS_PYMEDPHYS:
            self.validation_step_outputs.clear()
            return
        
        output = self.validation_step_outputs[0]
        pred = output['pred'][0, 0].numpy() * self.hparams.rx_dose_gy
        target = output['target'][0, 0].numpy() * self.hparams.rx_dose_gy
        
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
        pred_sub = pred[::subsample, ::subsample, ::subsample]
        target_sub = target[::subsample, ::subsample, ::subsample]
        
        spacing_sub = tuple(s * subsample for s in DEFAULT_SPACING_MM)
        
        axes = tuple(
            np.arange(s) * sp for s, sp in zip(pred_sub.shape, spacing_sub)
        )
        
        gamma_map = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=target_sub,
            axes_evaluation=axes,
            dose_evaluation=pred_sub,
            dose_percent_threshold=3.0,
            distance_mm_threshold=3.0,
            lower_percent_dose_cutoff=dose_threshold_pct,
        )
        
        valid = np.isfinite(gamma_map)
        if not valid.any():
            return 0.0
        
        return float(np.mean(gamma_map[valid] <= 1.0) * 100)
    
    def predict_full_volume(
        self,
        condition: torch.Tensor,
        constraints: torch.Tensor,
        patch_size: int = 128,
        overlap: int = 32,
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
        
        # Sliding window
        stride = patch_size - overlap
        positions = []
        
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                for z in range(0, D - patch_size + 1, stride):
                    positions.append((y, x, z))
        
        # Add edge positions
        for y in [0, H - patch_size]:
            for x in [0, W - patch_size]:
                for z in [0, D - patch_size]:
                    if (y, x, z) not in positions:
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
        config = {
            'script': 'train_baseline_unet.py',
            'version': SCRIPT_VERSION,
            'model': 'BaselineUNet3D (Direct Regression)',
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
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--exp_name', type=str, default='baseline_unet')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rx_dose_gy', type=float, default=70.0)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')
    
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
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BaselineUNet3D (Direct Regression)")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
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
        deterministic=True,
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
