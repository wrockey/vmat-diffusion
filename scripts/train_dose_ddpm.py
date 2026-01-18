"""
DEPRECATED: Use train_dose_ddpm_v2.py instead.

This script is kept for reference only. The v2 version includes:
- Patch-based training (fits on RTX 3090)
- SDFs instead of binary masks
- FiLM conditioning for timestep and constraints
- Proper cosine noise schedule
- DDIM sampling for fast inference
- Publication-ready logging

Conditional DDPM Trainer for VMAT Dose Prediction (Goal 4 Feasibility)

Usage: python train_dose_ddpm_v2.py --data_dir ./processed --epochs 200
"""
import warnings
warnings.warn(
    "train_dose_ddpm.py is deprecated. Use train_dose_ddpm_v2.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import RandomRotation, RandomHorizontalFlip
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import pymedphys
import time

# Custom Callback for Global ETA
class ETACallback(Callback):
    def __init__(self):
        self.epoch_times = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - trainer.profiler.summary['start_time']  # Approx per epoch
        self.epoch_times.append(epoch_time)
        if len(self.epoch_times) > 1:
            avg_time = np.mean(self.epoch_times)
            remaining_epochs = trainer.max_epochs - trainer.current_epoch
            eta_hours = (avg_time * remaining_epochs) / 3600
            trainer.logger.log_metrics({'eta_remaining_hours': eta_hours})

# ----------------------------- Dataset -----------------------------
class VMATDoseDataset(Dataset):
    """
    Loads .npz files: condition = CT (1ch) + masks (Nch); target = dose (1ch normalized).
    Constraints vector broadcast to spatial dims or embedded (here: concat as channels).
    Augmentations: Random rot/flip for robustness.
    """
    def __init__(self, data_dir, constraint_channels=True, augment=False):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.constraint_channels = constraint_channels
        self.augment = augment
        self.rot = RandomRotation(degrees=15)
        self.flip = RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        ct = torch.from_numpy(data['ct']).float().unsqueeze(0)          # [1, H, W, D]
        masks = torch.from_numpy(data['masks']).float()                 # [N, H, W, D]
        dose = torch.from_numpy(data['dose']).float().unsqueeze(0)      # [1, H, W, D]
        constraints = torch.from_numpy(data['constraints']).float()     # [C]

        # Condition: CT + masks
        condition = torch.cat([ct, masks], dim=0)  # [1+N, H, W, D]

        # Optional: Broadcast constraints as extra channels (repeat spatially)
        if self.constraint_channels:
            c_map = constraints.view(1, -1, 1, 1, 1).expand(-1, -1, condition.shape[2], condition.shape[3], condition.shape[4])
            condition = torch.cat([condition, c_map], dim=1)

        # Augmentations (apply same to condition and dose)
        if self.augment:
            combined = torch.cat([condition, dose], dim=0)  # Temp stack
            combined = self.rot(combined)
            combined = self.flip(combined)
            condition = combined[:-1]
            dose = combined[-1].unsqueeze(0)

        return {'condition': condition, 'dose': dose, 'file': self.files[idx]}

# ----------------------------- 3D U-Net -----------------------------
class UNet3D(nn.Module):
    """
    Simple 3D U-Net for noise prediction. Input channels = 1 (CT) + N_masks (+ C_constraints).
    Output: 1 channel (noise or dose residual).
    """
    def __init__(self, in_channels=9, base_filters=32, timesteps=1000):
        super().__init__()
        # Time embedding (sinusoidal)
        self.time_emb = nn.Sequential(
            nn.Embedding(timesteps, base_filters),
            nn.Linear(base_filters, base_filters * 4),
            nn.SiLU(),
            nn.Linear(base_filters * 4, base_filters * 4)
        )

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self.conv_block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(base_filters * 8, base_filters * 16)

        # Decoder
        self.dec4 = self.conv_block(base_filters * 24, base_filters * 8)
        self.dec3 = self.conv_block(base_filters * 12, base_filters * 4)
        self.dec2 = self.conv_block(base_filters * 6, base_filters * 2)
        self.dec1 = self.conv_block(base_filters * 3, base_filters)

        self.out = nn.Conv3d(base_filters, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_emb(t).view(-1, self.time_emb[1].out_features, 1, 1, 1)  # Adjusted for base*4

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e4, 2))
        b = b + t_emb.expand(-1, -1, b.shape[2], b.shape[3], b.shape[4])  # Add time

        # Decoder with skip connections
        d4 = F.interpolate(b, scale_factor=2, mode='trilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# ----------------------------- DDPM Module -----------------------------
class DoseDDPM(pl.LightningModule):
    """
    Conditional DDPM using U-Net noise predictor.
    Standard Gaussian forward process; predicts epsilon (noise).
    Added: Augmentations, DVH metric (pymedphys), DDIM sampler.
    """
    def __init__(self, data_dir, in_channels=9, batch_size=1, timesteps=1000, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet3D(in_channels=in_channels, base_filters=32, timesteps=timesteps)
        self.beta_schedule = torch.linspace(1e-4, 0.02, timesteps).to(self.device)  # Linear
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        return self.model(x, t)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def training_step(self, batch, batch_idx):
        condition = batch['condition']
        dose = batch['dose']
        t = torch.randint(0, self.hparams.timesteps, (dose.shape[0],), device=self.device).long()
        noise = torch.randn_like(dose)
        x_noisy = self.q_sample(dose, t, noise)
        x_input = torch.cat([x_noisy, condition], dim=1)  # [B, 2+N+C, H, W, D]
        pred_noise = self(x_input, t)
        loss = F.mse_loss(pred_noise, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        condition = batch['condition']
        dose = batch['dose']
        # Use DDIM for better denoised pred (faster than full reverse)
        pred_dose = self.ddim_sample(condition, shape=dose.shape[1:], steps=50, eta=0.0)
        mae_gy = F.l1_loss(pred_dose * 70.0, dose * 70.0)
        self.log('val_mae_gy', mae_gy, prog_bar=True)

        # DVH metric (pymedphys; simple mean dose proxy, extend to full DVH)
        pred_mean_gy = torch.mean(pred_dose * 70.0).item()
        gt_mean_gy = torch.mean(dose * 70.0).item()
        dvh_diff = abs(pred_mean_gy - gt_mean_gy)  # Placeholder; use pymedphys.dicom for full
        self.log('val_dvh_mean_diff_gy', dvh_diff, prog_bar=True)

        # Gamma on full vol (expensive; run on batch_idx==0)
        if batch_idx == 0:
            dose_pred = pred_dose[0, 0].cpu().numpy() * 70
            dose_gt = dose[0, 0].cpu().numpy() * 70
            spacing = (1.0, 1.0, 2.0)  # mm
            gamma_pass = pymedphys.gamma(
                [spacing], [3, 3], dose_gt, dose_pred,
                dose_percent_threshold=3, distance_mm_threshold=3
            )
            valid = np.isfinite(gamma_pass)
            pass_rate = np.mean(gamma_pass[valid] <= 1) * 100 if valid.any() else 0
            self.log('val_gamma_3mm3%', pass_rate, prog_bar=True)

    def ddim_sample(self, condition, shape, steps=50, eta=0.0):
        """
        DDIM sampler for faster inference (deterministic if eta=0).
        """
        x = torch.randn((1, 1, *shape), device=self.device)
        for i in reversed(range(0, self.hparams.timesteps, self.hparams.timesteps // steps)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            x_input = torch.cat([x, condition], dim=1)
            pred_noise = self(x_input, t)
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_cumprod_t_prev = self.alpha_cumprod[max(t- self.hparams.timesteps // steps, torch.tensor(0))]
            sigma = eta * ((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)).sqrt()
            pred_x0 = (x - (1 - alpha_cumprod_t).sqrt() * pred_noise) / alpha_cumprod_t.sqrt()
            dir_xt = (1 - alpha_cumprod_t_prev - sigma**2).sqrt() * pred_noise
            x = alpha_cumprod_t_prev.sqrt() * pred_x0 + dir_xt + sigma * torch.randn_like(x)
        return x.clamp(0, 1)  # Normalized dose

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ----------------------------- Main Trainer -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed_npz')
    parser.add_argument('--log_dir', type=str, default='runs/ddpm_dose')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to ckpt for resume')
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)  # Reproducibility

    # Dataset (augment train only)
    dataset = VMATDoseDataset(args.data_dir, constraint_channels=True, augment=False)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_ds.dataset.augment = True  # Enable aug on train subset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = DoseDDPM(
        data_dir=args.data_dir,
        in_channels=1 + 8 + 13,  # CT + ~8 masks + ~13 constraints broadcast
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        lr=args.lr
    )

    # Callbacks & Logger
    checkpoint = ModelCheckpoint(save_top_k=3, monitor='val_mae_gy', mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop = EarlyStopping(monitor='val_mae_gy', patience=10, min_delta=0.1, mode='min', check_finite=True)
    eta_callback = ETACallback()
    logger = TensorBoardLogger(args.log_dir, name='prostate_dose_feasibility')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # AMP for 3090 memory
        logger=logger,
        callbacks=[checkpoint, lr_monitor, early_stop, eta_callback],
        log_every_n_steps=10,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Best model: {checkpoint.best_model_path}")

if __name__ == '__main__':
    main()

# TODO: DVH loss term (e.g., weighted MSE on PTV/OAR voxels via pymedphys)
# TODO: After feasibility, append RP machine params and extend model