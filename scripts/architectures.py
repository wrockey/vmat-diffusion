"""
Architecture variants for VMAT dose prediction.

Contains alternative 3D U-Net architectures for the Phase 2 ablation study
(architecture comparison conditions C11-C16). All variants share the same
ConvBlock3D and FiLM conditioning from the baseline, and have identical
forward signatures: forward(x, constraints) -> dose.

Architectures:
    - AttentionUNet3D: Oktay et al. (2018) attention gates at all 4 skip connections
    - BottleneckAttnUNet3D: Multi-head self-attention at bottleneck only
    - BaselineUNet3D: Imported from train_baseline_unet.py (wider variant via --base_channels)

References:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," 2018.
    Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from train_baseline_unet import ConvBlock3D, BaselineUNet3D


# =============================================================================
# Attention Gate (Oktay et al., 2018)
# =============================================================================

class AttentionGate3D(nn.Module):
    """
    Attention gate for 3D U-Net skip connections (Oktay et al., 2018).

    Projects skip (encoder) and gate (decoder) features to an intermediate
    dimension via 1x1x1 conv, applies ReLU + sigmoid to produce spatial
    attention coefficients, then element-wise multiplies on skip features.

    This learns to suppress irrelevant spatial regions in skip connections,
    focusing the decoder on clinically relevant areas (PTV/OAR boundaries).
    """

    def __init__(self, skip_channels: int, gate_channels: int):
        """
        Args:
            skip_channels: Number of channels in the skip connection (encoder output)
            gate_channels: Number of channels in the gating signal (decoder output)
        """
        super().__init__()

        inter_channels = skip_channels // 2

        # Project skip and gate to intermediate dimension
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False)

        # Attention coefficient: intermediate -> 1 channel
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip: (B, skip_channels, H, W, D) encoder features
            gate: (B, gate_channels, H', W', D') decoder features (may be smaller)

        Returns:
            (B, skip_channels, H, W, D) attention-weighted skip features
        """
        # Project both to intermediate dim
        skip_proj = self.W_skip(skip)
        gate_proj = self.W_gate(gate)

        # Upsample gate to match skip spatial dims if needed
        if gate_proj.shape[2:] != skip_proj.shape[2:]:
            gate_proj = F.interpolate(
                gate_proj, size=skip_proj.shape[2:],
                mode='trilinear', align_corners=False,
            )

        # Additive attention + sigmoid
        attn = self.relu(skip_proj + gate_proj)
        attn = self.psi(attn)  # (B, 1, H, W, D)

        return skip * attn


# =============================================================================
# Attention U-Net (Oktay et al., 2018)
# =============================================================================

class AttentionUNet3D(nn.Module):
    """
    3D U-Net with attention gates at all 4 skip connections.

    Same encoder/bottleneck/decoder as BaselineUNet3D, but inserts
    AttentionGate3D at each skip connection before concatenation.
    The gate signal comes from the upsampled decoder feature map.

    ~5-8% parameter increase over baseline due to attention gate projections.
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
        ch = [base_channels * (2**i) for i in range(5)]

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, ch[0], cond_dim)
        self.enc2 = ConvBlock3D(ch[0], ch[1], cond_dim)
        self.enc3 = ConvBlock3D(ch[1], ch[2], cond_dim)
        self.enc4 = ConvBlock3D(ch[2], ch[3], cond_dim)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(ch[3], ch[3], cond_dim)

        # Attention gates (skip_channels from encoder, gate_channels from decoder/bottleneck)
        self.attn4 = AttentionGate3D(skip_channels=ch[3], gate_channels=ch[3])
        self.attn3 = AttentionGate3D(skip_channels=ch[2], gate_channels=ch[2])
        self.attn2 = AttentionGate3D(skip_channels=ch[1], gate_channels=ch[1])
        self.attn1 = AttentionGate3D(skip_channels=ch[0], gate_channels=ch[0])

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

        # Decoder with attention-gated skip connections
        up4 = self.up4(b)
        e4_attn = self.attn4(e4, up4)
        d4 = self.dec4(torch.cat([up4, e4_attn], dim=1), cond)

        up3 = self.up3(d4)
        e3_attn = self.attn3(e3, up3)
        d3 = self.dec3(torch.cat([up3, e3_attn], dim=1), cond)

        up2 = self.up2(d3)
        e2_attn = self.attn2(e2, up2)
        d2 = self.dec2(torch.cat([up2, e2_attn], dim=1), cond)

        up1 = self.up1(d2)
        e1_attn = self.attn1(e1, up1)
        d1 = self.dec1(torch.cat([up1, e1_attn], dim=1), cond)

        # Output
        out = self.out_conv(d1)

        return out


# =============================================================================
# Bottleneck Self-Attention
# =============================================================================

class BottleneckSelfAttention3D(nn.Module):
    """
    Pre-norm multi-head self-attention for 3D feature maps.

    Reshapes 3D feature map (B, C, H, W, D) to a sequence of spatial tokens
    (B, H*W*D, C), applies standard QKV multi-head attention, then reshapes
    back. Includes residual connection and layer normalization.

    At the bottleneck (8x8x8 spatial after 4 pooling stages from 128^3),
    this produces 512 tokens — trivially cheap for attention.
    Captures long-range beam-path dependencies that convolutions miss.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of feature channels (C dimension)
            num_heads: Number of attention heads (must divide channels)
        """
        super().__init__()

        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Pre-norm
        self.norm = nn.LayerNorm(channels)

        # QKV projection (single linear for efficiency)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)

        # Output projection
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D) 3D feature map

        Returns:
            (B, C, H, W, D) attention-refined feature map (with residual)
        """
        B, C, H, W, D = x.shape

        # Reshape to sequence: (B, C, H, W, D) -> (B, H*W*D, C)
        x_seq = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        N = x_seq.shape[1]

        # Pre-norm
        x_normed = self.norm(x_seq)

        # QKV projection
        qkv = self.qkv(x_normed).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)

        # Recombine heads
        attn = attn.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Output projection
        out = self.proj(attn)

        # Residual connection
        out = x_seq + out

        # Reshape back to 3D: (B, N, C) -> (B, C, H, W, D)
        out = out.permute(0, 2, 1).reshape(B, C, H, W, D)

        return out


# =============================================================================
# Bottleneck Attention U-Net
# =============================================================================

class BottleneckAttnUNet3D(nn.Module):
    """
    3D U-Net with multi-head self-attention at the bottleneck only.

    Same as BaselineUNet3D but inserts BottleneckSelfAttention3D after the
    bottleneck ConvBlock. At 128^3 input with 4 pooling stages, the bottleneck
    is 8^3 = 512 tokens, making attention trivially cheap (~2-3% param increase).

    Captures long-range spatial dependencies (beam paths, opposing field
    interactions) that purely convolutional bottlenecks cannot model.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_channels: int = 48,
        constraint_dim: int = 13,
        attention_heads: int = 4,
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
        ch = [base_channels * (2**i) for i in range(5)]

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, ch[0], cond_dim)
        self.enc2 = ConvBlock3D(ch[0], ch[1], cond_dim)
        self.enc3 = ConvBlock3D(ch[1], ch[2], cond_dim)
        self.enc4 = ConvBlock3D(ch[2], ch[3], cond_dim)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck: conv block + self-attention
        self.bottleneck = ConvBlock3D(ch[3], ch[3], cond_dim)
        self.bottleneck_attn = BottleneckSelfAttention3D(ch[3], num_heads=attention_heads)

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

        # Bottleneck with self-attention
        b = self.bottleneck(self.pool(e4), cond)
        b = self.bottleneck_attn(b)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), cond)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), cond)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)

        # Output
        out = self.out_conv(d1)

        return out


# =============================================================================
# Architecture Registry & Factory
# =============================================================================

ARCHITECTURE_REGISTRY = {
    'baseline': BaselineUNet3D,
    'attention_unet': AttentionUNet3D,
    'bottleneck_attn': BottleneckAttnUNet3D,
}

ARCHITECTURE_DESCRIPTIONS = {
    'baseline': 'BaselineUNet3D (standard 3D U-Net with FiLM conditioning)',
    'attention_unet': 'AttentionUNet3D (Oktay et al. 2018, attention gates at all skip connections)',
    'bottleneck_attn': 'BottleneckAttnUNet3D (multi-head self-attention at bottleneck only)',
}


def build_model(
    architecture: str,
    in_channels: int = 9,
    out_channels: int = 1,
    base_channels: int = 48,
    constraint_dim: int = 13,
) -> nn.Module:
    """
    Factory function to create architecture variants.

    All models have identical forward signatures:
        model(x, constraints) -> dose
    where x is (B, in_channels, H, W, D) and constraints is (B, constraint_dim).

    Args:
        architecture: One of 'baseline', 'attention_unet', 'bottleneck_attn'
        in_channels: Input channels (CT + SDFs)
        out_channels: Output channels (dose)
        base_channels: Base filter count (channel progression: bc, 2*bc, 4*bc, 8*bc, 16*bc)
        constraint_dim: Dimension of constraint conditioning vector

    Returns:
        nn.Module with forward(x, constraints) -> dose

    Raises:
        ValueError: If architecture name not in registry
    """
    if architecture not in ARCHITECTURE_REGISTRY:
        valid = ', '.join(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Valid options: {valid}")

    model_cls = ARCHITECTURE_REGISTRY[architecture]

    return model_cls(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        constraint_dim=constraint_dim,
    )
