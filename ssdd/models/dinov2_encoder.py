# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DINOv2-based encoder adapter for SSDD.

Wraps a pretrained DINOv2 model to produce latent codes compatible with SSDD decoder.
DINOv2 is a vision transformer that produces patch embeddings, which we project to
the latent space required by the diffusion decoder.
"""

import torch
import torch.nn as nn
from typing import Optional

from .blocks.diag_gauss import DiagonalGaussianDistribution


class DinoV2Encoder(nn.Module):
    """
    DINOv2-based encoder for SSDD.

    Architecture:
        Input image [B, 3, H, W]
            ↓
        DINOv2 backbone (frozen ViT)
            ↓
        Patch embeddings [B, N_patches, embed_dim]
            ↓
        Reshape to spatial grid [B, embed_dim, h, w]
            ↓
        Projection conv to z_dim
            ↓
        DiagonalGaussianDistribution [B, z_dim*2, h, w]

    Args:
        z_dim: Target latent dimension (must match decoder)
        model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        patch_size: Patch size of DINOv2 (14 for DINOv2 models)
        freeze_backbone: Whether to freeze DINOv2 weights (default: True)
        use_registers: Whether to use DINOv2 with registers (default: False)
    """

    def __init__(
        self,
        z_dim: int = 4,
        model_name: str = 'dinov2_vitb14',
        patch_size: int = 14,
        freeze_backbone: bool = True,
        use_registers: bool = False,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.patch_size = patch_size
        self.model_name = model_name

        # Load pretrained DINOv2
        print(f"Loading pretrained DINOv2 model: {model_name}")

        # Force torch.hub to avoid Hugging Face downloads
        # Set trust_repo to skip verification and speed up loading
        try:
            # Try to load with registers if requested
            if use_registers:
                model_name_reg = model_name + '_reg'
                self.dinov2 = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name_reg,
                    trust_repo=True,
                    skip_validation=True
                )
            else:
                self.dinov2 = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name,
                    trust_repo=True,
                    skip_validation=True
                )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print("Trying without registers...")
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                trust_repo=True,
                skip_validation=True
            )

        # Get DINOv2 embedding dimension
        self.embed_dim = self.dinov2.embed_dim
        print(f"DINOv2 embedding dimension: {self.embed_dim}")

        # Freeze DINOv2 if requested
        if freeze_backbone:
            print("Freezing DINOv2 backbone")
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()

        # Projection layers to convert DINOv2 features to latent space
        # We use double_z=True for compatibility with VAE-style encoders
        self.double_z = True
        out_channels = 2 * z_dim if self.double_z else z_dim

        # Projection: embed_dim -> z_dim
        # Use a small conv network to adapt features
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.embed_dim // 2),
            nn.SiLU(),
            nn.Conv2d(self.embed_dim // 2, out_channels, kernel_size=3, padding=1),
        )

        # Initialize projection layers
        self._init_projection_weights()

    def _init_projection_weights(self):
        """Initialize projection layers with small weights."""
        for m in self.feature_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Forward pass through DINOv2 encoder.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            DiagonalGaussianDistribution with parameters [B, z_dim*2, h, w]
            where h = H // patch_size, w = W // patch_size
        """
        B, C, H, W = x.shape
        assert C == 3, f"Expected 3 input channels, got {C}"

        # Ensure input size is compatible with patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}x{W}) must be divisible by patch_size ({self.patch_size})"

        # Extract features from DINOv2
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dinov2.parameters())):
            # Get patch embeddings (without CLS token)
            features = self.dinov2.forward_features(x)
            # features['x_norm_patchtokens']: [B, N_patches, embed_dim]
            patch_features = features['x_norm_patchtokens']

        # Reshape to spatial grid
        # N_patches = (H // patch_size) * (W // patch_size)
        h = H // self.patch_size
        w = W // self.patch_size

        # Reshape: [B, N_patches, embed_dim] -> [B, embed_dim, h, w]
        spatial_features = patch_features.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)

        # Project to latent space
        latent = self.feature_proj(spatial_features)  # [B, z_dim*2, h, w]

        # Return as DiagonalGaussianDistribution
        return DiagonalGaussianDistribution(latent, deterministic=False)

    @staticmethod
    def get_config(
        z_dim: int,
        patch_size: int = 14,
        model_name: str = 'dinov2_vitb14',
        **kwargs
    ):
        """Get default config for DinoV2Encoder."""
        return {
            'z_dim': z_dim,
            'patch_size': patch_size,
            'model_name': model_name,
            'freeze_backbone': True,
            **kwargs
        }

    @classmethod
    def make(cls, z_dim: int, patch_size: int = 14, model_name: str = 'dinov2_vitb14', **kwargs):
        """Factory method to create DinoV2Encoder."""
        cfg = cls.get_config(z_dim=z_dim, patch_size=patch_size, model_name=model_name, **kwargs)
        return cls(**cfg)
