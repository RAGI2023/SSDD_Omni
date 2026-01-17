# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-view SSDD for 360° panorama reconstruction.

Architecture:
    4 fisheye views [B, 4, 3, H, W]
        ↓
    Shared-weight Encoder (applied 4 times)
        ↓
    Individual latents [B, 4, C, zH, zW]
        ↓
    Fusion Conv (multi-view → single latent)
        ↓
    Fused latent z [B, C, zH, zW]
        ↓
    Decoder (Teacher frozen / Student trainable)
        ↓
    Panorama reconstruction [B, 3, H_pano, W_pano]
"""

import torch
import torch.nn as nn
from typing import Optional, Mapping, Union
from pathlib import Path

from .ssdd import SSDD
from ..blocks.diag_gauss import DiagonalGaussianDistribution
from ..model_utils import TrainStepResult


class ViewEncoding(nn.Module):
    """
    Learnable view embeddings for multi-view inputs.

    Provides positional/directional encoding for each view to help the model
    distinguish between different viewing directions (front/right/back/left).

    Supports multiple encoding strategies:
        - 'learnable': Learnable embeddings (default)
        - 'sinusoidal': Sinusoidal positional encoding
        - 'directional': Based on 3D direction vectors
    """

    def __init__(
        self,
        z_dim: int,
        n_views: int = 4,
        encoding_type: str = "learnable",
        view_directions: Optional[list] = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_views = n_views
        self.encoding_type = encoding_type

        if encoding_type == "learnable":
            # Learnable embeddings for each view (most flexible)
            self.view_embeddings = nn.Parameter(torch.randn(n_views, z_dim, 1, 1))
            nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)

        elif encoding_type == "sinusoidal":
            # Sinusoidal encoding based on view index
            # Similar to Transformer positional encoding
            position = torch.arange(n_views).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, z_dim, 2) * (-torch.log(torch.tensor(10000.0)) / z_dim))

            pe = torch.zeros(n_views, z_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('view_embeddings', pe.unsqueeze(-1).unsqueeze(-1))

        elif encoding_type == "directional":
            # Encoding based on 3D direction vectors
            # view_directions: list of [x, y, z] for each view
            if view_directions is None:
                # Default: front, right, back, left
                view_directions = [
                    [0.0, 0.0, 1.0],   # front
                    [1.0, 0.0, 0.0],   # right
                    [0.0, 0.0, -1.0],  # back
                    [-1.0, 0.0, 0.0],  # left
                ]

            assert len(view_directions) == n_views

            # Project 3D directions to z_dim
            self.direction_proj = nn.Linear(3, z_dim)
            directions = torch.tensor(view_directions, dtype=torch.float32)
            self.register_buffer('directions', directions)

        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

    def forward(self, z_views: torch.Tensor) -> torch.Tensor:
        """
        Add view encodings to latent codes.

        Args:
            z_views: [B, N_views, C, zH, zW]

        Returns:
            z_views_encoded: [B, N_views, C, zH, zW]
        """
        B, N, C, zH, zW = z_views.shape
        assert N == self.n_views, f"Expected {self.n_views} views, got {N}"

        if self.encoding_type == "learnable" or self.encoding_type == "sinusoidal":
            # Broadcast and add embeddings
            # view_embeddings: [N_views, C, 1, 1] -> [1, N_views, C, zH, zW]
            embeddings = self.view_embeddings.unsqueeze(0).expand(B, -1, -1, zH, zW)
            z_views_encoded = z_views + embeddings

        elif self.encoding_type == "directional":
            # Project directions and add
            # directions: [N_views, 3] -> [N_views, C]
            embeddings = self.direction_proj(self.directions)  # [N_views, C]
            embeddings = embeddings.view(1, N, C, 1, 1).expand(B, -1, -1, zH, zW)
            z_views_encoded = z_views + embeddings

        return z_views_encoded


class FusionModule(nn.Module):
    """
    Fuses multi-view latent codes into a single panorama latent code.

    Now supports view encoding to inject directional information before fusion.

    Strategies:
        - 'concat_conv': Concatenate along channel dimension and apply conv
        - 'average': Simple average (baseline)
        - 'attention': Learned attention-based fusion
    """

    def __init__(
        self,
        z_dim: int,
        n_views: int = 4,
        fusion_type: str = "concat_conv",
        hidden_dim: Optional[int] = None,
        use_view_encoding: bool = True,
        view_encoding_type: str = "learnable",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_views = n_views
        self.fusion_type = fusion_type
        self.use_view_encoding = use_view_encoding

        # View encoding module
        if use_view_encoding:
            self.view_encoding = ViewEncoding(
                z_dim=z_dim,
                n_views=n_views,
                encoding_type=view_encoding_type,
            )

        if fusion_type == "concat_conv":
            # Concatenate views and project back to z_dim
            fusion_hidden = hidden_dim or (z_dim * 2)
            # Calculate appropriate number of groups (must divide fusion_hidden evenly)
            num_groups = min(32, fusion_hidden)
            while fusion_hidden % num_groups != 0 and num_groups > 1:
                num_groups -= 1

            self.fusion_conv = nn.Sequential(
                nn.Conv2d(z_dim * n_views, fusion_hidden, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, fusion_hidden),
                nn.SiLU(),
                nn.Conv2d(fusion_hidden, z_dim, kernel_size=3, padding=1),
            )

            # Aspect ratio adjustment for 2:1 panorama output
            # Upsample width by 2x: scale_factor=(height_scale, width_scale) = (1.0, 2.0)
            # Input: [B, C, zH, zW] -> Output: [B, C, zH, zW*2] for 2:1 panorama
            num_groups_z = min(32, z_dim)
            while z_dim % num_groups_z != 0 and num_groups_z > 1:
                num_groups_z -= 1

            self.aspect_adjust = nn.Sequential(
                nn.Upsample(scale_factor=(1.0, 2.0), mode='nearest'),  # (H, W) = (1x, 2x)
                nn.Conv2d(z_dim, z_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups_z, z_dim),
                nn.SiLU(),
            )

        elif fusion_type == "attention":
            # Attention-based fusion
            self.query_proj = nn.Conv2d(z_dim, z_dim, kernel_size=1)
            self.key_proj = nn.Conv2d(z_dim, z_dim, kernel_size=1)
            self.value_proj = nn.Conv2d(z_dim, z_dim, kernel_size=1)
            self.out_proj = nn.Conv2d(z_dim, z_dim, kernel_size=1)
        elif fusion_type == "average":
            # Simple average, no parameters
            pass
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, z_views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_views: [B, N_views, C, zH, zW]

        Returns:
            z_fused: [B, C, zH, zW*2] for 2:1 panorama (width doubled for aspect ratio)
        """
        B, N, C, zH, zW = z_views.shape
        assert N == self.n_views, f"Expected {self.n_views} views, got {N}"

        # Apply view encoding first (inject directional information)
        if self.use_view_encoding:
            z_views = self.view_encoding(z_views)

        if self.fusion_type == "concat_conv":
            # Concatenate along channel dimension
            z_concat = z_views.reshape(B, N * C, zH, zW)
            z_fused = self.fusion_conv(z_concat)  # [B, C, zH, zW]
            z_fused = self.aspect_adjust(z_fused)  # [B, C, zH, zW*2] for 2:1 panorama

        elif self.fusion_type == "attention":
            # Multi-head attention across views
            # Treat views as sequence: [B, N, C, zH, zW] -> [B, N, C, zH*zW]
            z_flat = z_views.reshape(B, N, C, zH * zW)

            # Compute attention weights across views
            # For simplicity, use average as query, each view as key/value
            q = self.query_proj(z_views.mean(dim=1))  # [B, C, zH, zW]
            q = q.reshape(B, C, zH * zW).transpose(1, 2)  # [B, zH*zW, C]

            attn_scores = []
            for i in range(N):
                k = self.key_proj(z_views[:, i])  # [B, C, zH, zW]
                k = k.reshape(B, C, zH * zW)  # [B, C, zH*zW]
                score = torch.bmm(q, k) / (C ** 0.5)  # [B, zH*zW, zH*zW]
                attn_scores.append(score.mean(dim=-1, keepdim=True))  # [B, zH*zW, 1]

            attn_weights = torch.softmax(torch.cat(attn_scores, dim=-1), dim=-1)  # [B, zH*zW, N]

            # Apply attention weights
            z_fused = torch.zeros(B, C, zH * zW, device=z_views.device, dtype=z_views.dtype)
            for i in range(N):
                v = self.value_proj(z_views[:, i])  # [B, C, zH, zW]
                v = v.reshape(B, C, zH * zW)  # [B, C, zH*zW]
                z_fused += v * attn_weights[:, :, i].unsqueeze(1)  # [B, C, zH*zW]

            z_fused = z_fused.reshape(B, C, zH, zW)
            z_fused = self.out_proj(z_fused)

        elif self.fusion_type == "average":
            # Simple average
            z_fused = z_views.mean(dim=1)

        return z_fused


class SSDDMultiView(nn.Module):
    """
    Multi-view SSDD for 360° panorama reconstruction from fisheye views.

    Input: 4 fisheye views [B, 4, 3, H, W]
    Output: Panorama reconstruction [B, 3, H_pano, W_pano]
    """

    def __init__(
        self,
        base_ssdd: Optional[SSDD] = None,
        fusion_type: str = "concat_conv",
        fusion_hidden_dim: Optional[int] = None,
        n_views: int = 4,
        use_view_encoding: bool = True,
        view_encoding_type: str = "learnable",
        # SSDD parameters (used if base_ssdd is None)
        encoder: Optional[Mapping | nn.Module] = None,
        encoder_checkpoint: Optional[str] = None,
        encoder_train: bool = False,
        decoder: Optional[Mapping] = None,
        decoder_image_size: Optional[list] = None,  # [W, H] for output panorama
        fm_trainer: Optional[Mapping] = None,
        fm_sampler: Optional[Mapping] = None,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()

        # Build base SSDD model (encoder + decoder)
        if base_ssdd is not None:
            self.base_ssdd = base_ssdd
        else:
            self.base_ssdd = SSDD(
                encoder=encoder,
                encoder_checkpoint=encoder_checkpoint,
                encoder_train=encoder_train,
                decoder=decoder,
                decoder_image_size=decoder_image_size,
                fm_trainer=fm_trainer,
                fm_sampler=fm_sampler,
                checkpoint=checkpoint,
            )

        self.n_views = n_views
        self.encoder_train = self.base_ssdd.encoder_train

        # Fusion module (with view encoding)
        z_dim = self.base_ssdd.encoder.z_dim
        self.fusion = FusionModule(
            z_dim=z_dim,
            n_views=n_views,
            fusion_type=fusion_type,
            hidden_dim=fusion_hidden_dim,
            use_view_encoding=use_view_encoding,
            view_encoding_type=view_encoding_type,
        )

    @property
    def encoder(self):
        return self.base_ssdd.encoder

    @property
    def decoder(self):
        return self.base_ssdd.decoder

    @property
    def fm_trainer(self):
        return self.base_ssdd.fm_trainer

    @property
    def fm_sampler(self):
        return self.base_ssdd.fm_sampler

    def encode_views(self, views: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Encode multiple views with shared-weight encoder.

        Args:
            views: [B, N_views, 3, H, W]

        Returns:
            z_views: [B, N_views, C, zH, zW]
            encoded_dists: List of DiagonalGaussianDistribution (length N_views)
        """
        B, N, C_in, H, W = views.shape
        assert N == self.n_views, f"Expected {self.n_views} views, got {N}"

        # Reshape to process all views in batch
        views_flat = views.reshape(B * N, C_in, H, W)

        # Encode with shared weights
        if self.encoder_train:
            encoded_flat = self.base_ssdd.encode(views_flat)
        else:
            with torch.no_grad():
                encoded_flat = self.base_ssdd.encode(views_flat)

        # Sample latent codes
        if self.training:
            z_flat = encoded_flat.sample()
        else:
            z_flat = encoded_flat.mode()

        # Reshape back to [B, N_views, C, zH, zW]
        _, C_z, zH, zW = z_flat.shape
        z_views = z_flat.reshape(B, N, C_z, zH, zW)

        # Split encoded distributions for KL computation
        encoded_dists = [
            DiagonalGaussianDistribution(encoded_flat.parameters[i::N])
            for i in range(N)
        ]

        return z_views, encoded_dists

    def encode(self, views: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode and fuse multi-view inputs.

        Args:
            views: [B, N_views, 3, H, W]

        Returns:
            Fused latent distribution (for compatibility)
        """
        z_views, encoded_dists = self.encode_views(views)
        z_fused = self.fusion(z_views)

        # Return as distribution for compatibility
        # Use mean of individual distributions
        mean = torch.stack([dist.mean for dist in encoded_dists]).mean(dim=0)
        logvar = torch.stack([dist.logvar for dist in encoded_dists]).mean(dim=0)

        return DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1))

    def decode(
        self,
        z: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode fused latent code to panorama."""
        return self.base_ssdd.decode(z, steps=steps, noise=noise)

    def forward(
        self,
        gt_views: torch.Tensor,
        gt_panorama: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        from_noise: bool = False,
        as_teacher: bool = False,
    ) -> Union[torch.Tensor, TrainStepResult]:
        """
        Forward pass for multi-view SSDD.

        Args:
            gt_views: [B, N_views, 3, H, W] - fisheye views
            gt_panorama: [B, 3, H_pano, W_pano] - ground truth panorama (for training)
            steps: Number of sampling steps (for inference)
            noise: Optional noise (for teacher/student)
            z: Optional pre-computed latent (for teacher/student)
            from_noise: Whether to train from noise (distillation)
            as_teacher: Whether to act as teacher (return z and noise)

        Returns:
            If training: TrainStepResult with losses
            If inference: Reconstructed panorama [B, 3, H_pano, W_pano]
        """
        # Use gt_panorama as reconstruction target
        if gt_panorama is None:
            # If no panorama provided, we can't train properly
            # This should come from EquiDataset's img_original
            raise ValueError("gt_panorama is required for training/inference")

        gt_x = gt_panorama

        # Encode multi-view inputs
        encoded_dists = None
        if z is None:
            z_views, encoded_dists = self.encode_views(gt_views)
            z = self.fusion(z_views)

        # Use base SSDD decoder for the rest
        if not self.training:
            # Inference mode
            if as_teacher:
                if noise is None:
                    noise = torch.randn_like(gt_x)
            x_gen = self.decode(z, steps=steps, noise=noise)
            if as_teacher:
                return x_gen, z, noise
            return x_gen
        else:
            # Training mode
            t = self.fm_trainer.sample_t(gt_x.shape[0], device=gt_x.device)
            if from_noise:
                t = torch.ones_like(t)

            # Diffusion loss
            diff_loss, (x_t, noise_used, noise_t, v_pred) = self.fm_trainer.loss(
                self.decoder, x=gt_x, t=t, fn_kwargs={"z": z}, noise=noise
            )

            # Compute x0 prediction
            x0_pred = self.fm_trainer.step(x_t, v_pred, noise_t)

            # Compute losses
            losses = {"diffusion": diff_loss}

            # Add KL loss for all views if encoder is trainable
            if encoded_dists is not None and self.encoder_train:
                kl_losses = [dist.kl().mean() for dist in encoded_dists]
                losses["kl"] = torch.stack(kl_losses).mean()

            return TrainStepResult(
                x0_gt=gt_x,
                x0_pred=x0_pred,
                xt=x_t,
                t=t,
                z=z,
                noise=noise_used,
                losses=losses,
            )

    def get_last_layer_weight(self):
        return self.base_ssdd.get_last_layer_weight()

    ### Loading / Checkpointing ###

    def load(self, weights: Union[str, Path, Mapping], strict: bool = True, freeze=False, eval=None):
        """Load weights (supports loading base SSDD weights)."""
        if not isinstance(weights, Mapping):
            from safetensors.torch import load_file as safe_load_file
            weights = safe_load_file(weights)

        # Try to load full state dict
        try:
            self.load_state_dict(weights, strict=strict)
        except RuntimeError as e:
            # If loading fails, try to load base_ssdd weights only
            print(f"Warning: Could not load full state dict ({e})")
            print("Attempting to load base_ssdd weights only...")
            base_weights = {k.replace("base_ssdd.", ""): v for k, v in weights.items() if k.startswith("base_ssdd.")}
            if not base_weights:
                base_weights = weights  # Assume weights are for base model
            self.base_ssdd.load_state_dict(base_weights, strict=False)

        if eval or (eval is None and freeze):
            self.eval()
        if freeze:
            self.requires_grad_(False)
        return self
