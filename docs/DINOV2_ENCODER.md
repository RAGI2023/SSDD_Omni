# DINOv2 Encoder for SSDD Multi-View

This document describes how to use pretrained DINOv2 as the encoder for SSDD multi-view panorama reconstruction.

## Overview

The DINOv2 encoder replaces the default VQEncoder with a pretrained vision transformer from Meta's DINOv2. The DINOv2 backbone is **frozen by default**, allowing you to leverage powerful pretrained features without fine-tuning.

## Architecture

```
Input Views [B, N_views, 3, H, W]
    ↓
Shared DINOv2 Encoder (frozen ViT)
    ↓
Patch Embeddings [B, N_views, N_patches, embed_dim]
    ↓
Reshape to Spatial Grid [B, N_views, embed_dim, h, w]
    ↓
Projection Conv (trainable)
    ↓
Latent Codes [B, N_views, z_dim, h, w]
    ↓
Fusion Module
    ↓
Fused Latent [B, z_dim, h, w*2]
    ↓
Decoder (Flow Matching)
    ↓
Panorama [B, 3, H_pano, W_pano]
```

## Configuration

### Encoder Format

The encoder is configured using the following pattern:

```
dinov2_{model}_p{patch_size}_c{z_dim}
```

Where:
- `model`: DINOv2 variant
  - `vits14`: ViT-Small (embed_dim=384)
  - `vitb14`: ViT-Base (embed_dim=768) **[Recommended]**
  - `vitl14`: ViT-Large (embed_dim=1024)
  - `vitg14`: ViT-Giant (embed_dim=1536)
- `patch_size`: Patch size (14 for all DINOv2 models)
- `z_dim`: Target latent dimension (typically 4)

**Note**: Add `_reg` suffix for models with registers (e.g., `dinov2_vitb14_reg_p14_c4`)

### Example Configuration

```yaml
ssdd:
  encoder: dinov2_vitb14_p14_c4  # ViT-Base with patch_size=14, z_dim=4
  encoder_checkpoint: null       # Not needed (loads from torch.hub)
  encoder_train: false           # Keep frozen (recommended)
```

### Image Size Requirements

Since DINOv2 uses patch_size=14, input images must have dimensions divisible by 14:

```yaml
dataset:
  im_size: 224  # 224 = 14 * 16 ✓
  # im_size: 128  # 128 is not divisible by 14 ✗
```

Recommended sizes: 224, 280, 336, 448, 560, etc.

### Output Panorama Size

The decoder output size must also be divisible by 14:

```yaml
ssdd:
  decoder_image_size: [448, 224]  # Width, Height (2:1 for panorama)
  # 448 = 14 * 32, 224 = 14 * 16 ✓
```

## Usage

### Training with DINOv2 Encoder

Use the provided configuration file:

```bash
accelerate launch ssdd/main_multiview.py \
    --config-name=SpiderEye_dinov2 \
    run_name=dinov2_multiview_train
```

Or override the encoder in an existing config:

```bash
accelerate launch ssdd/main_multiview.py \
    --config-name=SpiderEye \
    ssdd.encoder=dinov2_vitb14_p14_c4 \
    dataset.im_size=224 \
    ssdd.decoder_image_size=[448,224]
```

### Evaluation

```bash
accelerate launch ssdd/main_multiview.py \
    --config-name=SpiderEye_dinov2 \
    task=eval \
    ssdd.checkpoint=path/to/checkpoint \
    ssdd.fm_sampler.steps=12
```

## Key Differences from VQEncoder

| Feature | VQEncoder | DinoV2Encoder |
|---------|-----------|---------------|
| Architecture | CNN (ResNet-style) | Vision Transformer |
| Pretraining | From scratch or checkpoint | Pretrained on ImageNet-1k/22k |
| Patch Size | 8 (f8c4) | 14 (fixed for DINOv2) |
| Parameters | ~20M | 86M (ViT-B) |
| Frozen | Optional | **Yes (recommended)** |
| Image Size | Any multiple of 8 | Any multiple of 14 |

## Benefits of DINOv2

1. **Strong Pretrained Features**: DINOv2 is trained on large-scale data with self-supervision
2. **Zero-Shot Transfer**: Works well without fine-tuning the encoder
3. **Semantic Understanding**: Better captures high-level visual semantics
4. **Fewer Trainable Parameters**: Only fusion and decoder need training

## Training Tips

1. **Start with Frozen Encoder**: Keep `encoder_train: false` initially
2. **Adjust Learning Rate**: May need lower LR since encoder is frozen
3. **Batch Size**: DINOv2 has larger memory footprint; reduce batch_size if OOM
4. **Model Selection**:
   - ViT-S: Faster, lower memory, good for prototyping
   - **ViT-B**: Best balance of performance and efficiency
   - ViT-L/G: Highest quality, but requires more memory

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient checkpointing:
```yaml
training:
  batch_size: 8  # Reduce from 12
  grad_accumulate: 2  # Compensate with accumulation
```

### Dimension Mismatch

Ensure all sizes are divisible by 14:
- Input images: `dataset.im_size`
- Output panorama: `ssdd.decoder_image_size`

### Download Issues

DINOv2 is downloaded from torch.hub automatically. If you have network issues:
```python
# Pre-download manually
import torch
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```

## Implementation Details

See:
- Encoder: [`ssdd/models/dinov2_encoder.py`](../ssdd/models/dinov2_encoder.py)
- Integration: [`ssdd/models/ssdd/ssdd.py`](../ssdd/models/ssdd/ssdd.py) (make_encoder method)
- Config: [`config/SpiderEye_dinov2.yaml`](../config/SpiderEye_dinov2.yaml)
