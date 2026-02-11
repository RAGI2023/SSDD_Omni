# DINOv2 Encoder Integration - Changes Summary

## Overview

This document summarizes the changes made to integrate pretrained DINOv2 as an encoder option for SSDD Multi-View, with the encoder **frozen by default**.

## Files Added

### 1. `ssdd/models/dinov2_encoder.py`
**New encoder implementation based on DINOv2**

- Wraps pretrained DINOv2 (ViT) from `torch.hub`
- Projects DINOv2 patch embeddings to latent space compatible with SSDD decoder
- Supports all DINOv2 variants: ViT-S/B/L/G with 14x14 patches
- Backbone frozen by default (`freeze_backbone=True`)
- Returns `DiagonalGaussianDistribution` for compatibility with VAE-style training

Key features:
- Automatic loading from torch.hub (no manual checkpoint needed)
- Learnable projection layers to adapt features
- Compatible with multi-view architecture

### 2. `config/SpiderEye_dinov2.yaml`
**Configuration file for training with DINOv2**

Key settings:
```yaml
ssdd:
  encoder: dinov2_vitb14_p14_c4  # ViT-B, patch_size=14, z_dim=4
  encoder_train: false            # Keep frozen

dataset:
  im_size: 224                    # Must be divisible by 14

decoder_image_size: [448, 224]    # Must be divisible by 14
```

### 3. `docs/DINOV2_ENCODER.md`
**Comprehensive documentation**

Covers:
- Architecture overview
- Configuration format
- Usage examples
- Comparison with VQEncoder
- Training tips
- Troubleshooting

### 4. `scripts/train_dinov2_multiview.sh`
**Training script example**

Provides command-line examples for:
- Basic training with ViT-B
- Alternative model variants (ViT-S, ViT-L)
- Different batch sizes

### 5. `scripts/test_dinov2_encoder.py`
**Integration test script**

Tests:
- DinoV2Encoder forward pass
- Shape compatibility
- Frozen backbone verification
- SSDDMultiView integration
- Multi-view encoding pipeline

## Files Modified

### 1. `ssdd/models/ssdd/ssdd.py`

**Import added:**
```python
from ..dinov2_encoder import DinoV2Encoder
```

**`make_encoder` method updated:**
- Added regex pattern to recognize DINOv2 config format
- Pattern: `dinov2_{model}_p{patch_size}_c{z_dim}`
- Example: `dinov2_vitb14_p14_c4`
- Supports `_reg` suffix for register variants

```python
dinov2_cfg_re = r"^dinov2_(vits14|vitb14|vitl14|vitg14)(?:_reg)?_p(\d+)_c(\d+)$"
if dinov2_cfg_match:
    model_variant = dinov2_cfg_match.group(1)
    patch_size = int(dinov2_cfg_match.group(2))
    z_dim = int(dinov2_cfg_match.group(3))
    use_registers = '_reg' in encoder
    model_name = f"dinov2_{model_variant}"
    encoder = DinoV2Encoder.make(
        z_dim=z_dim,
        patch_size=patch_size,
        model_name=model_name,
        use_registers=use_registers,
        freeze_backbone=True
    )
```

**Backward compatibility:** VQEncoder still works with `f{patch_size}c{z_dim}` format

## Usage

### Quick Start

1. **Test the integration:**
   ```bash
   python scripts/test_dinov2_encoder.py
   ```

2. **Train with DINOv2:**
   ```bash
   accelerate launch ssdd/main_multiview.py \
       --config-name=SpiderEye_dinov2 \
       run_name=my_dinov2_experiment
   ```

3. **Evaluate:**
   ```bash
   accelerate launch ssdd/main_multiview.py \
       --config-name=SpiderEye_dinov2 \
       task=eval \
       ssdd.checkpoint=path/to/checkpoint
   ```

### Configuration Options

#### Available DINOv2 Models

| Config String | Model | Embed Dim | Params |
|---------------|-------|-----------|--------|
| `dinov2_vits14_p14_c4` | ViT-Small | 384 | 21M |
| `dinov2_vitb14_p14_c4` | ViT-Base | 768 | 86M |
| `dinov2_vitl14_p14_c4` | ViT-Large | 1024 | 304M |
| `dinov2_vitg14_p14_c4` | ViT-Giant | 1536 | 1.1B |

Add `_reg` for register variants: `dinov2_vitb14_reg_p14_c4`

#### Image Size Requirements

**IMPORTANT:** All image dimensions must be divisible by 14 (DINOv2's patch size)

Valid sizes: 224, 280, 336, 392, 448, 560, etc.

Example:
```yaml
dataset:
  im_size: 224  # ✓ 224 = 14 * 16

ssdd:
  decoder_image_size: [448, 224]  # ✓ 448=14*32, 224=14*16
```

## Architecture Comparison

### Before (VQEncoder)
```
Views → CNN Encoder (ResNet-style) → Latent → Fusion → Decoder → Panorama
        [trainable, ~20M params]
```

### After (DinoV2Encoder)
```
Views → DINOv2 ViT (frozen) → Projection (trainable) → Latent → Fusion → Decoder → Panorama
        [frozen, 86M params]  [trainable, ~1M params]
```

## Key Benefits

1. **Pretrained Features**: Leverage DINOv2's powerful representations
2. **Frozen Backbone**: No need to train 86M parameters
3. **Better Semantics**: ViT captures high-level visual understanding
4. **Reduced Training**: Only fusion + decoder need training
5. **Plug-and-Play**: Easy to switch between VQEncoder and DINOv2

## Migration Path

To convert existing experiments to DINOv2:

1. **Update encoder config:**
   ```yaml
   # Old
   encoder: f8c4  # VQEncoder with patch_size=8

   # New
   encoder: dinov2_vitb14_p14_c4  # DINOv2 with patch_size=14
   ```

2. **Adjust image sizes (divisible by 14):**
   ```yaml
   dataset:
     im_size: 224  # Changed from 128

   ssdd:
     decoder_image_size: [448, 224]  # Changed from [256, 128]
   ```

3. **Optionally reduce batch size (DINOv2 uses more memory):**
   ```yaml
   dataset:
     batch_size: 8  # Reduced from 12

   training:
     grad_accumulate: 2  # Compensate
   ```

## Testing

Run the test script to verify everything works:

```bash
python scripts/test_dinov2_encoder.py
```

Expected output:
```
✓ DinoV2Encoder works correctly
✓ SSDDMultiView integration successful
✓ Multi-view encoding pipeline functional
```

## Known Limitations

1. **Patch Size**: Fixed at 14 (cannot use 8 or 16)
2. **Memory**: DINOv2 ViT-B uses ~2x memory of VQEncoder
3. **Image Size**: Must be multiples of 14
4. **Download**: First run downloads ~300MB model from torch.hub

## Future Improvements

Potential enhancements:
- [ ] Support for DINOv1
- [ ] Adaptive patch size projection
- [ ] LoRA fine-tuning of DINOv2
- [ ] Mixed precision optimization
- [ ] Cached feature extraction

## Support

For issues or questions:
- See documentation: `docs/DINOV2_ENCODER.md`
- Run tests: `python scripts/test_dinov2_encoder.py`
- Check configuration: `config/SpiderEye_dinov2.yaml`

## References

- DINOv2 Paper: https://arxiv.org/abs/2304.07193
- DINOv2 GitHub: https://github.com/facebookresearch/dinov2
- Torch Hub: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`
