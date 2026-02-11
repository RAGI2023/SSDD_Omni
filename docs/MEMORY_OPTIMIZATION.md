# Memory Optimization Guide for DINOv2 Multi-View

This guide helps you configure SSDD Multi-View with DINOv2 encoder based on your GPU memory.

## Memory Usage Breakdown

| Component | Memory Usage (approx) |
|-----------|----------------------|
| **DINOv2 ViT-B** | ~3.5 GB (frozen, inference mode) |
| **Decoder (size M)** | ~2-3 GB |
| **Decoder (size S)** | ~1-2 GB |
| **Fusion (concat_conv)** | ~500 MB |
| **Fusion (attention)** | ~1-2 GB |
| **Activations (batch=4)** | ~8-12 GB |
| **Compile overhead** | ~2-4 GB |

**Total estimate**: 15-25 GB depending on configuration

## Recommended Settings by GPU

### 24GB GPU (e.g., RTX 3090, RTX 4090)

**Conservative (Recommended)**:
```yaml
dataset:
  batch_size: 4

ssdd:
  compile: false
  decoder: S
  fusion_type: concat_conv

training:
  grad_accumulate: 3  # Effective batch = 12
  mixed_precision: bf16
```

**Aggressive (May OOM)**:
```yaml
dataset:
  batch_size: 6

ssdd:
  compile: false
  decoder: S
  fusion_type: concat_conv

training:
  grad_accumulate: 2  # Effective batch = 12
```

### 40GB GPU (e.g., A100)

```yaml
dataset:
  batch_size: 8

ssdd:
  compile: true
  decoder: M
  fusion_type: concat_conv

training:
  grad_accumulate: 2  # Effective batch = 16
  mixed_precision: bf16
```

### 80GB GPU (e.g., A100 80GB)

```yaml
dataset:
  batch_size: 12

ssdd:
  compile: true
  decoder: M
  fusion_type: attention  # Can use attention fusion

training:
  grad_accumulate: 1
  mixed_precision: bf16
```

## Memory Reduction Strategies

### 1. Reduce Batch Size
**Impact**: Most effective
**Trade-off**: Need higher `grad_accumulate`

```yaml
dataset:
  batch_size: 2  # Start small if OOM
```

### 2. Use Smaller Decoder
**Impact**: Moderate
**Trade-off**: Slightly lower quality

```yaml
ssdd:
  decoder: S  # Instead of M or L
```

### 3. Disable Compile
**Impact**: Moderate (2-4 GB saved)
**Trade-off**: 10-20% slower training

```yaml
ssdd:
  compile: false
```

### 4. Use Simpler Fusion
**Impact**: Moderate
**Trade-off**: Minimal quality impact

```yaml
ssdd:
  fusion_type: concat_conv  # Instead of attention
```

### 5. Reduce Image Size
**Impact**: Large
**Trade-off**: Lower resolution

```yaml
dataset:
  im_size: 168  # Instead of 224 (must be divisible by 14)

ssdd:
  decoder_image_size: [336, 168]  # Instead of [448, 224]
```

### 6. Use Gradient Checkpointing
Add to encoder (requires code modification):

```python
# In dinov2_encoder.py
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Use gradient checkpointing for DINOv2
    features = checkpoint(self.dinov2.forward_features, x)
    ...
```

### 7. Enable Memory-Efficient Settings

Set environment variables before training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
```

## Current Optimized Config

The current `SpiderEye_dinov2.yaml` is optimized for **24GB GPUs**:

```yaml
dataset:
  batch_size: 4

ssdd:
  compile: false
  decoder: S
  fusion_type: concat_conv

training:
  grad_accumulate: 3  # Effective batch = 12
```

This should fit comfortably in 24GB with ~2-4GB headroom.

## Troubleshooting OOM

### Error: "CUDA out of memory"

**Quick fixes** (try in order):

1. **Reduce batch_size**:
   ```yaml
   dataset:
     batch_size: 2  # Or even 1
   ```

2. **Disable compile**:
   ```yaml
   ssdd:
     compile: false
   ```

3. **Use smaller decoder**:
   ```yaml
   ssdd:
     decoder: S
   ```

4. **Clear cache before training**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

5. **Set memory expansion**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   accelerate launch ssdd/main_multiview.py ...
   ```

### Error: "fragmentd memory"

Add to training script:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

## Monitor Memory Usage

During training, monitor with:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

Or add to code:

```python
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

## Alternative: Use Smaller DINOv2

If ViT-B is too large, use ViT-S:

```yaml
ssdd:
  encoder: dinov2_vits14_p14_c4  # Instead of vitb14
```

**Memory savings**: ~1.5 GB
**Trade-off**: Slightly lower quality

## Summary

**Default (24GB GPU)**:
- batch_size: 4
- decoder: S
- compile: false
- fusion: concat_conv
- grad_accumulate: 3

**Expected memory**: ~18-20 GB
**Training speed**: ~80% of compiled version
**Quality**: Minimal impact vs. full config
