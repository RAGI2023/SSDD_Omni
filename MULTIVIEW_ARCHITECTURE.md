# Multi-View SSDD Architecture for 360° Panorama Reconstruction

## 概述

本文档详细说明多视角 SSDD 架构，用于从 4 个鱼眼视图重建 360° 全景图。

---

## 架构设计

### 完整数据流

```
输入: 4 个鱼眼视图
[B, 4, 3, H, W]
    │
    ├─> View 0 (front)  ─┐
    ├─> View 1 (right)  ─┤
    ├─> View 2 (back)   ─┤  共享权重 Encoder
    └─> View 3 (left)   ─┘  (VQEncoder: f8c4)
          │
          ├─> z0 [B, C, zH, zW]  ─┐
          ├─> z1 [B, C, zH, zW]  ─┤
          ├─> z2 [B, C, zH, zW]  ─┤  Fusion Module
          └─> z3 [B, C, zH, zW]  ─┘  (Conv / Attention / Average)
                │
                ├─> z_fused [B, C, zH, zW]
                │
                ├─> Teacher Decoder (冻结, 可选)
                │   (UViT + Flow Matching)
                │
                └─> 输出: 全景图
                    [B, 3, H_pano, W_pano]
```

---

## 核心模块详解

### 1. Shared-Weight Encoder

**实现**: [ssdd_multiview.py:135-162](ssdd/models/ssdd/ssdd_multiview.py#L135-L162)

```python
def encode_views(self, views: torch.Tensor):
    """
    使用共享权重的 Encoder 编码所有视角。

    输入: views [B, N_views, 3, H, W]
    输出: z_views [B, N_views, C, zH, zW]
    """
    B, N, C_in, H, W = views.shape

    # Reshape: [B, N, 3, H, W] -> [B*N, 3, H, W]
    views_flat = views.reshape(B * N, C_in, H, W)

    # 共享权重编码 (一次前向传播处理所有视角)
    z_flat = self.encoder(views_flat)

    # Reshape back: [B*N, C, zH, zW] -> [B, N, C, zH, zW]
    z_views = z_flat.reshape(B, N, C_z, zH, zW)

    return z_views
```

**关键优势**:
- ✅ 参数高效：4 个视角共享同一个 Encoder
- ✅ 批处理优化：一次前向传播处理所有视角
- ✅ 视角不变性：每个视角使用相同的特征提取器

---

### 2. Fusion Module

**实现**: [ssdd_multiview.py:30-121](ssdd/models/ssdd/ssdd_multiview.py#L30-L121)

提供 3 种融合策略：

#### 策略 A: Concatenation + Convolution (默认)

```python
fusion_type = "concat_conv"

# 1. 拼接所有视角的隐变量
z_concat = [z0, z1, z2, z3] -> [B, 4*C, zH, zW]

# 2. 使用 Conv 网络融合
z_fused = Conv([B, 4*C, zH, zW] -> [B, C, zH, zW])
```

**架构**:
```python
nn.Sequential(
    nn.Conv2d(z_dim * 4, z_dim * 2, kernel_size=3, padding=1),
    nn.GroupNorm(32, z_dim * 2),
    nn.SiLU(),
    nn.Conv2d(z_dim * 2, z_dim, kernel_size=3, padding=1),
)
```

**优势**:
- ✅ 简单有效
- ✅ 能学习视角间的空间关系
- ✅ 参数适中

#### 策略 B: Attention-Based Fusion

```python
fusion_type = "attention"

# 1. 计算每个视角的注意力权重
attn_weights = softmax(Query @ Keys) -> [B, N_views]

# 2. 加权求和
z_fused = Σ(attn_weights[i] * Values[i])
```

**优势**:
- ✅ 自适应权重：不同位置关注不同视角
- ✅ 可解释性：可视化注意力权重
- ⚠️ 参数较多

#### 策略 C: Simple Average (Baseline)

```python
fusion_type = "average"

z_fused = mean(z0, z1, z2, z3)
```

**优势**:
- ✅ 无参数
- ✅ 计算高效
- ⚠️ 性能可能受限

---

### 3. Teacher-Student Distillation (可选)

**实现**: [SpiderTask_MultiView.py:329-344](ssdd/SpiderTask_MultiView.py#L329-L344)

```python
if "teacher" in self.models:
    # Teacher: 直接在全景图空间工作 (多步采样)
    with torch.no_grad():
        target_panorama, z, noise = self.models["teacher"](panorama, as_teacher=True)

    # Student: 从多视角输入学习 (单步采样)
    ssdd_out = self.models["ae"](views, gt_panorama=target_panorama, z=z, noise=noise, from_noise=True)
```

**流程**:
```
Teacher (冻结):
    全景图 -> Encoder -> z -> Decoder (12 steps) -> 高质量全景图

Student (训练):
    4 视角 -> 共享 Encoder -> Fusion -> z -> Decoder (1 step) -> 模仿 Teacher
```

**优势**:
- ✅ Student 学习 Teacher 的高质量输出
- ✅ 单步采样，推理快速
- ✅ 保持 Teacher 的重建质量

---

## 数据流详解

### Training Loop

```python
for batch in train_loader:
    views, panorama = batch
    # views: [B, 4, 3, 128, 128]      # 4 个鱼眼视图
    # panorama: [B, 3, 256, 128]      # 目标全景图

    # 1. Multi-view encoding
    z_views = encoder(views)          # [B, 4, C, zH, zW]

    # 2. Fusion
    z_fused = fusion(z_views)         # [B, C, zH, zW]

    # 3. Decoding
    panorama_pred = decoder(z_fused)  # [B, 3, 256, 128]

    # 4. Loss
    loss = mse_loss(panorama_pred, panorama) + aux_losses
```

### Inference

```python
# 输入: 4 个鱼眼视图
views = load_fisheye_views()  # [1, 4, 3, 128, 128]

# 前向推理
with torch.no_grad():
    panorama = model(views, steps=1)  # [1, 3, 256, 128]

# 输出: 重建的全景图
save_panorama(panorama)
```

---

## 配置文件

### SpiderEye_MultiView.yaml

```yaml
defaults:
  - _self_
  - override hydra/job_logging: colorlog

seed: 0
task: train

# Paths
runtime_path: ${hydra:runtime.cwd}
ckpt_dir: ${runtime_path}/runs
run_name: ${now:%Y-%m-%d}/${now:%H-%M-%S}
cache_dir: ${ckpt_dir}/cache
run_dir: ${ckpt_dir}/jobs/${run_name}
checkpoint_path: ${run_dir}/checkpoints

# Dataset (EquiDataset for multi-view)
dataset:
  imagenet_root: /data/360SP-data
  im_size: 128
  batch_size: 256
  aug_scale: null
  limit: null
  return_all_views: true  # CRITICAL: Enable multi-view mode

  # EquiDataset UCM parameters
  f_pix: 220.0
  xi: 0.9
  mask_mode: inscribed

# Model (Multi-View SSDD)
distill_teacher: true  # Enable teacher-student distillation

ssdd:
  compile: true
  checkpoint: null  # Path to teacher model (required if distill_teacher=true)

  encoder: f8c4
  encoder_checkpoint: null
  encoder_train: false
  decoder: M

  # Multi-view specific
  fusion_type: concat_conv  # "concat_conv" | "attention" | "average"
  fusion_hidden_dim: null   # Optional: override default hidden dim
  n_views: 4

  fm_sampler:
    steps: 12  # Teacher steps (12 for distillation, 1 for inference)
    t_pow_shift: 2.0

  ema:
    decay: 0.999
    start_iter: 50_000

# Auxiliary losses
aux_losses:
  compile: ${ssdd.compile}
  repa:
    i_extract: 4
    n_layers: 2
  lpips: true

# Training
training:
  mixed_precision: bf16
  grad_accumulate: 1
  grad_clip: 0.1
  epochs: 300
  eval_freq: 4
  save_on_best: FID
  log_freq: 200

  lr: 8e-4
  weight_decay: 1e-2

# Losses
losses:
  diffusion: 1
  repa: 0.25
  lpips: 0.5
  kl: 1e-6

# Evaluation
show_samples: 8
```

---

## 使用方法

### 1. 数据准备

```bash
/data/360SP-data/
├── train/
│   ├── pano001.jpg  # 等距圆柱投影全景图
│   ├── pano002.jpg
│   └── ...
└── val/
    ├── pano_val001.jpg
    └── ...
```

### 2. 训练 Teacher 模型 (可选)

```bash
# 先训练一个标准的 SSDD 作为 Teacher
accelerate launch ssdd/main.py \
    run_name=teacher_f8c4_M \
    training.epochs=100 \
    dataset.im_size=128 \
    ssdd.fm_sampler.steps=12
```

### 3. 训练 Multi-View Student

创建主文件 `ssdd/main_multiview.py`:

```python
import hydra
from omegaconf import DictConfig
from ssdd.SpiderTask_MultiView import SpiderTasksMultiView

@hydra.main(version_base=None, config_path="../config", config_name="SpiderEye_MultiView")
def main(cfg: DictConfig):
    task = SpiderTasksMultiView(cfg)
    task()

if __name__ == "__main__":
    main()
```

训练命令:

```bash
# 从头训练 (不使用 Teacher)
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_student \
    distill_teacher=false \
    training.epochs=100

# 蒸馏训练 (使用 Teacher)
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_distill \
    distill_teacher=true \
    ssdd.checkpoint=teacher_f8c4_M@best \
    training.epochs=10 \
    training.lr=1e-4
```

### 4. 评估

```bash
accelerate launch ssdd/main_multiview.py \
    task=eval \
    ssdd.checkpoint=multiview_distill@best \
    ssdd.fm_sampler.steps=1  # Single-step inference
```

---

## 关键参数调优

### Fusion Module

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `fusion_type` | `"concat_conv"` | 默认选择，平衡性能和效率 |
| `fusion_hidden_dim` | `z_dim * 2` | Concat+Conv 的隐藏层维度 |

**实验建议**:
1. 先用 `"average"` 建立 baseline
2. 切换到 `"concat_conv"` 提升性能
3. 如需进一步提升，尝试 `"attention"`

### UCM Camera Parameters

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `f_pix` | 220.0 | 焦距，控制视野范围 |
| `xi` | 0.9 | 镜面参数，控制畸变 |
| `mask_mode` | `"inscribed"` | 圆形遮罩 (鱼眼标准) |

### Training Hyperparameters

| 参数 | Teacher | Student (Distill) |
|------|---------|-------------------|
| `epochs` | 100-300 | 10-20 |
| `lr` | 8e-4 | 1e-4 |
| `fm_sampler.steps` | 8-12 | 12 (train), 1 (eval) |

---

## 性能优化

### 内存优化

1. **减小批量大小**:
   ```yaml
   dataset:
     batch_size: 128  # 从 256 降低到 128
   ```

2. **减小图像尺寸**:
   ```yaml
   dataset:
     im_size: 96  # 从 128 降低到 96
   ```

3. **使用 gradient checkpointing** (需修改代码):
   ```python
   from torch.utils.checkpoint import checkpoint
   z_views = checkpoint(self.encoder, views_flat)
   ```

### 速度优化

1. **启用编译**:
   ```yaml
   ssdd:
     compile: true
   ```

2. **增加 num_workers**:
   ```python
   loader = DataLoader(..., num_workers=16)
   ```

3. **使用混合精度**:
   ```yaml
   training:
     mixed_precision: bf16  # or fp16
   ```

---

## 调试指南

### 检查数据加载

```python
from ssdd.dataset_equi import load_equirect

cfg = {
    'imagenet_root': '/data/360SP-data',
    'im_size': 128,
    'batch_size': 4,
    'limit': 10,
    'return_all_views': True,
}

(train_ds, test_ds), (train_loader, test_loader) = load_equirect(cfg)

for views, panorama in train_loader:
    print(f"Views: {views.shape}")      # [4, 4, 3, 128, 128]
    print(f"Panorama: {panorama.shape}") # [4, 3, 256, 128]
    break
```

### 检查模型前向传播

```python
from ssdd.models.ssdd.ssdd_multiview import SSDDMultiView

model = SSDDMultiView(
    encoder="f8c4",
    decoder="M",
    fusion_type="concat_conv",
)

# Dummy input
views = torch.randn(2, 4, 3, 128, 128)
panorama = torch.randn(2, 3, 256, 128)

# Forward
output = model(views, gt_panorama=panorama)
print(f"Output shape: {output.x0_pred.shape}")  # [2, 3, 256, 128]
print(f"Losses: {output.losses}")
```

### 常见问题

#### Q1: "Expected 4 views, got X"

**原因**: `return_all_views` 未设置为 `True`

**解决**:
```yaml
dataset:
  return_all_views: true  # 确保启用
```

#### Q2: "Shape mismatch in fusion"

**原因**: 视角数量不匹配

**解决**:
```yaml
ssdd:
  n_views: 4  # 确保与 EquiDataset.VIEWS 一致
```

#### Q3: Memory OOM

**解决方案**:
1. 减小 `batch_size`
2. 减小 `im_size`
3. 使用 `gradient_accumulation`
4. 使用 `fp16` 混合精度

---

## 架构优势

### 相比单视图方法

| 特性 | 单视图 | 多视图 (本架构) |
|------|--------|-----------------|
| **信息完整性** | ❌ 单一透视视角 | ✅ 360° 全覆盖 |
| **遮挡处理** | ❌ 无法处理 | ✅ 多视角互补 |
| **重建质量** | ⚠️ 受限于单视角 | ✅ 更丰富的特征 |
| **参数效率** | ✅ 简单 | ✅ 共享权重 |

### 设计亮点

1. **共享权重 Encoder**: 4 个视角共享，参数高效
2. **灵活的 Fusion**: 3 种策略可选，适应不同场景
3. **Teacher-Student**: 可选蒸馏，提升推理速度
4. **即插即用**: 兼容原 SSDD 架构，无缝集成

---

## 下一步改进

### 短期优化
- [ ] 添加视角选择机制 (动态选择最相关的视角)
- [ ] 实现 cross-attention fusion (视角间交互)
- [ ] 优化 GPU 内存使用 (gradient checkpointing)

### 长期研究
- [ ] 扩展到 6 视角 (包含上/下视角)
- [ ] 端到端联合优化 UCM 参数
- [ ] 引入深度信息 (stereo fisheye)

---

## 参考文献

1. **SSDD**: Single-Step Diffusion Decoder for Efficient Image Tokenization
2. **UCM**: Unified Camera Model for fisheye projection
3. **Flow Matching**: Conditional Flow Matching for generative modeling

---

## 总结

本架构实现了：
- ✅ 多视角输入 (4 个鱼眼视图)
- ✅ 共享权重 Encoder
- ✅ 灵活的 Fusion Module
- ✅ Teacher-Student 蒸馏
- ✅ 完整的训练/评估流程

**核心价值**: 将 360° 全景重建从单视图透视提升到多视角鱼眼，提供更完整的信息和更高的重建质量。
