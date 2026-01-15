# Multi-View SSDD Quick Start Guide

## 快速开始：5 分钟上手多视角 SSDD

---

## 你的需求回顾

**目标**: 4 个鱼眼视角 → 共享权重 Encoder → Fusion Conv → Teacher Decoder → 全景图

**实现状态**: ✅ 已完成！

---

## 文件清单

我已经为你创建了以下文件：

### 核心实现
1. **`ssdd/models/ssdd/ssdd_multiview.py`** - 多视角 SSDD 模型
   - `SSDDMultiView`: 主模型类
   - `FusionModule`: 融合模块 (3 种策略)

2. **`ssdd/dataset_equi.py`** - EquiDataset 包装器 (已更新)
   - 支持 `return_all_views=True` 返回所有 4 个视角

3. **`ssdd/SpiderTask_MultiView.py`** - 多视角训练任务
   - 完整的训练/评估循环
   - Teacher-Student 蒸馏支持

4. **`ssdd/main_multiview.py`** - 主入口文件

### 文档
5. **`MULTIVIEW_ARCHITECTURE.md`** - 完整架构文档
6. **`QUICK_START_MULTIVIEW.md`** - 本文件

---

## Step 1: 检查配置

修改 `config/SpiderEye.yaml` (已经部分配置好):

```yaml
dataset:
  imagenet_root: /data/360SP-data
  im_size: 128
  batch_size: 256
  return_all_views: true  # ← 添加这一行 (关键!)

  # EquiDataset 参数 (可选，使用默认值)
  f_pix: 220.0
  xi: 0.9
  mask_mode: inscribed

distill_teacher: true  # 如果要使用 teacher，设为 true

ssdd:
  checkpoint: null  # teacher checkpoint 路径 (如果 distill_teacher=true)
  fusion_type: concat_conv  # ← 添加: "concat_conv" | "attention" | "average"
  n_views: 4  # ← 添加: 视角数量
```

---

## Step 2: 准备数据

确保数据目录结构正确：

```bash
/data/360SP-data/
├── train/
│   ├── pano001.jpg  # 等距圆柱投影全景图 (2:1 宽高比)
│   ├── pano002.jpg
│   └── ...
└── val/
    ├── pano_val001.jpg
    └── ...
```

**验证数据**:
```bash
ls /data/360SP-data/train | head -5
ls /data/360SP-data/val | head -5
```

---

## Step 3: 训练

### 选项 A: 从头训练 (不使用 Teacher)

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_baseline \
    distill_teacher=false \
    dataset.return_all_views=true \
    ssdd.fusion_type=concat_conv \
    training.epochs=100 \
    dataset.limit=null
```

### 选项 B: Teacher-Student 蒸馏 (推荐)

**步骤 1**: 先训练一个 Teacher (可选，如果已有 teacher checkpoint 跳过)
```bash
accelerate launch ssdd/main.py \
    run_name=teacher_model \
    training.epochs=100 \
    dataset.im_size=128 \
    ssdd.fm_sampler.steps=12
```

**步骤 2**: 蒸馏训练 Student
```bash
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_student \
    distill_teacher=true \
    ssdd.checkpoint=teacher_model@best \
    dataset.return_all_views=true \
    ssdd.fusion_type=concat_conv \
    ssdd.fm_sampler.steps=12 \
    training.epochs=10 \
    training.lr=1e-4
```

### 选项 C: 快速测试 (小数据集)

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_test \
    distill_teacher=false \
    dataset.return_all_views=true \
    dataset.limit=100 \
    training.epochs=5 \
    training.eval_freq=1 \
    training.log_freq=10
```

---

## Step 4: 监控训练

### TensorBoard

```bash
tensorboard --logdir=tensorboard_logs
```

访问 `http://localhost:6006` 查看：
- 训练损失曲线
- 评估指标 (FID, PSNR, etc.)
- 生成的全景图样本

### 日志文件

```bash
# 查看最新的训练日志
tail -f runs/jobs/multiview_*/main_multiview.log

# 查看训练进度
grep "End of epoch" runs/jobs/multiview_*/main_multiview.log
```

---

## Step 5: 评估

```bash
accelerate launch ssdd/main_multiview.py \
    task=eval \
    ssdd.checkpoint=multiview_student@best \
    ssdd.fm_sampler.steps=1 \
    dataset.return_all_views=true
```

**评估指标**:
- FID (Fréchet Inception Distance)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)

---

## 架构流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    EquiDataset                               │
│  输入: 全景图 (2048x1024)                                     │
│  输出: 4 个鱼眼视图 + 原始全景图                              │
│  - front, right, back, left                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  4 鱼眼视图                 │
         │  [B, 4, 3, 128, 128]       │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  共享权重 Encoder          │
         │  VQEncoder (f8c4)          │
         │  - 一次前向传播处理 4 视角  │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  4 个隐变量                 │
         │  [B, 4, C, zH, zW]         │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  Fusion Module             │
         │  - concat_conv (默认)      │
         │  - attention (可选)        │
         │  - average (baseline)      │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  融合隐变量                 │
         │  z [B, C, zH, zW]          │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  Decoder (UViT)            │
         │  + Flow Matching           │
         │  - Teacher: 12 steps       │
         │  - Student: 1 step         │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  重建全景图                 │
         │  [B, 3, 256, 128]          │
         └───────────────────────────┘
```

---

## 关键参数说明

### Fusion 策略

| 策略 | 参数 | 优势 | 劣势 |
|------|------|------|------|
| `concat_conv` | `ssdd.fusion_type=concat_conv` | 平衡，推荐 | 参数适中 |
| `attention` | `ssdd.fusion_type=attention` | 自适应权重 | 参数较多 |
| `average` | `ssdd.fusion_type=average` | 无参数，快速 | 性能受限 |

**建议**: 从 `concat_conv` 开始，性能不足时尝试 `attention`。

### UCM 相机参数

| 参数 | 默认值 | 说明 | 调整方向 |
|------|--------|------|----------|
| `f_pix` | 220.0 | 焦距 | 减小 → 更宽视野 <br> 增大 → 更窄视野 |
| `xi` | 0.9 | 镜面参数 | 0.05-3.0 范围 |
| `mask_mode` | `inscribed` | 遮罩模式 | `inscribed` / `diagonal` / `none` |

### 训练参数

| 阶段 | `epochs` | `lr` | `fm_sampler.steps` |
|------|----------|------|--------------------|
| Teacher 训练 | 100-300 | 8e-4 | 8-12 |
| Student 蒸馏 | 10-20 | 1e-4 | 12 (训练) / 1 (推理) |

---

## 故障排除

### ❌ "No image files found"

**原因**: 数据目录不存在或为空

**解决**:
```bash
ls /data/360SP-data/train
ls /data/360SP-data/val
```

---

### ❌ "Expected 4 views, got 1"

**原因**: `return_all_views` 未设置

**解决**: 在配置或命令行中添加:
```bash
dataset.return_all_views=true
```

---

### ❌ Memory OOM

**解决方案**:

1. **减小批量大小**:
   ```bash
   dataset.batch_size=128  # 从 256 降低
   ```

2. **减小图像尺寸**:
   ```bash
   dataset.im_size=96  # 从 128 降低
   ```

3. **启用梯度累积**:
   ```bash
   training.grad_accumulate=2
   ```

4. **使用混合精度**:
   ```bash
   training.mixed_precision=fp16
   ```

---

### ❌ "Teacher checkpoint not found"

**原因**: `distill_teacher=true` 但未指定 teacher checkpoint

**解决**: 设置 teacher checkpoint 或禁用蒸馏:
```bash
# 方案 1: 指定 checkpoint
ssdd.checkpoint=path/to/teacher

# 方案 2: 禁用蒸馏
distill_teacher=false
```

---

## 性能预估

### 内存使用 (per GPU)

| 配置 | Batch Size | 内存占用 | GPU 要求 |
|------|-----------|----------|----------|
| 小 | 32 | ~8 GB | RTX 3080 |
| 中 | 64 | ~12 GB | RTX 3090 |
| 大 | 128 | ~20 GB | A100 40GB |
| 超大 | 256 | ~40 GB | A100 80GB |

### 训练时间 (估算)

| 阶段 | Epochs | 样本数 | 时间 (4x A100) |
|------|--------|--------|----------------|
| Teacher 训练 | 100 | 1M | ~24 小时 |
| Student 蒸馏 | 10 | 1M | ~3 小时 |
| 快速测试 | 5 | 1K | ~5 分钟 |

---

## 下一步

### 实验建议

1. **Baseline**: 先用 `fusion_type=average` 建立基线
2. **改进**: 切换到 `fusion_type=concat_conv` 提升性能
3. **蒸馏**: 使用 teacher-student 加速推理
4. **调优**: 调整 UCM 参数 (`f_pix`, `xi`) 优化视角

### 进阶功能

- 添加更多视角 (上/下视角)
- 实现动态视角选择
- 引入深度信息
- 端到端优化 UCM 参数

---

## 获取帮助

- **架构文档**: 查看 `MULTIVIEW_ARCHITECTURE.md`
- **集成文档**: 查看 `EQUI_DATASET_INTEGRATION.md`
- **代码注释**: 所有关键函数都有详细注释

---

## 总结

你现在拥有：

✅ 完整的多视角 SSDD 实现
- 4 个鱼眼视角输入
- 共享权重 Encoder
- 3 种 Fusion 策略
- Teacher-Student 蒸馏支持

✅ 开箱即用的训练/评估流程
- 一键启动训练
- TensorBoard 监控
- 自动 checkpoint 管理

✅ 灵活的配置系统
- Hydra 配置管理
- 命令行参数覆盖
- 多种融合策略可选

**开始训练**:
```bash
accelerate launch ssdd/main_multiview.py \
    run_name=my_first_multiview \
    dataset.return_all_views=true \
    dataset.limit=100
```

🚀 Good luck with your 360° panorama reconstruction!
