# 多视角 SSDD 训练完整指南

## 🚀 快速开始

从零开始训练多视角 SSDD 模型。

---

## 📋 前置检查清单

### 1. 环境检查

```bash
# 检查 Python 版本 (需要 >= 3.11)
python --version

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查 accelerate
accelerate --version
```

### 2. 数据检查

```bash
# 检查数据目录
ls -lh /data/360SP-data/train | head -10
ls -lh /data/360SP-data/val | head -10

# 统计数据量
echo "训练集图片数量: $(ls /data/360SP-data/train/*.jpg | wc -l)"
echo "验证集图片数量: $(ls /data/360SP-data/val/*.jpg | wc -l)"
```

**期望输出**:
```
训练集图片数量: 1000000
验证集图片数量: 50000
```

### 3. GPU 检查

```bash
# 查看 GPU 信息
nvidia-smi

# 检查可用 GPU 数量
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
```

---

## 🎯 三种训练模式

### 模式 1: 从头训练 (不使用 Teacher)

**适用场景**: 首次训练，没有预训练模型

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_from_scratch \
    distill_teacher=false \
    training.epochs=100 \
    dataset.im_size=128 \
    dataset.batch_size=256
```

**预估时间**: 24-48 小时 (4×A100)

---

### 模式 2: Teacher-Student 蒸馏 (推荐)

**适用场景**: 已有 teacher 模型，想训练快速的 student

#### 步骤 2.1: 训练 Teacher (如果还没有)

```bash
# 训练多步 teacher 模型
accelerate launch ssdd/main.py \
    run_name=teacher_12steps \
    training.epochs=100 \
    dataset.im_size=128 \
    ssdd.fm_sampler.steps=12 \
    ssdd.encoder=f8c4 \
    ssdd.decoder=M
```

**时间**: ~24 小时

#### 步骤 2.2: 蒸馏训练 Student

```bash
# 使用 teacher 蒸馏训练 student (单步)
accelerate launch ssdd/main_multiview.py \
    run_name=student_1step_distill \
    distill_teacher=true \
    ssdd.checkpoint=teacher_12steps@best \
    ssdd.fm_sampler.steps=12 \
    training.epochs=10 \
    training.lr=1e-4 \
    training.eval_freq=1
```

**时间**: ~3 小时

---

### 模式 3: 快速测试 (小数据集)

**适用场景**: 验证流程、调试代码

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=test_run \
    distill_teacher=false \
    dataset.limit=1000 \
    training.epochs=5 \
    training.eval_freq=1 \
    training.log_freq=10 \
    dataset.batch_size=32
```

**时间**: ~10 分钟

---

## 🔧 详细配置说明

### 关键参数

#### 数据集参数

```yaml
dataset:
  imagenet_root: /data/360SP-data  # 数据根目录
  im_size: 128                      # 图像大小 (128 推荐)
  batch_size: 256                   # 总批量大小 (会自动分配到多 GPU)
  limit: null                       # 限制样本数 (null=全部, 1000=测试)
  return_all_views: true            # 必须为 true (多视角模式)

  # EquiDataset 参数
  f_pix: 220.0                      # UCM 焦距
  xi: 0.9                           # UCM 镜面参数
  mask_mode: inscribed              # 圆形遮罩模式
```

#### 模型参数

```yaml
ssdd:
  encoder: f8c4                     # f8=patch_size_8, c4=z_dim_4
  encoder_train: false              # 是否训练 encoder (通常 false)
  decoder: M                        # 解码器大小 (XS/S/M/L/XL)

  # 多视角参数
  n_views: 4                        # 视角数量
  fusion_type: concat_conv          # 融合策略
  use_view_encoding: true           # 启用视角编码
  view_encoding_type: sinusoidal    # 编码类型 (当前配置)

  # Flow Matching 参数
  fm_sampler:
    steps: 12                       # 采样步数 (teacher用12, eval用1)
```

#### 训练参数

```yaml
training:
  mixed_precision: bf16             # 混合精度 (bf16/fp16/no)
  grad_accumulate: 1                # 梯度累积步数
  grad_clip: 0.1                    # 梯度裁剪
  epochs: 300                       # 训练轮数
  eval_freq: 4                      # 评估频率 (每4个epoch)
  save_on_best: FID                 # 保存最佳模型的指标
  log_freq: 200                     # 日志频率 (每200步)

  lr: 8e-4                          # 学习率
  weight_decay: 1e-2                # 权重衰减
```

---

## 📊 监控训练

### 方式 1: TensorBoard (推荐)

```bash
# 启动 TensorBoard
tensorboard --logdir=tensorboard_logs --port=6006

# 在浏览器打开
# http://localhost:6006
```

**监控指标**:
- `Loss/average`: 平均训练损失
- `Loss/diffusion`: 扩散损失
- `Loss/repa`: REPA 感知损失
- `Loss/lpips`: LPIPS 感知损失
- `metric/FID`: Fréchet Inception Distance
- `metric/PSNR`: Peak Signal-to-Noise Ratio

---

### 方式 2: 查看日志文件

```bash
# 查看最新的训练日志
tail -f runs/jobs/*/main_multiview.log

# 查看训练进度
grep "End of epoch" runs/jobs/*/main_multiview.log

# 查看最佳指标
grep "Best metrics" runs/jobs/*/main_multiview.log
```

---

### 方式 3: 实时输出

训练时会自动显示进度：

```
[T_total=00:15:32 | T_train=00:12:45 | T_epoch=00:02:15]
Epoch 10, batch 200 / 1000 (step 2000)
loss=0.0235 (avg=0.0245) [[all losses: diffusion=0.0180 ; repa=0.0035 ; lpips=0.0020]]
```

---

## 🎛️ 常用命令行参数覆盖

### 调整批量大小

```bash
# 减小批量大小 (内存不足时)
accelerate launch ssdd/main_multiview.py \
    dataset.batch_size=128

# 使用梯度累积
accelerate launch ssdd/main_multiview.py \
    dataset.batch_size=128 \
    training.grad_accumulate=2  # 等效于 batch_size=256
```

### 调整学习率

```bash
# 使用更大的学习率 (从头训练)
accelerate launch ssdd/main_multiview.py \
    training.lr=1e-3

# 使用更小的学习率 (微调/蒸馏)
accelerate launch ssdd/main_multiview.py \
    training.lr=5e-5
```

### 切换 View Encoding 类型

```bash
# 使用 Learnable Encoding
accelerate launch ssdd/main_multiview.py \
    ssdd.view_encoding_type=learnable

# 使用 Sinusoidal Encoding (当前默认)
accelerate launch ssdd/main_multiview.py \
    ssdd.view_encoding_type=sinusoidal

# 使用 Directional Encoding
accelerate launch ssdd/main_multiview.py \
    ssdd.view_encoding_type=directional

# 禁用 View Encoding (消融实验)
accelerate launch ssdd/main_multiview.py \
    ssdd.use_view_encoding=false
```

### 切换 Fusion 策略

```bash
# Concat+Conv (默认)
accelerate launch ssdd/main_multiview.py \
    ssdd.fusion_type=concat_conv

# Attention-based
accelerate launch ssdd/main_multiview.py \
    ssdd.fusion_type=attention

# Simple Average (baseline)
accelerate launch ssdd/main_multiview.py \
    ssdd.fusion_type=average
```

---

## 🗂️ 输出文件结构

训练过程中会生成以下文件：

```
runs/
└── jobs/
    └── multiview_from_scratch/  # run_name
        ├── checkpoints/
        │   ├── checkpoint_epoch_10.pt
        │   ├── checkpoint_epoch_20.pt
        │   └── checkpoint_best.pt  # 最佳模型
        ├── plots/
        │   ├── generation_epoch=10.png
        │   └── generation_epoch=20.png
        ├── config.yaml  # 实际使用的配置
        ├── main_multiview.log  # 训练日志
        └── task_result.json  # 评估结果

tensorboard_logs/
└── events.out.tfevents.*  # TensorBoard 日志
```

---

## 🔄 恢复训练

如果训练中断，可以自动恢复：

```bash
# 使用相同的 run_name 会自动加载 checkpoint
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_from_scratch
```

系统会自动：
1. 检测 `runs/jobs/multiview_from_scratch/checkpoints/` 是否存在
2. 加载最新的 checkpoint
3. 从中断的 epoch 继续训练

---

## 🎯 完整训练流程示例

### 场景: 从零开始训练多视角模型

#### 第 1 步: 数据验证

```bash
# 测试数据加载 (1分钟)
python -c "
from ssdd.dataset_equi import load_equirect

cfg = {
    'imagenet_root': '/data/360SP-data',
    'im_size': 128,
    'batch_size': 4,
    'limit': 10,
    'return_all_views': True,
}

(train_ds, test_ds), (train_loader, test_loader) = load_equirect(cfg)
print(f'✓ 训练集: {len(train_ds)} 样本')
print(f'✓ 测试集: {len(test_ds)} 样本')

for views, panorama in train_loader:
    print(f'✓ Views shape: {views.shape}')
    print(f'✓ Panorama shape: {panorama.shape}')
    break
"
```

**期望输出**:
```
✓ 训练集: 10 样本
✓ 测试集: 10 样本
✓ Views shape: torch.Size([4, 4, 3, 128, 128])
✓ Panorama shape: torch.Size([4, 3, 256, 128])
```

---

#### 第 2 步: 快速测试 (可选)

```bash
# 小规模测试 (10分钟)
accelerate launch ssdd/main_multiview.py \
    run_name=quick_test \
    dataset.limit=100 \
    training.epochs=2 \
    training.eval_freq=1 \
    dataset.batch_size=8
```

检查是否有报错。

---

#### 第 3 步: 完整训练

```bash
# 完整训练 (24-48小时)
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_production_v1 \
    distill_teacher=false \
    training.epochs=100 \
    dataset.batch_size=256 \
    training.eval_freq=4 \
    ssdd.view_encoding_type=learnable \
    ssdd.fusion_type=concat_conv
```

---

#### 第 4 步: 监控和调优

在训练过程中：

```bash
# 终端 1: 查看日志
tail -f runs/jobs/multiview_production_v1/main_multiview.log

# 终端 2: TensorBoard
tensorboard --logdir=tensorboard_logs

# 终端 3: 查看 GPU 使用率
watch -n 1 nvidia-smi
```

---

#### 第 5 步: 评估

```bash
# 评估最佳模型
accelerate launch ssdd/main_multiview.py \
    task=eval \
    ssdd.checkpoint=multiview_production_v1@best \
    ssdd.fm_sampler.steps=1  # 单步推理
```

---

## 🐛 故障排除

### 问题 1: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

```bash
# 方案 A: 减小批量大小
accelerate launch ssdd/main_multiview.py \
    dataset.batch_size=128

# 方案 B: 使用梯度累积
accelerate launch ssdd/main_multiview.py \
    dataset.batch_size=64 \
    training.grad_accumulate=4

# 方案 C: 减小图像尺寸
accelerate launch ssdd/main_multiview.py \
    dataset.im_size=96

# 方案 D: 使用 fp16 混合精度
accelerate launch ssdd/main_multiview.py \
    training.mixed_precision=fp16
```

---

### 问题 2: 数据加载慢

**症状**: 训练卡在数据加载

**解决方案**:

```bash
# 减少 num_workers (如果 CPU 不够)
# 修改 dataset_equi.py:173
num_workers=4  # 从 10 改为 4

# 或禁用 persistent_workers
persistent_workers=False
```

---

### 问题 3: Teacher checkpoint 未找到

**症状**:
```
FileNotFoundError: teacher checkpoint not found
```

**解决方案**:

```bash
# 检查 checkpoint 是否存在
ls runs/jobs/teacher_12steps/checkpoints/

# 使用正确的路径
accelerate launch ssdd/main_multiview.py \
    ssdd.checkpoint=runs/jobs/teacher_12steps/checkpoints/checkpoint_best.pt

# 或禁用 teacher
accelerate launch ssdd/main_multiview.py \
    distill_teacher=false
```

---

### 问题 4: 训练损失不下降

**可能原因**:
1. 学习率过大或过小
2. 数据问题
3. 模型配置错误

**解决方案**:

```bash
# 尝试调整学习率
accelerate launch ssdd/main_multiview.py \
    training.lr=5e-4  # 或 1e-3, 1e-5

# 检查数据
python utils/EquiDataset.py  # 运行测试

# 检查梯度
# 在训练日志中查找 "grad_norm"
```

---

## 📈 性能优化建议

### 1. GPU 利用率优化

```bash
# 最大化批量大小
accelerate launch ssdd/main_multiview.py \
    dataset.batch_size=512  # 根据 GPU 内存调整

# 启用 TF32 (Ampere+ GPU)
# 已在 SpiderTask_MultiView.py 中默认启用
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

### 2. 多 GPU 训练

```bash
# 使用 accelerate 配置
accelerate config

# 选择:
# - multi-GPU
# - 使用的 GPU 数量
# - 混合精度: bf16

# 然后正常启动训练
accelerate launch ssdd/main_multiview.py \
    run_name=multigpu_training
```

---

### 3. 编译优化

```bash
# 启用 torch.compile (PyTorch 2.0+)
# 已在配置中默认启用
ssdd.compile=true

# 如果遇到编译问题，可以禁用
accelerate launch ssdd/main_multiview.py \
    ssdd.compile=false
```

---

## 📝 训练检查清单

开始训练前，确认：

- [ ] 数据路径正确: `/data/360SP-data/train` 和 `/data/360SP-data/val` 存在
- [ ] 数据格式正确: 等距圆柱投影全景图，`.jpg` 格式
- [ ] GPU 可用: `nvidia-smi` 显示 GPU
- [ ] 环境正确: PyTorch, accelerate 已安装
- [ ] 配置已更新: `config/SpiderEye.yaml` 中的参数符合预期
- [ ] 磁盘空间充足: 至少 100GB (checkpoints + logs)
- [ ] TensorBoard 已启动: 实时监控训练

---

## 🚀 推荐训练命令

### 生产环境 (推荐)

```bash
# 完整训练，Learnable View Encoding，Concat+Conv Fusion
accelerate launch ssdd/main_multiview.py \
    run_name=prod_learnable_concat_$(date +%Y%m%d_%H%M%S) \
    distill_teacher=false \
    training.epochs=100 \
    training.eval_freq=4 \
    ssdd.view_encoding_type=learnable \
    ssdd.fusion_type=concat_conv \
    dataset.batch_size=256
```

### 快速实验

```bash
# 快速迭代，Sinusoidal Encoding
accelerate launch ssdd/main_multiview.py \
    run_name=exp_sinusoidal_$(date +%Y%m%d_%H%M%S) \
    dataset.limit=10000 \
    training.epochs=20 \
    training.eval_freq=2 \
    ssdd.view_encoding_type=sinusoidal \
    dataset.batch_size=128
```

---

## 📞 获取帮助

遇到问题？查看：

1. **日志文件**: `runs/jobs/*/main_multiview.log`
2. **配置文件**: `runs/jobs/*/config.yaml`
3. **文档**:
   - `VIEW_ENCODING_GUIDE.md` - View Encoding 详解
   - `MULTIVIEW_ARCHITECTURE.md` - 架构说明
   - `QUICK_START_MULTIVIEW.md` - 快速上手

---

## ✅ 成功训练的标志

如果看到以下输出，说明训练正常：

```
✓ Loaded EquiDataset (multi-view): {'train': ..., 'test': ...}
✓ ae parameters count: Total: #... (trainable: #...)
✓ [Epoch 1] End of epoch ... train loss ...
✓ [Epoch 1] Test metrics: FID=... PSNR=... LPIPS=...
✓ Saved checkpoint to .../checkpoint_epoch_1.pt
✓ [Epoch 1] Best metrics: FID=... (best)
```

---

🎉 **开始训练吧！**

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=my_first_multiview_training
```
