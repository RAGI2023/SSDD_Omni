# 修复总结 - 多视角全景图评估

## 已修复的问题

### 1. ✅ 2:1 全景图输出比例问题
**问题**: 模型输出是正方形 (128×128)，但应该输出 2:1 全景图 (256×128)

**原因**: Fusion 模块输出的 latent 是正方形的，decoder 基于 latent 尺寸计算输出

**修复**:
- 文件: `ssdd/models/ssdd/ssdd_multiview.py`
- 在 FusionModule 添加 `aspect_adjust` 层
- 使用 `nn.Upsample(scale_factor=(1.0, 2.0))` 将宽度翻倍
- 输入: [B, C, zH, zW] → 输出: [B, C, zH, zW*2]
- 最终输出: [B, 3, 128, 256] ✓

```python
self.aspect_adjust = nn.Sequential(
    nn.Upsample(scale_factor=(1.0, 2.0), mode='nearest'),
    nn.Conv2d(z_dim, z_dim, kernel_size=3, padding=1),
    nn.GroupNorm(num_groups_z, z_dim),
    nn.SiLU(),
)
```

### 2. ✅ MetricsManager 调用错误
**问题**: `TypeError: MetricsManager.update() takes 1 positional argument but 3 were given`

**原因**: update() 需要关键字参数，而代码传递的是位置参数

**修复**:
- 文件: `ssdd/SpiderTask_MultiView.py:424-426`
- 改为: `self.logger.metrics.update(x_gt=panorama_01, x_pred=rec_panorama_01)`
- 同时添加归一化到 [0, 1] 范围: `panorama_01 = ((panorama + 1) / 2).clamp(0, 1)`

### 3. ✅ TensorBoard 实验名显示问题
**问题**: 所有实验的 TensorBoard 日志都记录到 `tensorboard_logs`，无法区分

**原因**: log_dir 硬编码为 `"tensorboard_logs"`

**修复**:
- 文件: `ssdd/log/loggers.py:153-157`
- 改为使用 `cfg.run_dir/tensorboard`
- 现在每个实验有独立目录: `runs/jobs/2026-01-17/00-20-56/tensorboard`

```python
tensorboard_dir = Path(self.cfg.run_dir) / "tensorboard"
self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
```

### 4. ✅ Eval 模式无 TensorBoard 日志
**问题**: 在 eval-only 模式下不创建 TensorBoard writer

**修复**:
- 文件: `ssdd/log/loggers.py:152-159`
- 移除 `and train` 条件，在 eval 模式也创建 writer

### 5. ✅ Teacher 模型参数不兼容
**问题**: `SSDDMultiView.__init__() got an unexpected keyword argument 'decoder_image_size'`

**原因**: Teacher 是标准 SSDD，不支持多视角参数

**修复**:
- 文件: `ssdd/SpiderTask_MultiView.py:241-243`
- 过滤掉多视角特有参数: `n_views`, `fusion_type`, `decoder_image_size` 等

### 6. ✅ 多视角可视化
**新增功能**:
- 文件: `ssdd/log/visualize.py:33-90`
- 新增 `show_multiview_result()` 函数
- 显示: 4个鱼眼视角 + GT全景 + 预测全景
- TensorBoard 标签: `MultiView/reconstruction`

## 测试结果

运行 `python test_eval_simple.py dataset.batch_size=2 dataset.limit=20 show_samples=2`

```
✓ [Step 1/5] Task initialized
✓ [Step 2/5] Data loader OK
    Views shape: [4, 4, 3, 128, 128]
    Panorama shape: [4, 3, 128, 256]
✓ [Step 3/5] Model forward OK
    Output shape: [4, 3, 128, 256] ← 正确的 2:1 比例!
✓ [Step 4/5] Metrics computation OK
    MSE, MAE, LPIPS, PSNR, SSIM, FID 全部正常
✓ [Step 5/5] Full evaluation OK
    13 batches 处理成功
    可视化保存到 plots/multiview_generation.png
    TensorBoard: runs/jobs/2026-01-17/00-20-56/tensorboard

ALL TESTS PASSED ✓
```

## 配置更新

### config/SpiderEye.yaml
```yaml
dataset:
  num_workers: 10  # 新增: 可配置 DataLoader workers

ssdd:
  decoder_image_size: [256, 128]  # 新增: 2:1 全景图输出尺寸
```

## 查看结果

### TensorBoard
```bash
# 查看所有实验
tensorboard --logdir runs/jobs

# 查看特定日期
tensorboard --logdir runs/jobs/2026-01-17

# 查看特定实验
tensorboard --logdir runs/jobs/2026-01-17/00-20-56/tensorboard
```

### 可视化图像
```bash
# 查看保存的图像
open plots/multiview_generation.png  # macOS
xdg-open plots/multiview_generation.png  # Linux
```

## 文件清单

### 修改的文件
1. `ssdd/models/ssdd/ssdd_multiview.py` - 添加 aspect_adjust
2. `ssdd/models/ssdd/ssdd.py` - 支持 decoder_image_size
3. `ssdd/SpiderTask_MultiView.py` - 修复 metrics 调用 + teacher 参数
4. `ssdd/log/loggers.py` - TensorBoard 路径 + eval 模式支持
5. `ssdd/log/visualize.py` - 多视角可视化函数
6. `config/SpiderEye.yaml` - 添加配置参数

### 新增的文件
1. `test_eval_simple.py` - 评估测试脚本
2. `test_eval.py` - 完整评估脚本(需 checkpoint)
3. `config/eval_test.yaml` - 评估配置
4. `TEST_EVAL_README.md` - 使用文档
5. `FIXES_SUMMARY.md` - 本文档

## 下一步

现在可以开始正常训练：

```bash
accelerate launch ssdd/main_multiview.py \
    run_name=multiview_experiment \
    distill_teacher=true \
    training.epochs=100 \
    dataset.batch_size=16 \
    ssdd.compile=true
```

训练过程中：
- 每个 epoch 结束会评估并记录到 TensorBoard
- 可视化会保存到 `plots/` 和 TensorBoard
- 实验日志独立保存在 `runs/jobs/<run_name>/`
