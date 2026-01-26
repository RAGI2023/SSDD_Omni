# 训练 vs Eval 快速对比

## 核心差异速查表

| 阶段 | 训练 (Training) | 验证 (Eval) |
|------|----------------|------------|
| **目标** | 学习从views重建panorama | 测试重建质量 |
| **数据集** | train split | val split |
| **Batch示例** | `views[12,4,3,128,128]`<br>`panorama[12,3,128,256]` | 同左 |
| **模型** | `EMAWrapper.model`<br>（训练中的模型） | `EMAWrapper.ema`<br>（EMA平均模型） |
| **UCM jitter** | ✅ xi±40%, f±10% | ❌ 固定参数 |
| **旋转jitter** | ✅ 随机扰动 | ❌ 无扰动 |

---

## Noise和t的生成

### 训练时

```python
# 1. 随机采样时间步
t = TimeSamplerLogitNormal()(batch_size)
# t ~ Sigmoid(Normal(0,1))
# 输出: [0.72, 0.31, 0.89, ...] shape=[12]

# 2. 随机生成噪声
noise = torch.randn_like(panorama)
# 输出: [12, 3, 128, 256] ~ N(0,1)

# 3. 加噪
x_t = (1-t) * panorama + t * noise
# 例如 t=0.72:
#   x_t = 0.28*panorama + 0.72*noise

# 4. 预测速度场
v_pred = decoder(x_t, t*1000, z)

# 5. 计算loss
target = panorama - noise  # = A(t)*pano + B(t)*noise
loss = MSE(v_pred, target)
```

### Eval时

```python
# 1. 固定seed生成噪声
generator = torch.Generator(device).manual_seed(0)
noise = reproducible_rand(generator, panorama.shape)
# 输出: [B, 3, 128, 256]
# 每次eval都相同！

# 2. 不需要t（采样过程内部使用）

# 3. Euler采样（12步迭代去噪）
x_t = noise  # 从纯噪声开始
t_steps = [1.0, 0.84, 0.69, ..., 0.007, 0.0]  # 13个点

for i in range(12):
    v = decoder(x_t, t_steps[i]*1000, z)
    x_t = x_t + v * (t_steps[i] - t_steps[i+1])

return x_t  # 最终重建结果
```

---

## 数据变换流程

### Dataset处理（EquiDataset）

```
原始全景图 [H, W, 3] RGB
    ↓
┌─────────────────────┐
│ 采样UCM参数          │
│ 训练: xi, f_pix with jitter │
│ Eval: 固定 xi=0.85, f=220  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ 生成4个鱼眼视角      │
│ 方向: F/R/B/L       │
│ 输出: [4,3,128,128] │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ 调整全景图尺寸       │
│ [3, 128, 256]       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ 归一化到 [-1, 1]    │
│ x = x*2 - 1         │
└─────────────────────┘
    ↓
返回 (views, panorama)
```

---

## 模型前向传播

### 训练模式 (training=True)

```
views [B,4,3,128,128]  panorama [B,3,128,256]
    ↓                       ↓
Encoder (shared)         (GT目标)
    ↓                       ↓
z_views [B,4,4,16,16]      |
    ↓                       |
Fusion (concat+conv)        |
    ↓                       |
z [B,4,16,32]              |
    ↓                       |
    ├───────────────────────┤
    ↓                       ↓
Sample t ~ [0,1]     Generate noise
    ↓                       ↓
    └───────> x_t = (1-t)*pano + t*noise
                    ↓
            Decoder(x_t, t, z)
                    ↓
                v_pred [B,3,128,256]
                    ↓
            Loss = MSE(v_pred, pano-noise)
```

### Eval模式 (training=False)

```
views [B,4,3,128,128]  noise [B,3,128,256] (固定seed)
    ↓                       ↓
Encoder                 x_t = noise (t=1.0)
    ↓                       ↓
z_views                 ┌─────────────────┐
    ↓                   │ Euler采样 (12步) │
Fusion                  │ for i in 12:    │
    ↓                   │   v = dec(x_t,t,z) │
z [B,4,16,32] ──────────>│   x_t += v*dt   │
                        └─────────────────┘
                                ↓
                        rec_panorama [B,3,128,256]
```

---

## Checkpoint结构

```
model.safetensors
├─ model.base_ssdd.encoder.*     (训练中的encoder)
├─ model.base_ssdd.decoder.*     (训练中的decoder)
├─ model.fusion.*                (训练中的fusion)
├─ ema.ema_model.base_ssdd.encoder.*  (EMA的encoder)
├─ ema.ema_model.base_ssdd.decoder.*  (EMA的decoder)
└─ ema.ema_model.fusion.*             (EMA的fusion)
```

**加载策略：**
- **训练恢复**：加载完整checkpoint（包括optimizer）
- **Eval**：只加载EMA部分（`ema.ema_model.*`）
- **Demo**：提取`model.*`部分（避免使用未充分训练的EMA）

---

## EMA更新机制

```python
# 每个batch后
if step >= start_iter:  # start_iter=1000
    for ema_param, model_param in zip(ema.parameters(), model.parameters()):
        ema_param = 0.999 * ema_param + 0.001 * model_param

# 训练9010步的情况
# step 0-999:   EMA不更新（保持随机初始化）
# step 1000-9010: EMA更新8010次
# 收敛程度：约 1 - (0.999)^8010 ≈ 99.98%
```

---

## 采样时间步调度

```python
t_steps = torch.linspace(1, 0, steps+1) ** t_pow_shift
# steps=12, t_pow_shift=2.0

原始（线性）:
[1.000, 0.917, 0.833, 0.750, 0.667, 0.583,
 0.500, 0.417, 0.333, 0.250, 0.167, 0.083, 0.000]

平方后（非线性）:
[1.000, 0.840, 0.694, 0.562, 0.444, 0.340,
 0.250, 0.174, 0.111, 0.062, 0.028, 0.007, 0.000]

效果：更多步数花在低噪声区域（t接近0），精细化重建
```

---

## 常见问题

### Q1: 为什么eval结果是噪声？
**A:** EMA未充分训练
- 检查 `start_iter` 是否太大（应 < 当前训练步数）
- 检查是否使用了EMA权重（eval时应该用）

### Q2: Demo和eval结果不一致？
**A:** 检查以下差异
- Demo用model权重，eval用EMA权重
- Noise生成方式不同（随机 vs 固定seed）
- Steps可能不同

### Q3: 训练loss降低但eval指标差？
**A:** 可能原因
- EMA更新不足（decay太大或start_iter太晚）
- 训练集和验证集分布不同
- 过拟合（训练太久）

### Q4: PSNR只有14，正常吗？
**A:** 不正常
- 正常应该 >25
- 检查EMA是否正常更新
- 检查loss权重配置
- 增加训练epoch

---

## 调试检查清单

- [ ] EMA start_iter < 当前步数
- [ ] Eval使用EMA权重
- [ ] Noise shape正确 `[B, 3, 128, 256]`
- [ ] decoder_image_size = `[256, 128]` (W, H)
- [ ] 数据集路径正确，文件存在
- [ ] Fusion层权重已加载（非零）
- [ ] 训练loss持续下降
- [ ] Eval PSNR > 20

---

## 性能优化建议

1. **学习率**: 默认8e-4 > 当前1e-4
2. **Batch size**: 增大可提升稳定性
3. **EMA start**: 1000步比50000步更好
4. **训练数据**: 增加数据量和多样性
5. **Steps**: eval时用12步，demo测试可用1步看速度
