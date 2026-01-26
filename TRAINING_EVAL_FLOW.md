# 训练与Eval数据流动完整分析

## 启动命令

```bash
accelerate launch ssdd/main_multiview.py -cn=SpiderEye \
    run_name=train_enc_f8c4_1 \
    dataset.im_size=128 \
    training.lr=1e-4 \
    ssdd.encoder_train=true
```

## 配置参数（SpiderEye.yaml + 命令行覆盖）

```yaml
# 数据集
dataset:
  imagenet_root: /data/360SP-data
  im_size: 128              # 鱼眼视角尺寸
  batch_size: 12
  return_all_views: true    # 返回所有4个视角
  f_pix: 220.0             # UCM焦距
  xi: 0.85                 # UCM畸变参数

# 模型
ssdd:
  encoder: f8c4            # 8x downsampling, 4 channels
  encoder_train: true      # 编码器可训练（命令行覆盖）
  decoder: M               # Medium decoder
  decoder_image_size: [256, 128]  # 全景图 W×H (2:1)
  n_views: 4
  fusion_type: concat_conv

  fm_sampler:
    steps: 12              # 推理采样步数
    t_pow_shift: 2.0

  ema:
    decay: 0.999
    start_iter: 1000       # EMA从1000步开始

# 训练
training:
  lr: 1e-4                 # 学习率（命令行覆盖）
  epochs: 300
  eval_freq: 1             # 每个epoch做eval
  mixed_precision: bf16
```

---

## 一、训练流程 (Training)

### 1.1 数据加载

#### 文件：`ssdd/SpiderTask_MultiView.py:175-192`

```python
def load_data(self):
    # 加载EquiDataset（多视角模式）
    cfg_copy = OmegaConf.to_container(self.cfg.dataset, resolve=True)
    cfg_copy['return_all_views'] = True

    (train_dataset, test_dataset), (self.train_loader, self.test_loader) = load_equirect(cfg_copy)
```

#### 文件：`ssdd/dataset_equi.py:85-116`

**Dataset返回格式：**
```python
def __getitem__(self, idx):
    # 1. 读取全景图
    img_path = self.image_files[idx]
    img = cv2.imread(img_path)  # [H, W, 3] RGB

    # 2. 采样UCM参数（训练时有jitter）
    xi, f_pix = self._sample_ucm_params()
    # xi_jitter=0.4 -> xi在 [0.85*(1-0.4), 0.85*(1+0.4)] = [0.51, 1.19]
    # f_jitter=0.1  -> f_pix在 [220/1.1, 220*1.1] = [200, 242]

    # 3. 生成4个鱼眼视角
    # 方向：front(0,0,1), right(1,0,0), back(0,0,-1), left(-1,0,0)
    views = []
    for _, base_dir in self.VIEWS:
        out = equirect_to_fisheye_ucm(
            img, out_w=128, out_h=128,
            base_dir=base_dir,
            xi=xi, f_pix=f_pix,
            jitter_cfg=jitter_cfg  # 训练时有旋转抖动
        )
        views.append(out)

    imgs = torch.stack(views)  # [4, 3, 128, 128]

    # 4. 调整全景图到输出尺寸
    img_original = resize(img, size=(128, 256))  # [3, 128, 256] (H, W)

    # 5. 归一化到[-1, 1]
    imgs = imgs * 2.0 - 1.0
    img_original = img_original * 2.0 - 1.0

    return imgs, img_original
```

**DataLoader批次输出：**
```python
batch = next(iter(train_loader))
views, panorama = batch
# views:    [B=12, N_views=4, 3, 128, 128]  鱼眼视角（输入）
# panorama: [B=12, 3, 128, 256]             全景图（目标）
```

---

### 1.2 训练前向传播

#### 文件：`ssdd/SpiderTask_MultiView.py:412-464`

```python
def task_train(self):
    for cur_epoch in range(epochs):
        for i_batch, batch in enumerate(self.train_loader):
            # 获取数据
            views, panorama = batch
            # views: [12, 4, 3, 128, 128]
            # panorama: [12, 3, 128, 256]

            # 前向传播
            batch_log.losses = self._train_do_step(self.optimizer, batch, train_ctx)
```

#### 文件：`ssdd/SpiderTask_MultiView.py:317-373`

```python
def _compute_train_loss(self, batch, train_ctx):
    views, panorama = batch

    # 调用模型（EMAWrapper -> EMA.model 因为training=True）
    ssdd_out = self.models["ae"](views, gt_panorama=panorama)
    # 返回 TrainStepResult
```

---

### 1.3 模型前向传播（训练模式）

#### 文件：`ssdd/models/blocks/ema.py:92-96`

```python
# EMAWrapper.forward
def forward(self, *args, **kwargs):
    if self.training:  # True
        return self.model(*args, **kwargs)  # 使用训练模型
    else:
        return self.ema(*args, **kwargs)    # 使用EMA模型
```

#### 文件：`ssdd/models/ssdd/ssdd_multiview.py:410-490`

```python
def forward(self, gt_views, gt_panorama, steps=None, noise=None,
            z=None, from_noise=False, as_teacher=False):
    """
    训练时：
      gt_views: [B, 4, 3, 128, 128]
      gt_panorama: [B, 3, 128, 256]
      noise: None (会自动生成)
      steps: None
    """

    # ==== 步骤1: 编码多视角 ====
    if z is None:
        z_views, encoded_dists = self.encode_views(gt_views)
        # z_views: [B, 4, C, zH, zW]
        # 对于f8c4编码器：C=4, 下采样8x
        # 128x128 -> 16x16
        # 所以 z_views: [12, 4, 4, 16, 16]

        z = self.fusion(z_views)
        # fusion做了2:1 aspect ratio调整
        # z: [12, 4, 16, 32]  (W加倍)

    # ==== 步骤2: 训练模式 ====
    if self.training:
        # 2.1 采样时间步 t
        t = self.fm_trainer.sample_t(gt_x.shape[0], device=gt_x.device)
        # 文件: ssdd/flow.py:72-73
        # t = TimeSamplerLogitNormal()(batch_size=12)
        # t ~ Sigmoid(N(0, 1)) -> [B] in (0, 1)
        # 例如: t = [0.72, 0.31, 0.89, 0.45, ...]  shape: [12]

        # 2.2 计算扩散loss
        diff_loss, (x_t, noise_used, noise_t, v_pred) = self.fm_trainer.loss(
            self.decoder, x=gt_panorama, t=t, fn_kwargs={"z": z}, noise=noise
        )
```

#### 文件：`ssdd/flow.py:56-70`

**Flow Matching Loss计算：**

```python
def loss(self, fn, x, t=None, fn_kwargs=None, noise=None):
    """
    参数：
      x: gt_panorama [12, 3, 128, 256]
      t: [12] 时间步
      fn: decoder
      fn_kwargs: {"z": z}  融合后的latent
      noise: None (会生成)
    """

    # 1. 生成噪声
    if noise is None:
        noise = torch.randn_like(x)
    # noise: [12, 3, 128, 256]  ~ N(0, 1)

    # 2. 加噪（Flow Matching）
    # alpha(t) = 1 - t
    # sigma(t) = sigma_min + t * (1 - sigma_min)  (sigma_min=0)
    # x_t = alpha(t) * x + sigma(t) * noise

    x_t, noise = self.add_noise(x, t, noise=noise)
    # 例如：t[0] = 0.72
    #   alpha(0.72) = 0.28
    #   sigma(0.72) = 0.72
    #   x_t[0] = 0.28 * gt_panorama[0] + 0.72 * noise[0]

    # 3. 预测速度场
    v_pred = fn(x_t, t=t * 1000, z=z)
    # decoder输入：
    #   x_t: [12, 3, 128, 256] 加噪图像
    #   t: [12] * 1000 = 时间步（缩放到[0,1000]）
    #   z: [12, 4, 16, 32] 融合latent
    # decoder输出：
    #   v_pred: [12, 3, 128, 256] 预测的速度场

    # 4. 计算目标
    # target = A(t) * x + B(t) * noise
    # A(t) = 1.0
    # B(t) = -(1 - sigma_min) = -1.0
    target = x + (-1.0) * noise  # = x - noise

    # 5. MSE loss
    loss = ((v_pred - target) ** 2).mean()

    return loss, (x_t, noise, t, v_pred)
```

**关键数据流（训练）：**

```
全景图 gt_panorama [12, 3, 128, 256]
    ↓
采样 t ~ Sigmoid(N(0,1)) -> [12]  例如 [0.72, 0.31, ...]
    ↓
生成 noise ~ N(0,1) -> [12, 3, 128, 256]
    ↓
加噪 x_t = (1-t)*gt + t*noise
    ↓  例如 t=0.72: x_t = 0.28*gt + 0.72*noise
    ↓
Decoder(x_t, t, z) -> v_pred [12, 3, 128, 256]
    ↓
Loss = MSE(v_pred, gt - noise)
```

---

### 1.4 辅助损失

#### 文件：`ssdd/SpiderTask_MultiView.py:350-352`

```python
# 计算辅助损失（在预测的x0上）
aux_losses = self.models["aux_losses"](
    panorama,           # GT
    ssdd_out.x0_pred,   # 从x_t和v_pred推导的x0
    target_x=None
)
# 返回：{"lpips": ..., "repa": ...}

# 总损失
losses = {
    "diffusion": diff_loss,    # 主损失
    "lpips": lpips_loss,       # 感知损失
    "repa": repa_loss,         # 重构损失
    "kl": kl_loss              # KL散度（编码器）
}
```

---

### 1.5 EMA更新

#### 文件：`ssdd/SpiderTask_MultiView.py:434-438`

```python
# 每个batch后更新EMA
if EMA.uses_ema(self.models["ae"]):
    self.set_train_state(False)
    EMA.update_ema_modules(self.models["ae"])
    # ema_model = decay * ema_model + (1-decay) * model
    # decay = 0.999
    self.set_train_state(True)
```

#### 文件：`ssdd/models/blocks/ema.py:48-61`

```python
def update(self):
    if self.num_updates >= self.start_iter:
        # 只有在step >= 1000后才更新
        decay = self.decay  # 0.999

        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            self.model.ref.parameters()
        ):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1-decay)
            # ema = 0.999*ema + 0.001*model

    self.num_updates += 1
```

**EMA关键点：**
- `start_iter=1000`：前1000步EMA不更新（保持随机初始化）
- `decay=0.999`：每步只更新0.1%
- 训练9010步时，EMA已更新约8010步

---

## 二、Eval流程 (Evaluation)

### 2.1 Eval入口

#### 文件：`ssdd/SpiderTask_MultiView.py:468-512`

```python
def task_eval(self):
    acc = self.accelerator
    self.set_train_state(False)  # model.eval()

    # 创建确定性generator
    self.generator = torch.Generator(device=acc.device)
    self.generator.manual_seed(self.cfg.seed)  # seed=0

    for batch in tqdm(self.test_loader):
        views, panorama = batch
        # views: [B, 4, 3, 128, 128]
        # panorama: [B, 3, 128, 256]

        with torch.no_grad(), acc.autocast():
            # 生成可复现噪声
            noise = reproducible_rand(acc, self.generator, panorama.shape)

            # 确定步数
            steps = 1 if "teacher" in self.models else self.cfg.ssdd.fm_sampler.steps
            # 非distillation模式：steps = 12

            # 前向传播
            rec_panorama = self.models["ae"](
                views,
                gt_panorama=panorama,
                noise=noise,
                steps=steps
            )

        # 更新指标
        self.logger.metrics.update(x_gt=panorama, x_pred=rec_panorama)
```

---

### 2.2 可复现噪声生成

#### 文件：`ssdd/mutils/torch_utils.py:242-246`

```python
def reproducible_rand(accelerator, generator, shape, fn=None):
    """
    在多GPU环境下生成可复现的随机噪声
    每个进程使用相同的generator seed生成噪声，然后选择对应进程的噪声
    """
    fn = fn or torch.randn

    # 生成所有进程的噪声（使用相同的generator）
    noise = [
        fn(shape, generator=generator, device=accelerator.device)
        for _ in range(accelerator.num_processes)
    ]
    # 例如：4个GPU，生成4份噪声，都用seed=0的generator

    # 选择当前进程对应的噪声
    noise = noise[accelerator.process_index]
    # GPU0拿第0份，GPU1拿第1份...

    return noise
```

**为什么这样设计？**
- DataLoader在多GPU下会对数据分片
- 为了让每个GPU上的batch都有确定的noise
- 但不同GPU的noise应该不同（避免重复）

**Eval时noise示例：**
```python
# panorama.shape = [1, 3, 128, 256] (batch_size=12在eval时可能只有1)
noise = reproducible_rand(acc, generator, (1, 3, 128, 256))
# noise: [1, 3, 128, 256]
# 由 generator(seed=0) 生成，每次eval都相同
```

---

### 2.3 模型推理（Eval模式）

#### 文件：`ssdd/models/blocks/ema.py:92-96`

```python
# EMAWrapper.forward
def forward(self, *args, **kwargs):
    if self.training:  # False (eval模式)
        return self.model(*args, **kwargs)
    else:
        return self.ema(*args, **kwargs)  # 使用EMA模型！
```

**关键：Eval使用EMA模型，不是训练模型！**

#### 文件：`ssdd/models/ssdd/ssdd_multiview.py:451-461`

```python
def forward(self, gt_views, gt_panorama, steps=12, noise=...):
    # 编码
    z_views, _ = self.encode_views(gt_views)
    z = self.fusion(z_views)
    # z: [1, 4, 16, 32]

    # 推理模式
    if not self.training:
        x_gen = self.decode(z, steps=steps, noise=noise)
        return x_gen
```

---

### 2.4 采样过程（Euler采样）

#### 文件：`ssdd/models/ssdd/ssdd.py:83-103`

```python
def decode(self, z, steps=None, noise=None):
    B, _, zH, zW = z.shape  # [1, 4, 16, 32]

    # 计算输出shape
    # patch_size = 8 (f8编码器)
    H, W = zH * 8, zW * 8  # 16*8=128, 32*8=256
    shape = (B, 3, H, W)   # [1, 3, 128, 256]

    ret = self.fm_sampler.sample(
        self.decoder,
        self.fm_trainer,
        shape=shape,
        steps=steps,  # 12
        fn_kwargs={"z": z},
        noise=noise,
    )
    return ret
```

#### 文件：`ssdd/flow.py:98-129`

**Euler采样核心：**

```python
def sample(self, fn, fm_trainer, shape, steps=12, fn_kwargs=None, noise=None):
    """
    从噪声反向生成图像

    参数：
      shape: [1, 3, 128, 256]
      steps: 12
      fn_kwargs: {"z": [1, 4, 16, 32]}
      noise: [1, 3, 128, 256] 从reproducible_rand来的
    """

    # 1. 初始化（从噪声开始）
    x_t = noise  # [1, 3, 128, 256]

    # 2. 生成时间步序列
    t_steps = torch.linspace(1, 0, steps + 1) ** self.t_pow_shift
    # t_pow_shift = 2.0
    # steps = 12 -> 13个点
    # 原始: [1.0, 0.917, 0.833, ..., 0.083, 0.0]
    # 平方: [1.0, 0.840, 0.694, ..., 0.007, 0.0]
    # 更多时间花在t接近0的区域

    # 3. 迭代去噪
    for i in range(steps):  # i = 0..11
        t = t_steps[i]  # 当前时间步
        # i=0:  t=1.000
        # i=1:  t=0.840
        # i=2:  t=0.694
        # ...
        # i=11: t=0.007

        # 3.1 预测速度场
        neg_v = fm_trainer.get_prediction(
            fn=self.decoder,
            x_t=x_t,
            t=t * 1000,  # 缩放到[0, 1000]
            fn_kwargs={"z": z}
        )
        # decoder(x_t, t, z) -> v_pred

        # 3.2 Euler步进
        next_t = t_steps[i + 1]
        x_t = fm_trainer.step(x_t, neg_v, t, next_t)
        # x_{t-dt} = x_t + v * (t - next_t)
        # 例如：i=0
        #   x_0.840 = x_1.0 + v * (1.0 - 0.840)
        #   x_0.840 = x_1.0 + 0.16 * v

    # 4. 返回最终图像
    return x_t  # x_0: [1, 3, 128, 256]
```

**采样过程示意：**

```
noise [1, 3, 128, 256]  (t=1.0, 纯噪声)
    ↓ decoder(x_t, t=1000, z) -> v
    ↓ x_t += v * 0.16
x_{t=0.840}
    ↓ decoder(x_t, t=840, z) -> v
    ↓ x_t += v * 0.146
x_{t=0.694}
    ↓ ...
    ↓
x_{t=0.007}
    ↓ decoder(x_t, t=7, z) -> v
    ↓ x_t += v * 0.007
x_{t=0.0}  重建结果 [1, 3, 128, 256]
```

---

## 三、关键参数对比

| 参数 | 训练时 | Eval时 |
|------|--------|--------|
| **数据来源** | train split | val split |
| **shuffle** | True | False |
| **jitter** | 有（UCM+旋转） | 无 |
| **模型状态** | `model.training=True` | `model.training=False` |
| **使用模型** | `EMAWrapper.model` (训练模型) | `EMAWrapper.ema` (EMA模型) |
| **t采样** | 随机 `Sigmoid(N(0,1))` | 无（推理不需要） |
| **noise生成** | 随机 `torch.randn_like(x)` | 确定 `reproducible_rand(seed=0)` |
| **前向模式** | `forward(...) -> TrainStepResult` | `forward(..., steps=12) -> Tensor` |
| **采样步数** | 训练时1步（加噪-去噪） | 推理时12步（迭代去噪） |
| **loss计算** | ✅ 计算diffusion+aux losses | ❌ 只计算metrics |
| **梯度更新** | ✅ optimizer.step() | ❌ 无梯度 |
| **EMA更新** | ✅ 每batch更新 | ❌ 冻结 |

---

## 四、为什么之前效果差？

### 问题1：EMA未充分训练

```yaml
ema:
  start_iter: 50000  # 需要50000步
```

但只训练了9010步，所以：
- 前1000步：EMA = 随机初始化（未更新）
- 1000-9010步：EMA逐渐更新（约8000步更新）
- **Eval时使用EMA**：质量差（PSNR=14）

**修复后（start_iter=1000）：**
- 前1000步：EMA = 随机初始化
- 1000-9010步：EMA充分更新（8000步）
- Eval效果改善

### 问题2：Demo未正确加载

Demo原本尝试加载完整checkpoint（包含EMA wrapper），但：
- 如果加载`model.safetensors`，可能只有训练模型权重
- EMA权重在`ema.ema_model.*`下
- 需要显式提取`model.*`权重

**修复：**
```python
# 检测EMA结构并提取训练模型权重
if has_ema_wrapper:
    model_weights = {k[6:]: v for k, v in weights.items() if k.startswith('model.')}
    model.load_state_dict(model_weights)
```

---

## 五、数据流图示

```
┌─────────────────────────────────────────────────────────────┐
│                        训练流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  全景图文件 (360SP-data/train/*.jpg)                         │
│      ↓                                                       │
│  EquiDataset                                                 │
│      ├─ 读取全景图 [H, W, 3]                                 │
│      ├─ 采样UCM参数 (xi, f_pix) with jitter                  │
│      ├─ 生成4个鱼眼视角 [4, 3, 128, 128]                      │
│      ├─ 调整全景图 [3, 128, 256]                             │
│      └─ 归一化到 [-1, 1]                                     │
│      ↓                                                       │
│  DataLoader (batch_size=12, shuffle=True)                    │
│      ↓                                                       │
│  batch = (views, panorama)                                   │
│      views: [12, 4, 3, 128, 128]                            │
│      panorama: [12, 3, 128, 256]                            │
│      ↓                                                       │
│  ┌──────────────────────────────────────────┐               │
│  │ EMAWrapper (training=True)               │               │
│  │   └─> model (非EMA)                      │               │
│  └──────────────────────────────────────────┘               │
│      ↓                                                       │
│  SSDDMultiView.forward(views, panorama)                      │
│      ├─ encode_views(views) -> z_views [12,4,4,16,16]       │
│      ├─ fusion(z_views) -> z [12,4,16,32]                   │
│      ├─ sample t ~ Sigmoid(N(0,1)) -> [12]                  │
│      ├─ generate noise ~ N(0,1) -> [12,3,128,256]           │
│      ├─ add_noise: x_t = (1-t)*pano + t*noise               │
│      ├─ decoder(x_t, t, z) -> v_pred                        │
│      └─ loss = MSE(v_pred, pano - noise)                    │
│      ↓                                                       │
│  losses = {diffusion, lpips, repa, kl}                       │
│      ↓                                                       │
│  optimizer.step()                                            │
│      ↓                                                       │
│  EMA.update() (if step > 1000)                              │
│      ema = 0.999*ema + 0.001*model                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Eval流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  全景图文件 (360SP-data/val/*.jpg)                           │
│      ↓                                                       │
│  EquiDataset                                                 │
│      ├─ 读取全景图                                           │
│      ├─ UCM参数固定 (xi=0.85, f_pix=220) NO jitter          │
│      ├─ 生成4个鱼眼视角 [4, 3, 128, 128]                      │
│      └─ 全景图 [3, 128, 256]                                 │
│      ↓                                                       │
│  DataLoader (batch_size=12, shuffle=False)                   │
│      ↓                                                       │
│  batch = (views, panorama)                                   │
│      ↓                                                       │
│  generator = Generator(seed=0)  # 固定seed                   │
│  noise = reproducible_rand(generator, shape=[B,3,128,256])   │
│      ↓                                                       │
│  ┌──────────────────────────────────────────┐               │
│  │ EMAWrapper (training=False)              │               │
│  │   └─> ema.ema_model (EMA权重)            │               │
│  └──────────────────────────────────────────┘               │
│      ↓                                                       │
│  SSDDMultiView.forward(views, panorama, noise, steps=12)     │
│      ├─ encode_views(views) -> z_views                      │
│      ├─ fusion(z_views) -> z [B,4,16,32]                    │
│      └─ decode(z, steps=12, noise=noise)                    │
│          └─ Euler采样 (12步迭代去噪)                         │
│              t = [1.0, 0.84, 0.69, ..., 0.0]                │
│              for i in range(12):                            │
│                  v = decoder(x_t, t[i], z)                  │
│                  x_t += v * (t[i] - t[i+1])                 │
│              return x_0                                      │
│      ↓                                                       │
│  rec_panorama [B, 3, 128, 256]                              │
│      ↓                                                       │
│  metrics.update(gt=panorama, pred=rec_panorama)              │
│      MSE, MAE, LPIPS, PSNR, SSIM, FID                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、关键takeaways

1. **训练用model，eval用ema**
   - EMA需要足够的更新步数（>start_iter）
   - start_iter设置很重要

2. **Noise的作用不同**
   - 训练：随机噪声用于data augmentation和flow matching
   - Eval：固定噪声保证可复现性

3. **t的含义**
   - 训练：随机采样，模拟不同噪声水平
   - Eval：从1递减到0，逐步去噪

4. **采样步数**
   - 训练：1步（forward + loss）
   - Eval：12步（迭代refinement）

5. **数据增强只在训练**
   - UCM jitter、旋转jitter、shuffle
   - Eval保持确定性
