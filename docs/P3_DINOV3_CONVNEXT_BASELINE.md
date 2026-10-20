# P3：DINOv3 ConvNeXt-Tiny 现代 Backbone 基线

## 1. 实验目的

本阶段只测量现代预训练 backbone 带来的收益，不加入论文创新模块。

固定项：

- LEVIR-CD manifest 和 train/val/test split；
- 256×256 训练裁块；
- RAM 缓存；
- 正样本感知裁剪；
- BCE + Dice；
- `pos_weight=2.0`；
- 0.5 评价阈值；
- 完整 1024×1024 validation；
- 相同 Precision、Recall、F1、IoU 和 OA 实现。

变化项：

- 将从零训练的四级 CNN 编码器替换为 DINOv3 ConvNeXt-Tiny；
- 使用 DINOv3 的四级特征 `[96, 192, 384, 768]`；
- 首个实验冻结 backbone，只训练差分 FPN 解码器。

## 2. 模型来源

配置中的 timm 模型名：

```text
convnext_tiny.dinov3_lvd1689m
```

该模型约 27.8M 参数，是 DINOv3 ViT-7B 在 LVD-1689M 网页图像数据上蒸馏得到的 ConvNeXt-Tiny。它不是 SAT-493M 卫星专用权重。

注意：

- `DINOv3 ViT-Tiny` 和 `DINOv3 ConvNeXt-Tiny` 不是同一个模型；
- 两类权重不能互换；
- 正式使用前必须记录权重文件名、来源和 SHA256；
- 权重使用受 DINOv3 License 约束。

## 3. 权重路径

推荐服务器目录：

```text
/root/autodl-tmp/rs_cd_work/pretrained/DINOv3/
```

支持两种输入：

### 3.1 timm/Hugging Face 完整模型目录

若下载的是 `timm/convnext_tiny.dinov3_lvd1689m` 完整目录，配置填写目录：

```yaml
model:
  backbone_name: convnext_tiny.dinov3_lvd1689m
  pretrained: true
  checkpoint_path: /root/autodl-tmp/rs_cd_work/pretrained/DINOv3/convnext_tiny.dinov3_lvd1689m
```

### 3.2 单个 timm 兼容 checkpoint

若下载的是单个 `.pth`、`.pt` 或 `.safetensors` 文件，配置填写文件：

```yaml
model:
  backbone_name: convnext_tiny.dinov3_lvd1689m
  pretrained: false
  checkpoint_path: /root/autodl-tmp/rs_cd_work/pretrained/DINOv3/<实际文件名>
```

如果出现 missing/unexpected keys，不允许使用 `strict=False` 静默跳过；应先确认下载的是否为 ConvNeXt-Tiny 以及是否为 timm 兼容格式。

### 3.3 在线加载

当 `checkpoint_path: null` 且 `pretrained: true` 时，timm 会尝试下载：

```text
timm/convnext_tiny.dinov3_lvd1689m
```

为了避免网络和缓存不稳定，正式实验优先使用本地权重。

## 4. 依赖

服务器执行：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
python -m pip install "timm==1.0.20"
```

确认：

```bash
python - <<'PY'
import timm
print(timm.__version__)
print("model exists:", "convnext_tiny.dinov3_lvd1689m" in timm.list_models())
PY
```

预期：

```text
1.0.20
model exists: True
```

## 5. 网络结构

```text
T1 ─ Shared DINOv3 ConvNeXt-Tiny ─ [F1, F2, F3, F4] ─┐
                                                      ├─ |F1-F2| at each level
T2 ─ Shared DINOv3 ConvNeXt-Tiny ─ [F1, F2, F3, F4] ─┘
                                                               │
                                                               ▼
                                                   Difference FPN Decoder
                                                               │
                                                               ▼
                                                       Binary Change Logits
```

实际计算是对两个时相相同尺度特征求绝对差，然后进行自顶向下的 FPN 融合。

该结构保持：

- backbone 权重共享；
- 推理阶段 T1/T2 交换不变性；
- 输出 contract 为 `{"change_logits": [B,1,H,W], "aux": None}`。

## 6. 首个配置

```text
configs/p3_dinov3_convnext_tiny_frozen.yaml
```

首个配置暂时使用：

```yaml
freeze_backbone: true
decoder_channels: 128
epochs: 60
samples_per_epoch: 8192
batch_size: 128
learning_rate: 0.001
```

GPU smoke 后正式选择 batch 128。其吞吐为 673.16 images/s，
显存 reserved 约 5.96 GiB。Batch 256 虽达到 743.79 images/s，
但相对 batch 128 只快约 10.5%，同时会显著减少固定样本预算下的更新次数。
`8192 / 128 = 64`，因此每 epoch 保留 64 次参数更新。

## 7. GPU smoke

先从 batch 8、16 开始：

```bash
python -m tools.gpu_smoke \
  --config configs/p3_dinov3_convnext_tiny_frozen.yaml \
  --batch-size 8 \
  --image-size 256 \
  --steps 5 \
  --output reports/p3_dinov3_frozen_bs8.json

python -m tools.gpu_smoke \
  --config configs/p3_dinov3_convnext_tiny_frozen.yaml \
  --batch-size 16 \
  --image-size 256 \
  --steps 5 \
  --output reports/p3_dinov3_frozen_bs16.json
```

若显存充足，再测 24、32。正式 batch 以吞吐边际收益和至少 20% 显存余量为准。

## 8. 训练阶段

### P3-A：冻结 backbone

目的：

- 检查 DINOv3 特征是否能直接支持变化检测；
- 测量预训练特征的线性/浅层适配能力；
- 降低训练显存；
- 建立后续微调的对照组。

### P3-B：解冻最后 stage

后续新增独立配置：

- 只解冻 ConvNeXt stage 4；
- backbone 学习率低于 decoder；
- 与冻结实验使用相同样本预算。

### P3-C：全量分层微调

后续实现：

- backbone 小学习率；
- decoder 大学习率；
- 分层学习率衰减；
- 必要时梯度检查点；
- 不直接复用冻结实验的统一学习率。

## 9. 结果门槛

冻结 backbone 首次实验只使用 validation 做模型选择：

- F1 < 0.84：检查权重、归一化和解码器；
- F1 0.84–0.88：特征可用，进入部分微调；
- F1 ≥ 0.88：冻结特征已经很强；
- F1 ≥ 0.895：达到当前传统基线 validation 水平；
- 配置冻结前不运行 test。

本阶段结果无论高低都要保留，因为它是判断“性能来自预训练 backbone 还是后续创新模块”的必要消融。

## 10. 实验命名

建议表格名称：

```text
DINOv3-CN-T (frozen) + Diff-FPN
```

不得将其命名为本文完整方法，也不得把 backbone 替换本身表述为主要创新。
