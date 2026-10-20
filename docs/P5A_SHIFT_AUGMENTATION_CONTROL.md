# P5a：随机错配增强强对照

## 目的

P4 发现 DINOv3 部分微调模型对亮度、模糊和噪声具有很强的自然鲁棒性，但 T2 单侧位移会造成明显退化。P5a 用最简单的训练时随机位移回答：

> 不增加任何网络模块，仅进行针对性增强，是否已经足以解决错配问题？

这是后续论文创新的必要对照。普通增强能够解决的问题，不应通过堆叠复杂模块包装为创新。

## 训练设置

配置：

```text
configs/p5a_dinov3_shift_aug_control.yaml
```

初始化：

```text
P3b DINOv3 ConvNeXt-Tiny partial fine-tuning best checkpoint
```

增强协议：

- 75% 训练样本仅对 T2 施加随机位移；
- 25% 保持 clean；
- 幅度为 2、4、8 px；
- 八方向随机；
- 反射填充；
- T1 和标签保持在原参考坐标；
- 原有正样本感知裁剪、翻转和 90 度旋转继续保留。

优化：

- 20 epochs；
- 8192 samples/epoch；
- batch size 128；
- decoder lr `1e-4`；
- backbone lr `1e-5`；
- Validation interval 2；
- 模型选择仍使用 clean Validation F1。

## 训练命令

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src

python -m tools.train \
  --config configs/p5a_dinov3_shift_aug_control.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --device cuda \
  --initial-checkpoint outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth
```

## 训练后鲁棒性评测

```bash
SUMMARY=$(find outputs/p5a_dinov3_shift_aug_control \
  -name summary.json -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-)

RUN_DIR=$(dirname "$SUMMARY")
cat "$SUMMARY"

python -m tools.evaluate_robustness \
  --checkpoint "$RUN_DIR/checkpoints/best.pth" \
  --protocol configs/p4_robustness_diagnostic_v1.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --split val \
  --device cuda \
  --output-dir reports/robustness/p5a_shift_aug_val
```

与未进行错配增强的 DINOv3 基线比较：

```bash
python -m tools.compare_robustness \
  --reference reports/robustness/p4_dinov3_partial_ft_val/results.json \
  --candidate reports/robustness/p5a_shift_aug_val/results.json \
  --reference-name dinov3_clean_ft \
  --candidate-name dinov3_shift_aug \
  --output reports/robustness/p5a_vs_dinov3.csv
```

## 通过门槛

- clean Validation F1 `>= 0.8973`；
- `shift:s3:t2` F1 `>= 0.7800`；
- 三档 T2 shift 平均 F1 相对原模型至少提高 `0.05`；
- 亮度、模糊和噪声场景不能出现系统性退化。

P5a 不访问 Test。只有创新方案和超参数在 Validation 上冻结后，才进行最终 Test 评测。
