# P4：双时相变化检测鲁棒性诊断

## 1. 阶段目标

本阶段不训练模型，而是在固定的 LEVIR-CD Validation 上比较两个已冻结 checkpoint：

1. FC-Siam-Diff 强训练基线；
2. DINOv3 ConvNeXt-Tiny 部分微调基线。

目标不是立即追求更高 clean F1，而是回答：

- 哪类输入扰动造成的性能下降最大；
- 一侧时相变化与两侧同步变化是否有不同影响；
- DINOv3 的增益是否能够延伸到困难条件；
- 下一阶段应设计配准鲁棒、辐射不变、退化感知还是其他模块。

## 2. 固定协议

配置：`configs/p4_robustness_diagnostic_v1.yaml`

| 扰动 | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| Shift | 2 px | 4 px | 8 px |
| Brightness | 10% | 20% | 30% |
| Gaussian blur | sigma 0.8 | sigma 1.6 | sigma 2.4 |
| Gaussian noise | std 0.01 | std 0.03 | std 0.05 |

每个非 clean 扰动包含三种模式：

- `t1`：仅扰动 T1；
- `t2`：仅扰动 T2；
- `common`：以相同参数同步扰动 T1、T2。

因此场景总数为：

```text
1 clean + 4 corruptions × 3 severities × 3 modes = 37
```

固定条件：

- split：Validation；
- 阈值：沿用 checkpoint 中的 0.5；
- seed：3407；
- 输入和标签协议不变；
- checkpoint 权重不变；
- 两个模型使用同一 protocol hash；
- 不使用 Test 选择创新方向。

## 3. 位移评价说明

单侧位移模拟双时相轻微错配，标签保持在原参考坐标。

同步位移模拟整对影像发生坐标偏移，T1、T2 和标签一起移动。

平移产生的无有效对应边界不会计入混淆矩阵。这样报告的 F1/IoU 反映有效重叠区域，而不是边界填充值。Loss 仍可作为运行诊断，但本阶段结论只基于掩膜后的 Precision、Recall、F1、IoU 和 OA。

## 4. 服务器执行

上传并解压增量包后：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
python -m pytest -q
```

评测 FC-Siam-Diff：

```bash
python -m tools.evaluate_robustness \
  --checkpoint outputs/p2_fc_siam_diff_recovery_bs96/20260724T135252Z_seed3407/checkpoints/best.pth \
  --protocol configs/p4_robustness_diagnostic_v1.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --split val \
  --device cuda \
  --output-dir reports/robustness/p4_fc_siam_diff_val
```

评测 DINOv3 部分微调模型：

```bash
python -m tools.evaluate_robustness \
  --checkpoint outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth \
  --protocol configs/p4_robustness_diagnostic_v1.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --split val \
  --device cuda \
  --output-dir reports/robustness/p4_dinov3_partial_ft_val
```

生成逐场景比较：

```bash
python -m tools.compare_robustness \
  --reference reports/robustness/p4_fc_siam_diff_val/results.json \
  --candidate reports/robustness/p4_dinov3_partial_ft_val/results.json \
  --reference-name fc_siam_diff \
  --candidate-name dinov3_partial_ft \
  --output reports/robustness/p4_fc_vs_dinov3.csv
```

## 5. 产物

每个模型目录会包含：

- `results.json`：元数据、checkpoint SHA256、protocol hash、汇总和全部场景；
- `results.csv`：逐场景指标；
- `protocol_resolved.yaml`：实际使用的协议。

比较脚本会生成：

- `p4_fc_vs_dinov3.csv`：逐场景 F1、性能下降和两模型差值；
- `p4_fc_vs_dinov3.json`：胜出场景数及平均差异。

## 6. 决策门槛

完成两个模型的 37 场景评测后再选择创新主线：

1. 某类扰动随 severity 增大应呈总体恶化趋势；
2. 至少存在一种实际有意义的扰动造成明显 F1 下降；
3. 该下降不能仅靠换成 DINOv3 backbone 自然消失；
4. 候选创新必须能建立“针对性机制—对应损失或训练策略—同协议消融”的证据链；
5. 若所有合成扰动都过弱或不稳定，先修订验证协议，不仓促增加网络模块。
