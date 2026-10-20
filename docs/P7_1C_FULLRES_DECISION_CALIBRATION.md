# P7-1C：全分辨率失配决策校准

## 1. 唯一目的

冻结已在 Shift-8 达到 92.19% PCK@1 的条件定位器，只重新校准
`q_shift`。不修改 P3b、descriptor、correlation、radius、offset head 或
0.5 阈值。

## 2. 触发证据

P7-1B：

```text
Conditional Shift-8 PCK@1 = 92.19%
Applied Shift-8 PCK@1     = 51.56%
Clean q decision accuracy = 96.88%
Shift-8 q detection       = 54.69%
```

训练使用 256×256 crop 并在训练时用 GT mask 排除变化区域；验证使用
1024×1024 全图且无法使用 GT mask。这是明确的决策特征分布错位。

## 3. 校准协议

```text
训练输入：完整 1024×1024 training images
训练时 GT mask exclusion：关闭
冻结：descriptor、reliability、surface refiner、conditional offset
训练：shiftedness head only
P3b forward：跳过
人工失配：50%，{2,4,8} px
epochs：8
samples/epoch：1024
batch size：4
threshold：固定 0.5
```

P3b forward 被跳过，因此训练过程中的 `val_f1` 没有意义；只使用
`last.pth` 做固定 Validation offset 评测。

## 4. 停止规则

门槛保持不变。若 P7-1C 仍不能同时满足：

```text
Clean hard PCK@1 >= 95%
Shift-8 hard PCK@1 >= 80%
Shift-8 hard PCK@2 >= 95%
```

则停止继续修改 selective gate，不再增加 epoch 或调阈值。
