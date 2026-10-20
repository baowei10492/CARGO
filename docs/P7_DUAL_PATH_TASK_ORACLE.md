# P7 双路径任务收益 Oracle

## 1. 目的

P7-1B 已证明条件非零位移定位器在 LEVIR-CD Shift-8 上达到
`PCK@1=92.19%`，P7-1C 又证明全分辨率校准可以缓解、但不能消除
`q_shift` 的 clean/shift 冲突。

因此本阶段不再继续调 `q_shift`，而是先回答：

> 冻结 Raw 与条件 Aligned 两个专家后，如果每张图都能选择任务损失更低的
> 分支，变化检测 F1 的上限是否足以支持训练一个安全路由器？

## 2. 四种固定评估方法

1. `raw`：冻结 P3b，直接处理原始图像对；
2. `always_aligned`：使用条件非零 hard offset 对齐 T2，再运行冻结 P3b；
3. `q_gate`：用当前 P7C `q_shift` 在 Raw/Aligned 之间选择；
4. `loss_oracle`：逐图读取 GT，以有效区域内 BCE+Dice 较小者为选择结果。

`loss_oracle` 使用标签，只是可行性上限，不能作为最终模型结果，也不能用于
测试集报告。

## 3. 公平性约束

- 只评估 LEVIR-CD validation；
- corruption seed 固定为 3407；
- 场景固定为 clean 与 T2 Shift-2/4/8；
- 四条分支使用同一个冻结 P3b；
- Aligned 只使用已经训练完成的 conditional non-zero hard offset；
- 所有分支使用 corruption valid mask 与固定 16 px interior mask 的交集；
- Oracle 逐图选择完整预测，不允许逐像素拼接；
- task loss 权重继承 checkpoint 中的 BCE、Dice 与 positive weight。

## 4. 预声明门槛

| 门槛 | 要求 |
|---|---:|
| clean Oracle F1 降幅 | `<= 0.001` |
| Shift-2/4/8 平均 Oracle F1 增益 | `>= 0.020` |
| Shift-8 Oracle F1 增益 | `>= 0.050` |
| Shift-8 Oracle 选择 Aligned 的比例 | `>= 0.50` |

全部通过才进入“训练 task-benefit router”。

若 Oracle 不通过，说明即便知道标签，Raw/Aligned 两个专家也没有足够互补性，
应停止路由器路线并回到配准或融合分支本身。

## 5. 输出

运行脚本：

```bash
bash scripts/run_p7_dual_path_oracle_levir.sh
```

结果位于：

```text
reports/p7_dual_path_oracle_levir/<UTC>/
├── decision.json
├── oracle_val/
│   ├── results.json
│   ├── decision.json
│   ├── scenario_metrics.csv
│   ├── sample_decisions.csv
│   └── protocol_resolved.yaml
├── oracle_val.log
└── pytest.log
```

`sample_decisions.csv` 保存逐样本 Raw/Aligned task loss、Oracle 选择、当前
`q_gate` 选择、两者一致率、条件位移 EPE 和门控概率，可直接用于下一阶段
设计路由器输入。
