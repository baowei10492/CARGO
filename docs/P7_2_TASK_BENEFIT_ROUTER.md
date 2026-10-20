# P7-2 风险受控任务收益路由器

## 1. Oracle 结论

P7 双路径 Oracle 在 LEVIR-CD validation 上全部通过预声明门槛：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| Clean Oracle F1 drop | -0.000079 | <= 0.001 |
| Shift 平均 Oracle F1 gain | +0.112478 | >= 0.020 |
| Shift-8 Oracle F1 gain | +0.214161 | >= 0.050 |
| Shift-8 Aligned selection rate | 93.75% | >= 50% |

核心分支：

| 场景 | Raw | Always Aligned | P7C q gate | Loss Oracle |
|---|---:|---:|---:|---:|
| Clean | 0.9032 | 0.8485 | 0.8931 | 0.9033 |
| Shift-2 | 0.8672 | 0.8952 | 0.8952 | 0.9002 |
| Shift-4 | 0.8053 | 0.8884 | 0.8884 | 0.8956 |
| Shift-8 | 0.6793 | 0.8934 | 0.8754 | 0.8934 |

这说明：

1. Aligned expert 对严重错位有很强的任务收益；
2. 无条件对齐会严重破坏 clean；
3. 当前 `q_shift` 已恢复大部分 shift 性能；
4. 但 clean 中仅 4/64 次错误启用，就造成约 0.0101 F1 损失；
5. 需要的是高精度、Raw 默认的任务收益路由，而不是普通位移分类器。

## 2. 监督目标

只在 LEVIR-CD training split 上生成监督：

```text
benefit = L_raw - L_aligned
```

其中 `L` 是固定有效区域内的 BCE+Dice。`benefit > 0` 表示采用 Aligned
降低任务损失。

训练和校准按原始 sample ID 的稳定 SHA256 哈希划分，确保同一原图的 clean、
Shift-2、Shift-4 和 Shift-8 版本不会跨区泄漏：

```text
80% fit
20% calibration
```

Validation 不参与拟合、早停、特征选择和阈值选择。

## 3. 两类路由器

### 3.1 Registration-only router

只输入配准侧证据：

- `q_shift` 与 zero probability；
- registration confidence、global/local margin、reliability；
- conditional offset 的方向、幅度和是否触及搜索边界。

### 3.2 Task-aware router

在以上证据基础上增加 Raw/Aligned 两次冻结分割输出之间的无标签分歧：

- 平均变化概率与变化面积；
- entropy 与预测置信度；
- probability disagreement；
- binary disagreement；
- Raw-only/Aligned-only change fraction；
- 两个预测掩码的 IoU。

这两种路由器在同一次运行中训练和验证，Registration-only 是预声明消融，
用于检验“任务分歧证据”是否真正有额外价值。

## 4. 风险控制

路由器回归预测 task benefit。最终规则：

```text
score <= threshold → Raw
score > threshold  → Aligned
```

阈值固定为：

```text
max(
  0,
  training calibration 中所有 clean 样本的最高 score + 1e-6
)
```

因此在 training calibration 上不允许任何 clean 样本切换到 Aligned。该规则
不保证 validation 零误切换，但明确把 clean 安全性置于 shift recall 之前。

## 5. Validation 预声明门槛

| 指标 | 门槛 |
|---|---:|
| Clean task-router F1 drop | <= 0.001 |
| Shift 平均 task-router F1 gain | >= 0.020 |
| Shift-8 task-router F1 gain | >= 0.050 |
| Shift-8 Oracle gain recovery | >= 50% |
| Shift-8 Aligned selection rate | >= 50% |

Validation 协议只运行一次。若失败，不根据 validation 调阈值；只能分析训练集
校准泛化失败的原因，再以新的预注册协议重新开始。

## 6. 运行

```bash
bash scripts/run_p7_task_benefit_router_levir.sh
```

主要输出：

```text
reports/p7_task_benefit_router_levir/<UTC>/
├── decision.json
└── router_experiment/
    ├── registration_router.pth
    ├── task_router.pth
    ├── registration_router_summary.json
    ├── task_router_summary.json
    ├── training_router_features.csv
    ├── validation_scenario_metrics.csv
    ├── validation_sample_decisions.csv
    ├── results.json
    ├── decision.json
    └── protocol_resolved.yaml
```
