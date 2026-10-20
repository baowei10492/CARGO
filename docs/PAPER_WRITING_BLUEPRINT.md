# 论文写作蓝图：Task-Aware Selective Alignment for Change Detection

更新时间：2026-07-28  
状态：实验主线冻结，进入表图与初稿阶段  
纪律：不再开启新数据集主线；LEVIR P12/P13 test 禁止重跑；DSIFN、
S2Looking test 保持封存。

## 1. 暂定标题

首选：

> **When and Whether to Align: Task-Aware Selective Registration for Robust
> Remote Sensing Change Detection**

备选：

> **Safe Selective Alignment under Residual Misregistration for Remote
> Sensing Change Detection**

标题避免使用 “universal”“general-purpose”“guaranteed”等超出证据范围的词。

## 2. 核心论点

遥感变化检测中的残余错位包含两个不同问题：

1. **能否对齐**：残差是否可近似为当前估计器能够求解的全局平移；
2. **是否值得对齐**：该残差是否真的造成足够大的变化检测损失。

只有两项同时满足，显式对齐才有价值。本文在样本级采用任务收益驱动的
选择性路由，在数据集级采用几何适用性与 input-oracle headroom 的分级
预检；不满足条件时回退 Raw 或停止部署，而不是强制对齐。

## 3. 方法主线

```text
T1/T2
→ 冻结/微调后的 DINOv3 ConvNeXt-Tiny 变化检测器
→ 条件式非零全局平移定位
→ Raw expert + Aligned expert
→ 基于任务收益监督的 safe router
→ 风险校准阈值
→ 有把握时选择 Aligned，否则回退 Raw
```

数据集级预检：

```text
Gate 0：无标签几何筛查（clean q + 几何匹配瓦片一致性）
→ 明确红灯停止；歧义黄灯不武断判死
Gate 1：validation-only、同 support input-oracle headroom
→ 判断是否存在足够任务收益上限
```

Gate 0 不负责预测任务收益；DSIFN 的黄灯案例必须用于说明两级判据缺一
不可。

## 4. 可主张贡献

1. 将“位移是否存在/能否估计”与“采用对齐是否改善任务”明确解耦；
2. 以变化检测任务收益而非单一几何误差监督选择性对齐；
3. 采用 Raw 默认、风险校准和显式弃权，限制错误对齐的净伤害；
4. 用同支撑量 Oracle headroom、risk-coverage/AURC 和一次性 test
   评价安全—收益权衡；
5. 在 LEVIR 上通过两套检测器权重、一次性 test 和 DiffRegCD
   同 support 强基线验证；
6. 通过 LEVIR、WHU、S2Looking、DSIFN 建立“几何可解性 × 任务效用”
   的两项必要条件，并给出干预或形态证据。

## 5. 四数据集的最终角色

| 数据集 | 几何可解性 | 任务效用 | 论文角色 |
|---|---|---|---|
| LEVIR-CD | 近全局平移，适配 | 高 | 主正例 |
| WHU-CD 原生 | 细 GSD 下非全局分量放大；粗化恢复 | 有 headroom | GSD/覆盖边界与干预证据 |
| S2Looking | 空间变化场，当前全局估计器不适配 | 同时有限 | 非全局场边界 |
| DSIFN-CD | Gate 0 黄灯，几何机制未决 | 明确低 | 任务对 8 px 扰动不敏感的低效用区间 |

禁止把 DSIFN 写成估计器“适用域负例”；禁止写 Gate 0 预测了 DSIFN
失败。正确表述是 Gate 0 弃权、Gate 1 作任务效用裁决。

## 6. 主结果表

### 表 1：LEVIR 一次性 test，同 support

| 方法 | Clean | Shift-2 | Shift-4 | Shift-8 |
|---|---:|---:|---:|---:|
| Raw P3b | 0.900673 | 0.863712 | 0.796434 | 0.663834 |
| 本文 safe_veto_task | **0.900673** | **0.897089** | **0.868934** | **0.819015** |
| DiffRegCD official | 0.870729 | 0.795491 | 0.613877 | 0.621907 |

本文相对 Raw：

```text
Clean +0.0000
Shift-2 +0.0334
Shift-4 +0.0725
Shift-8 +0.1552
Mean shift gain +0.0870
```

本文相对 DiffRegCD：

```text
Clean +0.0299
Shift-2 +0.1016
Shift-4 +0.2551
Shift-8 +0.1971
Mean shift advantage +0.1846
```

### 表 2：LEVIR 消融与风险控制

待从最终证据包统一导出：

```text
Raw
Always Aligned
q gate
Registration-only safe router
Task-aware safe router（primary）
Oracle
AUROC / AURC / excess AURC
aligned selection rate
harmful accepted rate
```

### 表 3：第二检测器权重复刻

至少报告：

```text
seed 3407 与 seed 1234 的 clean drop
shift mean gain
Shift-8 gain
Shift-8 oracle recovery
绝对 tau 不跨检测器权重迁移
```

### 表 4：适用性与停止判据

列出 LEVIR、WHU、S2Looking、DSIFN 的：

```text
clean q
几何一致性/尺度证据
Shift-8 Raw drop
input-oracle headroom
最终路由
```

## 7. 主图

1. **方法结构图**：Raw/Aligned 双专家、任务收益路由、Raw fallback；
2. **LEVIR test shift 曲线**：Raw、本文、DiffRegCD；
3. **风险—覆盖图**：Task-veto 与 Registration-veto；
4. **两项必要条件示意图**：
   横轴 geometric solvability，纵轴 task utility；
   LEVIR、WHU、S2Looking、DSIFN 放入相应位置；
5. **DSIFN vs LEVIR 形态证据图**：
   change fraction、面积加权直径、Shift-8 self-IoU、边界占比；
6. **WHU GSD/覆盖剂量响应图**。

## 8. DSIFN 形态证据（val-only）

| 指标 | DSIFN | LEVIR | 比值/差值 |
|---|---:|---:|---:|
| change fraction | 0.364599 | 0.041966 | 8.688× |
| 面积加权中位等效直径 | 286.07 px | 37.46 px | 7.637× |
| Shift-8 mask self-IoU | 0.888024 | 0.519975 | +0.368049 |
| 8 px 边界占变化区比例 | 0.227182 | 0.839063 | −0.611881 |

论文措辞：

> The validation-mask morphology is strongly consistent with a
> target-scale explanation: DSIFN changes are substantially larger and
> have a lower boundary-to-area ratio, making the same 8-pixel
> translation much less disruptive to pixel-wise change detection.

紧接限制：

> This cross-dataset descriptive audit supports but does not establish a
> causal relationship.

## 9. 论文结构

```text
1 Introduction
2 Related Work
  2.1 Remote-sensing change detection
  2.2 Registration-aware change detection
  2.3 Selective prediction and risk-aware routing
3 Method
  3.1 Problem formulation
  3.2 Raw and aligned experts
  3.3 Task-aware alignment benefit supervision
  3.4 Safe selective router and Raw fallback
  3.5 Dataset-level applicability preflight
4 Experiments
  4.1 Datasets, splits, leakage audit
  4.2 Misregistration protocol and paired support
  4.3 Main LEVIR results
  4.4 Ablation, seeds, risk-coverage
  4.5 External DiffRegCD comparison
  4.6 Applicability boundaries across four datasets
  4.7 Efficiency and perturbation limitations
5 Discussion
  5.1 When alignment is geometrically invalid
  5.2 When alignment has insufficient task utility
  5.3 Limitations
6 Conclusion
```

## 10. 必须主动披露的限制

```text
主正例只有 LEVIR；LEVIR-CD+ 不作为独立数据集；
Task-veto 在部分旋转/亚像素域外场景存在净伤害；
Registration-veto 更安全但域内排序能力较弱；
全局平移假设不覆盖空间变化场；
DSIFN 机制证据是跨数据集描述性关联，不是因果实验；
条件执行的延迟收益依赖覆盖率，Task-veto 不能完全早退；
P12/P13 test 各只运行一次，不能给 test seed 方差。
```

## 11. 禁止主张

```text
首次联合配准与变化检测；
首次使用合成位移监督；
适用于任意几何畸变；
Gate 0 可以单独预测所有失败；
DSIFN 是估计器适用域负例；
LEVIR-CD 与 LEVIR-CD+ 是两个独立正例；
所有相对 DiffRegCD 的优势都来自 router；
已经保证能被二区录用。
```
