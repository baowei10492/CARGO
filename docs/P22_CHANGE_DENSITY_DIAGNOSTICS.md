# P22 诊断：变化密度与定位器失效（预注册判读，2026-09-14）

> 背景：主记录 §157 发现 WHU val 条带上冻结 P7C 定位器的失效与分析窗口内变化占比强相关
> （< 5% 零失效，>= 15% 大多失效），但失效几乎全在 val 条带中段，**变化占比与位置混杂**。
> 用户 2026-09-14 选定「先做两个廉价诊断，再决定是否换定位器」。
> 本文件在两个诊断运行之前写定判读；看到结果后不得修改门槛、分箱与选样规则。
> 两个诊断都不改变 P22 已作出的 Gate 1 判定（recovery_insufficient，止损）。

---

## A1：train 区变化占比剂量-响应（解开位置混杂）

```text
问题：val 条带之外，变化密集的窗口是否同样让定位器失效？
选样：WHU train 栅格 train 列 0..33（x < 17408）、行 0..28（y < 14848）。
  以 512 px 网格为步长枚举全部 3072 px 窗口（29 × 24 = 696 个），用 train 变化标签
  计算窗口内变化占比；按四个分箱 [0, 5%)、[5%, 10%)、[10%, 15%)、[15%, 100%]
  各自按行优先顺序贪心选取，窗口原点切比雪夫间距 >= 1536 px，每箱最多 20 个。
  只读 train 区标签用于分层；不读 val 条带、不开 test 栅格。
测量：与 P22 Gate 0 完全相同的无标签定位——窗口 6 倍 BOX 降采样，冻结 LEVIR P7C
  （sha 389c65e1…），场景 clean 与 B 族 12 / 24 / 48 px；每窗口记 S2/S3 是否在 1 粗像素内、
  误差、clean q 是否触发。
判读（按窗口计数，每箱 n = 选到的窗口数）：
  「能用」：S3 coarse PCK@1 >= 0.90 且 clean q 触发率 <= 0.10
  「失效」：S3 coarse PCK@1 <= 0.50 或 clean q 触发率 >= 0.50
  最低箱 [0, 5%) 须「能用」（n >= 8），否则 inconclusive_low_bin_failed（仪器在 train 区也不稳）
  最高箱 [15%, 100%] 若 n < 8，与 [10%, 15%) 合并后再判（标注 merged_high_bin）；仍 < 8 则
    inconclusive_insufficient_high_change_windows
  高箱「失效」→ reproduced_outside_val：变化密集的效应在 val 之外复现，位置混杂解除
  高箱「能用」→ not_reproduced：val 条带中段存在位置特有因素（区域性错位/视差等）更可能
  其余 → partial
描述量：各箱 S2 PCK@1、误差均值；窗口级 Spearman（变化占比 vs S2/S3 误差、clean q）
```

## A2：跨数据集——P21 已有逐样本定位误差（零 GPU）

```text
问题：DSIFN、GZ-CD 上的定位器长尾（主记录 §151、§152）是否与瓦片内变化占比相关？
数据：P21 的 BIT × {LEVIR-CD, DSIFN-CD, GZ-CD} 三格已写出的
  safe_alignment_experiment/validation_sample_decisions.csv（逐样本 conditional_epe_pixels、
  q_gate）。定位器输出与检测器无关，任一行的三格均可；取 BIT 行。
  逐瓦片变化占比读各数据集 val 掩膜（按各自 mask_threshold），这些标签已被 P21 使用过。
量：每数据集 Spearman（变化占比 vs S3（8 px）conditional EPE；vs clean q），以及同样四箱的
  EPE 均值与 q 触发率。
判读（每数据集）：
  associated：Spearman(变化占比, S3 EPE) >= 0.30，且 [15%, 100%] 箱与 [0, 5%) 箱各 >= 10 个样本、
    高箱 S3 EPE 均值 >= 2 × 低箱均值
  not_associated：Spearman <= 0.10
  其余：weak_or_inconclusive
总判读：DSIFN 与 GZ-CD 均 associated → cross_dataset_supported；均 not_associated →
  cross_dataset_not_supported；其余 → mixed。LEVIR 作参照，不参与总判读。
```

## 结果到行动的映射（写于运行前）

```text
A1 reproduced_outside_val 且 A2 cross_dataset_supported
  → 「变化密集破坏全局相关」跨区域、跨数据集成立：写入论文适用条件；
    P23 变化感知定位（匹配前遮掉预测变化区）有充分依据，另写预注册协议
A1 reproduced_outside_val 且 A2 非 supported
  → 机制在 WHU 内成立、位置混杂解除，但不能推广到其他数据集：P23 仅以 WHU 为目标
A1 not_reproduced
  → val 中段有位置特有因素，换定位器缺乏依据：WHU 线收尾，写入论文边界
A1 partial / inconclusive
  → 不开新线，WHU 收尾；如实报告诊断结果
```

## 限制（事先声明）

```text
A1 用真值标签分层（部署时不可得）；窗口间距 1536 仍允许约 25% 面积重叠；
train 区平均变化占比 0.036，高变化箱可能窗口不足（已预设合并与不足规则）。
A2 的「窗口」即瓦片本身（512 或 1024 px），定位器与尺度设置不同于 P22，只检验关联方向。
两者都是相关性证据，不构成因果证明。
```
