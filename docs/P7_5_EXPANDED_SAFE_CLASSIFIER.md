# P7-5：瓦片一致性特征扩充安全分类器（预注册协议）

> 预注册时间：2026-07-26，在扩充分类器任何训练之前  
> 前置：筛查两道关均已裁决（主记录 69.2 预注册的按场景规则 +
> 主记录 80 的裁决）：6 个候选中 **4 个入主特征集**，
> tile_magnitude_spread 仅限消融，tile_mean_offset_magnitude 出局  
> 目标：吃下 P7-3 天花板分解中的 classifier over-veto 份额
> （LEVIR S8 上限 0.2141 = 0.1396 已回收 + ~0.0565 过度否决 +
> 0.0180 q 召回架构缺口）

## 1. 入选特征与依据（已裁决，不得追加）

```text
入主集（4，按场景 AUROC：clean / S8）：
  tile_max_pairwise_distance   0.674 / 0.796
  tile_dispersion_radius       0.689 / 0.802
  tile_agreement_within_2px    0.646 / 0.801
  tile_agreement_within_4px    0.649 / 0.784
仅消融：tile_magnitude_spread（pooled 0.698 过杆，但 clean 内 0.517
  → 按 69.2 规则判为场景代理，这正是该规则要抓的情形）
出局：tile_mean_offset_magnitude（pooled 0.585 与 S8 0.536 双不过）
诚实注记：所有候选在 S1 内都弱（~0.56）——瓦片一致性对 2 px 位移
  无分辨力；其价值在 clean 与大位移端，作为与
  registration_confidence（clean 0.741 / S8 0.876）互补的
  空间一致性轴，而非替代。
```

按场景规则**不追溯**适用于在位特征（shift_probability 在 S1 内仅
0.502 等观察照实记录，但 P7-3 判定与特征集冻结不变）。

## 2. 结构：唯一变化是特征集

```text
veto-registration 集：10 → 14
veto-task 集：       23 → 27
特征全集（张量列序）：ROUTER_FEATURE_NAMES(35) + 4 瓦片特征追加在尾部
  ——前 35 列不动，冻结的 P7-2 task-router bundle 索引保持有效
训练侧：瓦片特征来自筛查已固化的 joined_training_rows.csv，按
  (id, scenario) 严格全覆盖连接（缺一行即报错），sha256 记录在案
验证侧：逐批实时计算，口径与筛查所用的卫生记录完全一致——
  非重叠 512 瓦片上 estimate_registration 的 conditional hard offset，
  以整图 conditional 估计为参照；归约公式由
  rs_cd.dual_path.tile_consistency_features 单点实现，
  测试断言与筛查实现逐值相等（防公式漂移使筛查证据失效）
其余全部字节不变：三分（同盐同种子 → 与 P7-3 同划分）、标签、权重、
  网络结构 (32,16)、tau 规则、门槛数值。P7-3 的协议文件不动（禁令 13），
  工具改动向后兼容（tile 块缺省关闭时行为与 P7-3 逐字节相同，有测试）。
```

## 3. 门槛与路由（预注册）

四条 P7-3 原门槛原数值照用（clean drop <= 0.001、shift mean >=
0.020、S8 >= 0.050、recovery >= 0.50，primary = safe_veto_task）。

新增**同种子配对改进门**，基线 = 最早的 P7-3 seed-3407 运行
（同划分盐 → 唯一差异就是特征集）：

```text
shift8_gain_improvement >= +0.010（F1 按像素池化，远超噪声）
shift_mean_gain_delta   >= −0.002（防"牺牲小位移买 S8"）
```

路由：

```text
原门槛全过 且 改进门过 → p7_5_phase_b_three_seeds（三种子复刻）
原门槛全过 但 改进门不过 → 扩充判 neutral，P7-3 保持最终版
  （这是合法且有价值的负结果，不得复调）
原门槛任一失败 → 扩充判 harmful，弃用，P7-3 保持最终版
```

消融（spread 版）仅在主判定后按需另行预注册,不与主运行混跑。

## 4. 成本与运行

```text
成本：分类器训练是 CPU 级 MLP；验证多 4 次/批的瓦片配准前向，
  总计约比 P7-3 验证慢一半，全程 << 1 h
bash scripts/run_p7_5_expanded_classifier.sh
产物 reports/p7_5_expanded_classifier_levir/<UTC>/
```
