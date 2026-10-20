# P7-5 前置：瓦片一致性候选特征筛选（预注册协议）

> 预注册时间：2026-07-26（卫生诊断 v2 判据成立之后、本筛选运行之前）  
> 目的：在写 P7-5 完整协议之前，先用**只有训练集**的证据决定
> 瓦片一致性特征是否值得进入特征集，避免又一次凭直觉加特征

## 1. 为什么现在做这一步

卫生诊断 v2（主记录 67）证实瓦片一致性在**数据集级**有判别力
（LEVIR D=0 vs S2Looking D>=4）。但 P7-5 需要的是**样本级**判别力：
在同一个 LEVIR 上，瓦片分歧能否区分"这一张图对齐有益 / 有害"。
数据集级有效不代表样本级有效，必须单独验证。

一个正面线索：LEVIR S8 的 median D 为 0，但只有 53.1% 的图像
max_pairwise <= 2 px，即约 47% 的图像存在瓦片分歧——样本间存在
真实差异可供判别，不是常量特征。

## 2. 候选特征（全部无标签、无需知道注入量）

```text
tile_max_pairwise_distance    瓦片两两最大距离
tile_dispersion_radius        瓦片到质心平均距离
tile_agreement_within_2px     与整图估计一致的瓦片比例（2 px）
tile_agreement_within_4px     同上（4 px）
tile_mean_offset_magnitude    瓦片估计幅度均值
tile_magnitude_spread         瓦片幅度极差
```

**明确排除**：任何用到注入真值的量（如 tile_error_*）。部署时不知道
注入量，凡用真值构造的特征都不可部署，不得进入候选集。

## 3. 数据与纪律

```text
样本：LEVIR train split，clean + Shift-2/4/8（与 P7-2 特征表同键）
标签：safe = aligned_loss_advantage > 0（沿用 P7-3 预注册定义）
监督来源：P7-2 冻结产物 training_router_features.csv
验证集：完全不使用（本步骤不消耗任何评估机会）
```

注意：clean 场景的瓦片特征处于 forced-choice 区（v1 教训），但该区
行为在部署时同样发生，因此作为特征照常纳入；若它在 clean 上恰好
指向"不对齐"，那正是 veto 需要的方向——由数据决定，不预设。

## 4. 预声明的纳入规则

对每个特征计算 **directional AUROC** = |AUROC − 0.5| + 0.5
（方向不可知时对下游 MLP 等价，故用方向无关幅度，并记录方向）。

```text
纳入门槛 bar = max(0.60, 十个现役 registration 特征 pooled
                    directional AUROC 的中位数)
候选特征 pooled directional AUROC >= bar → 纳入 P7-5 特征集
```

用现役特征的中位数作为参照，保证"新特征至少不比现有证据更弱"；
0.60 下限防止在现役特征整体很弱时把噪声也放进来。

结果分三种：

```text
有候选达标 → 进入 P7-5：写完整预注册协议，特征只增不改、
  定位器与专家冻结、沿用 P7-3 的划分与阈值规则，与 P7-3 构成
  严格 ablation 对；建议同时采纳主记录 57.2 的教训改用分位数
  tau 规则并预声明
无候选达标 → 正式放弃瓦片一致性作为 P7-5 特征，转向评审 1
  第 4.2 节解法 B 的其余三类证据（循环/幂等一致性、方向对称性、
  相关面结构），仍需各自筛选
```

无论哪种结果，本筛选的 AUROC 表全部记录，不得只报达标项。

## 5. 运行

第一步：在 LEVIR **train** 上采集瓦片记录（约 445 图 × 4 场景 ×
5 次前向，registration-only，十几分钟量级）：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
p7c="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
P7_CKPT="$(cat "$p7c/CHECKPOINT")"
tag="$(date -u +%Y%m%dT%H%M%SZ)"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$P7_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --manifest manifests/datasets/levir_cd.csv \
  --split train --label levir_train_tiles \
  --severities 0 1 2 3 \
  --device cuda \
  --output-dir "reports/p7_5_screening/$tag/tiles"
```

第二步：与 P7-2 训练特征表联接并应用纳入规则（秒级，纯 CPU）：

```bash
router="$(cat reports/p7_task_benefit_router_levir/LATEST)"

python -u -m tools.screen_tile_consistency_features \
  --tile-records "reports/p7_5_screening/$tag/tiles/tile_offset_records.csv" \
  --training-features "$router/router_experiment/training_router_features.csv" \
  --output-dir "reports/p7_5_screening/$tag/screening"
```

把 decision.json 的完整输出（含两张 AUROC 表）交回。
