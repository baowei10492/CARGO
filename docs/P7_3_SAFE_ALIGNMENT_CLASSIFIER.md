# P7-3 风险受控 safe-alignment veto classifier

> 预注册时间：2026-07-26（在任何服务器运行之前）  
> 前置裁决：正间隔 suppression 审计已失败，手工 suppression 规则正式终止。

## 1. 背景与动机

Training-calibration 正间隔审计（阈值 = clean q-active 最大 suppression
+ 1e-6）结果：

| 规则 | clean 选择 | total retained | Shift-8 retained | harmful rate |
|---|---:|---:|---:|---:|
| q_area_margin | 0 | 14.5% | 19.4% | 12.5% |
| q_probability_margin | 0 | 16.5% | 22.5% | 11.1% |
| q_asymmetry_margin | 0 | 14.5% | 19.4% | 12.5% |
| q_any_margin | 0 | 16.5% | 22.5% | 11.1% |
| q_all_margin | 0 | 14.5% | 19.4% | 12.5% |

预声明门槛为 retained >= 60%、harmful <= 10%，全部失败。因此按固定决策树
进入独立 safe-alignment classifier。

同时，P7-2 的 benefit 回归已证明会过度关注少量大收益 Shift-8 并产生严重
错序（val_34：true benefit +0.767，predicted −0.822）。因此 P7-3 不再回归
完整 benefit，改为二分类“对齐是否安全”。

## 2. 部署规则

```text
align = q_gate AND P(safe) >= tau
否则 Raw
```

- 只对 q 候选执行 veto，分类器不能新增任何 q 之外的选择；
- Raw 是默认路径；
- q gate、定位器、双专家全部冻结，唯一新训练参数是小型 MLP 分类头。

## 3. 训练数据

- 唯一来源：P7-2 报告的 `training_router_features.csv`（LEVIR training
  445 张原图 × clean/Shift-2/4/8，冻结网络的确定性输出）；
- 不重新前向训练集，不接触 validation/test 生成任何监督；
- 运行脚本记录该 CSV 的 SHA256；
- P7-2 的 80/20 fit/calibration 分区弃用，本协议使用全新划分。

## 4. ID 级三段划分

按原图 ID 的稳定 SHA256 哈希（盐 `safe-alignment-split`，split_seed 3407）：

```text
fit               70%
early_stop        15%
risk_calibration  15%
```

同一原图的四个场景永远在同一段。split_seed 与训练 seed 分离：后续多种子
稳定性实验只改训练 seed（`--seed`），划分保持固定。

## 5. 标签与代价加权（全部预声明）

```text
safe = aligned_loss_advantage > benefit_margin
benefit_margin = 0.0
```

margin 取 0 的理由：决策效用的分界就是 benefit = 0；风险不对称性通过
样本权重和 risk 阈值表达，而不是移动标签边界。

```text
weight = clamp(|advantage| / 0.05, 0.05, 4.0)
unsafe 样本            × 3.0
clean 且 unsafe 样本   再 × 2.0
```

- |advantage| 近零的样本标签本质是噪声，magnitude 权重把它们压低；
- unsafe × 3 表达“把有害样本判成 safe”的不对称代价；
- clean 有害误判 safe 是灾难错误，再乘 2（对应 15.4 的
  clean harmful false negative 代价加权）。

## 6. 特征（方向不变）

移除 signed dx/dy 后：

```text
SAFE_REGISTRATION_FEATURE_NAMES（10 个）：
  shift_probability, zero_probability, registration_confidence,
  registration_margin, registration_local_margin,
  registration_reliability, conditional_abs_dx_normalized,
  conditional_abs_dy_normalized, conditional_magnitude_normalized,
  conditional_boundary

SAFE_TASK_FEATURE_NAMES（23 个）：
  上述 10 个 + 13 个 Raw/Aligned task-disagreement 特征
```

两个分类器同协议训练：registration-only 是预声明消融，task-aware 是
预声明主方法（`primary_method = safe_veto_task`）。正式 pass/fail 只由
主方法决定，消融只报告。

## 7. 训练纪律

```text
标准化：fit 段均值/方差
结构：MLP (32,16)，GELU+LayerNorm+Dropout 0.1，输出 1 个 logit
损失：加权 BCEWithLogits（第 5 节权重）
优化：AdamW lr 1e-3，weight decay 1e-3，batch 128
上限 300 epoch；早停只看 early_stop 段加权 BCE，patience 60
保存最佳 epoch 状态，绝不使用 last epoch
在 CPU 上确定性训练
```

## 8. 风险阈值（只用 risk_calibration 段）

```text
tau = max(
  0.5,
  max P(safe) over {risk_calibration ∧ clean ∧ label unsafe} + 1e-6
)
```

- 构造上保证 risk_calibration 段所有 clean 有害样本被 veto；
- 下限 0.5：分类器自己认为 unsafe 时永不放行；
- 若 clean unsafe 集合为空（实际不会发生），回退为对全部 clean 取 max，
  并在 summary 里记录 fallback 标志；
- 同时在 risk_calibration 段输出 tau ∈ {0.00,…,1.00}（101 点）的
  risk-coverage 曲线 CSV，以及 chosen tau 下的 q-veto 审计
  （逐场景 selected/harmful/benefit 与 retained ratio）。

## 9. Validation 固定协议（只运行一次）

评估分支：

```text
raw
always_aligned
q_gate
safe_veto_registration（消融）
safe_veto_task（主方法）
p72_task_router（历史对照，同时校验评估管线复现 P7-2 数值）
loss_oracle（上限，不可部署）
```

预声明门槛（对 safe_veto_task）：

| 门槛 | 要求 |
|---|---:|
| Clean F1 drop | <= 0.001 |
| Shift mean F1 gain | >= 0.020 |
| Shift-8 F1 gain | >= 0.050 |
| Shift-8 Oracle gain recovery | >= 0.50 |

不再设 aligned selection rate 门槛：selective router 以 Oracle gain
recovery 和 clean 风险衡量；selection rate 照常报告但不判定。这不是事后
改判 P7-2——P7-2 的正式状态永久保持 failed。

失败处理：不得依据 validation 调整 tau、margin、权重或特征；只能分析
训练侧泛化失败原因，再以新的预注册协议重新开始。test split 继续不碰。

## 10. 运行

```bash
bash scripts/run_p7_safe_alignment_classifier_levir.sh
```

主要输出：

```text
reports/p7_safe_alignment_classifier_levir/<UTC>/
├── decision.json
├── SOURCE_FEATURES_CSV(.sha256)
└── safe_alignment_experiment/
    ├── safe_alignment_registration.pth / _summary.json / _risk_coverage.csv
    ├── safe_alignment_task.pth / _summary.json / _risk_coverage.csv
    ├── training_split_assignments.csv
    ├── validation_scenario_metrics.csv
    ├── validation_sample_decisions.csv
    ├── results.json
    ├── decision.json
    └── protocol_resolved.yaml
```
