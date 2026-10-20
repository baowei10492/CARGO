# P7 Stage C：S2Looking 直迁移预注册协议

> 预注册时间：2026-07-26（在任何 S2Looking 评估运行之前）  
> 前置：P7-3 正式 passed；Phase B 三种子全部 passed 且未触发止损

## 1. 研究问题

LEVIR 训练的整套选择性对齐机制——输入级 descriptor、conditional
localizer、q gate、safe-alignment veto classifier 及其 LEVIR 校准的
概率阈值——在完全不接触 S2Looking 训练数据的条件下，能否在
S2Looking validation 上安全地恢复错位性能？

这是第 16 节阶段 C 的第 1 种验证（真正跨数据集泛化）。若失败，才进入
第 2 种（dataset-specific 重训练），且两者不得混写。

## 2. 混合构成（全部冻结）

```text
Raw/Aligned 专家（backbone.* / decoder.*）：
  S2Looking P6 P3b best checkpoint
  outputs/p6_s2looking_dinov3_partial_ft/20260726T014320Z_seed3407/
    checkpoints/best.pth（best epoch 30，val F1 0.7062）

配准栈（registration.*）：
  LEVIR P7C checkpoint（经 reports/p7c_fullres_calibration_levir/LATEST）

safe classifiers 与 tau：
  LEVIR P7-3 seed-3407 运行
  reports/p7_safe_alignment_classifier_levir/20260726T080834Z/
    safe_alignment_experiment/safe_alignment_{registration,task}.pth
  故意不用 LATEST：Phase B 复跑已把 LATEST 移到 seed 3409，
  预注册迁移工件是原协议种子的冻结产物。
```

混合加载由 `merge_detector_state` 完成：detector 前缀键必须与
S2Looking P3b state 严格一一对应，registration.* 保持 LEVIR 权重，
任何键不匹配直接报错，防止两数据集专家被静默混装。损失定义
（BCE+Dice+pos_weight）与数据配置取自 S2Looking P3b checkpoint 的
config（那是该数据集任务的定义），评估协议数取本协议 yaml。

## 3. 主方法选择（预声明规则的执行结果）

此前已声明：reg-vs-task 取舍以 Phase B 三种子均值为准。Phase B 结果
（LEVIR val）：veto-registration 在全部三个种子上 shift mean gain 与
Shift-8 gain 均优于 veto-task（mean 0.0845±0.0033 vs 0.0791±0.0051；
S8 0.1583±0.0123 vs 0.1455±0.0133），clean 均零损。因此：

```text
primary_method = safe_veto_registration
safe_veto_task = 同一运行内的预声明消融
```

registration-only 特征不依赖检测器输出统计，跨数据集迁移时也更少
暴露于 S2Looking 检测器分布差异。

## 4. 无标签前置门（先于任何带标签指标）

```text
对 S2Looking val 只跑 estimate_registration（不执行 P3b 路径）：
  逐场景 conditional hard PCK@1/@2/@4（相对注入位移，L2 <= k px）
  q 触发率

门槛：Shift-8 conditional PCK@1 >= 0.50
```

不通过 → 立即停止，decision 状态 `stopped_at_unlabeled_gate`，
带标签评估完全不执行（不消耗 validation），进入 dataset-specific
localizer 分支（需另写预注册协议）。该门只使用注入位移真值，
不读取任何变化标签。

## 5. 带标签固定协议（只运行一次）

评估分支：raw / always_aligned / q_gate / safe_veto_registration（主）/
safe_veto_task（消融）/ loss_oracle。场景：clean + Shift-2/4/8（t2）。

预声明门槛（对主方法）：

| 门槛 | 要求 | 理由 |
|---|---:|---|
| Clean F1 drop | <= 0.001（绝对） | clean 安全不打折 |
| Shift-8 transferred oracle gain | >= 0.005 | headroom 保底：oracle≈raw 时 recovery 无意义 |
| Shift-8 oracle recovery | >= 0.50 | 相对同运行 loss_oracle headroom |
| Shift-4 oracle recovery | >= 0.30 | S4 headroom 更小，比例更噪 |
| Shift-2 F1 gain | >= -0.001 | 轻位移不许受损 |

不使用绝对 F1 gain 门槛：主记录 45.2 节已证明 S2Looking 输入级
Oracle headroom 仅 S2 +0.004 / S4 +0.013 / S8 +0.033，LEVIR 式绝对
门槛（如 S8 >= 0.050）在此物理不可达，沿用即重演 P5c 式注定失败。
selection rate 照常报告、不判定。

## 6. 纪律

1. 运行一次；失败不得依据 S2Looking validation 调整 tau、特征、
   阈值规则或换 classifier 种子重试；
2. 失败诊断只读已产出物：unlabeled_diagnostics（定位器域差 vs
   分类器域差）、transfer_sample_decisions.csv（clean 误放行还是
   shift 误 veto）、两 veto 变体对照；
3. 之后按诊断进入 dataset-specific 分支或 P7-5 一致性证据路线，
   均需新预注册协议；
4. S2Looking test 与 LEVIR test 继续不碰；
5. 本协议与实现在服务器运行前已通过本地 49 个 pytest。

## 7. 运行

```bash
bash scripts/run_p7_stage_c_s2looking_transfer.sh
```

产物：

```text
reports/p7_stage_c_s2looking_transfer/<UTC>/
├── decision.json
├── SOURCE_*（四个来源 checkpoint 路径；SHA256 在 results.json）
└── transfer_experiment/
    ├── unlabeled_diagnostics.csv / unlabeled_gate.json
    ├── transfer_scenario_metrics.csv
    ├── transfer_sample_decisions.csv
    ├── results.json / decision.json / protocol_resolved.yaml
```
