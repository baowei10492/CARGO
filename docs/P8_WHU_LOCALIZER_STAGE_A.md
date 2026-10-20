# P8-A：WHU 原生定位器（预注册协议）

> 预注册时间：2026-07-26，在任何 WHU 定位器训练之前  
> 前置：决定性判据已通过（主记录 71，Shift-8 input-oracle headroom
> 0.10880 >= 0.05），WHU 存在可恢复的失配空间  
> 本阶段唯一目的：把"筛查为红"的**歧义**解开，并给出与 LEVIR 同判据
> 的第二个域内定位器结果

## 1. 本阶段回答的问题

70.7 的筛查用 **LEVIR P7C 定位器**测 WHU，得到 clean q 触发率 0.833、
tile_error_pck4 三数据集最低。协议第 3 节当时就声明这类"红"是歧义的，
两种解释在观测上不可分：

```text
解释 A 描述子域差：LEVIR 描述子在 0.075 m 航空影像上不работает，
  换成 WHU 原生训练即可修复
解释 B 空间变化场：WHU 的固有失配本身不是全局平移，
  任何逐图全局估计都无法表示，属方法适用域之外（S2Looking 型）
```

决定性判据**无法**区分二者：Oracle 逆位移把图像对送回注入前的原始
状态，那个状态含不含固有失配，逆变换既不知道也不消除（主记录 71.4）。
因此区分二者必须训练一个 WHU 原生定位器——clean-anchor 监督会把
"零位移"重新定义为 WHU 自身的固有失配（任务坐标）。

判读规则（预注册，写入 decision.json）：

```text
原生定位器 clean q 触发率 <= 0.20（LEVIR 自身 0.0625）
  → attribution = descriptor_domain_gap，解释 A 成立
> 0.20
  → attribution = intrinsic_field_or_unresolved，解释 B 获得支持，
    GSD 机制假设（主记录 70.7 第 4 条 / 71.3）获得支持
```

## 2. 配方：除数据段外与 C2b 逐项一致

`configs/p8_whu_decoupled_stage_a.yaml` 对照
`configs/c2b_s2looking_decoupled_stage_a.yaml`（后者当时声明与已验证的
LEVIR P7-1B 配方逐项一致），**唯一偏离**：

```text
data 段指向 WHU-CD / manifests/datasets/whu_cd.csv
evaluation.batch_size 8（C2b 为 4）——由 512 图幅导致，非调参
```

`search_radius` 保持 4，这是刻意的：本阶段的价值来自与 LEVIR P7-1B
在**同一套门槛**下比较，而 radius 改变候选数（80 → 168）从而改变强制
选择的难度。禁令 #8 继续有效；`boundary_fraction` 会被报告，任何关于
radius 的讨论必须基于该证据并取得用户明确同意。

不做任何针对 WHU 的超参搜索。训练为 registration-only（检测器冻结），
15 epoch × 8192 crops。

初始化：`tools/build_c2b_init_checkpoint.py` 合并 LEVIR P7C 配准栈
（warm start，随后在 WHU 上重训）与 **WHU P3b 检测器 best.pth**
（冻结专家）。radius 未变，不需要 `--drop-prefixes`。

## 3. 门槛（预注册，数值与 C2b 完全相同）

复用 `tools/summarize_c2b_stage_a.py`，同一份门槛代码、同一组数值，
使两个数据集由同一仪器裁决：

```text
条件式（对照注入量，WHU val）
  Shift-4 conditional PCK@1 >= 0.70   PCK@2 >= 0.85
  Shift-8 conditional PCK@1 >= 0.70   PCK@2 >= 0.85
clean
  gated hard within 1 px >= 0.90
```

路由（与 C2b 同构）：

```text
全通过         → 进入下一阶段（P7-1C 式全分辨率 q 校准按需 →
                 特征收集 → P7-3 复刻），各需新协议
条件通过/clean 失败 → 全分辨率 q 校准（P7-1C 先例），不算失败
条件失败       → WHU 定位器分支止损；此时解释 B 成立，适用域条件
                 获得第二个独立负例，论文形态为"LEVIR 单域内成功 +
                 两个边界案例 + 可测量的部署判据"
```

## 4. 事先声明的解读规则（不改门槛）

**4.1 钳制伪命中的双向声明。** `boundary_fraction`（S8，对照 ±8 px 域）
必须报告。阈值 0.30（S2Looking 实测 0.706，主记录 63.3）：

```text
高 boundary_fraction 且 S8 通过 → 该通过被声明为含钳制成分，
  在有更大 radius 的对照之前不予采信
高 boundary_fraction 且 S8 失败 → 诊断为搜索域不足而非描述子失效
两种情况都不改变门槛记录的正式通过/失败，也都不授权改 radius
```

这一条在 `boundary_contamination_annotation` 中实现，并有测试断言
它**不影响任何 gate**（`test_boundary_contamination_is_annotation_never_a_gate`）。

**4.2 阈值不可迁移，故卫生诊断只作描述。** 本阶段会用 WHU 原生定位器
跑一次瓦片离散度（tile_size 256，native 512），但 1024/512 配置下校准的
门槛（D<=2 绿 / >=4 红）**不得搬用**——LEVIR 自身 S8 的 D 在 512/256 下
从 0.000 变成 7.606（主记录 70.7 第 3 条 / F4）。只报告免阈值的相对量
（tile_error_pck2/4、median_tile_error_magnitude）与匹配配置下的 LEVIR
参照值，不下绿红结论。

**4.3 LEVIR 参照必须并列。** 报告 PCK 时同时给出 LEVIR P7-1B 的对应值，
且按 F6 给出图像数或置信区间（WHU val n=174，比 LEVIR 的 n=64 好，
1 张 = 0.575 个百分点）。

**4.4 F8：跨 GSD 的严重度不等价。** 8 px 在 WHU 上是 0.60 m，在
LEVIR/S2Looking 上是 4.00 m。任何跨数据集的 PCK / headroom 比较都必须
附上物理位移量，不得读成方法适用性的强弱（主记录 71.5）。

## 5. 成本与禁令

```text
成本：init（秒级）+ gpu smoke + 15 epoch registration-only（约 1–2 h）
     + 三项 val 诊断（分钟级）
禁令：不碰 test split；不调 q_shift 阈值；不扩 search radius；
     不为 WHU 搜超参；结果不通过时不得事后改门槛
```

## 6. 运行

```bash
bash scripts/run_p8_whu_localizer.sh
```

流程：pytest → 合并初始检查点 → gpu smoke → 训练 → `evaluate_registration`
（clean 门）→ `diagnose_transfer_offset_consistency`（条件式 PCK 与
boundary_fraction）→ `summarize_c2b_stage_a`（固化判据 + 歧义判读）→
卫生诊断（描述性）。产物在 `reports/p8_whu_localizer/<UTC>/`。
