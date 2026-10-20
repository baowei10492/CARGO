# C2b 第一步：S2Looking 解耦配准栈重训练（预注册协议）

> 预注册时间：2026-07-26（C2 诊断完成之后、本训练运行之前）  
> 路由依据：C2 无标签诊断未满足 C2a 条件
> （delta_pck1(S8)=0.064 < 0.80；residual 两两 pck2 最高 0.682 < 0.80）

## 1. C2 诊断的机制读数（本协议的设计输入）

```text
S1(2px)：raw pck2 0.478 → delta pck2 0.732，EPE 2.96 → 1.99
S2(4px)：raw pck2 0.528 → delta pck2 0.656，EPE 2.93 → 2.28
S8(8px)：raw pck1 0.486 但 delta pck1 0.064（反转！）
clean 固有估计：median 2.83 px，仅 63% <= 4.1 px
residual 两两一致性 pck2 0.53–0.68
```

三个结论：

1. **描述子没有死**（H2 强形式不成立）：在搜索域内（S1/S2），逐图
   扣除 clean 估计后 2px 内命中率升到 0.66–0.73，定位器确实在跟随
   注入位移；
2. **固有失配是宽分布、非单一稳定向量**（H1 强形式不成立）：clean
   固有估计中位数 2.83 px 但分布很宽，residual 跨严重度一致性只有
   中等水平——更像空间变化的视差/配准残差场，单一全局平移只能
   部分刻画。这同时解释了 45.2 节 S2Looking oracle headroom 很小：
   检测器在训练中已适应固有失配；
3. **S8 存在光度坐标搜索域饱和**：注入 8 px 恰在 ±8 搜索边界上，
   固有分量使真实光度位移越界、估计被钳到边界附近——这解释了
   S8 raw pck1 反常偏高（0.486，部分是钳制伪命中）而 delta 崩溃
   （0.064）。任务坐标系的正确答案（8 px）本身在域内，重训练需要
   学会把"越界光度证据"映射回任务答案；能学到什么程度是本步
   要测量的问题之一（新增 boundary_fraction 指标）。

## 2. 方案

沿用 LEVIR 已验证的 P7-1B 解耦配方，仅数据换成 S2Looking：

```text
配置：configs/c2b_s2looking_decoupled_stage_a.yaml
  = p7b_decoupled_input_registration_stage_a.yaml，除 data 节外逐项一致
  （decoupled_shift_decision、shiftedness_weight 1.0、
   clean_anchor_weight 2.0、lr 3e-4、15 epoch × 8192 crops）

初始化：tools/build_c2b_init_checkpoint.py 生成
  S2Looking P3b 检测器（backbone/decoder，训练中冻结）
  + LEVIR P7C 配准栈（registration.*，作为热启动被重训练）
  关键机制：clean-anchor 在 S2Looking 训练对上把"零点"重定义为
  该数据集的固有失配（任务坐标系再锚定）

训练只更新 registration.*（registration_only: true）；
产出 checkpoint 自带 S2Looking 检测器，后续步骤无需再混装。
```

## 3. 评估与预声明门槛（S2Looking val，与 C2 诊断同协议同扰动种子）

评估两套：`evaluate_registration`（gated 指标）+
`diagnose_transfer_offset_consistency`（conditional/delta/boundary/q）。
判定由 `tools/summarize_c2b_stage_a.py` 固化。

Conditional 定位门槛（估计器是否可用）：

```text
Shift-4 conditional PCK@1 >= 0.70    PCK@2 >= 0.85
Shift-8 conditional PCK@1 >= 0.70    PCK@2 >= 0.85
（LEVIR 参照：conditional PCK@1(S8) = 92.19%）
```

Clean 门槛（q 头是否守住 clean）：

```text
clean gated hard within-1px >= 0.90
```

路由（预声明）：

```text
conditional 全过 且 clean 过
  → 第 3 步：P7-2 式特征收集 + P7-3 协议复刻（跳过 q 重校准）
conditional 全过 但 clean 不过
  → 第 2 步：P7-1C 式全分辨率 q 重校准（LEVIR 有先例）
conditional 不过
  → S2Looking 侧止损评审：论文收窄为 LEVIR 图像级 +
    S2Looking 失败机制分析（固有失配场 + 域饱和证据本身入文），
    区域级列 future work
```

诊断性报告（不判定）：clean q 触发率、S8 boundary_fraction、
delta/residual 一致性变化（对比 C2 迁移基线：0.204/0.204/0.486）。

## 4. 纪律

1. 单次运行；失败不得改 lr/epoch/权重重试后当作同一协议；
2. 本步属 dataset-specific training，与 Stage C 直迁移结论分开表述；
3. 后续第 2/3 步各需新的预注册协议；
4. 一切 test split 不碰；
5. 运行前本地 pytest 56 passed。

## 5. 运行

```bash
bash scripts/run_c2b_stage_a_s2looking.sh
```

流程：pytest → 构建混合 init（键严格校验）→ gpu smoke → 训练
（15 epoch，约 1–2 小时量级）→ gated 配准评估 → offset 一致性
诊断 → 固化判定。产物在 `reports/c2b_stage_a_s2looking/<UTC>/`。
