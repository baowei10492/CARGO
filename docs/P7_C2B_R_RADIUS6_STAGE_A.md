# C2b-R：搜索域扩大（radius 4→6）的单变量重试预注册协议

> 预注册时间：2026-07-26（C2b 阶段 A 止损评审之后、本运行之前）  
> 授权：止损评审三选项经用户裁决，选择"radius 4→6 重试一次"

## 1. 为什么可以推翻禁止事项 #8

原禁令"不扩大 offset search radius"来自 LEVIR P7-1 诊断：当时扩域
（0.641）差于现状（0.594），失败模式是零候选竞争。S2Looking 的证据
指向完全不同的失败模式——**光度坐标搜索域饱和**：

```text
C2b 阶段 A（radius 4，±8 px）实测：
  S8 boundary_fraction = 0.794（重训前 LEVIR 栈为 0.706）
  逐严重度梯度与边界占比同向：S1 0.06 / S2 0.17 / S8 0.794
  域内证据充分的 S1：delta_pck2 已达 0.882（架构能学会扣除固有）
  clean q 0.672→0.240、固有估计 ≤4.1px 63%→90.4%（再锚定起效）
```

注入 8 px + 固有 ~2–3 px 的真实相关峰位于 10–12 px，落在 ±8 相关
体积之外——训练无法修复模型看不见的证据。该豁免仅适用于
S2Looking C2b-R 一次运行；LEVIR 侧结构继续冻结。

## 2. 变更（严格单变量）

```text
search_radius: 4 → 6（可搜索域 ±12 px，覆盖 8 px 注入 + 90% 固有）
其余与 c2b_s2looking_decoupled_stage_a.yaml 逐项一致
（已用脚本核验 diff 只有 experiment.name 与 search_radius）
```

热启动兼容性：`zero_anchor_head`（Linear 81→169）与
`shiftedness_head`（Linear 166→342）形状依赖 radius，init 构建时
按前缀丢弃、由零初始化重新学习（两头末层本就零初始化，起点良性）；
descriptor、reliability、surface_refiner、logit scales 照常从
LEVIR P7C 热启动；检测器仍为 S2Looking P3b 且冻结。

评估协议 `c2b_r_s2looking_eval_protocol.yaml`：场景与扰动种子与
此前完全一致，仅 maximum_offset_pixels 12.0（boundary_fraction 按
新域界 11.9 计）。

## 3. 预声明门槛与路由（与 C2b 阶段 A 完全相同的门槛）

```text
conditional：S4 与 S8 的 PCK@1 >= 0.70 且 PCK@2 >= 0.85
clean：gated hard within-1px >= 0.90
路由三分支不变（直进特征收集 / 先 q 重校准 / 止损）
```

预声明的归因判读（结果出来前写死）：

```text
S8 大幅改善但 S4 仍不过
  → 域饱和被证实为 S8 主因；S4 剩余失败归因于固有场异质性
    （宽分布、空间变化，单全局平移不可完全刻画）
S8 仍被 boundary_fraction 高值卡住
  → 固有分布尾部超出 ±12，或钳制行为已被训练内化；不再扩域，
    直接止损
两者都过
  → 进入路由分支，S2Looking 链继续
```

本次是 S2Looking 定位器的**最后一次结构性重试**：无论结果，之后
要么进入下一链步，要么执行止损收窄，不再有第三次结构调整。

## 4. 运行

```bash
bash scripts/run_c2b_r_stage_a_s2looking.sh
```

流程与 C2b 阶段 A 相同（pytest → 带头部丢弃的混合 init → smoke →
15 epoch 训练 → gated 评估 → offset 一致性诊断（含新域界
boundary_fraction）→ 固化判定）。radius 6 相关体积 169 候选约 2×
计算，显存余量充足（阶段 A 实测仅 ~6 GiB）。
