# P8-C：图幅匹配归因控制实验（预注册协议）

> 预注册时间：2026-07-26，在控制实验运行之前  
> 前置：WHU 定位器阶段 A 全部门槛失败（主记录 73），判定已冻结  
> 本实验**不能**重开任何门槛。它只决定那次失败在论文里可以归因于什么。

## 1. 为什么必须先做这个

WHU 的所有定位器数字测在 **512 px** 瓦片上，而被用作对照的 LEVIR
数字全部测在 **1024 px** 影像上。全局位移估计器在 512 上可投票的
支撑面积只有 1/4。已有的直接警告：

```text
LEVIR 自身 S8 的瓦片一致性 D
  1024 图幅 / 512 瓦片 → 0.000（完美）
   512 图幅 / 256 瓦片 → 7.606（自己判红）
（主记录 70.7）
```

即"把 LEVIR 缩到 512"这一个动作就足以让参照数据集自己崩掉。在这种
情况下拿 512 的 WHU 去和 1024 的 LEVIR 比，就是 F4（仪器不匹配）。

另有一个反常提示：WHU 在 delta 抵消（0.057–0.293）与原生 clean q
（0.414）两项上**比已被判在适用域之外的 S2Looking 更差**
（0.662–0.884 与 0.240）。若 WHU 只是"又一个空间变化场数据集"，
应当落在 S2Looking 量级而不是更差。存在额外结构性障碍的嫌疑，
而 512 图幅是唯一已知候选。

## 2. 设计：只动一个变量

```text
同一个检查点（LEVIR P7C）
同一个工具（tools.diagnose_transfer_offset_consistency）
同一份协议（configs/p8_matched_control_eval_protocol.yaml，与
  configs/p8_whu_eval_protocol.yaml 逐字段相同）
同一 radius 4、同一 maximum_offset_pixels 8.0
唯一差异：--image-size（不给 = 原生 1024；给 512 = 中心裁剪）
```

数据集在 `augment=False` 时对 `image_size` 执行**中心裁剪**（见
`src/rs_cd/datasets/levir.py` 的 `_joint_transform`），因此不引入
缩放插值，不改变像素尺度，只改变可视范围——正是需要的那一个变量。

只用 val split。不训练。分钟级。

## 3. 主判据与预注册判读

主场景取 **Shift-4**，因为它的 boundary_fraction 最低（WHU 0.057），
受钳制伪命中污染最小；Shift-8（WHU boundary 0.466）照常报告但
**不参与判决**。

```text
令 native = LEVIR@1024 的 S4 conditional PCK@1
   matched = LEVIR@512 的 S4 conditional PCK@1
   retention = matched / native

图幅无罪 image_size_not_the_cause
  matched >= 0.60 且 retention >= 0.80
  → 归因 = 数据集属性，WHU 作为第二个负例成立
图幅有罪 image_size_is_the_dominant_confound
  matched <= 0.30（落入 WHU 的 0.132 量级附近）
  → 归因 = WHU 的结论仅限于 512 图幅配置，
    **不得**声称第二个适用域负例
歧义 ambiguous_partial_confound
  0.30 < matched < 0.60，或 matched >= 0.60 但 retention < 0.80
  → 两个解释都不得单独主张；凡引用 WHU 结论必须附带
    "512 图幅下 LEVIR 自身降到 X"这一限定
```

`retention` 这一项是刻意加的：只看绝对值会让"LEVIR 从 0.95 掉到
0.65"被判成无罪，而那明显说明图幅有实质影响。两个条件同时满足才算
无罪；两个带互斥（重叠即报错）。

## 4. 附带的免费证据（同一脚本，第一部分）

三个数据集的定位器**都用 256 px 训练 crop**，所以训练侧曲线不受
评估图幅混淆。脚本会打印三条 `train_shift_pck1` 曲线：

```text
若 LEVIR/S2Looking 的 train_shift_pck1 显著高于 WHU 的 0.43
  → 训练侧就已分化，图幅（评估侧）不能解释全部差距
若三者相近                      → WHU 的问题主要在评估配置或泛化
```

这是解读用的观察，不是门槛，不进 decision.json。

## 5. 明确的边界

```text
本实验不重开 WHU 阶段 A 的任何门槛：decision.json 里
  gates_reopened 恒为 false，target 的数字原样透传。
本实验不授权扩大 search radius。S4 的 boundary_fraction 只有 0.057
  却仍然只有 0.132，已经排除"域不足"这个解释（主记录 73.2）。
本实验不碰任何 test split。
若判为"图幅有罪"，后续选项（1024 重切 WHU）需要新协议与用户批准，
  不在本协议范围内。
```

## 6. 运行

```bash
bash scripts/run_p8_matched_image_size_control.sh
```

产物 `reports/p8_matched_image_size_control/<UTC>/`：
`insample_curves.txt`、`levir_native_1024/`、`levir_matched_512/`、
`decision.json`。
