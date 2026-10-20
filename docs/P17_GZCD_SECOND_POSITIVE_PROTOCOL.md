# P17 GZ-CD 第二正例数据集：预注册协议（2026-07-29）

> 目的：为论文补一个**真正独立**的正例数据集，解决「端到端只有 LEVIR
> 一个正例」这条最大的审稿伤。  
> 授权：用户批准（记录 118.5 之后的裁决）。  
> 纪律：不修改 P12/P13 任何冻结产物，不重读 LEVIR test，GZ-CD test 在
> Gate 2 全部通过前保持封存。任一关失败即止损，不放宽门槛、不做结构重试。

---

## 0. 为什么是 GZ-CD，以及它可能失败

选它的理由：它是候选中**独立性最强**的一个——不同国家、不同影像提供
路径、不同时间跨度，与 LEVIR（美国 Texas，Google Earth）没有同源关系。
记录 116 已排除 LEVIR-CD+（985 对与 637 对同源、同 20 个地区，无法支撑
跨数据集泛化），WHU 粗化是构造配置（只能作为适用域条件的端到端验证，
不能算第二正例）。

先验风险，写在前面：

1. **时相跨度极长**（2–13 年，2005–2019）且季节光照差异大，检测器可能
   因外观变化而困难，这是任务难度风险，不是几何风险；
2. **变化占比 9.0%**（LEVIR 4.2%），若目标偏大则 Condition B 存疑——
   但按记录 118，标签形态**只能排序不能当门槛**，故仍以 headroom 裁决；
3. **只有 20 个栅格**，按栅格划分后方差偏大；
4. test 支撑仅 28.6 M 有效像素，约为 LEVIR test 的 21%，置信度不可同级
   表述（与 DSIFN 的 48 对同类问题）。

## 1. 数据来源与已核实事实

```text
来源：Peng et al., "SemiCDNet", IEEE TGRS 2020, DOI 10.1109/TGRS.2020.3011913
仓库：github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery
分发：百度网盘（README 提取码 8quw）；本地包 CD_Data_GZ.zip（799 MB）
结构：CD_Data_GZ/{T1,T2}/*.tif（20 对大幅影像）+ labels_change/*.png + train_list.txt
标签：单通道，取值仅 {0, 255}，mask_threshold = 127
尺寸：1006x1168 至 5184x6736，总计约 2.33 亿像素
变化占比：单幅 5.44%–19.09%，整体 9.0%
时相：2005–2019，命名含日期，例如 P_GZ_test1_2013_1221 与 2018_0119
```

**两项未核实、必须在论文中标注为推断而非官方值**：

1. **GSD 由文件名 `Level_18`（Google Earth 缩放层级）推断约 0.55 m**，
   官方 README 未给出。因此全程以**像素**为口径报告，跨数据集比较不做
   物理单位换算（记录 71.5 的约束）；
2. 文献常引「19 对，1006x1168 至 4936x5224」，而本包为 **20 对**、最大
   5184x6736，疑为版本差异。以本地包实测为准并如实说明。

**没有官方 train/val/test 划分**，仅有一个列出全部 20 对的 `train_list.txt`。
因此划分由本协议定义，并声明为 dataset-specific split。

## 2. 冻结的构建参数（跑之前定死）

```text
瓦片尺寸       = 512 x 512，非重叠，floor 切分，不足一格的边缘丢弃
每瓦片覆盖     ≈ 282 m（按推断 GSD 0.55 m），高于剂量-响应阈值
                 （154 m 时 PCK@1 0.933，230 m 时 1.000）
几何瓦片总数   = 729（仅按尺寸计算，未看像素内容）
mask_threshold = 127
sample_id      = <raster>_<row>_<col>，全局唯一
parent_id      = 栅格名（如 test12）
无数据过滤     = 若 T1 或 T2 中纯黑像素占比 >= 0.50，丢弃该瓦片
```

**关于无数据过滤（在看到任何像素内容之前预注册）**：Google Earth 拼接
影像常带纯黑填充边。本协议第 3 节表中的 511/109/109 是**纯几何上限**，
构建工具会报告实际保留数与丢弃数，两者都记入报告。设定 0.50 而不是更
严的阈值，是为了只剔除以填充为主的瓦片，保留边缘处仍有大量有效地物的
瓦片；该阈值不因结果调整。

**为什么不用 1024 瓦片**：1024 覆盖 563 m、与 LEVIR 几何最接近，但全库
只能切出 155 个瓦片，训练样本不足（LEVIR 训练集有 445 对）。512 在覆盖
仍远超阈值的前提下给出 729 个瓦片，且与 WHU 的既有口径一致。代价是几何
与 LEVIR@1024 不同，故 Gate 0 必须使用几何匹配对照（见第 4 节）。

## 3. Gate −1：按栅格划分与数据卫生

**划分规则（已按栅格瓦片数确定性生成，此处冻结，不得事后调整）**：

| split | 栅格 | 几何瓦片上限 | 占比 | 平均变化占比 |
|---|---|---:|---:|---:|
| train | test1, test12, test18, test2, test27, test28, test4, test5, test6 | 511 | 70.1% | 0.091 |
| val | test13, test15, test17, test21, test22 | 109 | 15.0% | 0.087 |
| test | test10, test19, test23, test24, test26, test9 | 109 | 15.0% | 0.091 |

瓦片数为几何上限；无数据过滤后的实际数量由构建工具报告，两者都留档。
栅格到 split 的映射写死在 `tools/build_gzcd_tiles.py` 的
`SPLIT_ASSIGNMENT` 中，与本表一致，运行时若出现未登记的栅格即报错退出。

**同一栅格的全部瓦片必属同一 split**，因此空间不相交由构建方式保证，
而不是靠事后审计——这是 WHU 的做法（记录 21），也是 DSIFN 泄漏事件
（记录 110.2）之后的强制要求。

审计项（沿用既有工具）：

```text
A1 尺寸/通道/标签二值性/无效样本 = 0        tools/audit_dataset.py
A2 cross-split 完全重复三元组 = 0
A3 增广泄漏（二面体哈希 + 感知哈希）        tools/audit_augmentation_leakage.py
   预期为 0：本数据集无增广，若非 0 说明栅格间存在重叠区域，必须查
A4 每个 parent_id 只出现在一个 split         构建后硬校验
```

同时运行 `tools/audit_change_target_scale.py`（标签形态），**仅作排序与
解释用途，不构成门槛**（记录 118 已证明标签代理无法复现门槛判定）。

## 4. Gate 0：无标签几何筛查

用 P11 冻结结构中的 `p7_checkpoint` 零适配跑 GZ-CD val，并在**同一次运行**
中产出几何匹配的 LEVIR 对照（center-crop 512，tile 256）。判据沿用
`configs/p15_dsifn_screening.yaml` 的三分支：

| 分支 | 条件 | 动作 |
|---|---|---|
| 绿 | `D_excess <= +1.0` 且 `clean q <= 0.15` | 进 Gate 1 |
| 红 | `D_excess >= +3.0` **且** `clean q >= 0.40` | 止损 |
| 黄 | 其余 | 进 Gate 1 |

止损要求两信号同时红；单信号红属歧义，走决定性判据（记录 71 的路由）。

## 5. Gate 1：决定性判据 headroom

```text
训练：P3b DINOv3 ConvNeXt-Tiny partial fine-tuning，配方沿用 LEVIR/WHU，
      唯一允许的偏离是 512 图幅强制的 batch size；不为本数据集搜任何超参
判据：Shift-8 input-oracle headroom >= 0.05
参照：LEVIR 0.2141 / WHU 0.10880 / S2Looking 0.0330 / DSIFN 0.0227
支撑：raw 与 oracle 必须在对称交集有效像素上评分（记录 114 的教训）
```

失败 → 止损，写成「GZ-CD 在输入级可恢复空间不足」的限制。

## 6. Gate 2：全链复刻与 P7-3 四门槛

```text
链路：P3b → 输入级定位器 stage A → q 头全分辨率校准 → P7-3 分类器
primary = safe_veto_task（与 LEVIR 一致，不得更换）
搜索半径固定 4；按 parent_id 做 fit / early-stop / risk-calibration 三段划分
```

四条门槛与 LEVIR / seed-1234 完全相同，**不得放宽**：

| 门槛 | 要求 |
|---|---|
| clean F1 掉幅 | <= 0.001 |
| shift mean gain | >= 0.020 |
| Shift-8 gain | >= 0.050 |
| Shift-8 Oracle recovery | >= 50% |

## 7. Gate 3：一次性 test

```text
marker = reports/p17_gzcd_final_test/GZCD_TEST_CONSUMED.json
allowed_reruns = 0，读 test 前写入，机制同 P12
场景与口径与 P12 一致：full-image shift → 非重叠瓦片 → 阈值 0.5 → 16 px border
test 结果只报告，不用于任何选择
```

**支撑量必须与 LEVIR 分开报告**：GZ-CD test 为 109 个 512² 瓦片
（28.6 M 有效像素，约 LEVIR test 的 21%），置信度不可写成同级。

## 8. 失败即结论

任一关失败都是允许的结局。GZ-CD 若失败，它会成为第四个边界案例；
但按记录 118 的经验，边界案例的边际价值已经不高，因此**失败即停止寻找
第二正例**，转入论文写作，不再启动新候选。

禁止：为通过门槛而调 tau、换 primary、改搜索半径、改划分或重跑。
