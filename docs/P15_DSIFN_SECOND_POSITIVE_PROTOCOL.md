# P15 DSIFN-CD 第二正例数据集：预注册协议（2026-07-27）

> 目的：为论文补一个**正向**数据集，同时争取覆盖「自然错位」这一伤。  
> 授权：用户明确要求补第二正例（超出 §0A 原定的「只允许 BiFA」，
> 属用户批准的范围扩展）。  
> 纪律：本协议**不修改** P12/P13 任何冻结产物，不重读 LEVIR test，
> DSIFN test 在 Gate 2 全部通过前保持封存。

## 0. 为什么是 DSIFN，以及它可能失败

选它的理由：该数据集以配准噪声著称，若本文方法在其上成立，则同时拿下
「第二个正例数据集」与「自然错位」两条证据；若不成立，它会成为第二个
**空间变化场**负例——那也是可发表的结果，但必须如实写成失败，
**不得**事后放宽门槛。

先验：已测三个数据集中只有 LEVIR 域内（S2Looking 真负例、WHU 原生
GSD 域外）。因此本协议把最便宜的判据放在最前面，训练放在最后。

## 1. 官方事实（引自作者仓库 `dataset/README.md`）

```text
来源：Google Earth，六城（北京/成都/深圳/重庆/武汉/西安）
五城大幅影像 → 394 对 512×512 子图；数据增广后 3940 对
西安大幅影像 → 48 对，作为 test
划分：train 3600 / val 340 / test 48
下载：Google Drive 或百度网盘（提取码 ru08）
许可：仅限非商业与科研用途
```

**GSD 官方未公开。** 因此本协议全程以**像素**为口径；若后续能确认
GSD，再按主记录 71.5 的要求补报物理位移量。跨数据集比较 headroom/PCK
时，在 GSD 未知期间**不得**做物理单位换算。

## 2. Gate −1：数据卫生审计（不训练，必须先过）

复用 `tools/audit_dataset.py` + `tools/build_manifest.py`，新增
`tools/audit_augmentation_leakage.py`（二面体规范化哈希 + 感知差异哈希），
执行入口 `scripts/run_p15_dsifn_gate_minus1.sh`：

```text
A1 尺寸/通道/标签二值性/无效样本 = 0
A2 cross-split 完全重复三元组 = 0
A3 增广泄漏：对每个样本做二面体规范化（8 向旋转+翻转取字典序最小的
   字节哈希）后比对 train / val，cross_split_augmented_duplicates = 0
A4 test（西安 48 对）只允许参与 A1–A3 的哈希/尺寸审计，
   禁止任何指标评估
```

路由（预注册，不得事后更改）：

- A3 = 0 → 使用官方 train/val 划分；
- A3 > 0 → **不得**在官方 val 上选 tau。改为按去增广后的源裁块
  （394 个）重新做 80/20 划分，并在论文中明确声明这是
  `dataset-specific split`，同时报告官方划分下的泄漏数量。

这一关的存在理由：本方法的 tau 是在 validation 上以 max 统计量确定的
（§57.2/§83 已两次记录该规则对分布变化敏感），val 若与 train 增广重复，
tau 会被系统性低估。

## 3. Gate 0：无标签适用域筛查（不训练，分钟级）

用**已冻结的 LEVIR 结构（P11 freeze manifest 里的 p7_checkpoint）零适配**
跑 DSIFN，工具复用 `tools/diagnose_registration_hygiene.py`，
路由由 `tools/summarize_p15_gate0.py` 按
`configs/p15_dsifn_screening.yaml` 判定。

**必须同时跑几何匹配的 LEVIR 对照**（val，center-crop 512，tile 256）。
这一条不是可选项：主记录中的旧值来自不同几何或不同统计量，不能与
本协议的 `median_max_pairwise_tile_distance` 混用；本轮几何匹配对照实测
LEVIR@512/256 为 **2.535**。因此 §67 那组绝对数
（LEVIR 0 / S2Looking 4.0–4.5）同样**不能**直接拿来判 512 图幅的 DSIFN；
主记录 75.2 的通则是「异常检测型统计必须先在已知良好对照上运行」。

两个信号：

```text
D      = 注入场景（1/2/3）的 median_max_pairwise_tile_distance 均值
         判据量 = D(DSIFN) − D(LEVIR@matched)，单位像素
clean q = clean 场景的 whole_image_q_trigger_rate（绝对值，512 几何）
         参照：LEVIR@512 0.0625 域内 / WHU@512 0.4138 域外
```

预注册判读（三分支，照抄 WHU 那次挡住错误止损的成功路由）：

| 分支 | 条件 | 动作 |
|---|---|---|
| 绿 | `D_excess <= +1.0` 且 `clean q <= 0.15` | 进 Gate 1 |
| 红 | `D_excess >= +3.0` **且** `clean q >= 0.40` | **止损**，记为第二个空间变化场负例 |
| 黄 | 其余 | 进 Gate 1（歧义一律走决定性判据） |

止损要求**两个信号同时红**：单信号红属于歧义，按 §71 的教训走决定性判据。

两条工具级硬约束（已写进 `summarize_p15_gate0.py`，违反即抛错）：

1. **clean 场景的离散度永远不参与判定**——条件定位器无法输出零 offset，
   clean 上测到的是强制选择噪声底且与真实失配强度反相关（§65）；
   工具遇到 `forced_choice_regime = true` 的场景被当作 D 来源就报错；
2. 候选与对照的 `tile_size` 与每图瓦片数必须一致，否则判 geometry 不匹配
   并以非零码退出。

另须报告 `excess = est_rate − gt_rate` 的边界占比（§75.2 的更正口径），
缺 gt 字段时报「不可判读」，不得单看 `boundary_fraction`。

## 4. Gate 1：决定性判据 headroom（需训练 DSIFN 的 P3b）

Gate 0 实际路由为黄灯：`clean q=0.09599`（绿），但注入场景
`D=8.000`，相对同轮几何匹配 LEVIR 对照 `2.535` 的 excess 为
`+5.465`（红）。这是项目中首次出现「q 好、离散度坏」的组合。
在 Gate 1 训练前必须把每个严重度的
`median_injected_offset_magnitude`、`median_tile_error_magnitude`、
`tile_error_pck2/4` 与离散度输出为**只诊断、不加门槛**的仪器报告，
用于识别搜索边界钳制或描述子域失效。该报告不改变黄灯进入 Gate 1
的预注册路由。

```text
训练：P3b DINOv3 ConvNeXt-Tiny partial fine-tuning，配方沿用 LEVIR，
      唯一允许的偏离是 512 图幅强制的 batch size（同 WHU 的口径）
判据：Shift-8 input-oracle headroom >= 0.05
参照：LEVIR 0.2141 / WHU 0.10880 / S2Looking 0.0330
```

Raw 与 input-oracle 必须使用完全相同的有效像素支撑量：对每个已知注入
位移，统一取正向平移与逆向恢复均有效的对称内部区域。若任一场景的
`valid_pixels` 不同，汇总工具必须拒绝判定，不得直接相减两个 F1。

必须同时记录 §71.5 的两条解读约束：该判据**对固有失配不敏感**，
它既不证实也不否证空间变化场假设；解开歧义靠 Gate 0 的 clean q。

失败 → 止损，写成「DSIFN 在输入级不可恢复」的限制。

即使通过也不得写成“方法已经适用”：Gate 1 只说明已知注入全局位移在
输入级存在可恢复上限。是否能由 DSIFN 原生定位器兑现，仍由 Gate 2
四门槛裁决。

## 5. Gate 2：全链复刻与 P7-3 四门槛

```text
链路：P3b（Gate 1 已得） → 输入级定位器 stage A → q 头全分辨率校准
      → P7-3 safe-alignment classifier（primary = safe_veto_task）
划分：按源裁块 ID 稳定哈希做 fit / early-stop / risk-calibration 三段
      （§9.3 教训：不得固定用最后一轮）
搜索半径固定 4，不得自行改动（禁令 #8/#15 的延伸）
```

四条门槛与 LEVIR/seed-1234 完全相同，**不得放宽**：

| 门槛 | 要求 |
|---|---|
| clean F1 掉幅 | <= 0.001 |
| shift mean gain | >= 0.020 |
| Shift-8 gain | >= 0.050 |
| Shift-8 Oracle recovery | >= 50% |

全过 → 进 Gate 3；任一失败 → 止损，如实写成「方法在 DSIFN 上未通过
预声明门槛」，**不得**调 tau、换 primary 或重跑。

## 6. Gate 3：一次性 test（仅在 Gate 2 全过后）

```text
marker = reports/p15_dsifn_final_test/DSIFN_TEST_CONSUMED.json
allowed_reruns = 0，写在读取 test 之前，状态 started_irreversible → completed
场景 = clean / Shift-2 / Shift-4 / Shift-8，协议与 P12 完全一致
（full image shift → 非重叠瓦片 → 阈值 0.5 → 16 px interior border）
test 结果只报告，不得用于任何选择
```

支撑量必须与 LEVIR 分开报告：DSIFN test 仅 48 对 512×512
（≈12.6M 有效像素，约为 LEVIR test 的 1/10），
论文中**不得**把两个数据集的置信度写成同级。

## 7. 可选加分：DSIFN 上的第二次外部对比

DiffRegCD 作者同时公开了 **DSIFN-CD 权重**（主记录 103 已登记）。
若 Gate 3 完成，可用**已验证的** P13 工具链在 DSIFN 上做同 support 对比：
先 validation clean 复现作者报告值 → 冻结 manifest → 一次性 test。
该分支需要独立预注册，不在本协议的授权范围内。

## 8. 失败即结论

本协议的每一关都允许失败，且失败是**可发表**的：DSIFN 若判为第二个
空间变化场负例，它与 S2Looking 一起把「适用域条件」从单例证据升级为
两例证据，同时使 Gate 0 的无标签筛查成为**可预测部署成败**的贡献。
禁止为了得到正例而修改任何门槛、阈值或 primary。
