# P21 跨检测器必要性验证：预注册协议（2026-08-15）

> 动机：导师（于晓升，2026-08-15）明确了论文的成立条件——「你的步骤
> 加进去之后，是不是比没这个步骤的原方法好；在多个方法上都好用，
> 才能证明有意义」。即证明选择性配准层是**检测器无关**的必要步骤，
> 而不是绑定在本文检测器上的工程改进。
> 授权：用户与导师在对话中达成方案共识（BIT / ChangeFormer /
> ChangeMamba 三个检测器，各自作为共享检测器跑 raw/aligned 双路径，
> 分歧特征重算、路由器重训、阈值重标定），导师回复「可以」。
> 纪律：P12/P13 冻结产物一字不动；LEVIR test 的既有消费记录不受影响；
> 本协议全部开发与主报告都在 validation 上，test 消费见第 7 节。

---

## 0. 主张与可证伪性

要证明的主张（导师版本）：

```text
对每一个代表性变化检测方法 D_i：
  (a) D_i 在错位输入下显著掉分（问题是真的）；
  (b) 接入选择性配准层后，错位场景显著恢复（步骤有效）；
  (c) clean 场景不受损伤（步骤无害）；
  (d) 以上不依赖 D_i 的架构（步骤必要且通用）。
```

预注册的失败处理：**任一检测器不满足即如实报告为该主张的边界**，
不得从检测器清单中静默移除；清单在跑前冻结于本文件。

## 1. 检测器清单（跑前冻结）

| 顺位 | 检测器 | 依据 | 风险 |
|---|---|---|---|
| 1 | BIT (chen2022bit) | 代码/权重公开，架构轻，社区标准基线 | 低 |
| 2 | ChangeFormer (bandara2022changeformer) | 代码/权重公开，transformer 代表 | 低 |
| 3 | ChangeMamba (chen2024changemamba) | 状态空间模型代表，架构差异最大 | 中：其公开二值结果基于 LEVIR-CD+，须在 LEVIR-CD 上重训；依赖 mamba-ssm/causal-conv1d |
| 备 | SNUNet | 若 ChangeMamba 环境不可行则顶替，并如实记录顶替原因 | 低 |

已有的第四个数据点：本文检测器（DINOv3 ConvNeXt-Tiny，P12 冻结结果）。
FC-Siam-Diff 的 P4 位移敏感性数据（记录 §17）作为背景证据引用，
不重跑。

## 2. 每个检测器的固定管线

```text
1. 架构移植：从官方仓库导入模型类（固定 commit，记录 SHA），
   包装为「(I1, I2) → change_logits」接口接入本项目训练框架；
   不改动网络结构。
2. 训练：本项目 LEVIR-CD 划分与协议（1024 瓦片、16 px 内边界），
   使用官方发表的优化超参；唯一允许的偏离是显存强制的 batch size。
   **不为任何检测器做超参搜索**（沿 P17「不为数据集搜超参」纪律）。
3. raw 敏感性：val 四场景（clean / S2 / S4 / S8）raw F1 → 判据 (a)。
4. input-oracle headroom：完美逆位移，paired support → 各检测器
   自己的可恢复上限（分母）。
5. aligned 路径：冻结的 P7C 定位器（不重训、不换半径），同一 warp。
6. 路由器：23 特征重算 → 重训 task router 与 registration router
   （两个都训，延续不可区分性的诚实报告）→ 按既有规则重标定 τ
   （clean 校准分区最高分 + ε）。
7. val 报告：clean drop、S2/S4/S8 增益、mean 增益、
   相对自身 oracle 的捕获率、AUROC。
```

## 3. 预注册门槛（每个检测器独立判定）

| 判据 | 门槛 | 性质 |
|---|---|---|
| (a) 掉分显著 | raw S8 相对 clean 掉幅 >= 0.05 | 前提检查 |
| (c) clean 无损 | 路由后 clean F1 drop <= 0.001 | **硬门槛** |
| (b) 恢复显著 | mean shift gain >= 0.02 | **硬门槛** |
| (b') 捕获质量 | mean capture >= 40% 自身 oracle headroom | 报告量，非门槛 |

(b) 沿用 LEVIR 尺度的绝对门槛是**有意的**：P21 全部在同一数据集、
同一注入协议上，损失尺度可比，第 120.6 节的「绝对门槛跨尺度失义」
问题不适用。捕获率仍按第 122 节口径作为描述量报告。

判据 (a) 若不满足（某检测器天生鲁棒），该检测器记为
「无需本步骤」——这**不是**方法失败，是适用域信息，如实报告。

## 4. 与既有纪律的关系

```text
P12/P13：不动。本文检测器的 test 数字继续引用冻结结果。
定位器：P7C 检查点冻结复用；若任何检测器需要改动定位器，
  即为协议违规，停下来重新预注册。
路由器阈值规则：不改（与 LEVIR 主线同一规则），
  使跨检测器结果可比。
```

## 5. 顺序与止损

```text
BIT 先行，全管线打通并出 val 报告后，再并行推进其余两个。
BIT 若在管线调通后判据 (b)/(c) 失败：暂停，先诊断是移植错误
  还是真实失败（守卫：raw clean F1 须落在其公开报告值的合理
  邻域内，偏差 > 3 pp 即先查移植）。
ChangeMamba 环境 2 天内装不通 → 启用 SNUNet 顶替并记录。
```

## 6. 每个检测器的仪器守卫（第 85 节纪律）

```text
训练完成后，raw clean F1 必须复现该方法公开报告的 LEVIR 量级
（考虑 256-crop 与 1024-tile 协议差异，容差 3 pp，与文献定位表
的转录口径一致对待）；不满足先查移植，不读下游数字。
```

## 7. test 消费（预注册，暂缓执行）

```text
三个检测器全部完成 val 报告并冻结后，才允许讨论 test：
  每检测器一次性 marker，allowed_reruns = 0，
  只跑「raw vs routed」两列，不做任何选择；
LEVIR test 图像在 P12/P13 已被读取过，新检测器的消费不影响
  已报告数字的有效性，但须在论文中写明消费历史；
在导师确认投稿版式前，test 保持未消费也是可接受的终态
  （主表用 val，注明）。
```

## 8. 论文重定位（结果出来之前不动笔）

导师给的包装方向：「变化检测新范式——前提是证明变化检测真的需要
这一步」。若三个检测器全部通过：

```text
主张升级为：选择性配准是检测器无关的必要组件；
现有 12 页稿的适用域协议与负结果部分保留为「何时需要这一步」；
新增跨检测器表成为主表；
在此之前不改论文正文——先有数字，后有叙事。
```

## 9. 修订 A（2026-09-14，任何格运行之前）

> 为什么需要：2026-08-15 起实际执行的是 3 检测器 × 3 数据集矩阵，但清单
> 只写进了接力文件与 ONBOARDING，本文件第 1 节仍是旧清单。按第 1 节
> 「清单跑前冻结于本文件」，本修订是开跑的前提；`scripts/run_p21_cell.sh`
> 检查本节存在，否则拒绝运行。写本节时服务器上尚无任何 P21 格的数字。
> 授权：导师 08-15 追加要求「最新方法 + 多个数据集」；以下各项选择由
> 用户于 2026-09-14 逐项确认。

### 9.1 检测器清单（替换第 1 节表）

| 行 | 检测器 | 模型与超参来源 | LEVIR-CD 仪器守卫 |
|---|---|---|---|
| 1 | BIT (TGRS 2021) | 模型类取自 rschange；超参取原作者仓库 BIT_CD | 原文 F1 89.31，±3 pp |
| 2 | CD-Lamba (JSTARS 2025) | rschange（即作者仓库） | 原文未报 LEVIR-CD，见 9.5 |
| 3 | CDxLSTM | rschange（即作者仓库） | 原文 F1 90.89，±3 pp |
| — | DINOv3 ConvNeXt-T（本文） | 冻结结果 | — |

```text
移出 ChangeFormer / ChangeMamba：导师要求改用最新方法（08-15）；
移出 CDMask：其头输出逐图 min-max 归一化的概率图（非 logit），且只能用
  Mask2Former 匈牙利匹配损失训练——纳入统一协议等于改写该方法，
  逐图归一化还会破坏 raw/aligned 两路 logit 的可比性（包装器现显式拒绝）；
备胎：STNet（rschange 内，普通 2 类输出）。顶替须另写修订，不得静默。
```

### 9.2 数据集（扩展第 2 节「LEVIR-CD」）

```text
LEVIR-CD   manifests/datasets/levir_cd.csv
DSIFN-CD   manifests/datasets/dsifn_cd_source_split.csv（无泄漏 source-crop 划分）
GZ-CD      manifests/datasets/gzcd.csv（source-crop 划分）
每个检测器在某数据集上的数据协议（划分、256 裁块、正样本裁块采样、
  增广、掩码阈值）与该数据集 DINOv3 行逐项一致
  （p3b_dinov3_convnext_tiny_partial_ft / p15_dsifn_p3b / p17_gzcd_p3b 的 data 块）。
```

第 3 节 (b) 绝对门槛的理由原写「同一数据集、损失尺度可比」。扩到三个
数据集后，这条理由对 DSIFN / GZ-CD 列**不再成立**（第 120.6 节的跨尺度
问题适用）。门槛数值不改——跑前既不放宽也不收紧——但判读这两列时必须
同时报告 capture 与自身 input-oracle headroom，不得只读门槛结论。

### 9.3 训练配方（细化第 2 节第 2 步）

```text
取官方：优化器、学习率、动量、权重衰减、batch、学习率调度的形状、无梯度裁剪
  BIT      SGD 0.01 / 0.9 / 5e-4，linear：lr·(1 − e/(E+1))，batch 8
  CD-Lamba SGD 0.01 / 0.9 / 5e-4，poly 0.9：lr·(1 − e/E)^0.9，batch 8
  CDxLSTM  SGD 0.05 / 0.9 / 5e-5，step：每 E/4 个 epoch ×0.1（官方 50/200），batch 8
取本项目：总预算 40 epoch × 8192 样本（调度按 epoch 比例缩放，两边都是
  每 epoch 步进一次）；损失统一 BCE + Dice（pos_weight 2），不用各自官方损失；AMP
不做任何超参搜索。
```

守卫失败、且冒烟与锚点均通过时，**唯一预注册的补救**：把该检测器训练
样本总量提到官方量级（BIT：200 epoch × 8144 块 ≈ 1.63M，约为当前 5 倍），
只做一次，其余不变，原结果保留在记录中。

### 9.4 定位器（替换第 4 节「P7C 冻结复用」）

```text
每数据集一个冻结定位器，数据集内所有检测器共享：
  LEVIR-CD、DSIFN-CD → P7C（LEVIR 训练；DSIFN 从未训练过适配定位器）
  GZ-CD              → P17 Gate 2 在 GZ-CD 上校准的检查点（与 DINOv3 行同一个；
                       脚本拒绝带 VOID.json 或缺暖启动记录的报告目录）
定位器配置从其检查点原样复制，只加 P12 路由工具施加的同样四项推理覆盖
  （tools/build_p21_dualpath_checkpoint.py；测试核对与路由工具源码逐字一致）。
```

理由：列内只让检测器变化，成败才能归因到检测器——这正是 P21 要检验的
「收益不足部分是 DINOv3 检测器的性质」。分类器逐样本记录的
`conditional_epe_pixels` 作为每列定位器的诊断：定位误差大且三行都失败，
瓶颈在定位器；误差小而仍失败，瓶颈在检测器或路由。

### 9.5 仪器守卫（细化第 6 节）

```text
LEVIR-CD 格：BIT、CDxLSTM 对照 9.1 的原文值（双侧 ±3 pp）；
  CD-Lamba 原文只报 WHU/SYSU/DSIFN/CLCD，检索未找到第三方 LEVIR-CD 数值
  → 守卫为「F1 ≥ 同协议 BIT LEVIR-CD 实测值 − 3 pp」（单侧），
    只能抓明显的移植错误，论文与记录中须写明这是较弱的守卫；
DSIFN-CD / GZ-CD 格：继承同检测器 LEVIR-CD 格的守卫结果
  （公开的 DSIFN/GZ-CD 数值用其他划分，不可比）；
训练前另加两条无标签守卫（tools/check_p21_detectors.py），任一不过不得训练：
  冒烟：本项目环境内 import、三种瓦片尺寸前向、反向、推理可重复；
  锚点：adapter 包住 P7 自带检测器时，registration 与 raw/aligned logit
        与 P7.forward_with_forced_offset 逐位相同；外部检测器零偏移 raw ≡ aligned。
```

### 9.6 路由与分类器协议

LEVIR-CD、DSIFN-CD 用 `configs/p7_task_benefit_router.yaml` 与
`configs/p7_safe_alignment_classifier.yaml`；GZ-CD 用 p17 版本（逐字复制
自 p7）。种子 3407，每格一次。汇总读分类器写出的
`validation_scenario_metrics.csv`。

### 9.7 DINOv3 行的实际状态（更正接力文件与 ONBOARDING 的「三格已完成」）

```text
LEVIR-CD  完整（P12 冻结）
GZ-CD     Gate 2 完成（capture 12.9%，场景内 AUROC≈0.5）
DSIFN-CD  停在 Gate 1（headroom 0.0227），没有路由结果；
          用户 2026-09-14 决定暂不补跑，矩阵中该格记为「Gate 1 止损」并注明口径不同
```

### 9.8 不变项

第 3 节四条门槛数值、第 5 节顺序（BIT × LEVIR-CD 先行）、第 7 节 test
暂缓、第 8 节「结果出来前论文不动笔」均不变。

## 10. 修订 B（2026-09-15，CD-Lamba 任何格运行之前）

> 为什么需要：队友 2026-09-15 首次在服务器上对 CD-Lamba 做训练前冒烟（第 9.5 节），
> 「推理可重复」与「零位移 raw ≡ aligned」两项未过：同输入两次前向最大差 0.0275，
> 零位移最大差 0.0131，关闭 AMP 仍不过。写本节时六格均未训练，没有读取任何 CD-Lamba 指标。
> 授权：用户于 2026-09-15 在三个方案中选定本节方案；另两个方案是换用备胎 STNet、推理时去掉噪声。

### 10.1 原因（对照 rschange@a3bc6ff 源码）

```text
cdloma.py 中阶段 0–1 的 SS2D（cdlamba 的 css.0–css.9）在 forward_core 里用 my_gumbel_softmax
（第 29–40 行定义，第 549 行调用）挑选局部扫描窗口：每次前向都以 torch.rand_like 生成 Gumbel 噪声，
在 4×4 共 16 个窗口中取 top-6，不区分训练与推理。
→ CD-Lamba 的官方推理本身是随机的：同一输入每次输出不同；双路径的 raw 与 aligned 是两次独立调用
  （dual_path_adapter.py），各抽一次噪声，零位移时也不相等。
其余模块推理时不消耗随机数（SeaFormer 的 DropPath 只在训练时生效，各处 Dropout 概率为 0）。
不是 mamba 扫描核、AMP 或本项目接线的问题。
```

### 10.2 变更

```text
rschange 检测器在 eval 模式下的每次前向，都在独立随机流中运行：
  以 INFERENCE_SEED = 3407 播种 CPU 与已初始化的全部 CUDA 设备，前向结束后恢复全局随机状态
  （torch.random.fork_rng；src/rs_cd/models/external_rschange.py，模块文档第 7 条）。
train 模式前向不变，保留官方的随机性。
不改 rschange 源码、网络结构、权重、配置与门槛；第 9.5 节冒烟守卫不放宽。
含义：每次评估前向是官方随机推理在固定种子下的一次抽样；raw 与 aligned 共用同一份噪声，
  两路差异只来自输入。
种子在读取任何 CD-Lamba 指标之前写定，不得更换种子重跑。
```

### 10.3 影响范围与口径

```text
BIT、CDxLSTM：评估前向不消耗随机数，数值与修订前逐位相同，全局随机状态也不变
  （tests/test_external_rschange.py 核对）。BIT 行结果不受影响；CDxLSTM 行按修订前代码运行，
  队友在该行三格结束后再同步本修订，避免一行之内换代码。
CD-Lamba：噪声取决于批内位置，同一工具内逐位可重复；批组成不同的两个工具之间，同一样本可能
  抽到不同窗口，两工具的 raw F1 可能有小差异，照报（BIT 行两工具之差 ≤ 0.0008）。
无梯度参数属官方结构，只记录、不处理（不冻结、不删除）：
  SeaFormer_L 第 5 阶段与 ImageNet 分类头 119 个张量（两检测器共有）；
  CD-Lamba 窗口打分器 score_predictor 100 个（输出只进入 top-k 下标，官方训练中同样得不到梯度）；
  CDxLSTM 中定义了但不调用的 ViLLayer.norm 12 个。
冒烟时无梯度张量数须为 CDxLSTM 131、CD-Lamba 219，数目不同即停（任务书 5.1）。
```

## 11. 修订 C（2026-09-15，CDxLSTM 行重跑之前）

> 为什么需要：CDxLSTM × LEVIR-CD 首次运行（队友服务器，整图推理）未通过仪器守卫：阶段一最佳验证 F1 0.6902，
> 原文 0.9089 ± 0.03。队列按协议自动停止，DSIFN、GZ-CD 未启动。本修订写于该行重跑之前；首次运行与诊断全部保留。
> 授权：用户于 2026-09-15 在三个方案中选定本节方案；另两个方案是换用备胎 STNet、如实报告不适用且不重跑。

### 11.1 诊断（只读，未训练，验证集）

```text
训练日志：训练 loss 0.64 → 0.22 平滑下降、无 nan；验证 F1 第 5 轮 0.6902 最高，之后降到 0.42–0.49，验证 loss 上升，
  学习率降到 5e-5 仍不回升 → 不是训练不稳定，也不是训练量不足（预注册补救不针对此原因）
同一组权重只改推理输入尺寸（整图 1024 / 512 小块 / 256 小块）：
  第 5 轮  0.6902 / 0.8420 / 0.8103（整图一项与阶段一记录逐位一致）
  第 40 轮 0.4748 / 0.7887 / 0.8864
源码中与绝对尺寸相关的结构（rscd/models/decoderheads/xformer.py）：LHBlock 以 AdaptiveMaxPool2d(12×12) 生成交叉注意力的键和值，
  xLSTM 序列长度随输入增长；CDxLSTM 原文训练和测试都用 256 裁块
结论：移植正确；失败来自输入尺寸不匹配。块尺寸取 256 的原则（原文输入尺寸，也是本项目训练裁块）写于看到第 40 轮结果之前
  （主记录 §169.1），不按分数挑选
```

### 11.2 变更

```text
只对 CDxLSTM：配置增加 model.inference_tile_size: 256。检测器在 eval 模式下的每次前向，都把输入切成不重叠的
  256×256 小块（右侧与下侧边缘用较小的块），逐块推理后按位置拼回 logits（src/rs_cd/models/external_rschange.py，
  模块文档第 8 条）。涵盖阶段一验证与最佳轮选择、raw/oracle 位移评估、路由器与分类器的双路径特征。
不变：训练方式（256 裁块、优化器与调度、40 轮 × 8192）；定位器、位移注入与 warp 仍在整图上进行；
  raw 与 aligned 两条路径使用相同切块；门槛；守卫仍为原文 0.9089 ± 0.03；BIT、CD-Lamba 仍为整图推理（均已通过守卫）。
执行：CDxLSTM 三格从头重跑（首次运行的最佳轮按整图验证选出，且只保存了第 5 与第 40 轮权重，无法重新选轮）；
  新运行写入新的报告目录，由新的守卫判定；只做一次，不再改块尺寸或其他设置。
```

### 11.3 口径

```text
论文中写明：CDxLSTM 以其原生输入尺寸（256 小块）推理，其余检测器为整图推理；首次整图运行未过守卫及诊断数值如实报告。
逐检测器的「加不加这一步」比较在同一推理方式内进行，不受影响；不同检测器之间的绝对 F1 不作直接比较。
```

## 12. 修订 D（2026-09-15，P21 任何格读取 test 之前）

> 为什么需要：第 7 节只允许「raw vs 路由后」两列，且写于 P21 只有 LEVIR-CD 一个数据集时；修订 A 扩到三个数据集后，
> 没有写 test 的范围。论文要比较「选择式 vs 全矫正」，这一列若要有 test 数字，必须在读取 test 之前写定。
> 写本节时，P21 没有任何格读取过 test；CD-Lamba 行的 DSIFN-CD、GZ-CD 两格尚未交回结果，CDxLSTM 行按修订 C 重跑的结果尚未核对。
> 授权：用户于 2026-09-15 决定加入全矫正列、只跑 LEVIR-CD。

### 12.1 前提：先冻结验证集

```text
第 7 节的前提不变：九格验证集报告全部完成后，才允许读取 test。「冻结」落成一个文件：
  python -m tools.freeze_p21_validation 逐格核对 LATEST、cell_summary.json、validation_scenario_metrics.csv 与该格
  SHA256SUMS.txt 记录的哈希；LEVIR-CD 三格另核 dual_path.pth，并补记该格未记录的两个安全分类器文件的哈希与 RSCHANGE_COMMIT；
  写出 reports/p21/VALIDATION_FROZEN.json，只写一次，不得覆盖。
另加两项检查与一项记录（因 CDxLSTM × GZ-CD 首次运行全 0 却被判 no_step_needed，主记录 §180）：
  拒绝 raw clean F1 为 0 或非有限的格——这种运行数值无效，不得作为裁决冻结；拒绝仪器守卫未通过的格；
  记录每格运行时哈希过的阶段一配置（configs/p21/*.yaml），使偏离共用配方的格在清单中可见（主记录 §182）。
任何一格缺报告或哈希不符，冻结工具拒绝写出。某格若未能完成（例如守卫失败），不在本修订范围内，须另写修订。
CDxLSTM 各格在队友服务器运行：先把三格报告目录与 LEVIR-CD 格的 dual_path.pth、两个分类器文件复制到用户服务器，再冻结；
  复制后的哈希由冻结工具对照该格 SHA256SUMS.txt 核对。
```

### 12.2 范围

```text
数据集：只有 LEVIR-CD test（manifests/datasets/levir_cd.csv 的 test 划分）；DSIFN-CD、GZ-CD 的 test 继续封存。
检测器：BIT、CD-Lamba、CDxLSTM 各一次；DINOv3 行不重跑，引用 P12 冻结的一次性 test。
一次性：每检测器一个标记 reports/p21_final_test/<slug>_LEVIR_TEST_CONSUMED.json，allowed_reruns = 0。
  标记在模型与分类器加载成功之后、读取第一张 test 图像之前写入，写入后不删除；已有任一标记时运行脚本拒绝启动。
```

### 12.3 输入

```text
全部取自冻结清单，运行前逐个重算 sha256，不一致即停：
  该检测器 LEVIR-CD 格的 dual_path.pth（P7C 冻结定位器 + 阶段一验证集最佳轮检测器）；
  safe_alignment_registration.pth 与 safe_alignment_task.pth（含验证集风险校准得到的阈值）。
rschange 检出须与该格记录的 commit 相同，且已跟踪文件无改动。
推理方式与各自验证格相同：CD-Lamba 固定推理随机流（修订 B），CDxLSTM 256 小块推理（修订 C），BIT 整图。
```

### 12.4 评估

```text
评估函数：tools/evaluate_safe_alignment_transfer.evaluate_transfer_validation（P12 test 所用）。它与验证格所用的
  tools/train_evaluate_safe_alignment_classifier._evaluate_validation 在同一合成输入上输出逐项相同
  （tests/test_p21_final_test.py 核对）。
设置与验证协议 configs/p7_safe_alignment_classifier.yaml 逐项相同：种子 3407、batch 4、AMP、阈值 0.5、边缘 16 像素不计分、
  最大偏移 8 像素、平移 2/4/8 像素（configs/p21_levir_final_test.yaml，测试逐键核对）。
```

### 12.5 报告内容（读取 test 之前写定）

```text
每检测器：
  raw、always_aligned、safe_veto_task（主路由）三列在 clean 与 2/4/8 像素下的 F1、三档平均，以及相对 raw 的差；
  主路由在各场景的采纳率；
  描述量：对 loss_oracle 的捕获率（逐场景与三档平均）；always_aligned 与主路由之间的 break-even 干净比例
    （与 tools/summarize_p21_matrix.py 同一公式）。
评估函数同时算出的 q_gate、safe_veto_registration 两列照样写入原始结果文件，不进入汇总，不用于任何选择。
```

### 12.6 口径

```text
只报告，不判门槛：各格裁决仍以验证集为准；test 结果无论方向如何都写进论文。
论文写明：LEVIR-CD test 此前已由 P12（本文检测器）与 P13（DiffRegCD）各读取一次，本次为三个外部检测器各一次。
运行中途因与模型无关的原因中断（进程被杀、实例关机、磁盘满）时，保留目录、日志与标记，由用户记录后另写修订决定；
  脚本不提供重跑。
```
