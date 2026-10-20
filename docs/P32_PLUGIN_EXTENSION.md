# P32：扩充下游检测器与混合分布评测（结果出来前写定）

依据 `paper/DIRECTION_2026-09-21.md`（导师面谈定案）。本文件在任何 P32 结果产生之前写定，之后不得修改。

---

## 1. 目的

论文定位为"即插即用的前端模块"，两项补充：

1. **扩充下游检测器**，从 4 个增至 6 个，且新增的两个与现有四个在原理上差异明显，用以支撑泛化性主张；
2. **混合分布评测**：在一个测试集上随机混合已配准、小幅错位、大幅错位三种状态，证明在"事先不知道输入是否错位"
   的现实条件下本文方法仍然最优。这正是选择器存在的意义。

---

## 2. 新增检测器

| 检测器 | 原理 | 来源 | 与现有四个的差异 |
|---|---|---|---|
| SNUNet | 密集连接的孪生 UNet，纯卷积 | rschange 已收录（`rscd/models`），走现有接入路径 | 现有四个分别是 Transformer、状态空间、xLSTM、DINOv3 特征，缺纯 CNN 一支 |
| BiFA | 网络内部做双时相特征对齐（TGRS 2024） | 官方仓库 `zmoka-zht/BiFA` | 唯一自带对齐机制的下游；用于检验"网络内隐式对齐"与"前端显式校正"是否互补 |

**BiFA 接入的技术说明**：其模型文件仅在加载预训练与日志处依赖 `mmcv.runner.load_checkpoint` 与
`mmseg.utils.get_root_logger`，不涉及编译算子。接入时提供垫片替换这两个符号，**不安装 mmcv**，
与 `src/rs_cd/models/external_rschange.py` 处理 rschange 的方式一致：只导入 `nn.Module` 定义，
训练循环、优化器、损失、采样、评测边界全部用本项目的。SegFormer 主干使用 ImageNet 预训练的 mit_b0。

### 2.1 训练配方

沿用 P21 协议（`docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md`）不变：

- 数据、划分、裁块、正样本采样与 DINOv3 行完全相同；
- 优化器与调度取各检测器原论文设置，按本项目 40 × 8192 的样本预算保持调度形状（协议修正 A）；
- 损失统一为 BCE + Dice（本项目协议，非原论文损失）；
- 三个数据集（LEVIR-CD、DSIFN-CD、WHU-CD）各训一遍，取干净验证集 F1 最佳的检查点。

### 2.2 接入与评测

沿用 P31 无黑边流程（`docs/P31_BORDER_FREE_REMAINDER.md`）不变：检测器冻结 → 提取选择器训练特征
（约 1 小时 GPU / 数据集）→ 按 P27 配方训练选择器 → 验证集一次 → 测试集一次。定位器沿用冻结的 v2，不重训。

---

## 3. 混合分布评测

### 3.1 定义

- **数据集**：WHU-CD 测试集的无黑边小块（2552 个），是三个数据集中小块最多、统计最稳的一个；
- **状态分配**：每个小块由其编号的确定性哈希分到三种状态之一，各占三分之一——
  **已配准**（第 0 档）、**小幅错位**（第 1 档）、**大幅错位**（第 3 档）；
- **关键**：所有方法在推理时都不知道某一对属于哪种状态，与真实部署一致；
- **指标**：整个混合集上的合并 F1（单一数字），另给三种状态的分解值用于诊断。

### 3.2 参与比较的方法

| 行 | 说明 |
|---|---|
| 不校正 | 检测器直接吃混合输入 |
| 总是校正 | 本文定位器，每一对都校正 |
| 增强训练基线 | 无黑边样本重训的 DINOv3 检测器，不校正（仅 DINOv3 行） |
| **本文方法** | 定位器 + 选择器 |

**DiffRegCD 不进混合集**：它的输入约定与本文相反（扰动前一期），混在同一集合里不可比。
与它的比较单列一组，见 `paper/DIRECTION_2026-09-21.md` 第一节。

### 3.3 计算方式

混合集的结果**从已有的逐块记录直接构造，不需要新的 GPU 运行**：每个小块按其分到的状态，
取该状态下已经记录好的混淆计数（`reports/p30/test/*/records_proposed_*.csv`、
`reports/p31/<row>/test/records_*.csv`、`reports/p31/final/*/border_counts/counts_*.csv`），
再合并成整体 F1。新增的两个检测器待其测试记录产生后同样处理。

### 3.4 判读

不设通过/不通过的关卡，只报告。预期本文方法在混合集上高于"不校正"与"总是校正"两种固定策略；
若不成立，照实写入论文并在讨论中分析。

---

## 4. 纪律

- 新增检测器的测试集仍是**每格一次**，标记按组件摘要，沿用 `tools/run_p31_plugins.py` 的机制；
- 混合集的状态分配种子写死在配置里，分配一次后不得更改；
- 本文件写定后，判读口径与状态比例不再调整。

---

## 5. 分工

| 事项 | 负责 |
|---|---|
| SNUNet、BiFA 的接入代码、配置、测试 | 本人 |
| 两个检测器 × 三个数据集的训练 | 队友 |
| 选择器特征提取、标定、测试集评测 | 本人 |
| 混合分布评测（CPU） | 本人 |

---

## 6. 预计工作量

| 项 | 机时 |
|---|---|
| SNUNet 训练（3 个数据集） | 约 6–9 小时 GPU |
| BiFA 训练（3 个数据集） | 约 9–12 小时 GPU |
| 选择器特征提取（2 个检测器 × 3 个数据集） | 约 6 小时 GPU |
| 验证 + 测试评测 | 约 2 小时 GPU |
| 混合分布评测 | CPU，分钟级 |

---

## 7. 接入实现（2026-09-21 补记，写在任何 P32 结果之前）

协议本身（第 1–4 节的判读口径、状态比例、数据集选择）不变；本节只记实现路径与分工变更。

| 项 | 实现 |
|---|---|
| SNUNet | rschange 的 `UNetpp`（backbones/SNUNet_UNetpp.py）+ `ECAM`（decoderheads/SNUNet_ECAM.py），只在 `external_rschange.KNOWN_MODULES` 加两行；骨干 `forward(xA, xB)`、头吃一个列表，与现有调用约定一致，无需垫片 |
| SNUNet 配方 | rschange `configs/SNUNet.py`：AdamW 1e-3、wd 1e-2、每 8 of 200 epoch 乘 0.5（→ `step_fraction 0.04`）；批量取数据集基础配置的 8，与其余 P21 行一致 |
| BiFA | 新文件 `src/rs_cd/models/external_bifa.py`：接管构造期的 `torch.load` 注入 ImageNet 权重、垫片 `mmcv.runner.load_checkpoint` 与 `mmseg.utils.get_root_logger`、把 forward 从不调用的 `SegFormerHead` 换成会报错的占位、注入后逐张量核对、评测按 256 分块 |
| BiFA 配方 | BiFA `config/levir.json` 与 `config/whu.json`：Adam 1e-4、linear、200 epoch、批量 8。本项目无 `adam` 选项，用 `adamw` + `weight_decay 0`，两者在无衰减时完全等价 |
| ImageNet 初始化 | 不走百度网盘：`tools/p32_convert_mit_b0.py` 把公开的 `nvidia/mit-b0` 改名成原始 SegFormer 命名（key 与 value 融成 `kv`），并与检出里真实的 `mit_b0` 逐键校验 |
| 运行入口 | `scripts/run_p32.sh`（预检前向 → 六格训练 → 两行评测）、`configs/p32_plugin_extension.yaml`；P31 的冻结配置不动，结果落在 `reports/p32` |

**分工变更**：第 5 节写的"训练由队友负责"作废。服务器改为我们自己的两台实例，队友出机时，全部步骤由本人执行。

**协议修正 D（2026-09-21，SNUNet 三格，在其任何结果产生之前写定）**：SNUNet 的三个配置用 `amp: false`，
与其余所有格子不同。两条理由：(1) rschange 训练 SNUNet 本来就是 fp32——他们 `train.py` 里的
`precision='16-mixed'` 是注释掉的；(2) 开 amp 时第一个 epoch 就溢出，污染了 BatchNorm 的滑动统计量，
训练与验证从此全是 NaN（val_f1 恒为 0）。关掉 amp 后同一配方两个 epoch 即 loss 0.96 → 0.65、
验证 F1 0.37 → 0.72。优化器、调度、样本预算、损失一律不变，只改数值精度。
