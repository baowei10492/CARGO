# P14 BiFA 第二外部基线：复现可行性核查（只读，2026-07-27）

> 依据接力文件 §0A 第 2/3 条：只允许按新协议接入 BiFA；若无可信公开
> 权重 / clean 复现条件，记录障碍后立即止损。  
> 本次核查**只读公开仓库元数据与源码文本**，未下载权重、未建环境、
> 未消费任何 test，也未改动任何冻结产物。

## 1. 核查到的事实

| 项 | 观测 | 来源 |
|---|---|---|
| 仓库 | `zmoka-zht/BiFA`，public，Apache-2.0 | GitHub API |
| 最近提交 | `4b439a449f27d51555273d4d5b8ac36b5befa40d`，2026-05-27（"fix bug"） | GitHub API |
| release / tag | 均为 0（只能按 commit 固定） | GitHub API |
| 论文 | TGRS 2024，DOI 10.1109/TGRS.2024.3376673；仓库内 `resources/BiFA.pdf`（14,266,351 B） | README / raw HEAD |
| 训练权重 | **仓库内直接提供**：`experiments/pretrain/LEVIR.pth`（62,804,667 B，git blob `eaa4c6a2f8be2c2393aefc588bb4e8b8ccb148f8`），另有 LEVIR+/WHU/SYSU/CLCD/DSIFN 各一 | git tree API |
| 骨干预训练 | SegFormer `mit_b0.pth`，**只给百度网盘链接**（提取码 i81p），仓库内没有 | README |
| 环境 | Python 3.7、torch 1.12.0 / torchvision 0.13.0、`mmcv-full==1.6.2`(cu113/torch1.12 wheel)、`mmseg 0.28.0`、`mmcls 0.25.0` | README / requirements.txt |
| 测试入口 | `test_cd.py` + `config/levir_test_bifa.json`，256×256 crop + `list/*.txt`，2 类 softmax `argmax`，`ConfuseMatrixMeter` 统计 | 仓库源码 |

README 里的 arXiv 链接 `2406.04207` **指向的是 CDMamba 而不是 BiFA**
（已核对 arXiv 摘要），所以作者报告的 LEVIR 指标只能从 TGRS 正文或仓库内
`resources/BiFA.pdf` 取，不能从该 arXiv 页面取。

## 2. 五条复现障碍（按严重度）

**B1（最严重，触及"不改网络"口径）**：`models/bifa.py` 第 494 行在
`BiFA.__init__` 里**无条件**执行

```python
self.ckpt = torch.load(r"E:\pertrain_weight\segformer\mit_b0.pth")
self.segformer.load_state_dict(self.ckpt, False)
...
self.ckpt.pop("head.weight"); self.ckpt.pop("head.bias")
```

即模型**构造**就依赖一个作者本机的 Windows 绝对路径。该文件只经百度网盘
分发，没有可校验的公开 SHA256。要跑起来只有两条路：
(a) 用户从百度网盘取 `mit_b0.pth`（来源可信度低于 P13 的 HuggingFace 权重，
无法做同级 provenance 审计）；(b) 改这两行源码跳过 ImageNet 初始化。
(b) **不能**再叫 `official_unmodified`——P13 之所以能进主表，正是因为它跑的
是作者公开、未改动的推理路径。

**B2（环境/硬件）**：作者环境是 py3.7 + torch 1.12 + `mmcv-full 1.6.2`；
服务器现为 torch 2.5.1+cu124、RTX 4090（sm_89）。`mmcv.runner` 在 mmcv 2.x
已移除，`mmcv-full 1.6.2` 也没有 torch 2.x 轮子，故必须另建 py3.7 环境；
而 torch 1.12 的官方二进制最高到 sm_86，4090 只能靠 PTX JIT，GPU 可用性
**不确定**，最坏情况退化为 CPU 推理。

**B3（靶值缺失）**：README 未给任何数据集的 F1/IoU，作者报告值只在
TGRS 正文/仓库 PDF 中。P13 式的 clean 复现门必须**先固定靶值再运行**，
所以在核查阶段就要把作者的 LEVIR 数字抄进协议。

**B4（配置路径不一致）**：`config/levir_test_bifa.json` 的 `resume_state`
是 `E:\PycharmProject\BiFA\experiments\LEVIR-train-bifa\checkpoint\best_cd_model`，
而 `test_cd.py` 会拼成 `{resume_state}_gen.pth`；仓库内实际提供的是
`experiments/pretrain/LEVIR.pth`。两者命名不对应，需要路径适配
（属"数据/路径适配器"，可接受，但必须写进协议并留档）。

**B5（静默降级陷阱）**：`test_cd.py` 用
`load_state_dict(cpkt_state, strict=False)` 加载权重，键不匹配也不会报错，
只打印 `missing_keys`。这与本项目 §85 记录的仪器教训同型：**能静默产出
"看起来正常"的低分**。若继续，协议必须加硬门：missing/unexpected keys
必须为空，且 clean 复现值必须命中作者报告值，否则判为复现失败而不是
"BiFA 很弱"。

## 3. 尚未确定的两件事（需要下载才能定）

1. `experiments/pretrain/LEVIR.pth`（60 MB）是否是**完整 state_dict**。
   若是，B1 的 (b) 路线在数值上是中性的（ImageNet 初始化会被完全覆盖），
   可写成"仅为绕过作者本机绝对路径的最小适配"；若不是（例如只存了部分
   分支），则必须拿到百度网盘的 `mit_b0.pth`。
2. 作者报告的 LEVIR clean F1/IoU（需读 `resources/BiFA.pdf`，14.3 MB）。

两件都需要下载文件，按用户工作方式（§1.1）由用户决定与执行。

## 4. 两条可选路线（供裁决）

**路线 A · 继续接入（成本明显高于 P13）**  
先由用户取 `mit_b0.pth`（百度）与 `resources/BiFA.pdf`，再建 py3.7 环境；
按 P13 两段式写 `docs/P14_BIFA_EXTERNAL_BASELINE_PROTOCOL.md`：
commit 固定为 `4b439a44…`、权重 SHA256 登记、clean validation 命中作者
报告值 + strict-key 工程门 → 冻结 manifest → 单独登记一次性 test。
若任一门失败即止损，且**不得**把失败写成"BiFA 比本文差"。

**路线 B · 立即止损（§0A 第 3 条）**  
把本文档作为复现障碍记录，论文中 BiFA 只做**定性**相关工作比较，明确写
"作者公开实现的模型构造依赖未公开分发的本机路径权重，且依赖 py3.7/
torch1.12/mmcv-full 1.6.2 栈，我们在 sm_89 硬件上无法在不修改网络源码的
前提下复现其公开推理路径"，不虚构任何 BiFA 数字。DiffRegCD（WACV 2026，
已精确复现作者 clean 指标）继续作为唯一外部强基线。

## 5. 本次核查的纪律声明

```text
未下载权重、未下载 PDF、未建环境、未运行任何模型；
未消费任何 test，未改动 P12/P13 冻结产物；
以上事实全部来自公开仓库 API 与 raw 源码文本，可复核。
```
