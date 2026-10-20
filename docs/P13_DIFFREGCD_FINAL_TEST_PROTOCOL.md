# P13 DiffRegCD 一次性外部基线 Test 协议

## 1. Validation 裁决

DiffRegCD 官方仓库固定 commit：

```text
6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38
```

在 LEVIR validation 上，`official_unmodified` 对作者公开结果的复现为：

| 指标 | 观测值 | 作者值 | 差值 |
|---|---:|---:|---:|
| change F1 | 0.872024 | 0.872 | +0.000024 |
| mF1 | 0.933376 | 0.929 | +0.004376 |
| mIoU | 0.881299 | 0.881 | +0.000299 |
| OA | 0.989872 | 0.987 | +0.002872 |

权重、源码 commit、warp 方向、四场景有效像素 SHA256 和有效像素数
全部通过工程门。因此该路径构成可信的官方公开基线复现。

`intended_warped` clean change F1 仅 0.439085，且四场景都显著低于官方
路径。它只作为源码 wiring 歧义诊断保留，不进入论文主表。路径选择依据
严格限定为“作者公开 test wiring + clean 作者指标复现”，没有使用 shift
结果选择对我们有利的版本。

## 2. 冻结 Test 设置

```text
split = LEVIR test
path = official_unmodified
scenarios = clean / Shift-2 / Shift-4 / Shift-8
full image shift before 256×256 non-overlap tiling
threshold = 0.5
evaluation border = 16 px
seed = 3407
allowed runs = 1
```

Test 只用于横向报告，不产生新门槛，不用于选择 DiffRegCD 路径，也不得
修改 P12 主方法、路由器、阈值、模型或论文假设。

## 3. 一次性保护

运行前先：

1. 读取 P13 validation 的唯一 `LATEST`；
2. 用 clean 指标和工程门生成 `freeze_manifest.json`；
3. 校验 validation 结果与 test 协议 SHA256；
4. 在读取 test 前写入
   `DIFFREGCD_LEVIR_TEST_CONSUMED.json`，状态为
   `started_irreversible`。

即使任务中断，也不得删除 marker 后擅自重跑。正常完成后 marker 改为
`completed`，`allowed_reruns` 始终为 0。

## 4. 公平比较

P12 与 P13 都使用 16 px interior border 和同一
clean/2/4/8 px corruption protocol。汇总工具从 TP/FP/FN/TN 重新计算
每个场景的有效像素数；只有四场景逐一相等时，才输出 P12
`safe_veto_task` 与 DiffRegCD `official_unmodified` 的正式横向表。

