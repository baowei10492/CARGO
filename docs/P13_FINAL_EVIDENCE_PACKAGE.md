# P13 最终证据包（打包口径与验证门）

> 状态：报告型工具，不重跑任何实验  
> 输入：已落盘的 P12 / P13 报告树  
> 输出：`p13_diffregcd_final_evidence.tar.gz`

## 1. 这个工具做什么、不做什么

做：

- 读取四棵已完成报告树的 `LATEST`（`p12_final_test`、
  `p13_diffregcd_validation`、`p13_diffregcd_external_freeze`、
  `p13_diffregcd_final_test`）；
- 从 `results.json` 的原始 TP/FP/FN/TN **重算**本文 `safe_veto_task` 与
  DiffRegCD `official_unmodified` 的同 support 对比；
- 生成论文主表/主图与外部对比表/图；
- 复制证据、协议、一次性 marker，逐文件写 SHA256，打成归档。

不做：

- 不加载任何模型、不读任何影像、不触碰 test split；
- 不产生新门槛、不改 primary/tau、不改 P12/P13 任何数字；
- 校验不通过时**不写半成品包**，直接抛错退出。

## 2. 五个验证门

| 门 | 判据 |
|---|---|
| `markers` | P12 `LEVIR_TEST_CONSUMED.json` 与 P13 `DIFFREGCD_LEVIR_TEST_CONSUMED.json` 均 `completed`、`allowed_reruns=0`；P13 另需 `reporting_only_no_selection=true`、`inference_path=official_unmodified` |
| `external_freeze` | `freeze_manifest.json` 为 `frozen_external_test`、路径 `official_unmodified`、commit `6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38`、`test_authorized=true` |
| `external_validation` | validation `split=val`、`test_consumed=false`、官方路径 clean change F1 = 0.872024（±1e-6） |
| `recorded_number_reproduction` | 四场景 F1 与主记录 102/107 已发布值一致（±1e-6）；mean shift advantage = 0.184588；两臂有效像素均为 125,960,192 且逐场景相等 |
| `recomputed_comparison_matches_recorded` | 由原始计数重算的对比与已落盘 `p12_vs_diffregcd.json` 逐项一致（≤1e-12） |

第四、五门是主记录 85 的仪器纪律在本工具上的落实：**任何新评估/汇总工具
必须先复现一个已知数值，复现不了就拒绝报告**。容差 1e-6 是对「已发布数字
被四舍五入到六位小数」的一致性检查，不是新的性能门槛。

## 3. 包结构

```text
p13_final_evidence/<tag>/
├── README.md                     主表 + 外部对比表 + 门结果 + 限制
├── verification.json             五个门的完整判据与数值
├── evidence_manifest.json        逐文件 bytes + SHA256
├── paper_assets/
│   ├── table_main_final_test.{csv,md}      P12 Raw vs Primary
│   ├── figure_main_final_test.svg
│   ├── table_external_comparison.{csv,md}  本文 vs DiffRegCD
│   └── figure_external_comparison.svg
├── evidence/
│   ├── p12_final_test/ p13_diffregcd_validation/
│   ├── p13_diffregcd_external_freeze/ p13_diffregcd_final_test/
│   ├── LEVIR_TEST_CONSUMED.json
│   └── DIFFREGCD_LEVIR_TEST_CONSUMED.json
└── protocol/                     P12/P13 协议 md 与三个 config yaml
```

权重文件（`.pth/.pt/.ckpt`）默认不入包；模型路径与哈希仍可在 P11 结构冻结
证据中查到。

## 4. 必须随数字一起报告的限制

包内 `README.md` 固定写入，论文引用时不得拆开：

1. 优势限定于受控全局平移，不外推到旋转、透视或局部形变；
2. P9-B：预注册 Task-veto 在 1.0 px 亚像素与 0.5° 旋转未过无损害门；
3. 两法骨干不同，**不得**把相对 DiffRegCD 的全部 +18.46 pp 归因于路由器。

## 5. 运行

```bash
bash scripts/run_p13_final_evidence.sh
```

脚本顺序：targeted pytest → marker 预检 → 打包 → 打印五门结果与外部对比表
→ 输出归档与 SHA256。归档已存在时默认拒绝覆盖（`FORCE=1` 才替换）。
