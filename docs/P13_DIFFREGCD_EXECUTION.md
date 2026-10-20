# P13 DiffRegCD 外部基线执行说明

## 1. 本阶段目标

本阶段只运行冻结的 LEVIR validation 工程审计，不运行 test，也不允许
根据 DiffRegCD 的 shift 结果修改本文模型、路由器、阈值或主张。

固定外部源码：

```text
DiffRegCD commit
6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38
```

固定权重：

| 组件 | SHA256 |
|---|---|
| DDPM backbone | `fc4482542ddadd704e6b8f95b6d6d778eb115001f89a4e988b3202fe6248b9f7` |
| LEVIR registration | `af9c8241e971b5c16469eb36e815b02b8ca054e8af223b3842699bdf5b3912bc` |
| LEVIR CD head | `842b3a83a764c26982fb93993ca30dce6e467fab3ccce247ebf7a3200cb38c02` |

## 2. 为什么必须报告两条路径

固定 commit 的作者测试函数把未变形的 `feats_A/feats_B` 直接送入
CD head，但训练函数先预测 B→A flow，再将 warped B 特征送入 CD head。
因此实验包同时报告：

- `official_unmodified`：作者仓库测试 wiring；
- `intended_warped`：只复用训练函数已经存在的 feature warp。

适配器不编辑外部仓库，不修改网络、权重、soft-argmax、阈值或损失。

## 3. 公平协议

- 先在完整 1024×1024 图上对 T2 注入确定性 2/4/8 px 平移；
- 再无重叠切成 16 个 256×256 tile；
- 作者输入保持 `[-1, 1]`；
- 两条推理路径复用同一批扩散特征；
- 每个样本和 tile 批次使用跨场景一致的扩散噪声；
- 固定裁去完整图四周 16 px，覆盖最大 8 px 注入边界；
- 对四场景逐像素 support 计算 SHA256，必须完全相同；
- 同一混淆矩阵同时报告变化类 F1/IoU、OA、作者兼容 mF1/mIoU。

## 4. 执行阶段

1. 运行单元测试和源码 commit/wiring 审计；
2. 用纯合成输入做 GPU smoke，不读取 validation；
3. 完整运行四场景 LEVIR validation；
4. 人工核对 clean 复现差异；
5. validation 工程门通过后，另行冻结一次性外部 test marker。

P13 主脚本不会运行 test。作者论文的 LEVIR 数字来自其合成仿射评测
分布，而本项目首先评估 clean 和受控纯平移，因此只报告数值差异，
不使用该差异自动选择 `official_unmodified` 或 `intended_warped`。

## 5. 服务器执行

首次安装外部依赖：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
bash scripts/setup_p13_diffregcd_environment.sh
```

运行完整 validation：

```bash
bash scripts/run_p13_diffregcd_validation.sh \
  2>&1 | tee reports/p13_diffregcd_validation_console.log
```

完成后：

```bash
REPORT_ROOT=$(cat reports/p13_diffregcd_validation/LATEST)
cat "$REPORT_ROOT/validation/decision.json"
cat "$REPORT_ROOT/validation/scenario_metrics.csv"
```

## 6. Validation 人工复核与一次性 Test

`official_unmodified` 在 clean validation 上得到 change F1
`0.872024`，与作者 `0.872` 几乎完全一致；mIoU 只差 `0.000299`。
工程门和同 support 门全部通过，因此正式冻结作者公开测试路径。

`intended_warped` clean change F1 为 `0.439085`，表明将训练路径中的
warp 直接接到公开测试权重并不是有效复现。该分支只保留为源码审计，
不进入主基线。路径选择只依据作者测试 wiring 与 clean 复现，不依据
shift 性能。

一次性外部 test：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
bash scripts/run_p13_diffregcd_final_test.sh \
  2>&1 | tee reports/p13_diffregcd_final_test_console.log
```

脚本会先冻结 validation 证据，再写不可逆 test marker，最后在相同有效
像素支持上自动比较 P12 主方法与 DiffRegCD。
