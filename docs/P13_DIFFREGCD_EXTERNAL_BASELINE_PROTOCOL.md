# P13：DiffRegCD 外部强基线预注册协议

> 写定时间：2026-07-27，在运行任何 DiffRegCD LEVIR validation/test 前  
> 本项目方法与 P12 数字已经冻结；P13 只增加外部对比，不得反向调参  
> DiffRegCD 官方仓库 commit：
> `6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38`

## 1. 为什么它是必须对比的方法

DiffRegCD（WACV 2026）与本文共享“配准服务于变化检测”的问题背景，
并公开了 LEVIR-CD 预训练权重。它采用冻结扩散特征、稠密分类式对应和
仿射扰动监督，是目前最直接、也最强的公开竞争方法。

论文：https://arxiv.org/abs/2511.07935  
代码：
https://github.com/Anita-Madani/DiffRegCD-Integrated-Registration-and-Change-Detection-with-Diffusion-Features  
权重：https://huggingface.co/Anita1379m/DiffRegCD

## 2. 源码审计发现：官方 inference wiring 有歧义

固定 commit 的 `model/cd_model_256.py` 中：

- `optimize_parameters()` 先预测 `pred_flow_coarse`，再用
  `warp_features_all(self.feats_B, pred_flow_coarse)`，CD head 接收
  `feats_A` 与 warped `feats_B`；
- 但 `test()` 计算的是拼写不同的 `pred_corase_flow`，随后直接调用
  `netCD(self.feats_A, self.feats_B)`，没有把预测流用于 CD；
- `get_current_visuals()` 又读取 `self.pred_flow_coarse`，与 `test()`
  写入的字段不一致。

这不是本文方法的结果，也不能据此贬低作者工作。为保证复现可审计，
P13 必须同时报告：

1. `official_unmodified`：原仓库 test wiring；
2. `intended_warped`：仅按训练路径，把预测流 warp 后送入 CD head；
3. 两者的 clean F1 与论文/作者给出的 clean 数字核对；
4. 论文主比较优先采用与作者方法描述一致、且 clean 复现更接近作者
   报告的路径；若仍有歧义，两组均报告，不擅自选择有利版本。

允许的最小修复只有：

```python
pred_flow_coarse, logits = netReg(fA, fB)
warped_feats_B = warp_features_all(feats_B, pred_flow_coarse)
pred_cm = netCD(feats_A, warped_feats_B)
```

不得改网络、soft-argmax、权重、阈值或损失。

## 3. 数据与公平口径

### 3.1 输入

- 数据：与本文相同的 LEVIR-CD 官方 split；
- DiffRegCD 原生输入为 256×256；
- 对 1024×1024 图先在完整图上注入 T2 位移，再切 256 瓦片；
- 首轮采用无重叠 4×4 瓦片，只作 validation 工程审计；
- 若瓦片边界导致 clean F1 与作者报告明显不符，允许在看到 shift
  结果前写定 overlap-stitch 方案；不得按 shift F1 选择 tiling。

### 3.2 场景

```text
clean:s0:common
shift:s1:t2 = 2 px
shift:s2:t2 = 4 px
shift:s3:t2 = 8 px
```

偏移方向、每样本随机种子和 P12 协议保持一致。不得另抽对 DiffRegCD
更有利或更不利的方向。

### 3.3 指标

作者代码主指标包含背景类与变化类的 mean F1；本文主指标是变化类
positive F1。为避免口径偷换，必须从同一混淆矩阵同时输出：

- change-class precision / recall / F1 / IoU（论文主对比）；
- OA；
- mean F1（仅用于核对作者复现）；
- TP / FP / FN / TN；
- 有效像素数。

所有方法使用同一 valid-pixel mask，排除人工平移产生的无效边界。

## 4. validation 门

先运行 validation，不运行 test。只有以下全部成立才允许登记外部
baseline test：

1. 权重 SHA256 和源码 commit 已记录；
2. clean preprocessing/label/方向审计通过；
3. `official_unmodified` 与 `intended_warped` 差异已记录；
4. 任一可解释路径的 clean F1 达到作者结果的合理复现范围，或已经
   获得足以解释差异的确切证据；
5. 四场景逐像素 support 完全一致；
6. 适配器测试通过。

失败时不得反复改作者模型。只记录复现障碍，并继续 BiFA 基线。

## 5. test 纪律

P12 已消费的是本文冻结方法的一次性 test。P13 外部基线若通过
validation 工程门，需在运行前生成新的只读 marker，声明：

- 外部权重和适配器已冻结；
- test 结果只用于横向报告；
- 不据此修改本文方法、阈值、主方法或论文假设；
- 每个外部基线只运行一次；
- 失败也完整保留。

## 6. 当前执行顺序

1. 下载仅 LEVIR 所需的 DiffRegCD 三个权重；
2. 在独立环境运行作者原始 clean validation；
3. 运行最小 intended-warped clean validation；
4. 冻结 tiling、方向和 valid support；
5. 执行四场景 validation；
6. 汇总后决定是否进入一次性外部基线 test；
7. 再以相同纪律接入 BiFA。

