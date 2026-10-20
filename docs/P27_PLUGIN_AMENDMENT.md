# P27 修订：接入其他检测器（2026-09-17）

> 授权：用户 2026-09-17 同意按「可接入任意检测器的结构」补实验，先做 LEVIR-CD 与 DSIFN-CD（主记录 §192.11）。
> 本文件与 `configs/p27_plugins/` 下的配置写于任何插件运行之前；看到插件结果后不得修改。

```text
检测器：P21 已冻结的 BIT、CD-Lamba、CDxLSTM（reports/p21/VALIDATION_FROZEN.json 中的 LEVIR / DSIFN 格），不重训；
  权重取该格的第一阶段检查点，服务器上没有时取该格双路检查点中的检测器部分（二者的 sha256 相同）
定位器：P27 第一版 best.pth，不变
流程（每个检测器，与本文检测器相同）：F1 关（只作对照）→ 选择器训练与校准 → 验证集一次 → 测试集一次
  配方、阈值规则、判据均与 configs/p27_f1_gate.yaml、configs/p27_selector.yaml 相同
自动放行：校准段上每个数据集的选择器在第 2、3 档都有收益才跑验证集，否则停下
验证集、测试集：每个检测器各一次（标记按检测器分开，reports/p27_localizer/plugins/<检测器>/）
WHU-CD：这三个检测器需要先训练，另写修订
```
