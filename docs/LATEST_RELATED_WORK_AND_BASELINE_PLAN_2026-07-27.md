# 最新相关工作与强基线计划（2026-07-27）

> 状态：P12 后的论文阶段检索  
> 目的：确定本项目能够诚实主张的创新，以及投稿前必须补齐的强基线  
> 纪律：不重训、不重跑 P12、不根据 test 结果调整本方法

## 1. 检索后的核心结论

“对未配准双时相图像联合执行配准与变化检测”已经不是空白方向。当前
最直接的相关工作包括：

1. **DiffRegCD，WACV 2026**：使用冻结扩散模型多尺度特征，把稠密
   对应建模为高斯平滑分类，并在 LEVIR-CD、DSIFN-CD、WHU-CD、
   SYSU-CD、VL-CMU-CD 上联合训练配准和变化检测。作者公开了代码及
   各数据集权重。
   - 论文：
     https://openaccess.thecvf.com/content/WACV2026/html/Madani_DiffRegCD_Integrated_Registration_and_Change_Detection_with_Diffusion_Features_WACV_2026_paper.html
   - arXiv：https://arxiv.org/abs/2511.07935
   - 代码：
     https://github.com/Anita-Madani/DiffRegCD-Integrated-Registration-and-Change-Detection-with-Diffusion-Features
   - 权重：https://huggingface.co/Anita1379m/DiffRegCD
2. **ChangeRD，ISPRS P&RS 2025**：预测透视变换，使用 APT 对输入与
   多层特征做空间变换，并联合变化检测监督。
   - DOI：https://doi.org/10.1016/j.isprsjprs.2024.11.019
3. **BiFA，IEEE TGRS 2024**：通过差分流场做双时相空间特征对齐，
   同时处理通道与多尺度对齐；代码和部分预训练权重公开。
   - DOI：https://doi.org/10.1109/TGRS.2024.3376673
   - 代码：https://github.com/zmoka-zht/BiFA
4. **AROSICS**：成熟的遥感全局/局部亚像素配准工具，可作为经典
   register-then-detect 工程基线。
   - 代码：https://github.com/GFZ/arosics
   - 论文：https://doi.org/10.3390/rs9070676

因此，本项目不能主张：

- 首次把配准整合进变化检测；
- 首次用合成几何扰动监督配准；
- 首次做双时相特征对齐；
- 在任意未配准遥感图像上普遍有效。

## 2. 本项目仍然成立的差异化贡献

本项目与上述工作的关键区别不是“能否对齐”，而是：

1. **任务收益选择**：显式比较 Raw/Aligned 两条任务路径，预测本次
   对齐是否真正有利于下游变化检测，而不是默认对齐一定有益。
2. **安全回退**：风险高或不确定时回退 Raw，从而把 clean 无损伤
   作为结构目标；P12 clean delta 为 0。
3. **选择性预测评价**：报告 per-scenario AUROC、ECE、risk-coverage、
   AURC 和部署工作点风险，而不仅报告平均 F1。
4. **适用域门控**：先测可恢复 headroom 和无标签定位器适用性，再
   决定是否消费目标数据集标签/test。
5. **失败边界实证**：S2Looking、旋转和中档亚像素扰动揭示全局平移
   与路由器的能力边界，而不是只展示有利案例。

建议论文方法名使用描述性名称，例如：

> Risk-aware Selective Alignment for Misregistration-Robust Change
> Detection

不要把 DINOv3 backbone 或全局平移定位器本身包装为主要创新。

## 3. 投稿前基线优先级

### A 级：必须运行

1. **DiffRegCD 作者公开 LEVIR 权重**
   - 原始 clean；
   - 与本文完全相同的 T2 方向 2/4/8 px 平移；
   - 同一有效像素支撑；
   - F1/IoU、干净损伤、各严重度恢复；
   - 参数量、延迟可以引用作者模型实测，不能仅从论文抄不同硬件数字。
2. **BiFA 作者公开 LEVIR 权重**
   - 同上；
   - 若其旧环境无法可靠复现，必须记录 commit、依赖和失败原因，
     不得用“无法运行”代替全部强基线。
3. 本项目已有经典基线：
   - Phase correlation；
   - ECC；
   - SIFT translation；
   - Always-aligned；
   - Raw detector。

### B 级：论文完整性建议

1. ChangeRD：若官方代码/权重可获得，按相同协议评估；否则只放文献
   定性比较，不虚构复现数字。
2. AROSICS：在原始 1024 图或地理影像口径上做一次 register-then-detect
   对照；若当前 PNG 裁块不具备地理元数据，应明确这不是其最佳用法。
3. 一个较强纯变化检测 backbone（ChangeMamba 或公开强模型）：
   证明“换强检测器”不能自然解决输入错配；这属于 backbone 对照，
   不是直接配准基线。

## 4. 新基线实验必须先冻结的协议

在运行 DiffRegCD/BiFA 前写出并保存：

- 仓库 URL 与 commit SHA；
- 作者权重 URL 与 SHA256；
- 原始 preprocessing 和输入尺寸；
- 本项目注入偏移的方向约定；
- valid-pixel 支撑生成方式；
- 阈值来源（作者固定阈值，不在本项目 val/test 搜索）；
- 只允许一次 LEVIR validation 适配检查；
- test 仅作报告，不用来修代码或选择模型；
- 若因输入格式需要适配，只允许数据适配器，不改网络和权重。

由于本项目 P12 test 已经消费，最稳妥的顺序是：

1. 先只在 LEVIR validation 完成 DiffRegCD/BiFA 适配与方向审计；
2. 冻结适配器和全部参数；
3. 单独登记“外部基线一次性 test 报告协议”；
4. 最后统一运行外部基线 test；不得因此更改本文方法或 P12 数字。

## 5. 当前判断

DiffRegCD 会显著提高投稿门槛，但不会让本项目失去价值。它更强于：

- 稠密、亚像素、仿射和大位移配准；
- 多数据集正向验证。

本项目当前更强、也更独特的证据在：

- clean 零损伤；
- 明确的 Raw 回退；
- 任务收益路由；
- 风险—覆盖评价；
- 适用域筛查和负例机制分析。

论文成败将取决于：在完全相同错配协议下，与 DiffRegCD/BiFA 相比，
本文能否展示更好的安全—收益权衡，而不是只与 FC-Siam-Diff 比较。

