# P10 效率—安全性协议

## 目的

在方法结构冻结前，量化 Raw、配准估计、恒定双路径和真正可条件执行
的 Registration-veto 部署路径。该实验不重新训练、不选阈值、不读取
标签、不触碰 test，也不改变 P9-B/P9-D 的正式结论。

## 关键口径

- Task-veto 使用 Raw/Aligned 任务分歧特征，因此必须先得到两条检测
  logits；它的部署成本按恒定双路径报告，不能宣称早退节省。
- Registration-veto 只用配准侧证据，可以在 Aligned 检测前决定是否
  弃权；它才使用条件执行曲线。
- vGPU 的绝对延迟受宿主调度影响，主报告同一次运行内的开销比与
  speedup；同时保留 p10/p50/p90 和原始设备信息。
- PyTorch FLOP 计数只覆盖支持的算子，属于一致口径的下界，不冒充
  完整硬件指令计数。

## 固定协议

见 `configs/p10_efficiency.yaml`：

- 输入：256×256；
- batch：1、8、32；
- warmup 20，正式 100 次；
- AMP；
- Registration-veto 覆盖率：0、25%、50%、75%、100%；
- batch=1 统计支持算子的 FLOPs。

## 输出

每个 batch 报告：

1. Raw detector；
2. registration-only；
3. constant dual-path（Task-veto 核心成本）；
4. Registration-veto conditional curve；
5. constant dual / Raw 开销比；
6. conditional 相对 constant dual 的 speedup；
7. 完整模型、registration、两个路由器参数量；
8. FLOPs 及计数范围说明。

## 决策

本阶段没有性能 pass/fail 门。它只回答：

- Task-veto 的域内排序收益付出了多少双路径成本；
- 更保守的 Registration-veto 除了 P9-B 域外安全性外，是否还提供
  可兑现的条件执行效率优势。

不得根据效率数字事后更换 P9-B primary；效率仅作为方法权衡和消融。
