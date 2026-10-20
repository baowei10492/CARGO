# P11 全结构冻结

## 冻结目的

P9-B、P9-D 和 P10 已分别给出域外安全、选择性预测与效率证据。此后
不再根据 validation 结果修改模型结构、特征集合、路由规则、阈值或
primary 身份。只有完成冻结，才能开始第二检测器种子以及最终 test。

## 冻结结构

- 检测器：DINOv3 ConvNeXt-Tiny P3b；
- 定位器：P7-1C task-aware input registration；
- 预注册主方法：P7-3 Task-veto；
- 安全/效率消融：P7-3 Registration-veto；
- 回退：Raw；
- Task-veto 必须按恒双路径部署；
- Registration-veto 可以按条件 Aligned 分支部署。

## 已冻结限制

- P9-B Task-veto 在 subpixel 1 px 和 rotate 0.5° 未通过无损害门；
- Registration-veto 的域内 AUROC/AURC 低于 Task-veto；
- 条件执行加速取决于 batch 与覆盖率；
- “LEVIR 存在真实约 3 px 残余错位”未被因果证明，不得写入论文；
- S2Looking 是空间变化场/定位器适用域负例；
- WHU 的历史 raw/oracle F1 support 不配对，不作为严格 headroom。

## 冻结后允许的工作

- 第二 P3b 检测器种子完整复刻一次 P7-3；
- 只读统计、制图和论文写作；
- 已预声明的最终 LEVIR/S2Looking test 各一次；
- 修复会导致崩溃或错误真值的代码缺陷，但必须记录且不得借机调指标。

## 冻结后禁止的工作

- 新增路由特征、重新挑阈值或换 primary；
- 根据第二种子结果只保留更优种子；
- 重复查看 test 或按 test 改模型；
- 把 Registration-veto 的 P9-B 通过改写成 Task-veto primary 通过。
