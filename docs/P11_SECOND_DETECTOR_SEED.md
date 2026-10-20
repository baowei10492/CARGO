# P11 第二检测器种子复刻协议

## 目的

验证 P7-3 的 clean/shift 结论是否依赖 seed-3407 的单一 P3b 检测器
权重。这是结构冻结后的稳定性检查，不是模型选择实验。

## 固定设置

- 第二种子：1234；
- P3b 配置从 seed-3407 的实际 checkpoint `config` 字段复制，只改
  `experiment.seed`、实验名和输出位置；
- P7-1A、P7-1B、P7-1C 使用冻结配置，只通过训练工具覆盖 seed；
- P7-2 特征和 task-router 使用同样配置的 seed=1234 副本；
- P7-3 使用冻结的特征集合、风险规则、门槛与 primary；
- 不做 Phase B 三路由器种子；
- 不运行 P9-B/P9-D/P10；
- 不触碰 LEVIR 或 S2Looking test。

## 统计纪律

- seed-1234 无论优于还是劣于 seed-3407 都必须完整保留；
- 不按 validation 选择“最好检测器种子”；
- 第二种子只作为独立复刻，最终报告两个检测器种子的并列结果；
- 若复刻失败，只能报告不稳定性，不得重新抽取第三种子替代。

## 执行链

```text
P3b seed1234
→ P7-1A seed1234
→ P7-1B seed1234
→ P7-1C seed1234
→ P7-2 training features/task router seed1234
→ P7-3 safe registration/task classifiers seed1234
→ validation decision
```

结构冻结清单会在训练前用 `sha256sum -c` 校验。任何冻结源文件变化都会
使流水线立即停止。
