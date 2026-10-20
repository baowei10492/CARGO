# P12 一次性最终测试协议

## LEVIR

- 使用 P11 冻结 manifest 指向的 seed-3407 P7C、Task/Registration
  classifiers；
- 检测器使用冻结 seed-3407 P3b；
- split 固定为 `test`；
- clean + Shift-2/4/8 各执行一次固定协议；
- primary 固定为 `safe_veto_task`；
- test 只报告，不产生 pass/fail，不做模型或阈值选择；
- 在启动 GPU 评估前写入全局 `LEVIR_TEST_CONSUMED.json`；即使中断也
  不得擅自重跑。

## S2Looking

Stage C 已在 validation 的预声明无标签门停止：

```text
Shift-8 conditional PCK@1 = 0.486 < 0.50
```

按原协议，门失败后不执行带标签评估。这个数据集的最终方法 test
因此继续不消费，并写入显式的 `NOT_CONSUMED` 记录。不得为了凑第二个
test 数据集而绕过适用域门。

## 纪律

- 最终 test 数字无论好坏全部保留；
- 不再训练、不调阈值、不换 primary；
- 不重复执行；
- seed-1234 是 validation 复刻，不用于选择最终 test 权重。
