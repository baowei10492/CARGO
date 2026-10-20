# 最终实验裁决

## 主结果

冻结方法：DINOv3 ConvNeXt-Tiny 变化检测器、任务感知输入级平移估计、
Raw/Aligned 双专家，以及以 Raw 为默认回退的风险控制 Task-veto。

LEVIR 一次性 test：

| 场景 | Raw F1 | Task-veto F1 | 提升 |
|---|---:|---:|---:|
| clean | 0.900673 | 0.900673 | 0 |
| Shift-2 | 0.863712 | 0.897089 | +0.033377 |
| Shift-4 | 0.796434 | 0.868934 | +0.072500 |
| Shift-8 | 0.663834 | 0.819015 | +0.155181 |

平均 shift 增益为 +0.087019。结果仅报告，未参与模型选择。

## 稳定性

- 第二套 P3b 检测器权重 seed-1234 完整复刻 P7-3；
- clean 掉幅 0；
- shift mean gain +0.098419；
- Shift-8 gain +0.178618；
- Shift-8 Oracle recovery 84.3%；
- 四个冻结门槛全部通过。

## 效率

- 完整模型 29.073M 参数，配准模块仅 34.8K；
- Task-veto 必须恒双路径，延迟为 Raw 的 2.51–3.27×；
- Registration-veto 可条件执行，batch=32 时相对恒双路径加速约
  1.09–1.44×，取决于覆盖率。

## 必须公开的限制

- P9-B 中 Task-veto 在 subpixel 1 px 和 rotate 0.5° 未通过预声明
  无损害门；
- Registration-veto 在几何 OOD 更安全，但域内排序不总是更优；
- Task/Registration 的绝对 tau 与检测器权重绑定；
- S2Looking 在 validation 的无标签适用域门失败，labeled test 未消费；
- “LEVIR 存在真实约 3 px 物理残余错位”未被因果证明，不能声称。

## 论文定位

可支持的核心主张是：

1. 把几何位移估计与“对齐是否有利于变化检测”分离；
2. 以任务收益和风险控制决定是否采用对齐；
3. clean 默认 Raw 回退，实现对齐场景零损；
4. 位移越严重，冻结方法带来的恢复收益越大；
5. 通过第二检测器权重、一次性 test、适用域负例和 OOD 失败共同界定
   方法的收益与边界。

实验链已经完成。后续不得再用 validation/test 调整结构或阈值。
