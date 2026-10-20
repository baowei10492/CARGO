# P7-1B：失配判断与条件位移定位解耦

## 1. 触发依据

P7-1 首轮 Shift-8 诊断：

```text
final PCK@1              59.4%
raw current PCK@1        75.0%
best non-zero PCK@1      79.7%
expanded ±12 px PCK@1    64.1%
```

扩大搜索域反而下降，因此保留 stride-2、radius-4 和 `±8 px` 支持。
强制去掉零候选可救回 13/26 个最终失败样本，证明主要矛盾是
zero anchor 与 offset localization 在同一个 softmax 中竞争。

## 2. 结构修改

旧结构：

```text
81-way softmax = zero + 80 non-zero offsets
```

P7-1B：

```text
q_shift = BinaryHead(score surface, local vote histogram, margin)
conditional offset = softmax over 80 non-zero offsets
```

固定推理规则：

```text
q_shift < 0.5  → offset=(0,0)，回退 Raw
q_shift >= 0.5 → 使用最佳非零 offset
```

## 3. 损失

Clean 样本：

```text
只监督 q_shift=0
```

人工失配样本：

```text
监督 q_shift=1
+ conditional non-zero offset distribution
+ conditional SmoothL1
```

不再要求 clean 样本参与 offset softmax，消除 zero 与非零定位的目标冲突。

## 4. 初始化与不可调整项

- 从 P7-1 `last.pth` 初始化；
- descriptor、reliability 和 surface refiner 全部继承；
- 独立 shiftedness head 随机初始化；
- P3b 继续冻结；
- radius 固定为 4；
- 决策阈值固定为 0.5；
- 不使用 Test split。

## 5. 服务器运行

覆盖更新包后：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
bash scripts/run_p7b_stage1_levir.sh
```

流水线自动执行：

```text
pytest
→ BS64 smoke
→ 从 P7-1 last.pth 初始化并训练 15 epoch
→ estimator Validation
→ 原预注册门槛
→ 仅通过后运行输入级 Warp F1
```
