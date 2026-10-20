# P7 Stage 1：输入级任务感知配准

## 1. 本阶段回答的问题

P7-0 已证明：

1. 几何符号和基础 Warp 正确；
2. 输入级 Oracle 优于后编码特征级 Oracle；
3. Phase correlation 会在 clean 双时相影像上追逐天然相关偏移，
   而不是遵守变化检测标注坐标系；
4. Phase score 不能可靠区分 clean 与人工 shift。

因此 P7-1 不再调 P5c 的四个融合强度，也不再以纯相关峰作为目标。
本阶段只判断：一个受 clean=zero 和人工 shift 监督的轻量输入匹配器，
能否学会任务坐标系中的额外失配量。

## 2. 新结构

```text
T1/T2
→ shared stride-2 descriptor
→ ±8 input-pixel local correlation
→ local margin + learned reliability
→ top-25% robust aggregation
→ global offset distribution
→ explicit shiftedness / zero anchor
```

训练时可以用 GT change mask 排除变化区域及其邻域；推理时绝不读取 mask，
不存在标签泄漏。

P3b backbone 和 decoder 全部冻结。Stage A 的 `change_logits` 只用于保持
训练器接口与 clean F1 恒定，梯度只来自 registration auxiliary loss。
因此训练日志中的 `val_f1` 基本不变是预期行为，不代表 estimator 没有学习；
应观察 `train_offset_epe`、`train_clean_pck1` 和 `train_shift_pck1`。

## 3. 损失

```text
L = L_offset_distribution
  + L_smooth_l1
  + 0.25 * L_shiftedness
```

clean 样本在前两项中使用 2 倍权重，明确要求天然双时相差异仍输出零位移。

训练样本：

```text
50% clean
50% synthetic T2 shift
shift pixels = {2, 4, 8}
8 个轴向/对角方向
```

## 4. 预注册门槛

先评 estimator，门槛全部通过才运行输入级对齐后的分割评测：

```text
Clean hard PCK@1       >= 95%
Shift-4 hard mean EPE  <= 1 px
Shift-4 hard PCK@1     >= 80%
Shift-4 hard PCK@2     >= 95%
Shift-8 hard mean EPE  <= 1 px
Shift-8 hard PCK@1     >= 80%
Shift-8 hard PCK@2     >= 95%
```

分割门槛：

```text
Clean F1 delta       >= -0.003
Shift-4 F1 gain      >= +0.020
Shift-8 F1 gain      >= +0.050
```

Test split 继续封存。

## 5. 服务器运行

先把更新包传到服务器项目父目录，覆盖解压后：

```bash
cd /root/autodl-tmp/rs_cd_work
unzip -o p7_stage1_code_update.zip -d rs_cd_project

cd rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
python -m pytest -q
```

确认测试通过后，前台一次运行：

```bash
bash scripts/run_p7_stage1_levir.sh
```

脚本会自动完成：

```text
pytest
→ BS64 GPU smoke
→ LEVIR Stage A 训练
→ last.pth offset 评估
→ 预注册门槛判定
→ 仅在 estimator 通过时运行输入级对齐 F1 评测
```

查看当前进度：

```bash
REPORT_ROOT=$(cat reports/p7_stage1_levir/LATEST)
echo "$REPORT_ROOT"
tail -f "$REPORT_ROOT/train.log"
```

完成后一次打印：

```bash
REPORT_ROOT=$(cat reports/p7_stage1_levir/LATEST)
cat "$REPORT_ROOT/train_summary.json"
cat "$REPORT_ROOT/registration_val.json"
cat "$REPORT_ROOT/decision.json"

if test -f "$REPORT_ROOT/input_alignment_val.json"; then
  cat "$REPORT_ROOT/input_alignment_val.json"
fi
```

## 6. 结果解释

- 若 clean PCK@1 失败：zero anchor 尚未压过天然双时相相关偏移；
- 若 clean 通过而 shift-4/8 失败：描述子或局部稳健聚合不可辨识；
- 若 estimator 全通过而分割门槛失败：检查门控、输入 Warp 边界或插值；
- 若 LEVIR 全通过：固定结构和阈值，再在 S2Looking 训练/验证，
  不先根据 S2Looking 结果反调 LEVIR 参数。
