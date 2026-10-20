# P7 Phase 0：几何因果链审计与经典控制组

## 1. 目的

本阶段不训练新网络，只回答三个问题：

1. 当前 offset 符号、输入像素/特征像素换算和 Warp 是否正确；
2. 把真实 offset 注入当前 P5c 特征路径后，能否恢复图像级 Oracle 的性能；
3. Phase Correlation 是否已经能解决合成全局平移。

若前两个问题失败，不得训练新匹配器；若第三个问题在两个数据集上均成立，则不再把“学习全局平移”作为论文主方向。

## 2. 新增工具

```text
tools/audit_geometry.py
tools/evaluate_feature_oracle.py
tools/evaluate_registration_controls.py
tools/summarize_p7_phase0.py
scripts/run_p7_phase0.sh
```

统一坐标约定：

```text
reference(x, y) 对应 target(x + dx, y + dy)
```

因此 T2 被向右移动 `dx > 0` 后，恢复时仍把正 `dx` 传给统一 Warp。

## 3. Feature-path Oracle 的四个分支

```text
raw
feature_aligned
checkpoint_blend
image_oracle
```

- `raw`：原 P3b 差异路径；
- `feature_aligned`：真实 offset 直接作用到所有特征尺度，完全使用 aligned difference；
- `checkpoint_blend`：真实 offset 加现有 Stage B alignment strengths；
- `image_oracle`：先在输入层逆移 T2，再运行 Raw P3b。

主要门槛：

```text
feature_aligned clean drop <= 0.001
feature_aligned shift-4 Oracle-gap recovery >= 80%
feature_aligned shift-8 Oracle-gap recovery >= 80%
```

评测统一去除 16 px 四周边界，避免各方法使用不同有效像素集合。

## 4. Phase Correlation 控制

第一轮只运行：

```text
raw
phase
oracle
```

每幅图输出：

- 预测 `(dx, dy)`；
- EPE；
- PCK@1/2/4；
- exact rate；
- Phase score；
- 分割 F1/IoU；
- Oracle-gap recovery。

如果 Phase 在 4/8 px 上均恢复至少 80% Oracle gap，且 clean 下降不超过 0.001，则合成全局平移已被经典方法解决。

ECC 已实现为第二控制组，但第一轮不安装、不运行。只有 Phase 不能给出明确结论时，才安装：

```bash
python -m pip install -e ".[geometry]"
```

并改用：

```text
configs/p7_registration_controls_t2.yaml
```

## 5. 服务器运行

上传并覆盖最新代码后：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate

bash scripts/run_p7_phase0.sh
```

脚本使用当前已知的四个 checkpoint 默认路径；如果实例中的路径不同，可在同一条命令前覆盖环境变量。

运行中查看：

```bash
REPORT_ROOT=$(cat reports/p7_phase0/LATEST)
echo "$REPORT_ROOT"
tail -f "$REPORT_ROOT/s2_phase_control.log"
```

完成后一次打印：

```bash
REPORT_ROOT=$(cat reports/p7_phase0/LATEST)
cat "$REPORT_ROOT/decision.json"
```

## 6. 决策

`decision.json` 只会给出下列三类后续动作之一：

```text
fix_geometry_or_coordinate_implementation
fix_feature_warp_or_fusion_before_new_matcher
skip_learned_global_translation_and_move_to_local_real_warps
begin_p7_1_robust_global_from_local_matcher
```

本阶段不得根据结果临时降低 80% recovery 门槛。
