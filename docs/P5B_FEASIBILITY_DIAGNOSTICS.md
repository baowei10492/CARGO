# P5b 前置可行性诊断

## 1. 为什么不直接训练新模块

P5a 表明，简单 T2 位移增强只改善三个目标位移场景，却使 clean 和其他扰动场景普遍下降。下一步可能需要显式空间对应，但在实现之前必须确认：

1. 已知逆变换能否恢复性能；
2. 当前 DINOv3 特征是否包含足够精细的位移对应信息。

这两项任一失败，都应先修正协议或更换描述子，而不是增加网络复杂度。

## 2. Oracle 逆位移

```bash
python -m tools.evaluate_robustness \
  --checkpoint outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth \
  --protocol configs/p5b_oracle_t2_shift.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --split val \
  --device cuda \
  --oracle-inverse-shift \
  --output-dir reports/robustness/p5b_oracle_inverse_val
```

输出包含 clean 和 2/4/8 px 三个经过 oracle 恢复的 T2 位移场景。

通过条件：

- 4/8 px 恢复后的 F1 接近 clean；
- 8 px oracle F1 原则上不低于 0.89；
- 若结果异常，优先检查位移符号、边缘掩膜和逆变换。

## 3. DINOv3 对应探针

```bash
python -m tools.probe_dino_correspondence \
  --checkpoint outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --split val \
  --device cuda \
  --feature-level 0 \
  --search-radius 3 \
  --spatial-step 2 \
  --severities 2 3 \
  --output reports/robustness/p5b_dino_correspondence_probe.json
```

探针定义：返回 `(dx, dy)` 表示参考特征 `(x, y)` 与 T2 特征 `(x+dx, y+dy)` 对应。单元测试已验证该坐标约定。

重点读取：

```bash
python - <<'PY'
import json

for path in (
    "reports/robustness/p5b_oracle_inverse_val/results.json",
    "reports/robustness/p5b_dino_correspondence_probe.json",
):
    data = json.load(open(path, encoding="utf-8"))
    print("\nFILE:", path)
    if "summary_by_severity" in data:
        print(json.dumps(data["summary_by_severity"], indent=2))
    else:
        print("clean_f1:", data["clean_f1"])
        for row in data["results"]:
            print(row["scenario"], f"F1={row['f1']:.6f}")
PY
```

对应探针初步通过条件：

- 4 px 与 8 px 的平均 EPE 不应接近随机搜索水平；
- 8 px 的精确 offset 命中率应明显高于零；
- `mean_score_margin` 应为正且不能系统性趋近于零。

若 level 0 成功，再开发单尺度原型；若失败，则先实现高分辨率轻量匹配描述子。

## 4. 暂定方法边界

只有诊断通过，才实现“变化保持的选择性对应”：

- 原始 raw-difference 路径完整保留；
- 对应分支仅作残差修正，初始残差系数为零；
- 匹配置信度与变化概率分开建模；
- 只在可信且疑似未变化的位置使用对齐证据；
- 真实变化和不确定位置保留 raw difference；
- GT 只用于训练辅助监督，推理 forward 不使用 GT；
- 必须与相同增强的 P5a、公平的 alignment-only 版本直接比较。

该设计仍有较高文献碰撞风险。正式论文必须直接比较现有联合配准检测和 mask-guided alignment 方法，并在真实错位数据上证明“对齐—变化保留冲突”真实存在。
