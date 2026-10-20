# P7 Stage C2：迁移失败来源的无标签 offset 一致性诊断

> 预注册时间：2026-07-26（Stage C stopped_at_unlabeled_gate 之后、
> 本诊断运行之前）  
> 性质：纯无标签诊断（只用注入位移真值，不读任何变化标签），
> 可以任意运行，不消耗 validation。

## 1. Stage C 停止时的现象

```text
scenario   pck1   pck2   pck4   epe    q_rate
clean      0.000  0.278  0.630  3.608  0.672
shift-2    0.204  0.478  0.744  2.956  0.858
shift-4    0.204  0.528  0.746  2.931  0.972
shift-8    0.486  0.602  0.676  2.853  0.998
```

注意四个事实：

1. clean pck1 = 0 是构造性的：conditional localizer 只在非零 offset
   中定位（最小步长 2 px），对 clean 永远不可能命中 1 px 内，该格
   不构成失败证据；
2. EPE 在注入 2/4/8 px 下几乎不变（2.85–2.96）。若定位器塌缩或
   偏向小 offset，误差应随注入幅度增长；恒定残差说明估计在跟随
   注入位移，只是带着一个 ~2–3 px 的稳定残差；
3. clean q 触发率 67.2%（LEVIR 为 6.25%）：q 头认为大部分"clean"
   S2Looking 对存在位移；
4. 主记录 45.3 节早已测得 S2Looking 的 phase correlation 固有残差
   ~4 px 且与注入严重度无关。

## 2. 待判别的两个假设

```text
H1 任务坐标系失配（描述子健康）：
   S2Looking 原始双时相对本身携带 ~2-3 px 固有错位；LEVIR 定位器
   如实估计"光度对应位移 = 注入 + 固有"；但 S2Looking P3b 检测器
   在训练中已吸收固有部分，任务坐标系里的"额外失配"≠光度位移。
   q 的 67% clean 触发在光度意义上是"对的"，在任务意义上是错的。

H2 描述子域差（描述子在 S2Looking 上失效）：
   侧视、阴影、农村弱纹理使输入级 descriptor 无法建立可靠对应，
   估计基本不跟随注入位移。
```

第 1 节的现象倾向 H1，但必须用逐图证据判别。

## 3. 诊断量（tools/diagnose_transfer_offset_consistency.py）

对每张 val 图记录四个场景的 conditional offset 估计，计算：

```text
delta-PCK：|| (est_shift - est_clean) - 注入offset || <= k 的比例
  （逐图扣除自身 clean 估计后，是否精确跟随注入位移）
residual 一致性：r_s = est_s - 注入offset；
  跨严重度两两 ||r_a - r_b||、以及 ||r_s - est_clean|| 的 PCK
clean conditional offset 幅度分布
```

H1 预期：delta-PCK 高、residual 跨严重度高度一致且 ≈ est_clean；
H2 预期：delta-PCK 低、residual 互不一致。

## 4. 预注册分支路由（诊断阈值，非成败门槛）

主判别量：Shift-8 delta-PCK@1（参考 LEVIR conditional PCK@1 = 92%）。

```text
delta_pck1(S8) >= 0.80 且 residual 两两 pck2 >= 0.80
  → H1 成立（描述子健康，坐标系失配）
  → 分支 C2a：冻结 descriptor/localizer，只做 S2Looking 决策层
    重建链——P7-1C 式 q 全分辨率重校准（S2Looking train）
    → P7-2 式特征收集 → P7-3 协议复刻（新 CSV、新 tau）。
    固有失配由 clean-anchor 监督重新定义"零点"。

否则
  → H2 或混合
  → 分支 C2b：按既有 stage-A 配置在 S2Looking train 重训整套
    配准栈（p7_s2looking_task_aware_input_registration_stage_a.yaml），
    之后同样走 P7-1C → P7-2 → P7-3 链。
```

两个分支都属"dataset-specific training"，论文中不得与直迁移混写；
各自需要新的预注册协议文件后才能运行训练。本文件只决定先写哪份。

## 5. 运行

先上传 `p7_stage_c2_diagnostic_update.zip`，然后：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
unzip -o p7_stage_c2_diagnostic_update.zip
python -m pytest -q
p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
python -u -m tools.diagnose_transfer_offset_consistency \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p7_stage_c_s2looking_transfer.yaml \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/S2Looking \
  --manifest manifests/datasets/s2looking.csv \
  --device cuda \
  --output-dir "reports/p7_stage_c2_offset_diagnostic/$(date -u +%Y%m%dT%H%M%SZ)"
```

输出 `offset_consistency_summary.json`（完整打印在终端）与逐图
`offset_records.csv`。
