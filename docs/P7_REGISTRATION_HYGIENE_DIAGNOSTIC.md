# 配准卫生诊断：瓦片级离散度（预注册协议）

> 预注册时间：2026-07-26（C2b-R 止损之后、本诊断运行之前）  
> 性质：纯无标签诊断，不读变化标签、不训练、不设成败门槛，
> 可随时运行；不属于 S2Looking 方法线的重启

## 1. 动机

C2b-R 给出了一个新的、干预性的结论（主记录 63）：定位器忠实估计
**光度位移**，而 S2Looking 的光度位移 = 注入 + 固有；且逐图扣除
clean 估计后的抵消质量随注入幅度单调退化（delta PCK@2：S1 0.884 →
S2 0.790 → S8 0.662）。单一全局平移无法表达的原因只有一个候选：
**固有失配是空间变化的场**，而非逐图常量向量。

该假设可以完全无标签地直接检验：把图像切成互不重叠的瓦片，各自
独立估计 offset。若残余失配接近全局平移，瓦片应彼此一致；若是空间
变化场（视差、侧视、局部配准误差），瓦片必然发散。

## 2. 三重用途

```text
1. 机制验证：证实"固有失配是空间变化场"，为 C2b-R 结论补上
   最后一环，并成为 P7-6 区域级方向的实测动机（不再是推测）。
2. 部署前判据（论文可交付组件）：对任意新数据集只需几分钟无标签
   诊断，即可预判选择性对齐是否well-posed。
3. P7-5 特征原型：多裁剪一致性正是评审 1 第 4.2 节解法 B 的
   第 4 类证据；本诊断先量化其判别力，再决定是否进特征集。
```

## 3. 度量（tools/diagnose_registration_hygiene.py）

对每图输出：整图估计（tile_index = -1）与各瓦片估计。聚合：

```text
median_max_pairwise_tile_distance   瓦片两两最大距离的中位数（主指标）
median_tile_dispersion_radius       瓦片到其质心的平均距离的中位数
fraction_images_max_pairwise_leq_2px 全局平移假设近似成立的图像占比
mean_tile_agreement_within_{1,2}px  瓦片与整图估计一致的平均比例
median_{tile,global}_offset_magnitude 固有幅度对照
```

默认 tile_size 512（1024 图 → 2×2 四瓦片），只跑 clean 场景。

## 4. 预声明的判读（写在运行之前）

```text
低离散（median_max_pairwise <= 2 px 且 agreement@2px >= 0.8）
  → 残余失配近似全局平移；选择性对齐 well-posed；
    预期该数据集上 LEVIR 式增益可复制
高离散（median_max_pairwise >= 4 px 或 agreement@2px <= 0.5）
  → 空间变化场；逐图全局平移在原理上不足；
    该数据集不适合作为本篇的"域内成功"目标，
    正确策略是拒绝对齐 + 区域级方向（future work）
中间区间
  → 不足以单独定性，需配合 oracle headroom 一并判断
```

预期（用于事后检验本判据的预测力，不是门槛）：LEVIR clean 低离散；
S2Looking clean 高离散。若预期不成立，说明该判据本身无效，据实记录。

## 5. 仪器选择（重要）

同一诊断有两种用法，结论含义不同：

```text
迁移仪器（部署判据用途）：统一使用 LEVIR P7C 定位器跑所有数据集，
  apples-to-apples；但"高离散"结论对该数据集是歧义的
  （可能是描述子域差，也可能是真的空间变化场）。
适配仪器（机制用途）：对 S2Looking 使用 C2b-R 重训后的定位器，
  描述子已适配该域，此时"高离散"直接支持空间变化场结论。
```

因此：LEVIR 与新数据集用 LEVIR 定位器；S2Looking 同时用两种仪器跑，
两组数字都记录。

## 6. 运行

LEVIR（对照基线，LEVIR 定位器）：

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
p7c="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
P7_CKPT="$(cat "$p7c/CHECKPOINT")"
tag="$(date -u +%Y%m%dT%H%M%SZ)"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$P7_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --manifest manifests/datasets/levir_cd.csv \
  --split val --label levir_levir_localizer \
  --device cuda \
  --output-dir "reports/p7_hygiene/$tag/levir"
```

S2Looking（迁移仪器 + 适配仪器各一次）：

```bash
c2br="$(cat reports/c2b_r_stage_a_s2looking/LATEST)"
C2BR_CKPT="$(cat "$c2br/CHECKPOINT")"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$P7_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/S2Looking \
  --manifest manifests/datasets/s2looking.csv \
  --split val --label s2looking_levir_localizer \
  --max-samples 200 --device cuda \
  --output-dir "reports/p7_hygiene/$tag/s2looking_levir_localizer"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$C2BR_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/S2Looking \
  --manifest manifests/datasets/s2looking.csv \
  --split val --label s2looking_adapted_localizer \
  --max-samples 200 --device cuda \
  --output-dir "reports/p7_hygiene/$tag/s2looking_adapted_localizer"
```

新数据集（如 WHU）到位后用同一条命令、换 dataset-root/manifest/label，
作为投入 P3b 训练之前的 go/no-go 参考。
