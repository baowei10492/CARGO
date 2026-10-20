# 配准卫生诊断 v2：修正强制选择混淆后的预注册协议

> 预注册时间：2026-07-26（v1 判据失效之后、v2 运行之前）  
> v1 正式状态：**判据无判别力（failed）**，永久保留，见主记录 65

## 1. v1 为什么失效（根因已定位）

v1 在 clean 场景上测量瓦片 offset 离散度。实测（主记录 65.1）：

```text
LEVIR / LEVIR 定位器        median_max_pairwise = 8.246
S2Looking / LEVIR 定位器    median_max_pairwise = 5.657
S2Looking / 适配定位器      median_max_pairwise = 4.000
```

方向与预声明预期完全相反（预期 LEVIR 低、S2Looking 高）。按 v1 判读
"两者都高 → 判据无判别力"，正式记为失效。根因有两条，均已用数值验证：

**（1）强制选择混淆（致命）。** conditional localizer 结构上无法输出
零 offset，只能在非零候选中选一个。clean 对上真实答案是 0，于是每个
瓦片被迫任选一个非零候选，离散度测的是**估计器的强制选择噪声底**，
而不是失配场。且噪声底与真实失配强度**反相关**：真实失配越小、
证据越弱，选择越任意，离散度越大。数值验证：

```text
从最小非零壳层 {(±2,0),(0,±2)} 随机取 4 个候选：
  模拟 median_max_pairwise = 4.000  ← 恰好等于 S2Looking/适配的观测值
加入约 25% 边界（|·|=8）命中：
  模拟 median_max_pairwise = 10.00  ← LEVIR 观测 8.246 落在其间
观测的 8.246 = sqrt(68)（含 magnitude-8 候选），
  5.657 = 4*sqrt(2)（如 (2,2) 与 (-2,-2)），
  4.000 = 最小壳层对角
三个观测值全部落在"任意选择"的预测区间内。
```

同时 median_tile_offset_magnitude 三组分别为 2.0 / 2.83 / 2.0——都在
最小壳层，正是"尽可能接近零"的表现。

**（2）阈值与栅格不匹配。** descriptor stride 2 使候选 offset 位于
2 px 栅格上，任意两个不同候选的最小距离恰为 2.0 px。因此
agreement@1px 与 agreement@2px 在原理上几乎等价（实测 LEVIR
0.5625 vs 0.5742），这两个阈值不携带独立信息。

## 2. v2 的修正

```text
（a）只在**已注入非零位移**的场景上测量（severities 1 2 3）。
    此时瓦片有真实相关信号可锁定，强制选择混淆消失：
    瓦片一致 → 残余失配近似全局平移；
    瓦片发散 → 空间变化场。
（b）记录注入真值，新增误差类指标：
    median_tile_error_magnitude、tile_error_pck2/pck4、
    median_global_error_magnitude。
    这把"瓦片各自是否准确"与"瓦片之间是否一致"分开。
（c）agreement 阈值改为栅格感知的 (2, 4) px。
（d）summary 输出 forced_choice_regime 布尔标志：注入幅度中位数为 0
    时置 true，提示该组数字不可用于场判断（防止本次错误复发）。
（e）clean 场景**不再作为判据输入**；如需运行仅作噪声底参照，
    且必须连同 forced_choice_regime=true 一起报告。
```

## 3. v2 预声明判读（写在运行之前）

主指标：注入场景下的 `median_max_pairwise_tile_distance`
（记作 D）与 `median_tile_error_magnitude`（记作 E）。

```text
空间近似常量（well-posed）：
  D <= 2 px 且 tile_error_pck4 >= 0.8
  → 残余失配可用逐图全局平移表达；该数据集适合本方法
空间变化场：
  D >= 4 px 或 tile_error_pck4 <= 0.5
  → 全局平移原理上不足；正确策略是拒绝对齐，区域级列 future work
中间区间：
  不足以单独定性，需与该数据集的 oracle headroom 一并判断
```

预期（用于事后检验判据预测力，非门槛）：LEVIR 注入场景 D 小、E 小；
S2Looking 注入场景 D 大或 E 大。若该预期再次不成立，则**正式放弃
瓦片离散度作为卫生判据**，改用第 4 节的替代信号，不再修补第三版。

## 4. 已在手的替代卫生信号（无需新工具）

无论 v2 结果如何，下面这个对照已经存在于既有产物中，且分离度很高：

```text
固定使用 LEVIR 定位器，测 clean 对的 q 触发率：
  LEVIR val clean      q = 0.0625   （P7-3 验证表）
  S2Looking val clean  q = 0.672    （Stage C 无标签诊断）
```

约 10 倍差异，方向符合直觉（"配准良好的数据集上，LEVIR 定位器认为
clean 对没有位移"）。局限已知且必须写明：它把描述子域差与真实固有
失配混在一起，因此只能作为**筛查信号**，不能作为机制结论。若 WHU
提供第三个数据点且落在 LEVIR 一侧，该信号即可作为论文的部署前
筛查判据（需要在 WHU 协议中预注册）。

## 5. 运行

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
p7c="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
P7_CKPT="$(cat "$p7c/CHECKPOINT")"
c2br="$(cat reports/c2b_r_stage_a_s2looking/LATEST)"
C2BR_CKPT="$(cat "$c2br/CHECKPOINT")"
tag="$(date -u +%Y%m%dT%H%M%SZ)"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$P7_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/LEVIR-CD \
  --manifest manifests/datasets/levir_cd.csv \
  --split val --label levir_v2 --severities 1 2 3 \
  --device cuda --output-dir "reports/p7_hygiene_v2/$tag/levir"

python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$C2BR_CKPT" \
  --dataset-root /root/autodl-tmp/rs_cd_work/data/S2Looking \
  --manifest manifests/datasets/s2looking.csv \
  --split val --label s2looking_v2_adapted --severities 1 2 3 \
  --max-samples 200 --device cuda \
  --output-dir "reports/p7_hygiene_v2/$tag/s2looking_adapted"
```

`--severities` 现在默认 1 2 3；clean 需显式传 0 且结果会带
`forced_choice_regime: true` 标志。
