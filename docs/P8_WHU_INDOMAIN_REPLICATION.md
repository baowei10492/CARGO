# P8：WHU-CD 作为第二个"域内"数据集（构建与 go/no-go 预注册协议）

> 预注册时间：2026-07-26（数据结构已实测，任何训练之前）  
> 动机：卫生诊断 v2（主记录 67）确立了方法的适用域条件——残余失配
> 需近似全局平移。因此正确的复制实验不是硬啃域外数据集，而是在
> 另一个**域内**数据集上验证可复制性。

## 1. 数据实测结构（服务器已解压）

```text
_raw_whu/Building change detection dataset_add/
  1. The two-period image data/
    2012/{whole_image,splited_images}/{train,test}/{image,label}
    2016/{whole_image,splited_images}/{train,test}/{image,label}
    change_label/{train,test}/change_label.tif
  2. The shape file of the images/{train,test}
```

要点：

```text
whole_image train = 21243 x 15354，test = 11265 x 15354（两幅独立大图
  → 官方 train/test 本身空间分离，予以沿用）
splited_images 为 512x512，但文件名是 0_<扁平序号>，不含空间坐标；
  网格已复原并验证：行优先，train 42 列 x 30 行 = 1260，
  test 23 列 x 30 行 = 690（6/6 逐像素匹配通过）
2012/2016 的 splited label 是**建筑物语义标注**，不是变化标注；
change_label 只有整幅大图版本（mode "1" 二值），没有切好的瓦片。
```

## 2. 构建约定（预注册）

因为变化标注必须自己裁，索性让三个通道用同一套约定从大图统一切，
这样官方边缘补零约定这个未知量被彻底消除：

```text
原点 (0,0)，步长 512，与官方网格同格；
**丢弃不完整的边缘行列**（不补零）——train 右侧余 251 px、下侧余
  506 px，test 右侧余 1 px 全部弃用，保证每个瓦片都是真实影像；
A = 2012 大图裁块，B = 2016 大图裁块，label = change_label 裁块；
标注读入后统一为 {0,255} uint8，非二值直接报错；
三个栅格尺寸不一致直接报错。
```

划分（预注册，不得事后调整）：

```text
沿用官方 train/test 栅格分离；
val 从 train 栅格中按**整列条带**切出，取最后 6 列；
train 与 val 之间丢弃 1 整列（512 px）作为缓冲带，
  保证没有任何 train 瓦片与 val 瓦片空间相邻；
绝不对瓦片做随机划分（同场景相邻瓦片强空间相关 → 严重泄漏）。
```

样本 id 编码空间坐标：`whu_<split>_r<行>c<列>`。这一条是刻意的——
它修掉了 WHU 自带的"扁平序号丢失空间信息"问题，使后续协议（例如
P7-3 复刻的 fit / early-stop / risk-calibration 三分）可以按**列条带**
划分而不是按 id 哈希，从而三个训练侧分区也彼此空间不相邻。
（LEVIR 侧当时按 id 哈希划分，未控制这一点；WHU 上可以做得更严，
差异需在论文中如实说明。）

预期计数（41 列 × 29 行可用网格）：

```text
train 34 列 x 29 行 = 986
buffer 1 列 = 29 个瓦片弃用
val 6 列 x 29 行 = 174
test 22 列 x 29 行 = 638
合计 1798
```

瓦片尺寸保持官方的 512（与已发表 WHU 结果同尺度；位移注入 2/4/8 px
是绝对像素量，不受图幅影响）。

## 3. 投入 P3b 之前的两项无标签 go/no-go 筛查

仪器统一使用 **LEVIR P7C 定位器**（迁移仪器口径，apples-to-apples）。

```text
筛查 1：clean 对的 q 触发率
  对照：LEVIR 0.0625 / S2Looking 0.672
  判读：<= 0.20 → LEVIR 型（绿）；>= 0.50 → S2Looking 型（红）；
        之间 → 歧义
筛查 2：注入场景（S1/S2/S8）的瓦片一致性 D
  （median_max_pairwise_tile_distance，tile_size 256，因图幅 512）
  对照：LEVIR 0.0 / S2Looking 4.0–4.47
  判读：D <= 2 → 绿；D >= 4 → 红；之间 → 歧义
```

**仪器的不对称性（必须写明）**：用 LEVIR 定位器测新数据集，"绿"是
强证据（说明描述子在该域有效**且**固有失配小）；"红"是歧义的
（可能是描述子域差，也可能是真的空间变化场），不能单独定性。

路由（预注册）：

```text
两项皆绿 → 直接进入 P3b 训练，按已验证链条复刻
任一为红或歧义 → 仍训练 P3b（数小时，任何 WHU 结果都需要它），
  随后以**双路径 Oracle headroom** 作为决定性判据：
    Shift-8 input-oracle headroom >= 0.05 → 该数据集存在可恢复空间，
      继续复刻链条；
    < 0.05（S2Looking 量级仅 0.033）→ 停止链条，WHU 与 S2Looking
      同列为适用域边界案例
```

这样最坏情况的投入被限定为 P3b（数小时）+ Oracle（分钟级）。

值得记下的一点：由于适用域条件已经确立，**WHU 的两种结局都有价值**
——绿并通过链条，则论文得到"两个域内成功"；红并止损，则适用域条件
获得第二个独立负例验证。不存在"白跑一趟"的分支。

## 4. 之后的链条（每步各需新预注册协议）

```text
P3b（WHU 检测器）→ 双路径 Oracle → P7-1B/1C → P7-3 复刻
  （建议采纳主记录 57.2 教训改用分位数 tau 规则，并按列条带做
   三分而非 id 哈希）→ 多种子
```

test split 在全部结构冻结之前不得触碰。

## 5. 运行

```bash
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
RAW="/root/autodl-tmp/rs_cd_work/data/_raw_whu/Building change detection dataset_add"
OUT=/root/autodl-tmp/rs_cd_work/data/WHU-CD

python -m pytest -q

# 先干跑核对计数与划分，不写盘
python -u -m tools.build_whu_tiles \
  --raw-root "$RAW" --output-root "$OUT" \
  --tile-size 512 --validation-columns 6 --buffer-columns 1 --dry-run

# 计数与预期一致（986 / 174 / 638）后再真写
python -u -m tools.build_whu_tiles \
  --raw-root "$RAW" --output-root "$OUT" \
  --tile-size 512 --validation-columns 6 --buffer-columns 1

python -m tools.build_manifest \
  --dataset-root "$OUT" \
  --output manifests/datasets/whu_cd.csv \
  --splits train val test

python -m tools.audit_dataset \
  --dataset-root "$OUT" \
  --manifest manifests/datasets/whu_cd.csv \
  --output-json reports/data_audit_WHU-CD.json \
  --output-md reports/data_audit_WHU-CD.md
```

审计通过后跑两项筛查（命令在 go/no-go 阶段给出，均为
registration-only 前向，分钟级）。

---

## 6. 已完成状态与 P3b 阶段（预注册补充，2026-07-26）

### 6.1 构建与审计：完成且经验证

```text
瓦片 1798（train 986 / val 174 / test 638），全部 512x512
change_fraction 0.041997，掩膜取值仅 {0,255}
掩膜像素计数算术闭合：451540223 + 19794689 = 1798 x 512 x 512
审计 passed = True，parent_id_coverage = 1.000，
  cross_split_parent_ids = 0（复合键 raster_split|column，这次是真零）
划分不变量已被独立验证：train 列 000..033、val 列 035..040、
  test 出自另一幅栅格，无任何 parent 跨划分
```

### 6.2 筛查结论：不据其判定，直接走决定性判据

匹配仪器（中心裁 512 / 瓦片 256）对照见主记录 70.7。两点结论：

```text
筛查 2 的预声明门槛在 512/256 配置下失效（LEVIR 自身 S8 的 D 从
  0.000 变为 7.606），记录为"阈值不可迁移"，不得就地改阈值再判；
免阈值的相对排序仍有效且方向一致：tile_error_pck4 在每个场景上
  LEVIR > S2Looking > WHU；clean q 率 0.063 / 0.310 / 0.833。
证据整体指向 WHU 为红，但按本协议第 3 节路由，"红或歧义"一律走
  决定性判据，不在筛查上纠缠。
```

GSD 机制假设（主记录 70.7 第 4 条）：WHU 的 0.075 m GSD 使同一物理
配准误差在像素上放大约 6.7 倍（0.15 m → LEVIR 0.30 px、WHU 2.00 px）。
适用域条件应表述为"固有失配的**像素**大小"，与 GSD 成反比。

### 6.3 P3b 配置：仅因 512 图幅而偏离的项

`configs/p8_whu_dinov3_partial_ft.yaml` 与 LEVIR/S2Looking 的 P3b
配方逐项一致，**唯二偏离**如下，均由 512 图幅（而非调参意图）导致：

```text
training.batch_size  32（LEVIR 8 / S2Looking 128）
evaluation.batch_size 8（LEVIR 2 / S2Looking 4）
```

其余全部保持：trainable_stages 2、decoder 128、lr 3e-4 /
backbone 2e-5、40 epoch × 8192 crops、crop 256、pos_weight 2.0、
positive_crop_probability 0.7、flip/rotate90、grad_clip 1.0。
`save_predictions` 设为 false（1798 张预测无必要落盘）。

不做任何针对 WHU 的超参搜索。若 P3b 的 val F1 明显低于同族数据集，
如实记录，不调参补救——本阶段的目的不是刷 WHU 的检测精度，而是
获得一个可用的冻结专家以测量 headroom。

### 6.4 决定性判据（预注册）

```text
量：输入级 Oracle 逆位移 headroom = F1(oracle) - F1(raw)，WHU val
门槛：Shift-8 headroom >= 0.05
判据的关键性质：只需检测器与已知注入量，**不涉及任何定位器**，
  因此对钳制伪命中、强制选择、瓦片尺寸、阈值迁移等全部仪器问题免疫
参照：LEVIR 0.2141（远超）；S2Looking 0.0330（远不足）
工具：tools/summarize_p8_whu_headroom.py（含 clean F1 一致性校验：
  raw 与 oracle 两次运行的 clean F1 必须逐位相同，否则报错）
路由：
  通过 → 继续链条：双路径 Oracle → P7-1B/1C → P7-3 复刻（各需新协议）
  不通过 → 停止 WHU 链条，WHU 与 S2Looking 并列为适用域边界案例；
    此时适用域条件获得**第二个独立负例**，且 GSD 机制假设获得支持，
    论文形态为"LEVIR 单域内成功 + 两个边界案例 + 可测量的部署判据"
```

### 6.5 运行

```bash
bash scripts/run_p8_whu_p3b.sh
```

流程：pytest → gpu smoke → P3b 训练（40 epoch × 8192 crops，
数小时）→ raw 与 oracle 两次鲁棒性评估（val）→ 固化判据。
产物在 `reports/p8_whu_p3b/<UTC>/`。
