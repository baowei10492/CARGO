# P24 WHU-CD：RoMa 直接进入 Gate 1（F1 判据）预注册协议（2026-09-16）

> 授权：用户 2026-09-16 在 P23 筛查止损、只读诊断之后选定「另立协议，用 RoMa 直接跑以 F1 为准的 Gate 1」。
> 本文件与 `configs/p24_whu_roma_gate1.yaml` 在任何 P24 运行之前写定；看到结果后不得修改任何设置。
> 纪律：P12/P13、P21、P22、P23 的冻结产物与判定一字不动；全程不打开 WHU test 栅格；不训练、不微调任何定位器。

---

## 0. 缘由与必须披露的事实

```text
P23 筛查（主记录 §188）：预注册判定 stop_no_candidate_tracks —— RoMa 在 val 条带 B 族原像素 PCK@2 仅 0.50（判据 >= 0.80）。
  该判定保持不变。
事后只读诊断（§188.2，本协议写于看到诊断之后）：
  WHU 两期影像本有约 2 px 的稳定固有错位（train 区 RoMa clean 偏移中位 1.79 px，60/60 块的残差在各档间稳定）；
  筛查以注入量为真值，把这部分算作误差。按相对 clean 估计的差分口径，RoMa 在 train 区全部跟住，
  val 条带约 79–80% 跟住；真失败集中在 val 行 11–21 与变化占比 >= 15% 的窗口，且多为未收敛
为什么仍以 Gate 1 为准：F1 判据不依赖「以什么为定位真值」，直接回答「用 RoMa 校正后，检测能追回多少」
必须披露：
  1. P24 是在 P23 筛查失败并看过诊断后才写的，属于第二次尝试
  2. val 条带标签用于判定的第二次（第一次为 P22 Gate 1；P23 筛查只为描述列读过变化占比）
  3. 最终结论须由 Gate 2 通过后一次性的 WHU test 支撑；val 上的结论只作为进入下一关的依据
```

## 1. 设计（与 P23 协议 §3 的 Gate 1 相同，只是定位器由本协议指定）

```text
定位器：roma_outdoor，设置全部取自 P23 配置（sha256 b15ef2ee…）：3 倍降采样、采样 5000 对、
  一致性阈值 1.0 工作像素、最少 8 个匹配、搜索先验 ±96 原像素、零位移判定 |偏移| > 6 原像素、
  整数取整后校正、融合局部相关扩展 fused-local-corr 0.1.1
  未收敛时偏移为 0，estimated 与 estimated_gated 即等于 raw（不校正）
工具核对：P24 配置 → P23 配置 sha256 → P22 配置 sha256；P23 筛查目录必须是 20260916T133941Z
  且其判定为 stop_no_candidate_tracks
样本与打分：P22 的 val 条带 174 瓦片、3072 px 窗口、7 个场景、48 px 共用内边界，冻结 P8 WHU DINOv3 检测器
每瓦片 × 7 场景 × 6 路径：
  raw / oracle / p7c_estimated / p7c_gated —— P22 Gate 1 的原计算，同批大小、同顺序
  estimated        RoMa 估计（取整）无条件校正
  estimated_gated  仅当 RoMa 判定「有错位」才校正，否则同 raw
锚点：前四条路径的 F1 必须复现 P22 Gate 1 的表，7 场景 × 4 路径全部 |差| <= 1e-4；
  不通过 → anchor_failed，RoMa 的数字不报告
```

## 2. 判据（与 P22 完全相同，作用于 RoMa 的 estimated）

```text
依次判定：
  no_step_needed          raw clean − raw B-S3 < 0.05
  headroom_insufficient   B-S3 oracle − raw < 0.05
  recovery_insufficient   B-S2 或 B-S3 的恢复率 (estimated − raw) / (oracle − raw) < 0.50
  proceed_to_gate2        以上都不成立
```

## 3. 描述量（不作判据）

```text
estimated 与 estimated_gated 在 clean 上相对 raw 的变化（P7C 为 −17.3 / −13.4 pp）
B-S1 的恢复率；RoMa 在 val 上的定位量（与筛查口径相同）
按 val 行段（0–10、11–21、22–28）分别给出 6 条路径的 F1、RoMa 与 P7C 的恢复率
注意：oracle 撤销的是注入量；RoMa 校正的是「注入 + 约 2 px 固有错位」，校正后的 T2 与检测器训练时见到的
  对齐状态相差约 2 px，因此 estimated 与 oracle 的差不全是定位误差
```

## 4. 通过之后

```text
Gate 2：门槛沿用 P22 §4（clean F1 drop <= 0.001；三档平均增益 >= 0.020；B-S3 增益 >= 0.050；
  B-S3 oracle 恢复率 >= 0.50）。选择器（窗口感知的路由特征、train 特征采集、阈值标定）须另写实现说明并落盘后才运行，
  且须先复现本协议 Gate 1 的 raw / oracle 数值
跨检测器（BIT、CD-Lamba、CDxLSTM 在 WHU 上训练）另写协议
WHU test 栅格：Gate 2 通过后按一次性纪律使用，另写修订
未通过：WHU 线在现成定位器下收尾，按边界写入论文；行 11–21 的失效作为自研定位器的目标
```

## 5. 禁止事项

```text
重跑 Gate 1（运行脚本只允许一次）；看到结果后修改任何设置、门槛或定位器
按 Gate 1 的 F1 改用其他定位器或其他 RoMa 设置后再判
把行段结果当作判据，或只报告有利的行段
打开 WHU test 栅格
```

## 6. 报告口径

```text
论文中如实写出 P23 筛查未通过、事后诊断的发现（固有约 2 px 错位、失败集中区域）、P24 在其后写定，以及 val 的第二次使用
数字同时给出像素与米；结论只能表述为「在物理量与 LEVIR 可比的位移下」
```

## 7. 事前预期（写于运行前，不作判据）

```text
锚点：逐位复现（P23 筛查中 P7C 的定位已逐行复现 P22，本关的检测计算与 P22 相同）
恢复率：行 0–10 与 22–28 大概率高（RoMa 在这些行几乎全部跟住）；行 11–21 低（多为未收敛，校正路径退化为 raw）；
  整体 B-S2 / B-S3 大概率在 0.6–0.8 之间，过 0.50 的可能性较大，但行 11–21 占 val 的 38%，是主要风险
clean：estimated_gated 的掉分应远小于 P7C（RoMa clean 触发 7.5%，P7C 33%）；
  estimated 在 clean 上会按约 2 px 的固有错位去校正，F1 变化方向不确定，量级预计在 ±2 pp 内
```
