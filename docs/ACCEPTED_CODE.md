# CARGO 录用版代码使用说明

对应 Remote Sensing 稿件 remotesensing-4659600。项目根目录包含原有训练、选择器拟合、评估和消融代码。下表说明录用稿补充实验的入口。所有命令均从项目根目录执行。

| 论文内容 | 入口 | 输入或产出 |
| --- | --- | --- |
| 共有的源瓦片抽样及 18 个组合的平均增益区间 | `tools/p33_statistics.py` | 各检测器逐裁块记录、按数据集共享的 bootstrap 抽样 |
| LEVIR-CD 标签坐标系诊断 | `round1_formal.py --stage E3` | 同一评分区域比较原始输出、已知变换重映射输出、已知变换配准输入、估计配准及 CARGO |
| WHU-CD 轻度模型失配 | `round1_formal.py --stage E2` | 原生、L1 平移缩放以及叠加旋转或局部形变 |
| WHU-CD 强度更高的复合扰动 | `round1_stress.py` 和 `round1_stress_protocol.json` | 成功完成的 v2 协议，在共同有效的 2,156 个裁块上评估 |
| WHU-CD 仅缩放、仅旋转、仅局部形变 | `tools/round1_pure_20261105/pure_conditions.py` 和同目录 `protocol.json` | 五个条件，含原生基线，2,552 个裁块。配套 `finish_results.py` 复算逐裁块记录汇总的 F1 |
| WHU-CD 检测器训练种子 | `round1_whu_seeds.py run` | 原始种子 3407，新增 42、2026；固定选择器训练种子。生成原生、L3、混合条件汇总 |
| 原生 S2Looking 检测 | `audit_s2_native_records.py` | 使用既有全测试集逐裁块预测记录，16,000 个裁块、1,000 个源瓦片 |
| 定性案例及图 6 | `round1_formal.py --stage E1`、`--stage E7`；`extend_qualitative_round1.py` | 导出 S2Looking/WHU-CD 案例；图 6 组图程序读取原 P31 归档案例和 E1 的原生 S2Looking 案例 |
| 选择器收敛及阈值 | `plot_selector_original_log.py`、`audit_original_thresholds.py` | 图 4 来自原始训练日志；18 个阈值来自原有选择器与测试记录 |
| 样本数、模型参数量、划分和环境 | `collect_round1_evidence.py`、`docs/accepted_supplement/`、`manifests/datasets/` | 五个数据集清单与最终补充材料一致，包含数据角色哈希规则和计时环境 |
| 校准单变量门控、错误选择分析及完整调用计时 | `tools/p33_calibrated_gate_readout.py`、`tools/analyse_selector_errors.py`、`tools/benchmark_m5_cost.py` | 原始训练、留出及测试记录，或相应冻结检查点。DiffRegCD 计时另见 `tools/p31_diffregcd_timing.py` |

未写目录的脚本位于 `tools/round1_revision_20261103/`。基础配准器、检测器及门控训练入口见根目录 README。

## 环境与输入

复算模型需要原始影像、DINOv3 预训练权重、相应检测器和选择器检查点。基于历史记录的统计还需要相应训练日志或逐裁块预测记录。外部基线按各自的依赖和权重配置。本 ZIP 提供项目代码、配置和清单；以上大型资产另行准备。

原实验在 Linux 上执行。项目环境安装方法见根目录 README；绘图还需要 matplotlib。轮次实验使用的 PyTorch 为 2.5.1+cu124。为能对照已有运行指纹，正式评估脚本与协议保留运行时的字节内容。原工程根目录为 `/root/autodl-tmp/rs_cd_work/rs_cd_project`，数据、预训练权重和部分第三方项目位于其同级目录。移植时可以在原路径建立指向实际目录的软链接。特别是纯扰动入口通过原绝对路径加载两个固定哈希的几何脚本。先在相应 protocol JSON 中核对资产路径和 SHA-256，再开始推理。

## 独立运行示例

```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=tools/round1_revision_20261103
P=tools/round1_pure_20261105
OUT=reports/round1_20261103
PY=.venv/bin/python

# 可先在 CPU 上检查重映射方向、前后向坐标、边界及 F1 计数
"$PY" "$R/check_geometry_local.py"

# 四个基础实验。每个实验都可单独运行
for stage in E3 E2 E1 E7; do
  "$PY" "$R/round1_formal.py" --project "$PWD" --protocol "$R/round1_protocol.json" --output "$OUT" --stage "$stage" || exit 1
done

# 复合扰动与纯扰动写入不同目录
"$PY" "$R/round1_stress.py" --project "$PWD" --protocol "$R/round1_stress_protocol.json" --output "$OUT/model_mismatch_stress_v2"
"$PY" "$P/pure_conditions.py" --project "$PWD" --protocol "$P/protocol.json" --output reports/round1_20261105/pure_conditions
"$PY" "$P/finish_results.py" reports/round1_20261105/pure_conditions completed

# 单独执行两次新种子的训练和选择器评估
"$PY" "$R/round1_whu_seeds.py" run

# 用已有预测计数复算原生 S2Looking
"$PY" "$R/audit_s2_native_records.py"
```

复合与纯扰动的裁块集合和计分掩膜不同，各自使用同一张表内的基线。选择器损失曲线可用 `plot_selector_original_log.py --log ... --summary ... --output ...` 重绘；S1 已保存曲线上实际记录的数值。交付中的运行示例只启动评估或训练，机器电源由使用者管理。

### 原生 S2Looking 历史记录版本

同名的 S2Looking 预测在不同时期的工程快照中存在不同版本。录用稿使用的 `records_S2Looking.csv` 的 SHA-256 是 `b2b4c4f51ac81109b108de7d3a4cb65c4d33efc1a81a4e01d4b64b53bc9955c1`；配套 `test_S2Looking.json` 是 `1762ea51de8bfae7c86b4157e8afc564b3387110d81be47aa822caf2dd8ea3bb`。在这份记录上运行本包的原生审计程序，16,000 个裁块中有 751 个采用配准，Raw、Always、CARGO 的 F1 分别为 63.51503%、62.36179%、63.98574%。用原始记录进行复算时，可用这组指纹确定版本。
