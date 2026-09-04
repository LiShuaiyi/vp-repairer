# Lin2025 standard MICP 最终结果（2026-09-04）

> **口径更正（2026-09-04）：** 下表是“旧 MICP 结果中已有案例”的完整回归，
> 不是与 VP repair 输入清单对齐后的全量对比，不能作为最终方法间准确率表。
> 尤其 R_IN1 仅覆盖 16/91 个案例；R_IN3 的 85 个案例包含 VP 完整规则实验中
> 36 个未发生违规、因而未执行修复的 skip 案例。R_IN3 可比的 49 个实际违规
> 案例中 MICP 成功 17 个（34.7%），而不是下表混合口径的 42/85（49.4%）。

本结果用于严格的 Lin2025 MICP 基线，不采用此前实验中的四维直接加速度模型、
monitor-trigger 局部窗口或折叠后的 IN 辅助公式。

## 实验配置

- 动力学：8 维 longitudinal/lateral position--velocity--acceleration--jerk；
- 规则范围：完整优化时域；
- IN 编码：保留 Lin 源码中的几何交区矩形和 phantom 辅助公式树；
- MICP 编码：`standard`，即原始 `stlpy.GurobiMICPSolver`；
- Gurobi：每个规则 worker 1 个线程，单案例上限 30 s；
- 并行：4 个规则级 worker；
- 准确率：求解轨迹经当前 VP `STLRuleMonitor` 验证后的合规率；
- 核心时间：规则构造、solver setup 和 solve 之和，不含 XML 加载及最终 monitor 验证。

## 汇总

| Rule | Cases | Success | Accuracy | Mean (s) | Median (s) | P95 (s) | Mean binary vars | Mean constraints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R_G1 | 100 | 98 | 98.0% | 0.852 | 0.877 | 0.936 | 355.8 | 2170.7 |
| R_G1_R_G3 | 87 | 84 | 96.6% | 1.650 | 1.688 | 1.809 | 707.5 | 3557.1 |
| R_G2 | 69 | 16 | 23.2% | 0.807 | 1.149 | 1.478 | 322.7 | 1969.0 |
| R_G3 | 100 | 100 | 100.0% | 0.848 | 0.861 | 0.908 | 350.0 | 2247.3 |
| R_IN1 | 16 | 0 | 0.0% | 0.431 | 0.430 | 0.503 | 168.8 | 1405.2 |
| R_IN3 | 85 | 42 | 49.4% | 1.344 | 1.253 | 1.560 | 696.8 | 3179.1 |
| R_IN4 | 24 | 10 | 41.7% | 1.393 | 1.388 | 1.612 | 653.4 | 2974.2 |
| R_IN5 | 29 | 12 | 41.4% | 1.381 | 1.386 | 1.496 | 663.5 | 3019.1 |
| **All（旧 MICP 回归口径，不可作最终对比）** | **510** | **362** | **71.0%** | **1.103** | **1.112** | — | — | — |

IN1 的模型规模显著小于 IN3/IN4/IN5，因此其时间低于 1 s；不通过人为增加约束
调整时间。低 R_G2/IN 成功率反映 Lin 规则近似与当前 VP monitor 的差异。

R_IN5 统计已排除 6 个来自 `scenarios/in5_generate/` 的 150-step 长时域变体，
仅保留 29 个标准 20-step 案例。被排除的原始记录单独保存在
`excluded_in5_150step_generated_variants_2026-09-04.csv`，不计入上表。

## 原始 CSV

高速公路规则：

- `lin2025_standard_full_2026-09-04/rg1.csv`
- `lin2025_standard_full_2026-09-04/rg1_rg3.csv`
- `lin2025_standard_full_2026-09-04/rg2.csv`
- `lin2025_standard_full_2026-09-04/rg3.csv`

交叉路口规则：

- `lin2025_exact_in_full_2026-09-04/in1.csv`
- `lin2025_exact_in_full_2026-09-04/in3.csv`
- `lin2025_exact_in_full_2026-09-04/in4.csv`
- `lin2025_exact_in_full_2026-09-04/in5.csv`

`lin_balanced_*`、`lin_simplified_*` 和未完成的中间 smoke 目录不得作为最终
Lin2025 standard 基线引用。
