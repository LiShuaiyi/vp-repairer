# 二维 MICP replan 对比实现

本目录独立实现 VP repair 对比所需的 MICP baseline，不修改 `crrepairer/` 或
现有 `comparison/` 下的源码。

MICP 在路线 CLCS 中同时优化 longitudinal 与 lateral 状态；reference path 仅作为
坐标图，不约束 `d=0`，因此不是 fixed-reference-path 速度规划。计划适配项目 VP
实验中的 `R_G1`、`R_G2`、`R_G3`、`R_G1+R_G3`、`R_IN1`、`R_IN3`、
`R_IN4`、`R_IN5`，并对每条求解轨迹使用原始 `STLRuleMonitor` 做最终验证。

动力学采用 Lin2025 仓库的完整二维 position--velocity--acceleration--jerk 链：
优化状态为 `[s,d,v_s,v_d,a_s,a_d,j_s,j_d]`，控制为 jerk 链末端输入，谓词输出为
10 维。规则默认保留原始完整时间范围和原始辅助公式树。

目录结构：

- `rules.py`：独立的二维 MICP 规则编码；
- `fewer_binary_solver.py`：Kurtz--Lin logarithmic SOS1 编码；
- `runner.py`：批量实验、分阶段计时与 monitor 验证；
- `analyze.py`：与 VP 结果的配对统计；
- `tests/`：规则映射和输出口径测试；
- `results/latest_full/`：最新完整结果与报告。

当前 704 个 VP 对齐 case 的完整结果见
[`results/latest_full/RESULTS.md`](results/latest_full/RESULTS.md)。旧回归、smoke、
分片和诊断结果均已清理。实验场景统一保存在
`scenarios/experiment_scenarios/`。

runner 默认使用 `standard`（原始 `stlpy.GurobiMICPSolver`）作为 Lin2025
复现基线。可通过 `--encoding fewer_binary` 在完全相同的动力学、规则和时域上
启用 logarithmic SOS1 编码；该选项只改变整数编码，不得再与简化模型混用。

正式 Lin2025 对比中的 R_G2 使用默认的 `--rule-semantics lin2025`，并在优化和
最终验证中保持初始违反对应的固定车辆绑定。逐时刻重新选择前车的实验版本已删除。

默认 Gurobi license 是仓库 Docker 配置中的 academic WLS license；它需要访问
`token.gurobi.com`。`repair-autoware/lib/gurobi.lic` 是 size-limited license，
只能用于小型单元测试，不能运行真实轨迹回归。

```bash
GRB_LICENSE_FILE=/data_linux/planning-sim/repairer/commonroad-repairer-vp/autoware-repair-docker/gurobi.lic \
/data_linux/conda-envs/repair-autoware/bin/python \
  -m micp_comparison.runner --dataset highd \
  --input evaluation/config/vp_temporal_full/vp_repairer_rg1_batch_result_updated.csv \
  --scenario-dir scenarios/experiment_scenarios/highd \
  --output /tmp/micp_rg1.csv --rule R_G1 --quiet
```

并行全量回归可用 `micp_comparison.regression` 的 `--all-cases`、`--workers`
和 `--rules` 参数；批处理会强制使用非交互 Matplotlib 后端，不弹出窗口。
