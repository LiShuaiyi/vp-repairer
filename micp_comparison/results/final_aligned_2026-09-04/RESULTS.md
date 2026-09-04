# 对齐口径 MICP 结果（2026-09-04）

## 配置

- MICP rule semantics: `lin2025`
- STL encoding: `standard`
- Gurobi threads: 1 per process
- time limit: 30 s per case
- success: repaired trajectory passes the selected VP `STLRuleMonitor`
- time: specification construction + solver setup + solve (`core_total_time`)

## 结果

| Cohort | Cases | Feasible | Success | Success rate | Mean (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_G1 highD | 100 | 100 | 98 | 98.0% | 0.852 | 0.877 | 0.937 |
| R_G1 MONA | 93 | 93 | 93 | 100.0% | 0.738 | 0.759 | 0.828 |
| **R_G1 highD + MONA** | **193** | **193** | **191** | **99.0%** | **0.797** | **0.814** | **0.926** |
| R_IN3_hand_draft | 49 | 36 | 16 | 32.7% | 1.526* | 1.407* | 2.661* |

`R_IN3_hand_draft` 的 49 个案例来自完整 R_IN3 表中 `tv` 非空的实际违规
案例，并强制用 hand-draft monitor 验证。一个案例
`DEU_AachenFrankenburg-1_215560_T-5579 / ego 10491` 的轨迹从 time step 4
开始，当前 MICP runner 假定从 0 开始，因此在建立初始状态时发生 `KeyError: 0`。
成功率仍按 49 个案例计（该案例为失败）；带星号的时间只统计有核心时间的 48 个。

两项本次新跑 cohort（MONA 93 + IN3 49）的描述性合计为 109/142（76.8%）；
141 个有计时案例的平均核心时间为 1.007 s。该合计仅用于核对本次新增运行，
不代表所有交通规则的总成功率。

## 原始结果

- `rg1_mona.csv`
- `in3_hand_draft.csv`
- `in3_hand_draft_part1.csv`
- `in3_hand_draft_part2.csv`
- `in3_hand_draft_part3.csv`
