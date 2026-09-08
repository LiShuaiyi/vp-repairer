# MICP replan：当前基线、固定路径降维版与比较口径

## 1. 当前用于 SMT repair 对比的 MICP replan

实验入口是：

- `comparison/batch_micp_rg1_3.py`：highD，实际构造 `RG123`；
- `comparison/batch_micp_rin1_4.py`：inD，按标签构造 `RIN1` 或 `RIN4`；
- `comparison/micp/traffic_rule_4d.py`：STL specification；
- `comparison/micp/vehicle_models_dt.py`：离散车辆模型；
- `comparison/micp/formula.py`、`constraints.py`：线性谓词和时变几何边界；
- `stlpy.solvers.GurobiMICPSolver`：STL 到 big-M MICP 的转换与 Gurobi 求解。

### 1.1 决策变量和动力学

旧 MICP 已经使用给定路线的 CLCS/Frenet 坐标，但仍允许横向运动，并非固定在 reference path 上：

\[
x_k=[s_k,d_k,v^s_k,v^d_k,a^s_k,a^d_k,j^s_k,j^d_k]^\top,
\qquad
u_k=[q^s_k,q^d_k]^\top .
\]

其中 `q` 是四阶积分模型的输入（按模型矩阵实际含义是 snap）。`dt=0.2 s`，输出为

\[
y_k=[x_k;u_k]\in\mathbb R^{10}.
\]

因此每个时刻有 8 个连续状态、2 个连续输入。STL 中的每个线性谓词还会由 stlpy 引入 binary variable，并用默认 `M=1000` 编码。即使某个叶子只是凸约束，stlpy 0.3.0 的通用递归编码也会给它建 binary；长时域和大量析取通常才是求解时间的主因。

### 1.2 目标函数

batch 使用

\[
\min -\rho+\sum_k x_k^\top Qx_k+u_k^\top Ru_k,
\]

其中

```text
Q = diag(0.1, 0.1, 0.5, 1, 0.1, 0.1, 0.5, 1)
R = I_2
u_min/u_max = -2000/+2000
```

这里惩罚的是绝对 `s/v/...`，不是相对原始轨迹或目标状态的 tracking error；也没有统一加入物理速度、加速度和 jerk 边界。因而它首先是一个“规则可行的全时域重规划”基线，不是与 SMT/VP 完全相同的轨迹质量目标。

### 1.3 规则编码

`RG123` 每个时刻大致编码为

\[
(\neg front\lor\neg sameLane\lor safeDistance)\land collisionFree,
\]

并全程施加非负速度和若干速度上界。`safeDistance` 用 10 个速度采样点的切线做线性化。当前 highD batch 不论输入标签都实例化 `RG123`，同时 monitor 被固定配置为 `R_G1`；仓库中的 clean MICP 表实际上只有 `R_G1`，不能据此声称已经比较了独立 `R_G3`。

`RIN1` 用一个停止线前 1 m 区间表达“进入该区间后下一步仍在区间内”，再叠加纵向碰撞走廊和非倒车约束。

`RIN4` 的当前 MICP 并没有完整编码 target vehicle 的时序优先权关系；在注释所表达的 intended semantics 下，它基本退化为“全程处于冲突区外”，再叠加碰撞走廊和非倒车约束。因此它只能作为现有近似基线，不能等同于 monitor 中完整的 `R_IN4`。

### 1.4 调用和计时

旧 batch 的顺序为：

```text
读取场景/建 World/建 monitor
    -> 构造规则几何和 STL specification
    -> 开始计时
    -> GurobiMICPSolver 初始化（动力学 + STL big-M）
    -> 添加二次代价和输入边界
    -> Solve
    -> 停止计时
```

所以旧 `total_time` 包含 solver model setup 和 solve，但不包含 `GetSpecification()` 的谓词/几何构造。SMT 表中的 `total_time` 是 `SAT reasoning + t_solver.total_runtime`，也排除了场景和 monitor 初始化，但包含其 theory solver 内的 TC/reachability/optimization。两者接近“方法 core time”，仍不是完全相同的阶段边界。

## 2. 旧结果能说明什么

仓库 `evaluation/plot/*clean*` 中成功行的描述统计如下。注意各方法的行数不同，并非严格按 `(scenario_id, ego_id, rule)` 取交集后的 paired comparison。

| 数据/规则 | MICP n / mean / median | SMT repair n / mean / median | 均值比 MICP/SMT |
|---|---:|---:|---:|
| highD `R_G1` | 57 / 1.734 s / 1.741 s | 72 / 0.120 s / 0.118 s | 14.48× |
| inD `R_IN1` | 80 / 0.686 s / 0.678 s | 55 / 0.091 s / 0.092 s | 7.53× |
| inD `R_IN4` | 13 / 2.333 s / 2.380 s | 13 / 0.182 s / 0.151 s | 12.85× |

这些数能支持“旧二维 MICP 明显更慢”，但还不能直接作为与 VP 的最终证据，原因包括 case 未配对、规则近似不完全一致、旧 MICP 未做最终 monitor validation，以及下面这些实现问题。

## 3. 旧基线中需要明确披露的问题

1. `phantom_false(index=1)` 实际构造的是 `-d >= 0`，并不是常假。固定 `d=0` 时它反而成立。当前 `RIN1` 的 3 个以及 `RIN4` 的 27 个 dummy 分支会放空 intended rule，不能机械地保留到降维版。
2. 当前 specification 多处使用 `0..T`，但公开的 stlpy 0.3.0 把传入 `T` 解释为样本数，只创建下标 `0..T-1`。旧 batch 又把 final time step 直接作为 `T`。当前源码与声明依赖存在 off-by-one 不一致；旧结果可能来自本地 patch 过的 stlpy，复现实验必须固定版本或修正口径。
3. `RIN4.GetSpecification()` 把同一个静态冲突区重复计算 `T` 次；虽然这段恰好在旧计时器外，仍是无意义开销。
4. 旧结果把 Gurobi 返回可行/最优视作 replanable，没有把解转换回 CommonRoad 后用同一 traffic-rule monitor 验证。
5. highD clean MICP 只有 `R_G1`；文件名中的 `rg1_3` 不能当成 `R_G3` 实验已经完成。
6. MICP 是从第一个时刻重规划整个 horizon；SMT repair 会搜索可保留前缀的 TC。与当前 VP 比较时应明确 VP 的固定 cutoff 是否也是首帧。

## 4. 已加入的固定 reference path 降维版本

新增代码：

- `fixed_path_micp_comparison/vehicle_models_fixed_path.py`；
- `fixed_path_micp_comparison/fixed_reference.py`；
- `fixed_path_micp_comparison/traffic_rule_fixed_path.py`；
- `fixed_path_micp_comparison/batch_micp_fixed_path.py`；
- `fixed_path_micp_comparison/tests/test_micp_fixed_path.py`。

### 4.1 降维后的系统

\[
\bar x_k=[s_k,v_k,a_k,j_k]^\top,\qquad \bar u_k=[q_k],
\qquad \bar y_k=[s_k,v_k,a_k,j_k,q_k]^\top.
\]

这正是旧 8-state/2-input 模型的纵向 block。横向状态和横向输入从模型中消失，而不是再加 `d=0`、`v_d=0` 等等式，因此每步连续状态和输入都减半。

默认 `--reference-path trajectory`：保留原始 ego trajectory 的几何，再沿已选 route 向前延伸并平滑回到 route center；这样首帧不会被吸到车道中心，且 reference 足够长，可表达制动或加速后的新进度。`--reference-path lane` 则使用旧 MICP 的 lane/route CLCS，便于做消融。

所有几何均投影到同一固定路径：

- rectangle/lane 与 path 相交后变成一维 `s` interval；
- `not same lane` 变成 `s` 在该 interval 外；
- 前车 rear position、安全距离线性化、collision corridor 都使用该 path 的 `s`；
- stop line 用线段中点/有效端点投影为一个 `s_stop`；
- R_IN4 冲突 polygon 与固定 path 相交为 conflict `s` interval。

此外，降维版做了几项逻辑等价或必要修正：

- 删除 intended-false dummy 分支，而不是把错误的 `phantom_false` 带入；
- 4 个同时成立的速度上界取最小值，减少 3 个冗余 binary predicate；
- 静态冲突几何只计算一次；
- `num_steps` 明确定义为信号样本数，所有 STL 下标限制在 `0..num_steps-1`；
- 把 `s` 限制在 reference path 的有效投影域内；
- 求解后重建 Cartesian state，并用同一个 rule monitor 输出 `monitor_compliant` 和 `updated_tv`。

这些优化会让 fixed-path MICP 成为比机械删维更强的对手；如果它仍慢于 VP，结论更有说服力。

### 4.2 新计时字段

新 batch 同时输出：

- `specification_time`：fixed reference、几何和 STL 构造；
- `solver_setup_time`：Gurobi model、STL big-M、代价和边界；
- `solve_wall_time` / `gurobi_runtime`；
- `legacy_total_time = solver_setup_time + solve_wall_time`，用于对接旧 MICP 表；
- `core_total_time = specification_time + legacy_total_time`，建议用于和 VP `core_runtime` 比较；
- `num_variables`、`num_binary_variables`、`num_constraints`；
- `micp_feasible` 表示降维 MICP 近似式有解；`replanability=yes` 只在重建轨迹也通过完整 rule monitor 时给出，避免把 specification mismatch 计作成功 repair；
- `validation_time` 单列且不计入 core time。

## 5. 运行方式

inD：

```bash
python -m fixed_path_micp_comparison.batch_micp_fixed_path \
  --dataset ind \
  --scenario-dir /path/to/ind_scenarios \
  --input evaluation/plot/inD_rin1_4_clean_micp.csv \
  --output comparison/inD_fixed_path_micp.csv \
  --monitor-config /path/to/commonroad-stl-monitor/crmonitor/config.yaml \
  --gurobi-license /home/shuaiyi/gurobi.lic \
  --reference-path trajectory \
  --quiet
```

highD（当前仅接受有明确旧基线映射的 `R_G1`）：

```bash
python -m fixed_path_micp_comparison.batch_micp_fixed_path \
  --dataset highd \
  --scenario-dir /path/to/highD_scenarios \
  --input evaluation/plot/highD_rg1_3_clean_micp.csv \
  --output comparison/highD_fixed_path_micp.csv \
  --monitor-config /path/to/commonroad-stl-monitor/crmonitor/config.yaml \
  --gurobi-license /home/shuaiyi/gurobi.lic \
  --reference-path trajectory \
  --quiet
```

先用 `--limit 1` 做环境和 license smoke test；必要时加 `--time-limit`，但正式统计中所有方法必须使用一致的 timeout 处理规则。

## 6. 与 VP 的推荐实验协议

1. 对 VP 和 fixed-path MICP 的成功结果按 `(scenario_id, ego_id, rule)` 取交集；另行报告各自全量成功率，不能只比较各自成功子集。
2. 两者使用同一原始 trajectory、同一 `dt`、同一 horizon、同一固定 path 构造和同一个最终 monitor validation。
3. 比较 `core_total_time` 与 VP `core_runtime`；monitor validation 两边都单列排除。
4. 每个 case 先 warm up，再重复至少 5 次；报告 paired median、IQR、geometric-mean speedup 和 bootstrap CI，而不只报告总体算术均值。
5. 同时报告 binary/constraint 数随 horizon 的增长。固定路径只减连续维度；RG1 的 safe-distance 切线和逻辑析取仍会产生大量 binary，因此理论上仍很可能比 VP 的 SAT/domain preprocessing + LP 慢。
6. 单独看 `R_IN4`：旧 MICP 的 27 个 dummy predicate 是最大的人为负担，修正后它预计降速最多。如果该规则接近或快于 VP，应如实分规则报告，不应靠旧 dummy 复杂度维持总体结论。

## 7. 当前验证状态

- 4 个降维动力学、谓词投影和 horizon 单测通过；
- 真实 highD `R_G1`、inD `R_IN1`、inD `R_IN4` case 均已成功构造 fixed-path specification；
- 已确认 `/home/shuaiyi/gurobi.lic` 是与当前主机 HostID 匹配的 Gurobi 13 academic license（有效期至 2027-01-13）；batch 会在导入 `gurobipy` 前显式设置该路径，也可用 `--gurobi-license` 覆盖。节点锁定 license 需要在可见主机网卡的环境运行；隔离网卡的命令沙箱会把 HostID 报为 `0`。

用该 license 完成的单 case paired smoke test（`core_total_time` 对 VP `core_total_time`）为：

| 规则 / case | fixed-path MICP | VP | MICP / VP | 最终 monitor |
|---|---:|---:|---:|---|
| `R_G1`, `DEU_LocationALower-11_102_T-1`, ego 16 | 0.9149 s | 0.01079 s | 84.77× | 两者通过 |
| `R_IN1`, `DEU_AachenBendplatz-1_155320_T-5339`, ego 10335 | 0.3403 s | 0.005697 s | 59.73× | MICP 未通过，VP 通过 |
| `R_IN4`, `DEU_AachenFrankenburg-1_265260_T-5279`, ego 10453 | 0.2252 s | 0.005658 s | 39.79× | 两者通过 |

这已经说明删掉横向维度后，MICP 在代表性 paired case 上仍明显慢于 VP；但这里只是功能和方向性的 smoke test，正式结论仍应按第 6 节协议跑完整交集和重复实验。
