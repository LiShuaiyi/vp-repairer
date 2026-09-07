# VP repair 与二维 sampling replan 对比

本目录独立实现并保存 Lin et al. (2025 TRO) 使用的 sampling baseline。它基于 CommonRoad Reactive Planner，在参考路线的 Frenet 坐标系中同时采样 longitudinal 与 lateral 多项式轨迹；参考路线只定义坐标系，并不把轨迹固定在 `d=0`，因此不是 Halder/Althoff 的 fixed-reference-path 方法。

## 基线实现与严格评测

论文实验依赖一个没有包含在本仓库环境中的历史 Reactive Planner 补丁。为使基线自包含，本目录原样 vendor 了论文期 commit `ab96a839`；它在 cost 排序后逐候选执行碰撞和规则检查。`RuleAwareReactivePlanner` 保留其候选生成、二维 lateral/longitudinal sampling、cost 排序、运动学和碰撞检测，并用完整 STL monitor 修正多规则覆盖与量词 witness 漂移问题。对于含车辆量词的规则，相关车辆在原始 violation 上确定并固定，避免重规划后换一个量词 witness。具体来源见 `vendor/README.md`。

R_IN1 默认使用论文批处理脚本的 velocity-keeping 配置（`--in1-strategy paper_batch`）。`paper_example` 可复现论文单例可视化脚本的零目标速度和 `[0.01, 15]` 速度采样区间；`stop_position` 只保留作敏感性实验，不属于论文批处理配置。

成功（本文中“准确率/成功率”）的严格定义为：planner 返回轨迹，且独立的后验完整规则监控得到 `updated_tv = +inf`。计算时间只比较 method core：VP 使用 `core_total_time`；sampling 使用包含候选规则检查的 `planning_time`。二者均排除场景/monitor 初始化和最终冗余验证。

默认覆盖 VP 的八个规则 cohort：`rg1,rg2,rg3,rg1_rg3,in1,in3,in4,in5`。`in3` 对应项目当前主结果的 `R_IN3_hand_draft`；另可显式运行 `in3_full`。`rg1_mona` 是额外数据集 cohort。

## 运行

快速 smoke test：

```bash
/data_linux/conda-envs/repairverse310_gpu/bin/python \
  sampling_comparison/run_experiment.py --groups rg2 --limit 1 --overwrite
```

完整实验（逐 rule 串行，最稳定）：

```bash
/data_linux/conda-envs/repairverse310_gpu/bin/python \
  sampling_comparison/run_experiment.py --groups all --jobs 1 --overwrite
```

`--jobs 2` 或更高可并行 cohort，但会改变 wall-clock 竞争条件；方法时间使用 `perf_counter`。默认不设置候选检查上限，与论文批处理一致。可用 `--max-rule-candidates N` 做明确预算下的消融实验，但这类成功率不能直接与论文的无上限结果比较。

输出全部位于 `sampling_comparison/results/<run>/`：逐组原始 CSV、日志、逐 case join、汇总 CSV 和 `REPORT.md`。

若 VP cohort 引用了已清理的 `/tmp` inD time-shift 场景，调度器会先四路并行调用官方 dataset converter，并把精确重建的 XML 与 manifest 保存到 `sampling_comparison/generated_scenarios/`。可用 `--skip-prepare` 显式关闭。

## 依赖与复现说明

- 主环境：`/data_linux/conda-envs/repairverse310_gpu/bin/python`
- Reactive Planner：`vendor/reactive_planner_ab96a839/`，来自论文期 commit `ab96a839`（source version 2024.1）。
- benchmark 会先导入主环境的 NumPy/SciPy，再把旧环境路径 append 到 `sys.path`，仅补充 `methodtools`/`wirerope` 等纯 Python 依赖，防止误载 Python 3.8 的 NumPy 二进制。
- 固定 `PYTHONHASHSEED=0`，图形后端为 `Agg`。
- 论文参数依据：`paper_source/content/experiments.tex` 的 sampling-based replanning 小节；除加入 rule monitoring 外保持 Reactive Planner 原始设置。
