# `monitor_wrapper.py` 相对 `original_monitor_wrapper.py` 的改动说明

本文档对比 `original_monitor_wrapper.py` 到 `monitor_wrapper.py` 的主要改动，重点说明新版本增加了哪些兼容逻辑、修复了哪些边界情况，以及这些改动对调用方的影响。

## 总览

`monitor_wrapper.py` 在原有 STL 规则监控器基础上主要做了四类增强：

1. 增加 MONA 场景的特殊兼容处理，使 MONA 地图可以复用 interstate 场景处理流程。
2. 增强 proposition/robustness 提取逻辑，避免某些 `crmonitor` monitor 结构下取不到命题值。
3. 修复 `time-to-violation` 为 `-inf` 或相对时间时的索引问题。
4. 增加若干容错逻辑，避免缺失车辆 ID、缺失 lanelet assignment、空序列等情况直接导致异常。

文件规模从 921 行增加到 1187 行，新增逻辑主要集中在 `STLRuleMonitor` 类中。

## 新增 MONA 场景处理

新版本新增了以下辅助方法：

- `_is_mona_scenario`
- `_prepare_mona_scenario_for_interstate`
- `_validate_mona_lanelet_assignments`
- `_patch_mona_large_step_clcs`
- `_create_world_for_config`

原始版本在初始化 world 时直接调用：

```python
World.create_from_scenario(config.scenario, config=world_config)
```

新版本改为：

```python
self._world = self._create_world_for_config(config, world_config)
```

如果场景 ID、source 或场景路径中包含 `MONA`/`mona`，新版本会先复制一份 scenario，然后做以下归一化：

- 清空 lanelet network 中的 intersection 信息。
- 从 lanelet 类型中移除 `intersection` 类型。
- 规范化 dynamic obstacle 的 `initial_shape_lanelet_ids` 为 `set`。
- 规范化或补全 prediction 的 `shape_lanelet_assignment`。
- 规范化或补全 prediction 的 `center_lanelet_assignment`。
- 创建 world 后为 lane 补充 `clcs_large_step`、`clcs_left_large_step`、`clcs_right_large_step` 属性。

这些改动的目的，是让 MONA 地图不走 intersection 相关逻辑，而是强制进入 interstate 风格的 pipeline，并补齐后续规则评估需要的 lanelet assignment 和 large-step CLCS 字段。

## Proposition 提取变得更稳健

新版本新增：

- `_extract_monitor_propositions`
- `_safe_get_propositions_all`

原始版本直接调用：

```python
evaluator.get_propositions_all()
```

新版本改为：

```python
self._safe_get_propositions_all(evaluator)
```

新逻辑会在 `evaluator._eval_visitor.all_props_all_ids` 为空时，从不同 monitor 结构中兜底提取 `_propositions`。它兼容以下结构：

- 节点本身有 `monitor`
- 节点有 `monitors`
- 节点有 `children`
- 有 `last_selected`
- 指定了 preferred vehicle id

这可以避免某些 evaluator 已经完成更新但 `get_propositions_all()` 返回结构不完整时，后续 `all_props_all_ids` 缺失导致计算中断。

## 车辆 ID 和命题序列增加容错

新版本新增：

- `_resolve_rule_other_id`
- `_safe_prop_sequence`

`_resolve_rule_other_id` 用来处理 `other_ids` 为空、不是列表/元组，或者规则为 `R_G3` 的特殊情况。原始版本默认从嵌套结构中取第一个元素，数据结构稍有变化就可能取错或抛异常。

`_safe_prop_sequence` 用来从 `prop_eval_by_vehicle` 中选择合适的 robustness 序列。优先级为：

1. 当前 `other_id`
2. ego vehicle id
3. `None`
4. 字典中的任意一个可用序列
5. 空列表

因此 `_update_prop_nodes` 中不再直接写：

```python
seq = self.all_props_all_ids_all[i][prop_name][other_id]
```

而是先安全获取 `prop_eval_by_vehicle`，再兜底选择序列。这样可以降低 KeyError 风险。

## `-inf` time-to-violation 的处理

原始版本在 `_cal_tv_def` 中，如果初始时间步已经违反规则，会直接返回：

```python
return -math.inf
```

但构造函数后续期望 `_cal_tv_def()` 返回五个值：

```python
violated_rules, min_rule_idx, tv, rule_to_tv, rule_to_other_id
```

新版本修复了这一点：当第一个时间步已经违例时，也会构造完整返回值：

- `violated_rules`
- `min_rule_idx`
- `_tv = -math.inf`
- `rule_to_tv`
- `rule_to_other_id`

同时，`prop_robust_ttv`、`_update_prop_nodes` 中的多个索引计算都增加了对 `-math.inf` 的判断。如果某条规则的 TV 是 `-inf`，索引统一回退到 `0`，避免出现 `-inf - start_time_step` 作为数组索引。

## TV 计算从绝对时间改为相对时间

在 `_cal_tv_def` 中，原始版本计算 proposition 的 satisfaction/violation time 时使用：

```python
idx - start_index - self._start_time_step
```

新版本改为：

```python
idx - start_index
```

也就是说，TV 列表现在表达的是相对于有效 evaluation 序列起点的时间，而不是再额外扣除 ego vehicle 的 `start_time_step`。

同时，原始版本有一处逻辑会把 `tv == self._start_time_step` 替换为 `inf`；新版本改为只在 `tv < 0` 时替换为 `inf`。结合前面的相对时间改动，这表示 `tv = 0` 会被保留下来，表示当前有效评估点已经触发违例；从修复角度看，这通常仍然属于无法通过后续时间步修复的情况，而不是无穷远处才会发生的违例。

## `_update_prop_nodes` 的索引和空序列保护

新版本在更新 proposition node 的 `ttv_value` 和 `ttv_h_min` 时增加了几处保护：

- 如果规则 TV 是 `-inf`，robustness/predicate 索引使用 `0`。
- 获取命题序列时使用 `_safe_prop_sequence`。
- 使用 `rule_to_tv[rule]` 计算 `tv_index`，不再直接依赖全局 `self._tv`。
- 对 `seq` 为空、索引越界的情况返回默认值 `-1`。

这些改动让多规则场景下每条规则可以使用自己的 TV，不再总是使用全局最小 TV 来切片 proposition robustness。

## `evaluate_initially` 中 proposition 获取方式变化

新版本在单进程和多进程逻辑中，都将：

```python
evaluator.get_propositions_all()
```

替换为：

```python
self._safe_get_propositions_all(evaluator)
```

因此初始评估阶段收集 `all_props_all_ids` 和 `all_rules_all_pre` 时会使用新增的兜底逻辑。这是 proposition 提取兼容性增强的主要接入点。

## 行为影响

对外接口基本保持不变，`STLRuleMonitor` 的构造方式、属性访问方式没有明显变化。主要行为差异包括：

- MONA 场景会被自动识别并特殊预处理。
- 初始时间步违例时，monitor 不再因为 `_cal_tv_def()` 返回值形状不匹配而失败。
- `time-to-violation` 更偏向相对时间语义。
- proposition node 的 robustness 更新更依赖每条 rule 自己的 TV。
- 当 monitor 内部 proposition 结构或车辆 ID 不完整时，新版本更倾向于使用可用兜底值继续执行。

## 需要注意的点

- MONA 场景识别依赖字符串包含 `MONA` 或 `mona`，如果路径或 scenario id 命名不包含该标记，就不会触发特殊处理。
- `_prepare_mona_scenario_for_interstate` 使用 `copy.deepcopy(config.scenario)`，不会直接修改原始 `config.scenario`，但会增加初始化时的拷贝成本。
- 多处异常被兜底处理后，程序更不容易中断，但也可能掩盖 lanelet assignment 计算失败的根因。
- `tv` 从绝对时间调整为相对时间后，下游如果假设它等于 scenario 的绝对 time step，需要重新确认使用方式。
