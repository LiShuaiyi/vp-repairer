"""Repeat paired CommonRoad cases and compare against recorded VP core times."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from .commonroad_runner import (
    _monitor,
    _states_on_reference,
    extract_problem,
    validate,
)
from .author_reference_planner import plan_with_author_engine


FIELDS = (
    "scenario_id", "scenario_path", "ego_id", "rule", "repeats",
    "dv_requested", "dv_actual", "ds_actual", "nodes_median",
    "generated_nodes_median", "rule_evaluations_median",
    "rule_evaluation_time_median_s",
    "ha_preprocess_median_s", "ha_search_median_s", "ha_core_median_s",
    "vp_core_time_s", "ha_over_vp", "monitor_compliant", "validation_time_s",
    "selected_tc_s", "selected_target_id", "validation_criterion",
    "search_state_key",
    "lattice_violation_costs", "error",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dv", type=float, default=1.0)
    parser.add_argument("--a-min", type=float, default=-10.0)
    parser.add_argument("--a-max", type=float, default=5.0)
    parser.add_argument("--max-expansions", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    root = args.manifest.resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for item in csv.DictReader(source):
            row = {field: "" for field in FIELDS}
            row.update({key: item.get(key, "") for key in row})
            row["repeats"] = args.repeats
            try:
                scenario = Path(item["scenario_path"])
                if not scenario.is_absolute():
                    scenario = root / scenario
                ego_id = int(item["ego_id"])
                rule = item["rule"]
                case_dv = float(item.get("dv") or args.dv)
                selected_tc = float(item.get("tc") or 0.0)
                row["dv_requested"] = case_dv
                monitor = _monitor(
                    scenario.resolve(), ego_id, rule, item.get("scenario_type"),
                    item.get("intersection_type") or "dataset",
                )
                preprocessing_times, search_times, cores, nodes = [], [], [], []
                generated_nodes, rule_evaluations, rule_times = [], [], []
                result = reference_lane = None
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    cfg, env, rules, s0, v0, reference_lane = extract_problem(
                        monitor, rule, case_dv, args.a_min, args.a_max,
                        args.max_expansions,
                    )
                    preprocessing = time.perf_counter() - start
                    ego = monitor.world.vehicle_by_id(ego_id)
                    planning_problem = SimpleNamespace(initial_state=ego.states_cr[0])
                    result = plan_with_author_engine(
                        monitor.world.scenario,
                        planning_problem,
                        reference_lane,
                        cfg,
                        env,
                        rules,
                    )
                    preprocessing_times.append(preprocessing)
                    search_times.append(result.runtime_s)
                    cores.append(preprocessing + result.runtime_s)
                    nodes.append(result.expanded_nodes)
                    generated_nodes.append(result.generated_nodes)
                    rule_evaluations.append(result.rule_evaluations)
                    rule_times.append(result.rule_evaluation_time_s)
                states = _states_on_reference(result, reference_lane)
                selected_target_id = monitor.rule_to_other_id.get(rule)
                compliant, validation_time, _ = validate(
                    monitor, ego_id, states, tc=selected_tc,
                    target_id=selected_target_id,
                )
                vp_time = float(item["vp_core_time_s"])
                core_median = statistics.median(cores)
                row.update(
                    {
                        "scenario_path": str(scenario.resolve()),
                        "dv_actual": cfg.dv,
                        "ds_actual": cfg.ds,
                        "nodes_median": statistics.median(nodes),
                        "generated_nodes_median": statistics.median(generated_nodes),
                        "rule_evaluations_median": statistics.median(rule_evaluations),
                        "rule_evaluation_time_median_s": statistics.median(rule_times),
                        "ha_preprocess_median_s": statistics.median(preprocessing_times),
                        "ha_search_median_s": statistics.median(search_times),
                        "ha_core_median_s": core_median,
                        "vp_core_time_s": vp_time,
                        "ha_over_vp": core_median / vp_time,
                        "monitor_compliant": compliant,
                        "validation_time_s": validation_time,
                        "selected_tc_s": selected_tc,
                        "selected_target_id": selected_target_id,
                        "validation_criterion": "selected_tc_target",
                        "search_state_key": "t_s_v_all_rule_memories",
                        "lattice_violation_costs": json.dumps(
                            dict(zip(result.rule_names, result.violation_costs)),
                            sort_keys=True,
                        ),
                    }
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            target.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
