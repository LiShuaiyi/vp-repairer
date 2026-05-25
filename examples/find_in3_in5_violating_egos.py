#!/usr/bin/env python3
"""Find ego ids whose trajectories violate R_IN3 and/or R_IN5 in scenarios."""

import argparse
import copy
import csv
import io
import math
import os
import re
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import (
    create_ego_vehicle_param,
    get_evaluation_config,
)
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import (
    IntersectionType,
    RepairerConfiguration,
    ScenarioType,
)


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/highD-cr-scenarios/13_inD/scenarios")
DEFAULT_OUTPUT = Path("output/in3_in5_violating_egos.csv")
RULES = {
    "in3": "R_IN3_hand_draft",
    "in5": "R_IN5",
}
TIMED_OPERATOR_PATTERN = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)

plt.ioff()
plt.show = lambda *args, **kwargs: None


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def normalize_other_ids(other_ids):
    if other_ids in (None, (), []):
        return ""
    if isinstance(other_ids, np.ndarray):
        other_ids = other_ids.tolist()
    if isinstance(other_ids, (list, tuple, set)):
        flattened = []
        for item in other_ids:
            if isinstance(item, (list, tuple, set)):
                flattened.extend(str(v) for v in item)
            else:
                flattened.append(str(item))
        return ";".join(flattened)
    return str(other_ids)


def debug_enabled_for_ego(debug_ego_ids, ego_id):
    return not debug_ego_ids or ego_id in debug_ego_ids


def print_debug(enabled, message):
    if enabled:
        print(f"    [debug] {message}", flush=True)


def align_bound_to_sampling_period(bound, dt, mode):
    if dt <= 0:
        return bound
    ratio = bound / dt
    if math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
        return round(ratio) * dt
    if mode == "floor":
        steps = math.floor(ratio)
    elif mode == "nearest":
        steps = round(ratio)
    else:
        steps = math.ceil(ratio)
    return max(0, steps) * dt


def format_time_bound(value):
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.12g}"


def align_rtamt_bounds_in_rule(rule_str, dt, mode="ceil"):
    changes = []

    def replace(match):
        low = float(match.group("low"))
        high = float(match.group("high"))
        low_aligned = align_bound_to_sampling_period(low, dt, mode)
        high_aligned = align_bound_to_sampling_period(high, dt, mode)
        if not (
            math.isclose(low, low_aligned, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(high, high_aligned, rel_tol=1e-9, abs_tol=1e-9)
        ):
            changes.append((match.group(0), low, high, low_aligned, high_aligned))
        return (
            f"{match.group('op')}["
            f"{format_time_bound(low_aligned)},"
            f"{format_time_bound(high_aligned)}{match.group('unit')}]"
        )

    return TIMED_OPERATOR_PATTERN.sub(replace, rule_str), changes


def align_traffic_rule_bounds(traffic_rules_config, dt, rules_to_align, mode, debug=False):
    traffic_rules = traffic_rules_config["traffic_rules"]
    all_changes = {}
    for rule in rules_to_align:
        if rule not in traffic_rules:
            continue
        aligned_rule, changes = align_rtamt_bounds_in_rule(
            traffic_rules[rule],
            dt,
            mode=mode,
        )
        traffic_rules[rule] = aligned_rule
        if changes:
            all_changes[rule] = changes
            for original, low, high, low_aligned, high_aligned in changes:
                print_debug(
                    debug,
                    (
                        f"aligned {rule} RTAMT bound {original}: "
                        f"[{low},{high}] -> [{low_aligned},{high_aligned}] for dt={dt}"
                    ),
                )
    return all_changes


def make_world(scenario):
    world_config = get_world_config()
    traffic_rules_config = copy.deepcopy(get_traffic_rule_config())
    traffic_rules_param = traffic_rules_config["traffic_rules_param"]

    world_config["scenario"] = traffic_rules_param["mpr_scenario"] = "intersection"
    world_config["intersection_road_network_param"]["map_type"] = "dataset"
    Cfg["common"]["scenario"] = "intersection"
    traffic_rules_param["use_mpr"] = False

    world = World.create_from_scenario(scenario, config=world_config)
    return world, traffic_rules_config


def evaluate_rule(world, traffic_rules_config, ego_id, rule, debug=False, debug_max_steps=10):
    ego_vehicle = world.vehicle_by_id(ego_id)
    if ego_vehicle is None:
        print_debug(debug, f"ego={ego_id} rule={rule}: ego_not_in_world")
        return {
            "violated": False,
            "tv": "",
            "min_robustness": "",
            "other_ids": "",
            "error": "ego_not_in_world",
        }

    ego_vehicle.vehicle_param = create_ego_vehicle_param(
        get_evaluation_config().get("ego_vehicle_param"),
        world.dt,
    )
    evaluator = PropositionRuleEvaluator.create_from_config(
        world,
        ego_id,
        rule,
        traffic_rules_config=traffic_rules_config,
    )

    samples = []
    other_ids_by_step = []
    print_debug(
        debug,
        (
            f"ego={ego_id} rule={rule}: direct evaluator "
            f"start={evaluator.ego_vehicle.start_time}, "
            f"end={evaluator.ego_vehicle.end_time}, dt={world.dt}"
        ),
    )
    for step_idx, time_step in enumerate(
        range(evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1)
    ):
        try:
            raw_value = evaluator.update()
        except Exception as exc:
            print_debug(
                debug,
                (
                    f"ego={ego_id} rule={rule}: update exception at "
                    f"idx={step_idx}, time={time_step}: {type(exc).__name__}: {exc}"
                ),
            )
            raise
        value = finite_float(raw_value)
        samples.append(value)
        other_ids_by_step.append(evaluator.other_ids)
        if debug and step_idx < debug_max_steps:
            print_debug(
                True,
                (
                    f"ego={ego_id} rule={rule}: sample idx={step_idx}, "
                    f"time={time_step}, raw={raw_value}, finite={value}, "
                    f"other_ids={normalize_other_ids(evaluator.other_ids)}"
                ),
            )

    valid_samples = [(idx, value) for idx, value in enumerate(samples) if value is not None]
    if not valid_samples:
        print_debug(debug, f"ego={ego_id} rule={rule}: no_valid_robustness_samples")
        return {
            "violated": False,
            "tv": "",
            "min_robustness": "",
            "other_ids": "",
            "error": "no_valid_robustness_samples",
        }

    min_idx, min_rob = min(valid_samples, key=lambda item: item[1])
    first_negative = next(
        ((idx, value) for idx, value in valid_samples if value < 0),
        None,
    )
    if first_negative is None:
        print_debug(
            debug,
            (
                f"ego={ego_id} rule={rule}: no violation, "
                f"min_idx={min_idx}, min_rob={min_rob:.12g}, "
                f"min_other_ids={normalize_other_ids(other_ids_by_step[min_idx])}"
            ),
        )
        return {
            "violated": False,
            "tv": "",
            "min_robustness": f"{min_rob:.12g}",
            "other_ids": normalize_other_ids(other_ids_by_step[min_idx]),
            "error": "",
        }

    violation_idx, violation_rob = first_negative
    tv = evaluator.ego_vehicle.start_time + violation_idx
    if violation_idx == 0:
        tv = "-inf"

    print_debug(
        debug,
        (
            f"ego={ego_id} rule={rule}: violation, idx={violation_idx}, "
            f"tv={tv}, rob={violation_rob:.12g}, "
            f"other_ids={normalize_other_ids(other_ids_by_step[violation_idx])}"
        ),
    )
    return {
        "violated": True,
        "tv": tv,
        "min_robustness": f"{violation_rob:.12g}",
        "other_ids": normalize_other_ids(other_ids_by_step[violation_idx]),
        "error": "",
    }


def evaluate_rule_with_monitor(
    scenario,
    planning_problem_set,
    scenario_path,
    ego_id,
    rule,
    align_mode="ceil",
    align_bounds=True,
    debug=False,
):
    original_rule = None
    global_traffic_rules_config = get_traffic_rule_config()
    if align_bounds:
        original_rule = global_traffic_rules_config["traffic_rules"].get(rule)
        align_traffic_rule_bounds(
            global_traffic_rules_config,
            scenario.dt,
            [rule],
            align_mode,
            debug=debug,
        )
    config = RepairerConfiguration()
    config.scenario = scenario
    config.planning_problem_set = planning_problem_set
    planning_problem_dict = getattr(planning_problem_set, "planning_problem_dict", {})
    if planning_problem_dict:
        config.planning_problem = next(iter(planning_problem_dict.values()))
    config.general.path_scenarios = str(scenario_path.parent)
    config.general.set_path_scenario(scenario_path.name)
    config.repair.rules = [rule]
    config.repair.ego_id = ego_id
    config.repair.scenario_type = ScenarioType.INTERSECTION
    config.repair.intersection_type = IntersectionType.DATASET
    config.repair.multiproc = False
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False

    try:
        monitor = STLRuleMonitor(config)
        violated_rules = list(getattr(monitor, "_violated_rules", []))
        rule_to_tv = getattr(monitor, "rule_to_tv", {}) or {}
        rule_to_other_id = getattr(monitor, "rule_to_other_id", {}) or {}
    finally:
        if original_rule is not None:
            global_traffic_rules_config["traffic_rules"][rule] = original_rule
    violated = rule in violated_rules
    tv = rule_to_tv.get(rule, "")
    if isinstance(tv, float) and math.isinf(tv):
        tv = ""
    print_debug(
        debug,
        (
            f"ego={ego_id} rule={rule}: monitor fallback "
            f"violated_rules={violated_rules}, rule_to_tv={rule_to_tv}, "
            f"rule_to_other_id={rule_to_other_id}"
        ),
    )
    return {
        "violated": violated,
        "tv": tv,
        "min_robustness": "",
        "other_ids": normalize_other_ids(rule_to_other_id.get(rule, "")),
        "error": "",
    }


def scan_scenario(path, quiet, selected_rules, args):
    output_context = (
        redirect_stdout(io.StringIO()) if quiet else nullcontext()
    )
    error_context = (
        redirect_stderr(io.StringIO()) if quiet else nullcontext()
    )

    with output_context, error_context:
        scenario, planning_problem_set = CommonRoadFileReader(str(path)).open(True)
        world, traffic_rules_config = make_world(scenario)
        if args.align_rtamt_bounds:
            align_traffic_rule_bounds(
                traffic_rules_config,
                world.dt,
                [RULES[short_rule] for short_rule in selected_rules],
                args.align_rtamt_mode,
                debug=args.debug_internal,
            )
        ego_ids = sorted(world.vehicle_ids())

    rows = []
    debug_rows = []
    for ego_id in ego_ids:
        rule_results = {}
        for short_rule in selected_rules:
            rule_name = RULES[short_rule]
            debug_this = args.debug_internal and debug_enabled_for_ego(args.debug_ego_id, ego_id)
            rule_output_context = nullcontext() if debug_this else output_context
            rule_error_context = nullcontext() if debug_this else error_context
            print_debug(debug_this, f"scenario={path.name}, ego={ego_id}, rule={rule_name}")
            try:
                with rule_output_context, rule_error_context:
                    rule_results[short_rule] = evaluate_rule(
                        world,
                        traffic_rules_config,
                        ego_id,
                        rule_name,
                        debug=debug_this,
                        debug_max_steps=args.debug_max_steps,
                    )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                print_debug(debug_this, f"ego={ego_id} rule={rule_name}: direct error={error_text}")
                if short_rule == "in5" and "multiple of the sampling period" in str(exc):
                    try:
                        with rule_output_context, rule_error_context:
                            fallback_result = evaluate_rule_with_monitor(
                                scenario,
                                planning_problem_set,
                                path,
                                ego_id,
                                rule_name,
                                align_mode=args.align_rtamt_mode,
                                align_bounds=args.align_rtamt_bounds,
                                debug=debug_this,
                            )
                        fallback_result["error"] = f"fallback_from_{type(exc).__name__}"
                        rule_results[short_rule] = fallback_result
                    except Exception as fallback_exc:
                        rule_results[short_rule] = {
                            "violated": False,
                            "tv": "",
                            "min_robustness": "",
                            "other_ids": "",
                            "error": (
                                f"{error_text}; fallback_failed="
                                f"{type(fallback_exc).__name__}: {fallback_exc}"
                            ),
                        }
                else:
                    rule_results[short_rule] = {
                        "violated": False,
                        "tv": "",
                        "min_robustness": "",
                        "other_ids": "",
                        "error": error_text,
                    }

            if (
                short_rule == "in5"
                and args.debug_monitor_fallback == "always"
                and "fallback_from_" not in str(rule_results[short_rule].get("error", ""))
            ):
                try:
                    with rule_output_context, rule_error_context:
                        fallback_result = evaluate_rule_with_monitor(
                            scenario,
                            planning_problem_set,
                            path,
                            ego_id,
                            rule_name,
                            align_mode=args.align_rtamt_mode,
                            align_bounds=args.align_rtamt_bounds,
                            debug=debug_this,
                        )
                    print_debug(
                        debug_this,
                        (
                            f"ego={ego_id} rule={rule_name}: compare direct="
                            f"{rule_results[short_rule]}, monitor={fallback_result}"
                        ),
                    )
                except Exception as fallback_exc:
                    fallback_result = {
                        "violated": False,
                        "tv": "",
                        "min_robustness": "",
                        "other_ids": "",
                        "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                    }
                    print_debug(
                        debug_this,
                        f"ego={ego_id} rule={rule_name}: monitor debug failed={fallback_result['error']}",
                    )
                debug_rows.append(
                    {
                        "scenario": path.stem,
                        "scenario_path": str(path),
                        "ego_id": ego_id,
                        "rule": rule_name,
                        "direct_violated": rule_results[short_rule]["violated"],
                        "direct_tv": rule_results[short_rule]["tv"],
                        "direct_min_robustness": rule_results[short_rule]["min_robustness"],
                        "direct_other_ids": rule_results[short_rule]["other_ids"],
                        "direct_error": rule_results[short_rule]["error"],
                        "monitor_violated": fallback_result["violated"],
                        "monitor_tv": fallback_result["tv"],
                        "monitor_other_ids": fallback_result["other_ids"],
                        "monitor_error": fallback_result["error"],
                    }
                )
            elif args.debug_output:
                debug_rows.append(
                    {
                        "scenario": path.stem,
                        "scenario_path": str(path),
                        "ego_id": ego_id,
                        "rule": rule_name,
                        "direct_violated": rule_results[short_rule]["violated"],
                        "direct_tv": rule_results[short_rule]["tv"],
                        "direct_min_robustness": rule_results[short_rule]["min_robustness"],
                        "direct_other_ids": rule_results[short_rule]["other_ids"],
                        "direct_error": rule_results[short_rule]["error"],
                        "monitor_violated": "",
                        "monitor_tv": "",
                        "monitor_other_ids": "",
                        "monitor_error": "",
                    }
                )

        if any(result["violated"] for result in rule_results.values()):
            rows.append(
                {
                    "scenario": path.stem,
                    "scenario_path": str(path),
                    "ego_id": ego_id,
                    "violates_in3": rule_results.get("in3", {}).get("violated", ""),
                    "in3_tv": rule_results.get("in3", {}).get("tv", ""),
                    "in3_other_ids": rule_results.get("in3", {}).get("other_ids", ""),
                    "in3_min_robustness": rule_results.get("in3", {}).get("min_robustness", ""),
                    "in3_error": rule_results.get("in3", {}).get("error", ""),
                    "violates_in5": rule_results.get("in5", {}).get("violated", ""),
                    "in5_tv": rule_results.get("in5", {}).get("tv", ""),
                    "in5_other_ids": rule_results.get("in5", {}).get("other_ids", ""),
                    "in5_min_robustness": rule_results.get("in5", {}).get("min_robustness", ""),
                    "in5_error": rule_results.get("in5", {}).get("error", ""),
                }
            )

    return rows, len(ego_ids), debug_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan CommonRoad inD scenarios for R_IN3/R_IN5 violating ego ids.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help=f"Directory containing scenario XML files. Default: {DEFAULT_SCENARIO_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV path for results. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=None,
        help="CSV path for scenarios that could not be checked. Default: <output>.failed.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N XML files, useful for a quick smoke test.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based scenario index to start from after sorting. Example: 41 skips the first 40.",
    )
    parser.add_argument(
        "--pattern",
        default="*.xml",
        help="Glob pattern under --scenario-dir. Default: *.xml",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Glob pattern to skip after matching. Can be repeated.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show noisy monitor/parser output while scanning.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush CSV files and print a progress summary every N scenarios. Default: 20.",
    )
    parser.add_argument(
        "--rules",
        default="in3,in5",
        help="Comma-separated subset of rules to evaluate: in3,in5. Default: in3,in5",
    )
    parser.add_argument(
        "--debug-internal",
        action="store_true",
        help="Print per-ego/per-rule evaluator internals while scanning.",
    )
    parser.add_argument(
        "--debug-ego-id",
        action="append",
        type=int,
        default=[],
        help="Only print debug internals for this ego id. Can be repeated.",
    )
    parser.add_argument(
        "--debug-max-steps",
        type=int,
        default=10,
        help="Number of initial robustness samples to print per ego/rule. Default: 10.",
    )
    parser.add_argument(
        "--debug-monitor-fallback",
        choices=("on-error", "always"),
        default="on-error",
        help="For R_IN5, compare STLRuleMonitor fallback only on RTAMT error or always. Default: on-error.",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-ego/per-rule debug summaries, including non-violations.",
    )
    parser.add_argument(
        "--no-align-rtamt-bounds",
        dest="align_rtamt_bounds",
        action="store_false",
        help="Disable automatic alignment of STL time bounds to the scenario sampling period.",
    )
    parser.add_argument(
        "--align-rtamt-mode",
        choices=("ceil", "nearest", "floor"),
        default="ceil",
        help="How to align non-multiple STL bounds to dt. Default: ceil.",
    )
    parser.set_defaults(align_rtamt_bounds=True)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_rules = [item.strip().lower() for item in args.rules.split(",") if item.strip()]
    invalid_rules = [rule for rule in selected_rules if rule not in RULES]
    if invalid_rules:
        raise ValueError(
            f"Unknown --rules values: {','.join(invalid_rules)}. Allowed: {','.join(RULES.keys())}"
        )
    if not selected_rules:
        raise ValueError("--rules cannot be empty")

    scenario_paths = sorted(args.scenario_dir.glob(args.pattern))
    for exclude_pattern in args.exclude_pattern:
        scenario_paths = [
            path for path in scenario_paths if not path.match(exclude_pattern)
        ]
    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")
    start_offset = args.start_index - 1
    total_matching_scenarios = len(scenario_paths)
    scenario_paths = scenario_paths[start_offset:]
    if args.limit is not None:
        scenario_paths = scenario_paths[: args.limit]

    failed_output = args.failed_output
    if failed_output is None:
        failed_output = args.output.with_suffix(".failed.csv")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed_output.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario",
        "scenario_path",
        "ego_id",
        "violates_in3",
        "in3_tv",
        "in3_other_ids",
        "in3_min_robustness",
        "in3_error",
        "violates_in5",
        "in5_tv",
        "in5_other_ids",
        "in5_min_robustness",
        "in5_error",
    ]

    total_egos = 0
    total_rows = 0
    failed_scenarios = []
    total_debug_rows = 0
    append_output = args.start_index > 1 and args.output.exists()
    append_failed_output = args.start_index > 1 and failed_output.exists()
    append_debug_output = (
        args.debug_output is not None
        and args.start_index > 1
        and args.debug_output.exists()
    )
    output_mode = "a" if append_output else "w"
    failed_output_mode = "a" if append_failed_output else "w"
    debug_output_mode = "a" if append_debug_output else "w"

    debug_file_context = (
        args.debug_output.open(debug_output_mode, newline="")
        if args.debug_output
        else nullcontext()
    )
    with args.output.open(output_mode, newline="") as csv_file, failed_output.open(
        failed_output_mode, newline=""
    ) as failed_file, debug_file_context as debug_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not append_output:
            writer.writeheader()
        failed_writer = csv.DictWriter(
            failed_file,
            fieldnames=["scenario", "scenario_path", "error"],
        )
        if not append_failed_output:
            failed_writer.writeheader()
        debug_writer = None
        if args.debug_output:
            debug_fieldnames = [
                "scenario",
                "scenario_path",
                "ego_id",
                "rule",
                "direct_violated",
                "direct_tv",
                "direct_min_robustness",
                "direct_other_ids",
                "direct_error",
                "monitor_violated",
                "monitor_tv",
                "monitor_other_ids",
                "monitor_error",
            ]
            debug_writer = csv.DictWriter(debug_file, fieldnames=debug_fieldnames)
            if not append_debug_output:
                debug_writer.writeheader()

        for local_idx, scenario_path in enumerate(scenario_paths, start=1):
            idx = start_offset + local_idx
            print(
                f"[{idx}/{total_matching_scenarios}] {scenario_path.name}",
                flush=True,
            )
            try:
                rows, ego_count, debug_rows = scan_scenario(
                    scenario_path,
                    quiet=not args.verbose,
                    selected_rules=selected_rules,
                    args=args,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failed_scenarios.append((scenario_path.name, error))
                failed_writer.writerow(
                    {
                        "scenario": scenario_path.stem,
                        "scenario_path": str(scenario_path),
                        "error": error,
                    }
                )
                plt.close("all")
                failed_file.flush()
                if args.flush_every > 0 and local_idx % args.flush_every == 0:
                    csv_file.flush()
                    print(
                        f"  progress: checked_this_run={local_idx}, last_index={idx}, "
                        f"egos={total_egos}, "
                        f"violating_rows={total_rows}, failed={len(failed_scenarios)}",
                        flush=True,
                    )
                continue

            total_egos += ego_count
            total_rows += len(rows)
            total_debug_rows += len(debug_rows)
            writer.writerows(rows)
            csv_file.flush()
            if debug_writer is not None:
                debug_writer.writerows(debug_rows)
                debug_file.flush()
            if rows:
                ego_list = ", ".join(str(row["ego_id"]) for row in rows)
                print(f"  violating ego ids: {ego_list}", flush=True)
            plt.close("all")
            if args.flush_every > 0 and local_idx % args.flush_every == 0:
                failed_file.flush()
                print(
                    f"  progress: checked_this_run={local_idx}, last_index={idx}, "
                    f"egos={total_egos}, "
                    f"violating_rows={total_rows}, failed={len(failed_scenarios)}",
                    flush=True,
                )

    print(f"Scanned scenarios this run: {len(scenario_paths)}", flush=True)
    print(f"World ego ids checked: {total_egos}", flush=True)
    print(f"Rows with IN3/IN5 violations: {total_rows}", flush=True)
    print(f"CSV written to: {args.output}", flush=True)
    if args.debug_output:
        print(f"Debug rows written: {total_debug_rows}", flush=True)
        print(f"Debug CSV written to: {args.debug_output}", flush=True)
    if failed_scenarios:
        print(f"Failed scenarios: {len(failed_scenarios)}", flush=True)
        print(f"Failed CSV written to: {failed_output}", flush=True)
        for name, error in failed_scenarios[:20]:
            print(f"  {name}: {error}", flush=True)
        if len(failed_scenarios) > 20:
            print("  ...", flush=True)


if __name__ == "__main__":
    main()
