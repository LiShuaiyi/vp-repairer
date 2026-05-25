#!/usr/bin/env python3
"""Fast IN3/IN5 scanner using crmonitor RuleEvaluator boolean evaluation."""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024")
DEFAULT_OUTPUT = Path("output/fast_in3_in5_violations.csv")
DEFAULT_FAILED_OUTPUT = Path("output/fast_in3_in5_failed.csv")
DEFAULT_RULES = ("R_IN3_hand_draft", "R_IN5")


def parse_rules(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def make_world(scenario, map_type):
    world_config = get_world_config()
    traffic_rules_config = get_traffic_rule_config()
    world_config["scenario"] = traffic_rules_config["traffic_rules_param"][
        "mpr_scenario"
    ] = "intersection"
    world_config["intersection_road_network_param"]["map_type"] = map_type
    Cfg["common"]["scenario"] = "intersection"
    traffic_rules_config["traffic_rules_param"]["use_mpr"] = False
    return World.create_from_scenario(scenario, config=world_config), traffic_rules_config


def first_violation_index(values, allow_initial_violation):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if not allow_initial_violation and values[0] < 0:
        return None
    negative_indices = np.flatnonzero(values < 0)
    if negative_indices.size == 0:
        return None
    return int(negative_indices[0])


def evaluate_with_rule_evaluator(world, traffic_rules_config, ego_id, rule, args):
    evaluator = RuleEvaluator.create_from_config(
        world,
        ego_id,
        rule=rule,
        traffic_rules_config=traffic_rules_config,
        use_boolean=args.use_boolean,
    )
    rule_values = evaluator.evaluate()
    violation_index = first_violation_index(
        rule_values,
        args.allow_initial_violation,
    )
    if violation_index is None:
        return None
    return {
        "first_violation_index": violation_index,
        "min_value": float(np.min(rule_values)),
        "num_steps": len(rule_values),
    }


def evaluate_with_proposition_evaluator(world, traffic_rules_config, ego_id, rule, args):
    evaluator = PropositionRuleEvaluator.create_from_config(
        world,
        ego_id,
        rule,
        traffic_rules_config=traffic_rules_config,
    )
    min_value = math.inf
    evaluated_steps = 0
    for index, _ in enumerate(
        range(evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1)
    ):
        value = float(evaluator.update())
        evaluated_steps += 1
        min_value = min(min_value, value)
        if value < 0:
            if index == 0 and not args.allow_initial_violation:
                return None
            return {
                "first_violation_index": index,
                "min_value": value,
                "num_steps": evaluated_steps,
            }
    return None


def scan_scenario(path, args):
    scenario, _ = CommonRoadFileReader(str(path)).open(lanelet_assignment=True)
    world, traffic_rules_config = make_world(scenario, args.map_type)

    rows = []
    for vehicle in world.vehicles:
        ego_id = vehicle.id
        for rule in args.rules:
            try:
                if args.evaluator_mode == "rule":
                    violation = evaluate_with_rule_evaluator(
                        world,
                        traffic_rules_config,
                        ego_id,
                        rule,
                        args,
                    )
                else:
                    violation = evaluate_with_proposition_evaluator(
                        world,
                        traffic_rules_config,
                        ego_id,
                        rule,
                        args,
                    )
            except Exception:
                continue

            if violation is None:
                continue

            rows.append(
                {
                    "scenario_id": str(scenario.scenario_id),
                    "scenario_path": str(path),
                    "ego_id": ego_id,
                    "rule_STL": rule,
                    "first_violation_index": violation["first_violation_index"],
                    "min_value": violation["min_value"],
                    "num_steps": violation["num_steps"],
                }
            )
            print(
                f"  violation scenario={scenario.scenario_id} ego={ego_id} "
                f"rule={rule} first_index={violation['first_violation_index']}",
                flush=True,
            )
    return rows, len(world.vehicles)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast boolean scan for IN3/IN5 violations in CommonRoad inD scenarios.",
    )
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED_OUTPUT)
    parser.add_argument("--pattern", default="*.xml")
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Glob pattern to skip after matching. Can be repeated.",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--map-type", default="dataset")
    parser.add_argument(
        "--evaluator-mode",
        choices=("proposition", "rule"),
        default="proposition",
        help=(
            "proposition is accurate for IN3/IN5 and early-stops on first violation; "
            "rule is closer to inD_evaluator_STL.py but may miss IN3/IN5 cases."
        ),
    )
    parser.add_argument(
        "--use-boolean",
        action="store_true",
        help="Only used with --evaluator-mode rule.",
    )
    parser.add_argument(
        "--rules",
        type=parse_rules,
        default=list(DEFAULT_RULES),
        help="Comma-separated rules. Default: R_IN3_hand_draft,R_IN5",
    )
    parser.add_argument(
        "--allow-initial-violation",
        action="store_true",
        help="Include cases where the rule is already violated at index 0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")

    scenario_paths = sorted(args.scenario_dir.glob(args.pattern))
    for exclude_pattern in args.exclude_pattern:
        scenario_paths = [
            path for path in scenario_paths if not path.match(exclude_pattern)
        ]
    total_matching = len(scenario_paths)
    scenario_paths = scenario_paths[args.start_index - 1 :]
    if args.limit is not None:
        scenario_paths = scenario_paths[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "scenario_path",
        "ego_id",
        "rule_STL",
        "first_violation_index",
        "min_value",
        "num_steps",
    ]
    failed_fieldnames = ["scenario_id", "scenario_path", "error"]

    total_vehicles = 0
    total_rows = 0
    failed_count = 0
    with args.output.open("w", newline="") as output_file, args.failed_output.open(
        "w", newline=""
    ) as failed_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        failed_writer = csv.DictWriter(failed_file, fieldnames=failed_fieldnames)
        failed_writer.writeheader()

        for local_idx, scenario_path in enumerate(scenario_paths, start=1):
            global_idx = args.start_index + local_idx - 1
            print(f"[{global_idx}/{total_matching}] {scenario_path.name}", flush=True)
            try:
                rows, vehicle_count = scan_scenario(scenario_path, args)
            except Exception as exc:
                failed_count += 1
                failed_writer.writerow(
                    {
                        "scenario_id": scenario_path.stem,
                        "scenario_path": str(scenario_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                failed_file.flush()
                continue

            total_vehicles += vehicle_count
            total_rows += len(rows)
            writer.writerows(rows)
            output_file.flush()

            if args.flush_every > 0 and local_idx % args.flush_every == 0:
                failed_file.flush()
                print(
                    f"  progress: scenarios={local_idx}, vehicles={total_vehicles}, "
                    f"violations={total_rows}, failed={failed_count}",
                    flush=True,
                )

    print(f"Scanned scenarios this run: {len(scenario_paths)}", flush=True)
    print(f"World vehicles checked: {total_vehicles}", flush=True)
    print(f"Violation rows: {total_rows}", flush=True)
    print(f"Failed scenarios: {failed_count}", flush=True)
    print(f"Output CSV: {args.output}", flush=True)
    print(f"Failed CSV: {args.failed_output}", flush=True)


if __name__ == "__main__":
    main()
