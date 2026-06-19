#!/usr/bin/env python3
"""Generate IN3/IN5 violations by speeding up existing ego trajectories."""

import argparse
import copy
import csv
import io
import math
import os
import signal
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.scenario.trajectory import Trajectory
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import (
    create_ego_vehicle_param,
    get_evaluation_config,
)
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024")
DEFAULT_OUTPUT_DIR = Path("output/generated_in3_in5_speedup_scenarios")
DEFAULT_REPORT = Path("output/generated_in3_in5_speedup_scenarios.csv")
RULES = {
    "in3": "R_IN3_hand_draft",
    "in5": "R_IN5",
}

plt.ioff()
plt.show = lambda *args, **kwargs: None


class CandidateTimeout(RuntimeError):
    pass


def _handle_timeout(signum, frame):
    raise CandidateTimeout("timed out")


@contextmanager
def time_limit(seconds):
    if seconds is None or seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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
        values = []
        for item in other_ids:
            if isinstance(item, (list, tuple, set)):
                values.extend(str(value) for value in item)
            else:
                values.append(str(item))
        return ";".join(values)
    return str(other_ids)


def make_world(scenario):
    world_config = get_world_config()
    traffic_rules_config = get_traffic_rule_config()
    traffic_rules_param = traffic_rules_config["traffic_rules_param"]

    world_config["scenario"] = traffic_rules_param["mpr_scenario"] = "intersection"
    world_config["intersection_road_network_param"]["map_type"] = "dataset"
    Cfg["common"]["scenario"] = "intersection"
    traffic_rules_param["use_mpr"] = False
    return World.create_from_scenario(scenario, config=world_config), traffic_rules_config


def evaluate_rule(world, traffic_rules_config, ego_id, rule):
    ego_vehicle = world.vehicle_by_id(ego_id)
    if ego_vehicle is None:
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
    for _ in range(evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1):
        samples.append(finite_float(evaluator.update()))
        other_ids_by_step.append(evaluator.other_ids)

    valid_samples = [(idx, value) for idx, value in enumerate(samples) if value is not None]
    if not valid_samples:
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
    return {
        "violated": True,
        "tv": tv,
        "min_robustness": f"{violation_rob:.12g}",
        "other_ids": normalize_other_ids(other_ids_by_step[violation_idx]),
        "error": "",
    }


def evaluate_rules(scenario, ego_id, quiet=True):
    output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    error_context = redirect_stderr(io.StringIO()) if quiet else nullcontext()
    with output_context, error_context:
        world, traffic_rules_config = make_world(scenario)

    results = {}
    for short_rule, rule in RULES.items():
        try:
            with output_context, error_context:
                results[short_rule] = evaluate_rule(
                    world,
                    traffic_rules_config,
                    ego_id,
                    rule,
                )
        except Exception as exc:
            results[short_rule] = {
                "violated": False,
                "tv": "",
                "min_robustness": "",
                "other_ids": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results, sorted(world.vehicle_ids())


def any_violation(rule_results):
    return any(result["violated"] for result in rule_results.values())


def state_at_time(obstacle, time_step):
    if obstacle.initial_state.time_step == time_step:
        return obstacle.initial_state
    return obstacle.state_at_time(time_step)


def all_obstacle_times(obstacle):
    times = [obstacle.initial_state.time_step]
    trajectory = getattr(getattr(obstacle, "prediction", None), "trajectory", None)
    if trajectory is not None:
        times.extend(state.time_step for state in trajectory.state_list)
    return sorted(set(times))


def copy_state_with_time(source_state, time_step, speed_factor):
    new_state = copy.deepcopy(source_state)
    new_state.time_step = int(time_step)
    if hasattr(new_state, "velocity"):
        new_state.velocity = float(new_state.velocity) * speed_factor
    if hasattr(new_state, "acceleration"):
        new_state.acceleration = float(getattr(new_state, "acceleration", 0.0)) * speed_factor
    return new_state


def speedup_obstacle_trajectory(scenario, ego_id, speed_factor, start_time):
    obstacle = scenario.obstacle_by_id(ego_id)
    if obstacle is None:
        raise ValueError(f"ego id {ego_id} is not in scenario")

    times = all_obstacle_times(obstacle)
    if len(times) < 2:
        raise ValueError(f"ego id {ego_id} has too few trajectory states")

    first_time = min(times)
    last_time = max(times)
    start_time = max(int(start_time), first_time)
    source_by_time = {time_step: state_at_time(obstacle, time_step) for time_step in times}

    new_states = {}
    for time_step in times:
        if time_step < start_time:
            source_time = time_step
        else:
            source_time = start_time + int(round((time_step - start_time) * speed_factor))
            source_time = min(source_time, last_time)
        source_state = source_by_time[source_time]
        new_states[time_step] = copy_state_with_time(source_state, time_step, speed_factor)

    obstacle.initial_state = new_states[first_time]
    trajectory = obstacle.prediction.trajectory
    state_list = [new_states[time_step] for time_step in times if time_step != first_time]
    obstacle.prediction.trajectory = Trajectory(state_list[0].time_step, state_list)
    update_lanelet_assignments(scenario, obstacle)


def update_lanelet_assignments(scenario, obstacle):
    lanelet_network = scenario.lanelet_network
    shape_assignment = {}
    center_assignment = {}
    for time_step in all_obstacle_times(obstacle):
        state = state_at_time(obstacle, time_step)
        position = getattr(state, "position", None)
        orientation = getattr(state, "orientation", 0.0)
        lanelet_ids = set()
        if position is not None:
            try:
                obstacle_shape = obstacle.obstacle_shape.rotate_translate_local(
                    position,
                    orientation,
                )
                lanelet_ids = set(lanelet_network.find_lanelet_by_shape(obstacle_shape))
            except Exception:
                lanelet_ids = set()
            if not lanelet_ids:
                try:
                    lanelet_ids = set(lanelet_network.find_lanelet_by_position([position])[0])
                except Exception:
                    lanelet_ids = set()
        shape_assignment[int(time_step)] = lanelet_ids
        center_assignment[int(time_step)] = set(lanelet_ids)

    initial_time = obstacle.initial_state.time_step
    obstacle.initial_shape_lanelet_ids = set(shape_assignment.get(initial_time, set()))
    obstacle.initial_center_lanelet_ids = set(center_assignment.get(initial_time, set()))
    obstacle.prediction.shape_lanelet_assignment = shape_assignment
    obstacle.prediction.center_lanelet_assignment = center_assignment


def write_scenario(scenario, planning_problem_set, output_path):
    writer = CommonRoadFileWriter(
        scenario,
        planning_problem_set,
        author="generated by commonroad-repairer",
        affiliation="",
        source="ego trajectory speedup perturbation",
        tags={Tag.CRITICAL, Tag.INTERSECTION},
    )
    writer.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)


def format_rule_summary(rule_results):
    parts = []
    for short_rule in ("in3", "in5"):
        result = rule_results[short_rule]
        if result["violated"]:
            parts.append(
                f"{short_rule}:tv={result['tv']}:other={result['other_ids']}:rob={result['min_robustness']}"
            )
    return "|".join(parts)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Traverse scenarios and generate modified XMLs where an existing ego "
            "vehicle violates R_IN3/R_IN5 after speeding up along its own trajectory."
        )
    )
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--pattern", default="*.xml")
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Glob pattern to skip after matching. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-generated", type=int, default=20)
    parser.add_argument("--max-per-scenario", type=int, default=1)
    parser.add_argument("--max-egos-per-scenario", type=int, default=4)
    parser.add_argument("--max-candidates-per-ego", type=int, default=8)
    parser.add_argument("--scenario-timeout", type=float, default=20.0)
    parser.add_argument("--candidate-timeout", type=float, default=15.0)
    parser.add_argument(
        "--speed-factors",
        default="1.5,2.0,2.5,3.0,4.0",
        help="Comma-separated speed/time-compression factors.",
    )
    parser.add_argument(
        "--start-times",
        default="0,3,5,10",
        help="Comma-separated time steps from which speedup starts.",
    )
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--include-original-violations",
        action="store_true",
        help="Also generate variants for egos that already violate before modification.",
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

    speed_factors = parse_float_list(args.speed_factors)
    start_times = parse_int_list(args.start_times)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_scenario",
        "source_path",
        "generated_path",
        "ego_id",
        "speed_factor",
        "start_time",
        "violated_rules",
    ]

    generated_count = 0
    failed_count = 0
    checked_egos = 0
    with args.report.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()

        for local_idx, scenario_path in enumerate(scenario_paths, start=1):
            global_idx = args.start_index + local_idx - 1
            print(f"[{global_idx}/{total_matching}] {scenario_path.name}", flush=True)
            try:
                with time_limit(args.scenario_timeout):
                    scenario, planning_problem_set = CommonRoadFileReader(str(scenario_path)).open(True)
                    _, world_ego_ids = evaluate_rules(
                        scenario,
                        scenario.dynamic_obstacles[0].obstacle_id,
                        quiet=not args.verbose,
                    )
            except Exception as exc:
                failed_count += 1
                print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
                continue

            generated_for_scenario = 0
            for ego_id in world_ego_ids[: args.max_egos_per_scenario]:
                checked_egos += 1
                try:
                    with time_limit(args.candidate_timeout):
                        original_results, _ = evaluate_rules(
                            scenario,
                            ego_id,
                            quiet=not args.verbose,
                        )
                except Exception:
                    continue
                if any_violation(original_results) and not args.include_original_violations:
                    continue

                found_for_ego = False
                candidates_for_ego = 0
                for speed_factor in speed_factors:
                    for start_time in start_times:
                        if candidates_for_ego >= args.max_candidates_per_ego:
                            break
                        candidates_for_ego += 1
                        candidate = copy.deepcopy(scenario)
                        try:
                            with time_limit(args.candidate_timeout):
                                speedup_obstacle_trajectory(
                                    candidate,
                                    ego_id,
                                    speed_factor,
                                    start_time,
                                )
                                modified_results, _ = evaluate_rules(
                                    candidate,
                                    ego_id,
                                    quiet=not args.verbose,
                                )
                        except Exception:
                            plt.close("all")
                            continue

                        if not any_violation(modified_results):
                            plt.close("all")
                            continue

                        output_name = (
                            f"{scenario_path.stem}_ego{ego_id}_speed{speed_factor:g}"
                            f"_t{start_time}_in_violation.xml"
                        )
                        output_path = args.output_dir / output_name
                        write_scenario(candidate, planning_problem_set, output_path)
                        summary = format_rule_summary(modified_results)
                        writer.writerow(
                            {
                                "source_scenario": scenario_path.stem,
                                "source_path": str(scenario_path),
                                "generated_path": str(output_path),
                                "ego_id": ego_id,
                                "speed_factor": speed_factor,
                                "start_time": start_time,
                                "violated_rules": summary,
                            }
                        )
                        report_file.flush()
                        generated_count += 1
                        generated_for_scenario += 1
                        found_for_ego = True
                        print(
                            f"  generated ego={ego_id} factor={speed_factor:g} "
                            f"start={start_time}: {summary}",
                            flush=True,
                        )
                        plt.close("all")
                        break
                    if found_for_ego:
                        break
                    if candidates_for_ego >= args.max_candidates_per_ego:
                        break

                if generated_for_scenario >= args.max_per_scenario:
                    break
                if generated_count >= args.max_generated:
                    break

            if args.flush_every > 0 and local_idx % args.flush_every == 0:
                print(
                    f"  progress: scenarios={local_idx}, checked_egos={checked_egos}, "
                    f"generated={generated_count}, failed={failed_count}",
                    flush=True,
                )
            if generated_count >= args.max_generated:
                break

    print(f"Scanned scenarios this run: {local_idx if scenario_paths else 0}", flush=True)
    print(f"Checked world ego ids: {checked_egos}", flush=True)
    print(f"Generated scenarios: {generated_count}", flush=True)
    print(f"Generated directory: {args.output_dir}", flush=True)
    print(f"Report CSV: {args.report}", flush=True)


if __name__ == "__main__":
    main()
