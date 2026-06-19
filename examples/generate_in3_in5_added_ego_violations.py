#!/usr/bin/env python3
"""Generate IN3/IN5 violations by adding a new ego vehicle to scenarios."""

import argparse
import copy
import csv
import io
import math
import os
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Tag
from commonroad.scenario.state import ExtendedPMState, InitialState
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
DEFAULT_OUTPUT_DIR = Path("output/generated_in3_in5_added_ego_scenarios")
DEFAULT_REPORT = Path("output/generated_in3_in5_added_ego_scenarios.csv")
RULES = {
    "in3": "R_IN3_hand_draft",
    "in5": "R_IN5",
}

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


def concat_center_vertices(lanelet_network, lanelet_ids):
    points = []
    for lanelet_id in lanelet_ids:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet is None:
            return None
        for point in lanelet.center_vertices:
            point = np.asarray(point, dtype=float)
            if not points or np.linalg.norm(point - points[-1]) > 1e-6:
                points.append(point)
    if len(points) < 2:
        return None
    return np.asarray(points)


def sample_polyline(points, count, start_offset=0.0, end_margin=0.0):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length <= start_offset + end_margin:
        return None

    distances = np.linspace(start_offset, total_length - end_margin, count)
    sampled_points = []
    orientations = []
    for distance in distances:
        seg_idx = np.searchsorted(cumulative, distance, side="right") - 1
        seg_idx = min(max(seg_idx, 0), len(segment_lengths) - 1)
        if segment_lengths[seg_idx] == 0:
            ratio = 0.0
        else:
            ratio = (distance - cumulative[seg_idx]) / segment_lengths[seg_idx]
        point = points[seg_idx] * (1.0 - ratio) + points[seg_idx + 1] * ratio
        tangent = points[seg_idx + 1] - points[seg_idx]
        sampled_points.append(point)
        orientations.append(math.atan2(tangent[1], tangent[0]))
    return sampled_points, orientations, total_length


def make_lanelet_path_candidates(scenario, max_candidates):
    lanelet_network = scenario.lanelet_network
    candidates = []
    seen = set()
    intersection_lanelets = [
        lanelet
        for lanelet in lanelet_network.lanelets
        if LaneletType.INTERSECTION in lanelet.lanelet_type
    ]
    for lanelet in intersection_lanelets:
        predecessors = list(lanelet.predecessor or [])
        successors = list(lanelet.successor or [])
        if predecessors and successors:
            for predecessor in predecessors:
                for successor in successors:
                    path = (predecessor, lanelet.lanelet_id, successor)
                    if path not in seen:
                        candidates.append(list(path))
                        seen.add(path)
        if successors:
            path = (lanelet.lanelet_id, successors[0])
            if path not in seen:
                candidates.append(list(path))
                seen.add(path)
        path = (lanelet.lanelet_id,)
        if path not in seen:
            candidates.append(list(path))
            seen.add(path)
    return candidates[:max_candidates]


def next_added_ego_id(scenario, preferred_id):
    existing_ids = {obstacle.obstacle_id for obstacle in scenario.obstacles}
    ego_id = preferred_id
    while ego_id in existing_ids:
        ego_id += 1
    return ego_id


def add_ego_on_lanelet_path(
    scenario,
    lanelet_path,
    new_ego_id,
    template_obstacle,
    state_count,
    start_offset,
    end_margin,
    speed_scale,
):
    points = concat_center_vertices(scenario.lanelet_network, lanelet_path)
    if points is None:
        return False
    sampled = sample_polyline(points, state_count, start_offset, end_margin)
    if sampled is None:
        return False
    positions, orientations, total_length = sampled
    speed = max(total_length / max((state_count - 1) * scenario.dt, scenario.dt), 0.1)
    speed *= speed_scale

    initial_state = InitialState(
        time_step=0,
        position=positions[0],
        orientation=orientations[0],
        velocity=speed,
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )
    state_list = [
        ExtendedPMState(
            time_step=time_step,
            position=positions[time_step],
            velocity=speed,
            orientation=orientations[time_step],
            acceleration=0.0,
        )
        for time_step in range(1, state_count)
    ]
    trajectory = Trajectory(1, state_list)
    shape = copy.deepcopy(template_obstacle.obstacle_shape)
    prediction = TrajectoryPrediction(trajectory, shape)
    obstacle = DynamicObstacle(
        new_ego_id,
        template_obstacle.obstacle_type,
        shape,
        initial_state,
        prediction,
    )
    scenario.add_objects(obstacle)
    update_lanelet_assignments(scenario, obstacle)
    return True


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


def update_lanelet_assignments(scenario, obstacle):
    lanelet_network = scenario.lanelet_network
    shape_assignment = {}
    center_assignment = {}
    for time_step in all_obstacle_times(obstacle):
        state = state_at_time(obstacle, time_step)
        lanelet_ids = set()
        try:
            obstacle_shape = obstacle.obstacle_shape.rotate_translate_local(
                state.position,
                getattr(state, "orientation", 0.0),
            )
            lanelet_ids = set(lanelet_network.find_lanelet_by_shape(obstacle_shape))
        except Exception:
            lanelet_ids = set()
        if not lanelet_ids:
            try:
                lanelet_ids = set(lanelet_network.find_lanelet_by_position([state.position])[0])
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
        source="added ego trajectory perturbation",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IN3/IN5 violation scenarios by adding a new ego id.",
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
    parser.add_argument("--max-candidates-per-scenario", type=int, default=20)
    parser.add_argument("--new-ego-id", type=int, default=90000)
    parser.add_argument("--state-count", type=int, default=30)
    parser.add_argument("--start-offsets", default="0,3,6,10")
    parser.add_argument("--speed-scales", default="1.0,1.5,2.0")
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
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

    start_offsets = parse_float_list(args.start_offsets)
    speed_scales = parse_float_list(args.speed_scales)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_scenario",
        "source_path",
        "generated_path",
        "new_ego_id",
        "lanelet_path",
        "start_offset",
        "speed_scale",
        "violated_rules",
    ]
    generated_count = 0
    failed_count = 0
    tried_candidates = 0

    with args.report.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()

        for local_idx, scenario_path in enumerate(scenario_paths, start=1):
            global_idx = args.start_index + local_idx - 1
            print(f"[{global_idx}/{total_matching}] {scenario_path.name}", flush=True)
            try:
                scenario, planning_problem_set = CommonRoadFileReader(str(scenario_path)).open(True)
                _, original_world_ids = evaluate_rules(
                    scenario,
                    scenario.dynamic_obstacles[0].obstacle_id,
                    quiet=not args.verbose,
                )
            except Exception as exc:
                failed_count += 1
                print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not original_world_ids:
                continue

            template_obstacle = scenario.obstacle_by_id(original_world_ids[0])
            if template_obstacle is None:
                template_obstacle = scenario.dynamic_obstacles[0]
            new_ego_id = next_added_ego_id(scenario, args.new_ego_id)
            lanelet_paths = make_lanelet_path_candidates(
                scenario,
                args.max_candidates_per_scenario,
            )

            generated_for_scenario = False
            for lanelet_path in lanelet_paths:
                for start_offset in start_offsets:
                    for speed_scale in speed_scales:
                        candidate = copy.deepcopy(scenario)
                        candidate_ego_id = next_added_ego_id(candidate, new_ego_id)
                        candidate_template = candidate.obstacle_by_id(
                            template_obstacle.obstacle_id
                        )
                        if candidate_template is None:
                            candidate_template = candidate.dynamic_obstacles[0]
                        tried_candidates += 1
                        try:
                            added = add_ego_on_lanelet_path(
                                candidate,
                                lanelet_path,
                                candidate_ego_id,
                                candidate_template,
                                args.state_count,
                                start_offset,
                                end_margin=0.0,
                                speed_scale=speed_scale,
                            )
                            if not added:
                                continue
                            rule_results, candidate_world_ids = evaluate_rules(
                                candidate,
                                candidate_ego_id,
                                quiet=not args.verbose,
                            )
                        except Exception:
                            plt.close("all")
                            continue
                        if candidate_ego_id not in candidate_world_ids:
                            plt.close("all")
                            continue
                        if not any_violation(rule_results):
                            plt.close("all")
                            continue

                        output_name = (
                            f"{scenario_path.stem}_addedEgo{candidate_ego_id}"
                            f"_path{'-'.join(str(i) for i in lanelet_path)}"
                            f"_off{start_offset:g}_speed{speed_scale:g}_in_violation.xml"
                        )
                        output_path = args.output_dir / output_name
                        write_scenario(candidate, planning_problem_set, output_path)
                        summary = format_rule_summary(rule_results)
                        writer.writerow(
                            {
                                "source_scenario": scenario_path.stem,
                                "source_path": str(scenario_path),
                                "generated_path": str(output_path),
                                "new_ego_id": candidate_ego_id,
                                "lanelet_path": "-".join(str(i) for i in lanelet_path),
                                "start_offset": start_offset,
                                "speed_scale": speed_scale,
                                "violated_rules": summary,
                            }
                        )
                        report_file.flush()
                        generated_count += 1
                        generated_for_scenario = True
                        print(
                            f"  generated ego={candidate_ego_id} path={lanelet_path} "
                            f"offset={start_offset:g} speed={speed_scale:g}: {summary}",
                            flush=True,
                        )
                        plt.close("all")
                        break
                    if generated_for_scenario:
                        break
                if generated_for_scenario:
                    break
            if args.flush_every > 0 and local_idx % args.flush_every == 0:
                print(
                    f"  progress: scenarios={local_idx}, tried={tried_candidates}, "
                    f"generated={generated_count}, failed={failed_count}",
                    flush=True,
                )
            if generated_count >= args.max_generated:
                break

    print(f"Scanned scenarios this run: {local_idx if scenario_paths else 0}", flush=True)
    print(f"Tried candidates: {tried_candidates}", flush=True)
    print(f"Generated scenarios: {generated_count}", flush=True)
    print(f"Generated directory: {args.output_dir}", flush=True)
    print(f"Report CSV: {args.report}", flush=True)


if __name__ == "__main__":
    main()
