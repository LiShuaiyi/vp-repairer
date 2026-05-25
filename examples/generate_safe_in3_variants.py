#!/usr/bin/env python3
"""Generate safe perturbed IN3/IN5 scenarios from original inD violation cases."""

import argparse
import copy
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import create_ego_vehicle_param, get_evaluation_config
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "scenarios" / "in3_original"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scenarios" / "generated_in3_tests"
DEFAULT_CASES_CSV = REPO_ROOT / "evaluation" / "config" / "ind_in3.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "evaluation" / "config" / "generated_in3_tests.csv"
RESULT_PREFIX = "__BATCH_RESULT__="
SUPPORTED_RULES = ("R_IN3_hand_draft", "R_IN5")
TIMED_OPERATOR_PATTERN = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)

PERTURBATIONS = [
    # ego_lat, target_lat, ego_lon, target_lon, ego_speed, target_speed, ego_shift, target_shift
    (-0.35, 0.25, 0.80, -0.60, 1.08, 0.94, 1, -1),
    (0.35, -0.25, -0.70, 0.70, 0.94, 1.08, -1, 1),
    (0.30, -0.20, -0.55, 0.55, 0.95, 1.06, -1, 1),
    (0.40, -0.20, -0.65, 0.65, 0.95, 1.07, -1, 1),
    (0.45, -0.25, -0.75, 0.75, 0.93, 1.09, -1, 1),
    (0.50, -0.20, -0.45, 0.55, 0.96, 1.05, 0, 1),
    (0.55, -0.20, -0.40, 0.60, 0.96, 1.05, 0, 1),
    (0.60, -0.18, -0.35, 0.55, 0.97, 1.04, 0, 1),
    (0.45, -0.15, -0.50, 0.70, 0.96, 1.06, 0, 1),
    (0.35, -0.15, -0.40, 0.60, 0.97, 1.04, 0, 1),
    (0.25, -0.15, -0.50, 0.50, 0.96, 1.06, -1, 1),
    (0.20, -0.10, -0.35, 0.45, 0.97, 1.03, 0, 1),
    (0.15, -0.10, -0.30, 0.40, 0.98, 1.02, 0, 1),
    (0.50, -0.30, -0.60, 0.80, 0.94, 1.08, -1, 1),
    (0.60, -0.25, -0.50, 0.75, 0.95, 1.06, 0, 1),
    (-0.45, 0.10, 1.10, -0.80, 1.12, 0.90, 1, -1),
    (0.45, -0.10, -0.90, 1.00, 0.90, 1.12, -1, 1),
    (-0.25, 0.35, 1.40, -1.00, 1.15, 0.88, 2, -1),
    (0.25, -0.35, -1.00, 1.40, 0.88, 1.15, -1, 2),
    (-0.55, 0.20, 0.60, -0.40, 1.05, 0.96, 1, 0),
    (0.55, -0.20, -0.40, 0.60, 0.96, 1.05, 0, 1),
    (-0.20, 0.45, 1.00, -1.20, 1.10, 0.92, 2, -2),
    (0.20, -0.45, -1.20, 1.00, 0.92, 1.10, -2, 2),
    (-0.30, 0.30, 1.60, -1.30, 1.18, 0.86, 2, -1),
    (0.30, -0.30, -1.30, 1.60, 0.86, 1.18, -1, 2),
    (-0.15, 0.15, 2.00, -1.60, 1.20, 0.84, 2, -2),
    (0.15, -0.15, -1.60, 2.00, 0.84, 1.20, -2, 2),
]


def all_obstacle_times(obstacle):
    times = [int(obstacle.initial_state.time_step)]
    trajectory = getattr(getattr(obstacle, "prediction", None), "trajectory", None)
    if trajectory is not None:
        times.extend(int(state.time_step) for state in trajectory.state_list)
    return sorted(set(times))


def state_at_time(obstacle, time_step):
    if int(obstacle.initial_state.time_step) == int(time_step):
        return obstacle.initial_state
    return obstacle.state_at_time(int(time_step))


def to_initial_state(state):
    return InitialState(
        time_step=int(state.time_step),
        position=np.asarray(state.position, dtype=float),
        orientation=float(getattr(state, "orientation", 0.0)),
        velocity=float(getattr(state, "velocity", 0.0)),
        acceleration=float(getattr(state, "acceleration", 0.0)),
        yaw_rate=float(getattr(state, "yaw_rate", 0.0)),
        slip_angle=float(getattr(state, "slip_angle", 0.0)),
    )


def to_trajectory_state(state):
    return CustomState(
        time_step=int(state.time_step),
        position=np.asarray(state.position, dtype=float),
        orientation=float(getattr(state, "orientation", 0.0)),
        velocity=float(getattr(state, "velocity", 0.0)),
        acceleration=float(getattr(state, "acceleration", 0.0)),
    )


def perturb_state(state, lateral_offset, longitudinal_offset, velocity_scale):
    new_state = copy.deepcopy(state)
    orientation = float(getattr(new_state, "orientation", 0.0))
    tangent = np.asarray([math.cos(orientation), math.sin(orientation)], dtype=float)
    normal = np.asarray([-math.sin(orientation), math.cos(orientation)], dtype=float)
    new_state.position = (
        np.asarray(new_state.position, dtype=float)
        + lateral_offset * normal
        + longitudinal_offset * tangent
    )
    if hasattr(new_state, "velocity"):
        new_state.velocity = max(0.0, float(new_state.velocity) * velocity_scale)
    if hasattr(new_state, "acceleration"):
        new_state.acceleration = float(getattr(new_state, "acceleration", 0.0)) * velocity_scale
    return new_state


def perturb_obstacle(
    scenario,
    obstacle_id,
    lateral_offset,
    longitudinal_offset,
    velocity_scale,
    time_shift,
):
    obstacle = scenario.obstacle_by_id(int(obstacle_id))
    if obstacle is None:
        raise ValueError(f"Obstacle {obstacle_id} not found")

    times = all_obstacle_times(obstacle)
    first_time = min(times)
    last_time = max(times)
    perturbed_states = {}
    for time_step in times:
        source_time = min(max(time_step + int(time_shift), first_time), last_time)
        source_state = state_at_time(obstacle, source_time)
        new_state = perturb_state(
            source_state,
            lateral_offset,
            longitudinal_offset,
            velocity_scale,
        )
        new_state.time_step = time_step
        perturbed_states[time_step] = new_state

    initial_time = int(obstacle.initial_state.time_step)
    obstacle.initial_state = to_initial_state(perturbed_states[initial_time])
    state_list = [
        to_trajectory_state(perturbed_states[time_step])
        for time_step in times
        if time_step != initial_time
    ]
    if not state_list:
        raise ValueError(f"Obstacle {obstacle_id} has no prediction states")
    obstacle.prediction.trajectory = Trajectory(state_list[0].time_step, state_list)
    update_lanelet_assignments(scenario, obstacle)


def update_lanelet_assignments(scenario, obstacle):
    lanelet_network = scenario.lanelet_network
    shape_assignment = {}
    center_assignment = {}
    for time_step in all_obstacle_times(obstacle):
        state = state_at_time(obstacle, time_step)
        try:
            obstacle_shape = obstacle.obstacle_shape.rotate_translate_local(
                state.position,
                float(getattr(state, "orientation", 0.0)),
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

    initial_time = int(obstacle.initial_state.time_step)
    obstacle.initial_shape_lanelet_ids = set(shape_assignment.get(initial_time, set()))
    obstacle.initial_center_lanelet_ids = set(center_assignment.get(initial_time, set()))
    obstacle.prediction.shape_lanelet_assignment = shape_assignment
    obstacle.prediction.center_lanelet_assignment = center_assignment


def rule_slug(rule):
    if rule == "R_IN5":
        return "in5"
    return "in3"


def align_bound_to_sampling_period(bound, dt, mode="ceil"):
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
    def replace(match):
        low = float(match.group("low"))
        high = float(match.group("high"))
        low_aligned = align_bound_to_sampling_period(low, dt, mode)
        high_aligned = align_bound_to_sampling_period(high, dt, mode)
        return (
            f"{match.group('op')}["
            f"{format_time_bound(low_aligned)},"
            f"{format_time_bound(high_aligned)}{match.group('unit')}]"
        )

    return TIMED_OPERATOR_PATTERN.sub(replace, rule_str)


def align_traffic_rule_bounds(traffic_rules_config, dt, rule):
    if rule != "R_IN5":
        return
    traffic_rules = traffic_rules_config["traffic_rules"]
    if rule in traffic_rules:
        traffic_rules[rule] = align_rtamt_bounds_in_rule(traffic_rules[rule], dt)


def make_world(scenario, rule):
    world_config = get_world_config()
    traffic_rules_config = copy.deepcopy(get_traffic_rule_config())
    traffic_rules_param = traffic_rules_config["traffic_rules_param"]
    world_config["scenario"] = traffic_rules_param["mpr_scenario"] = "intersection"
    world_config["intersection_road_network_param"]["map_type"] = "dataset"
    Cfg["common"]["scenario"] = "intersection"
    traffic_rules_param["use_mpr"] = False
    world = World.create_from_scenario(scenario, config=world_config)
    align_traffic_rule_bounds(traffic_rules_config, world.dt, rule)
    return world, traffic_rules_config


def evaluate_rule(scenario, ego_id, rule):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        world, traffic_rules_config = make_world(scenario, rule)
        ego_vehicle = world.vehicle_by_id(int(ego_id))
        if ego_vehicle is None:
            return False, math.inf, None, ""
        ego_vehicle.vehicle_param = create_ego_vehicle_param(
            get_evaluation_config().get("ego_vehicle_param"),
            world.dt,
        )
        evaluator = PropositionRuleEvaluator.create_from_config(
            world,
            int(ego_id),
            rule,
            traffic_rules_config=traffic_rules_config,
        )
        other_ids_by_step = []
        values = []
        for _ in range(evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1):
            value = float(evaluator.update())
            values.append(value)
            other_ids_by_step.append(evaluator.other_ids)

    for idx, value in enumerate(values):
        if value < 0:
            other_ids = normalize_other_ids(other_ids_by_step[idx])
            first_other = other_ids.split(";")[0] if other_ids else ""
            return True, evaluator.ego_vehicle.start_time + idx, first_other, f"{value:.12g}"
    min_value = min(values) if values else math.inf
    min_idx = values.index(min_value) if values else 0
    other_ids = normalize_other_ids(other_ids_by_step[min_idx]) if values else ""
    first_other = other_ids.split(";")[0] if other_ids else ""
    return False, math.inf, first_other, f"{min_value:.12g}" if values else ""


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


def lanelet_drivable_polygon(scenario):
    polygons = []
    for lanelet in scenario.lanelet_network.lanelets:
        vertices = np.asarray(lanelet.polygon.vertices, dtype=float)
        if len(vertices) >= 3:
            polygons.append(Polygon(vertices))
    return unary_union(polygons)


def obstacle_polygon(obstacle, state):
    shape = obstacle.obstacle_shape.rotate_translate_local(
        state.position,
        float(getattr(state, "orientation", 0.0)),
    )
    if hasattr(shape, "shapely_object"):
        return shape.shapely_object
    return Polygon(np.asarray(shape.vertices, dtype=float))


def obstacle_on_road(scenario, obstacle, drivable_polygon, margin):
    safe_polygon = drivable_polygon.buffer(-margin) if margin > 0 else drivable_polygon
    for time_step in all_obstacle_times(obstacle):
        state = state_at_time(obstacle, time_step)
        poly = obstacle_polygon(obstacle, state)
        if poly.is_empty or not safe_polygon.covers(poly):
            return False, f"offroad obstacle={obstacle.obstacle_id} time={time_step}"
    return True, ""


def obstacle_center_on_road(scenario, obstacle, drivable_polygon, margin):
    safe_polygon = drivable_polygon.buffer(-margin) if margin > 0 else drivable_polygon
    for time_step in all_obstacle_times(obstacle):
        state = state_at_time(obstacle, time_step)
        point = np.asarray(state.position, dtype=float)
        if not contains_xy(safe_polygon, point[0], point[1]):
            return False, f"offroad_center obstacle={obstacle.obstacle_id} time={time_step}"
    return True, ""


def scenario_collision_free(scenario, margin, obstacle_ids=None):
    obstacles = list(scenario.dynamic_obstacles)
    obstacle_id_set = set(int(obstacle_id) for obstacle_id in obstacle_ids or [])
    all_times = sorted(
        {
            time_step
            for obstacle in obstacles
            for time_step in all_obstacle_times(obstacle)
        }
    )
    for time_step in all_times:
        polys = []
        ids = []
        for obstacle in obstacles:
            state = state_at_time(obstacle, time_step)
            if state is None:
                continue
            poly = obstacle_polygon(obstacle, state)
            if margin > 0:
                poly = poly.buffer(margin)
            polys.append(poly)
            ids.append(obstacle.obstacle_id)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if obstacle_id_set and ids[i] not in obstacle_id_set and ids[j] not in obstacle_id_set:
                    continue
                if polys[i].intersects(polys[j]) and polys[i].intersection(polys[j]).area > 1e-6:
                    return False, f"collision time={time_step} ids={ids[i]},{ids[j]}"
    return True, ""


def validate_candidate(scenario, ego_id, target_id, args):
    drivable_polygon = lanelet_drivable_polygon(scenario)
    for obstacle_id in (ego_id, target_id):
        obstacle = scenario.obstacle_by_id(int(obstacle_id))
        if args.road_check == "none":
            continue
        if args.road_check == "shape":
            ok, reason = obstacle_on_road(
                scenario,
                obstacle,
                drivable_polygon,
                args.road_margin,
            )
            if not ok:
                return False, reason
        elif args.road_check == "center":
            ok, reason = obstacle_center_on_road(
                scenario,
                obstacle,
                drivable_polygon,
                args.road_margin,
            )
            if not ok:
                return False, reason
        assignments = getattr(obstacle.prediction, "shape_lanelet_assignment", {})
        if any(not lanelets for lanelets in assignments.values()):
            return False, f"empty_lanelet_assignment obstacle={obstacle_id}"

    collision_ids = None
    if args.collision_scope == "modified":
        collision_ids = [ego_id, target_id]
    ok, reason = scenario_collision_free(
        scenario,
        args.collision_margin,
        obstacle_ids=collision_ids,
    )
    if not ok:
        return False, reason
    return True, ""


def load_cases(cases_csv, source_dir):
    cases = []
    by_scenario = {}
    seen = set()
    with cases_csv.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            by_scenario.setdefault(row["scenario"], []).append(row)

    for scenario_path in sorted(source_dir.glob("*.xml")):
        scenario_name = scenario_path.stem
        for row in by_scenario.get(scenario_name, []):
            key = (scenario_name, int(row["ego_id"]))
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                {
                    "scenario_id": scenario_name,
                    "scenario_path": scenario_path,
                    "ego_id": int(row["ego_id"]),
                }
            )
    return cases


def write_scenario(scenario, planning_problem_set, output_path, rule):
    writer = CommonRoadFileWriter(
        scenario,
        planning_problem_set,
        author="generated by commonroad-repairer",
        affiliation="",
        source=f"safe perturbed {rule} trajectories",
        tags={Tag.CRITICAL, Tag.INTERSECTION},
    )
    writer.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)


def load_existing_output_rows(output_csv):
    if not output_csv.exists():
        return []
    with output_csv.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def existing_variant_keys(rows, rule):
    keys = set()
    for row in rows:
        if row.get("rule_STL") != rule:
            continue
        try:
            key = (
                row.get("source_scenario_id", ""),
                str(int(row.get("ego_id", ""))),
                str(int(row.get("target_id", row.get("other_id", "")))),
                str(float(row.get("ego_lateral_offset", "nan"))),
                str(float(row.get("target_lateral_offset", "nan"))),
                str(float(row.get("ego_longitudinal_offset", "nan"))),
                str(float(row.get("target_longitudinal_offset", "nan"))),
                str(float(row.get("ego_velocity_scale", "nan"))),
                str(float(row.get("target_velocity_scale", "nan"))),
                str(int(float(row.get("ego_time_shift", "nan")))),
                str(int(float(row.get("target_time_shift", "nan")))),
            )
        except (TypeError, ValueError):
            continue
        keys.add(key)
    return keys


def perturbation_key(case, target_id, perturbation):
    (
        ego_lat,
        target_lat,
        ego_lon,
        target_lon,
        ego_speed,
        target_speed,
        ego_shift,
        target_shift,
    ) = perturbation
    return (
        case["scenario_id"],
        str(int(case["ego_id"])),
        str(int(target_id)),
        str(float(ego_lat)),
        str(float(target_lat)),
        str(float(ego_lon)),
        str(float(target_lon)),
        str(float(ego_speed)),
        str(float(target_speed)),
        str(int(ego_shift)),
        str(int(target_shift)),
    )


def next_variant_index(existing_rows, output_dir, slug):
    max_idx = -1
    pattern = re.compile(rf"_{re.escape(slug)}_safe_variant_(\d+)$")
    for row in existing_rows:
        scenario_id = row.get("scenario_id", "")
        match = pattern.search(scenario_id)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    for path in output_dir.glob(f"*_{slug}_safe_variant_*.xml"):
        match = pattern.search(path.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def vp_repair_succeeds(scenario_id, ego_id, rule, scenario_path):
    batch_script = REPO_ROOT / "examples" / "batch_test_vp_repairer_in3_in5_generated.py"
    cmd = [
        sys.executable,
        str(batch_script),
        "--case",
        scenario_id,
        str(int(ego_id)),
        rule,
        str(scenario_path),
        "domain_dpll",
        "vp",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    result = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX):])
            break
    if result is None:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-8:])
        return False, f"VP check produced no result (exit={completed.returncode}): {tail}"
    if not result.get("success"):
        return False, result.get("error") or "VP repair returned success=False"
    return True, f"iterations={result.get('iterations')} updated_tv={result.get('updated_tv')}"


def generate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_existing_output_rows(args.output_csv) if args.append_existing else []
    slug = rule_slug(args.rule)
    seen_keys = existing_variant_keys(existing_rows, args.rule)
    variant_idx = next_variant_index(existing_rows, args.output_dir, slug) if args.append_existing else 0

    cases = load_cases(args.cases_csv, args.source_dir)
    if args.max_source_cases is not None:
        cases = cases[: args.max_source_cases]
    if not cases:
        raise RuntimeError(f"No matching cases from {args.cases_csv} under {args.source_dir}")
    print(
        f"Loaded {len(cases)} unique source case(s) from {args.cases_csv}",
        flush=True,
    )

    rows = []
    attempts = 0
    case_infos = []
    generated_by_case = {}
    for case in cases:
        scenario, planning_problem_set = CommonRoadFileReader(str(case["scenario_path"])).open(True)
        original_violated, original_tv, target_id, original_min_rob = evaluate_rule(
            scenario,
            case["ego_id"],
            args.rule,
        )
        if not original_violated or not target_id:
            print(
                f"skip {case['scenario_id']} ego={case['ego_id']}: "
                f"no original {args.rule} target (min_rob={original_min_rob})",
                flush=True,
            )
            continue
        target_id = int(target_id)
        case_infos.append(
            {
                "case": case,
                "scenario": scenario,
                "planning_problem_set": planning_problem_set,
                "original_tv": original_tv,
                "target_id": target_id,
            }
        )
        generated_by_case[len(case_infos) - 1] = 0

    if not case_infos:
        raise RuntimeError(f"No source cases with original {args.rule} violations were usable")

    def try_generate(info, case_idx, perturbation):
        nonlocal attempts, variant_idx
        case = info["case"]
        target_id = info["target_id"]
        if args.max_per_source is not None and generated_by_case[case_idx] >= args.max_per_source:
            return False
        key = perturbation_key(case, target_id, perturbation)
        if key in seen_keys:
            return False
        attempts += 1
        (
            ego_lat,
            target_lat,
            ego_lon,
            target_lon,
            ego_speed,
            target_speed,
            ego_shift,
            target_shift,
        ) = perturbation
        candidate = copy.deepcopy(info["scenario"])
        while True:
            scenario_id = f"{case['scenario_id']}_{slug}_safe_variant_{variant_idx:02d}"
            output_path = args.output_dir / f"{scenario_id}.xml"
            variant_idx += 1
            if not output_path.exists():
                break
        candidate.scenario_id = scenario_id
        print(
            f"try {scenario_id}: source={case['scenario_id']} ego={case['ego_id']} "
            f"target={target_id} attempt={attempts} generated={len(rows)}/{args.count}",
            flush=True,
        )
        try:
            perturb_obstacle(candidate, case["ego_id"], ego_lat, ego_lon, ego_speed, ego_shift)
            perturb_obstacle(candidate, target_id, target_lat, target_lon, target_speed, target_shift)
            ok, reason = validate_candidate(candidate, case["ego_id"], target_id, args)
            if not ok:
                print(f"reject {scenario_id}: {reason}", flush=True)
                return False
            violated, tv, other_id, min_rob = evaluate_rule(candidate, case["ego_id"], args.rule)
            if not violated:
                print(
                    f"reject {scenario_id}: {args.rule} no longer violated (min_rob={min_rob})",
                    flush=True,
                )
                return False
        except Exception as exc:
            print(f"reject {scenario_id}: {type(exc).__name__}: {exc}", flush=True)
            return False

        if args.require_vp_success:
            temp_path = args.output_dir / f".{scenario_id}.vpcheck.xml"
            try:
                write_scenario(candidate, info["planning_problem_set"], temp_path, args.rule)
                vp_ok, vp_reason = vp_repair_succeeds(
                    scenario_id,
                    case["ego_id"],
                    args.rule,
                    temp_path,
                )
            finally:
                if temp_path.exists() and not output_path.exists():
                    temp_path.unlink(missing_ok=True)
            if not vp_ok:
                print(f"reject {scenario_id}: VP repair failed ({vp_reason})", flush=True)
                return False
            write_scenario(candidate, info["planning_problem_set"], output_path, args.rule)
            print(f"vp-ok {scenario_id}: {vp_reason}", flush=True)
        else:
            write_scenario(candidate, info["planning_problem_set"], output_path, args.rule)
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_path": str(output_path),
                "ego_id": case["ego_id"],
                "target_id": target_id,
                "rule_STL": args.rule,
                "source_scenario_id": case["scenario_id"],
                "source_scenario_path": str(case["scenario_path"]),
                "original_tv": info["original_tv"],
                "tv": tv,
                "other_id": other_id,
                "min_robustness": min_rob,
                "ego_lateral_offset": ego_lat,
                "target_lateral_offset": target_lat,
                "ego_longitudinal_offset": ego_lon,
                "target_longitudinal_offset": target_lon,
                "ego_velocity_scale": ego_speed,
                "target_velocity_scale": target_speed,
                "ego_time_shift": ego_shift,
                "target_time_shift": target_shift,
                "road_margin": args.road_margin,
                "collision_margin": args.collision_margin,
                "road_check": args.road_check,
                "collision_scope": args.collision_scope,
            }
        )
        seen_keys.add(key)
        generated_by_case[case_idx] += 1
        print(
            f"generated {scenario_id}: ego={case['ego_id']} target={target_id} tv={tv}",
            flush=True,
        )
        return True

    for perturbation in PERTURBATIONS:
        for case_idx, info in enumerate(case_infos):
            if len(rows) >= args.count:
                break
            try_generate(info, case_idx, perturbation)
        if len(rows) >= args.count:
            break

    fieldnames = [
        "scenario_id",
        "scenario_path",
        "ego_id",
        "target_id",
        "rule_STL",
        "source_scenario_id",
        "source_scenario_path",
        "original_tv",
        "tv",
        "other_id",
        "min_robustness",
        "ego_lateral_offset",
        "target_lateral_offset",
        "ego_longitudinal_offset",
        "target_longitudinal_offset",
        "ego_velocity_scale",
        "target_velocity_scale",
        "ego_time_shift",
        "target_time_shift",
        "road_margin",
        "collision_margin",
        "road_check",
        "collision_scope",
    ]
    mode = "a" if args.append_existing and args.output_csv.exists() else "w"
    with args.output_csv.open(mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} safe {args.rule} variants after {attempts} attempts")
    print(f"Scenario output dir: {args.output_dir}")
    print(f"CSV output: {args.output_csv}")
    if len(rows) < args.count:
        raise RuntimeError(f"Only generated {len(rows)}/{args.count} requested variants")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate collision-free and on-road IN3/IN5 variants from original inD cases.",
    )
    parser.add_argument(
        "--rule",
        choices=SUPPORTED_RULES,
        default="R_IN3_hand_draft",
        help="Traffic rule to keep violated in generated scenarios. Default: R_IN3_hand_draft.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-source-cases", type=int, default=None)
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Maximum generated variants per original source scenario.",
    )
    parser.add_argument(
        "--road-check",
        choices=("shape", "center", "assignment", "none"),
        default="shape",
        help=(
            "Road-boundary strictness for modified ego/target: shape keeps the full "
            "vehicle shape inside lanelets, center only checks the center point, "
            "assignment only requires non-empty lanelet assignment, none skips road "
            "and lanelet-assignment checks. Default: shape."
        ),
    )
    parser.add_argument(
        "--collision-scope",
        choices=("all", "modified"),
        default="all",
        help=(
            "Collision strictness: all checks every vehicle pair; modified only checks "
            "pairs involving the modified ego or target. Default: all."
        ),
    )
    parser.add_argument(
        "--road-margin",
        type=float,
        default=0.0,
        help="Meters to shrink drivable lanelet polygons for road-edge clearance. Default: 0.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0,
        help="Meters to inflate vehicle rectangles for conservative collision checks. Default: 0.",
    )
    parser.add_argument(
        "--require-vp-success",
        action="store_true",
        help="Only keep generated scenarios whose VP repair succeeds.",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append generated rows to an existing CSV and continue variant numbering.",
    )
    return parser.parse_args()


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
