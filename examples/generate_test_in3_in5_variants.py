#!/usr/bin/env python3
"""Generate perturbed IN3/IN5 test scenarios from the existing hand-picked tests."""

import copy
import csv
import io
import math
import os
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

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCENARIO_ROOT = REPO_ROOT / "scenarios"
OUTPUT_ROOT = REPO_ROOT / "scenarios" / "generated_in3_in5_tests"
CSV_PATH = REPO_ROOT / "evaluation" / "config" / "generated_in3_in5_tests.csv"

BASE_CASES = [
    {
        "rule_name": "in3",
        "rule": "R_IN3_hand_draft",
        "scenario_id": "DEU_TestIntersectionInteract-3_1_T-1",
        "ego_id": 30,
        "target_id": 31,
        "scenario_type": "intersection",
        "intersection_type": "hand_draft",
        "n_r": 50,
    },
    {
        "rule_name": "in5",
        "rule": "R_IN5",
        "scenario_id": "DEU_AAH1-2_7900_T-1049",
        "ego_id": 10020,
        "target_id": 10019,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
        "n_r": 149,
    },
]

PERTURBATIONS = [
    # ego_lat, target_lat, ego_lon, target_lon, ego_speed, target_speed, ego_time_shift, target_time_shift
    (-0.70, 0.45, 1.20, -0.60, 1.18, 0.88, 1, -1),
    (0.70, -0.45, -0.80, 1.10, 0.88, 1.18, -1, 1),
    (-0.95, -0.30, 1.80, 0.40, 1.25, 0.95, 2, 0),
    (0.95, 0.30, -1.20, -0.70, 0.95, 1.25, 0, 2),
    (-0.55, 0.85, 2.40, -1.20, 1.35, 0.82, 2, -2),
    (0.55, -0.85, -1.60, 1.80, 0.82, 1.35, -2, 2),
    (-1.15, 0.00, 0.90, 0.90, 1.12, 1.12, 1, 1),
    (1.15, 0.00, -0.90, -0.90, 1.12, 1.12, -1, -1),
    (0.00, -1.15, 2.20, -0.40, 1.30, 0.90, 3, -1),
    (0.00, 1.15, -0.40, 2.20, 0.90, 1.30, -1, 3),
    (-0.85, 0.65, 3.00, -1.80, 1.40, 0.78, 3, -2),
    (0.85, -0.65, -1.80, 3.00, 0.78, 1.40, -2, 3),
    (-1.35, 0.35, 1.50, -1.50, 1.22, 0.84, 2, -1),
    (1.35, -0.35, -1.50, 1.50, 0.84, 1.22, -1, 2),
    (-0.45, 1.35, 2.80, -2.20, 1.45, 0.75, 4, -2),
    (0.45, -1.35, -2.20, 2.80, 0.75, 1.45, -2, 4),
    (-0.60, 0.40, 1.60, -0.90, 1.16, 0.92, 1, -1),
    (0.60, -0.40, -0.90, 1.60, 0.92, 1.16, -1, 1),
    (-0.75, 0.55, 2.00, -1.10, 1.20, 0.90, 2, -1),
    (0.75, -0.55, -1.10, 2.00, 0.90, 1.20, -1, 2),
    (-0.50, 0.70, 1.30, -1.30, 1.18, 0.88, 1, -2),
    (0.50, -0.70, -1.30, 1.30, 0.88, 1.18, -2, 1),
]


def all_obstacle_times(obstacle):
    times = [obstacle.initial_state.time_step]
    trajectory = getattr(getattr(obstacle, "prediction", None), "trajectory", None)
    if trajectory is not None:
        times.extend(state.time_step for state in trajectory.state_list)
    return sorted(set(times))


def state_at_time(obstacle, time_step):
    if obstacle.initial_state.time_step == time_step:
        return obstacle.initial_state
    return obstacle.state_at_time(time_step)


def perturb_state(state, lateral_offset, longitudinal_offset, velocity_scale):
    new_state = copy.deepcopy(state)
    orientation = float(getattr(new_state, "orientation", 0.0))
    normal = np.asarray([-math.sin(orientation), math.cos(orientation)], dtype=float)
    tangent = np.asarray([math.cos(orientation), math.sin(orientation)], dtype=float)
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


def to_initial_state(state):
    return InitialState(
        time_step=state.time_step,
        position=state.position,
        orientation=getattr(state, "orientation", 0.0),
        velocity=getattr(state, "velocity", 0.0),
        acceleration=getattr(state, "acceleration", 0.0),
        yaw_rate=getattr(state, "yaw_rate", 0.0),
        slip_angle=getattr(state, "slip_angle", 0.0),
    )


def to_trajectory_state(state):
    return CustomState(
        time_step=state.time_step,
        position=state.position,
        orientation=getattr(state, "orientation", 0.0),
        velocity=getattr(state, "velocity", 0.0),
        acceleration=getattr(state, "acceleration", 0.0),
    )


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


def perturb_obstacle(
    scenario,
    obstacle_id,
    lateral_offset,
    longitudinal_offset,
    velocity_scale,
    time_shift,
):
    obstacle = scenario.obstacle_by_id(obstacle_id)
    if obstacle is None:
        raise ValueError(f"Obstacle {obstacle_id} not found")

    times = all_obstacle_times(obstacle)
    first_time = min(times)
    last_time = max(times)
    source_time_by_time = {
        time_step: min(max(time_step + int(time_shift), first_time), last_time)
        for time_step in times
    }
    perturbed_states = {}
    for time_step in times:
        source_state = state_at_time(obstacle, source_time_by_time[time_step])
        new_state = perturb_state(
            source_state,
            lateral_offset,
            longitudinal_offset,
            velocity_scale,
        )
        new_state.time_step = time_step
        perturbed_states[time_step] = new_state
    initial_time = obstacle.initial_state.time_step
    obstacle.initial_state = to_initial_state(perturbed_states[initial_time])
    state_list = [
        to_trajectory_state(perturbed_states[time_step])
        for time_step in times
        if time_step != initial_time
    ]
    obstacle.prediction.trajectory = Trajectory(state_list[0].time_step, state_list)
    update_lanelet_assignments(scenario, obstacle)


def verify_case(scenario_id, case):
    config = RepairerConfiguration()
    config.general.path_scenarios = str(OUTPUT_ROOT) + "/"
    config.general.set_path_scenario(scenario_id + ".xml")
    config.update()
    config.repair.scenario_type = case["scenario_type"]
    config.repair.intersection_type = case["intersection_type"]
    config.repair.rules = [case["rule"]]
    config.repair.ego_id = case["ego_id"]
    config.repair.N_r = case["n_r"]
    config.repair.multiproc = False
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        monitor = STLRuleMonitor(config)
    tv = monitor.rule_to_tv.get(case["rule"], math.inf)
    other_id = monitor.rule_to_other_id.get(case["rule"])
    violated = case["rule"] in getattr(monitor, "_violated_rules", [])
    return violated and tv not in (math.inf, -math.inf), tv, other_id


def write_scenario(scenario, planning_problem_set, output_path):
    writer = CommonRoadFileWriter(
        scenario,
        planning_problem_set,
        author="generated by commonroad-repairer",
        affiliation="",
        source="perturbed IN3/IN5 test trajectories",
        tags={Tag.CRITICAL, Tag.INTERSECTION},
    )
    writer.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)


def generate_case(case, count):
    source_path = SOURCE_SCENARIO_ROOT / f"{case['scenario_id']}.xml"
    scenario, planning_problem_set = CommonRoadFileReader(str(source_path)).open(True)

    rows = []
    generated = 0
    for idx, (
        ego_offset,
        target_offset,
        ego_longitudinal_offset,
        target_longitudinal_offset,
        ego_speed,
        target_speed,
        ego_time_shift,
        target_time_shift,
    ) in enumerate(PERTURBATIONS):
        candidate = copy.deepcopy(scenario)
        scenario_id = f"{case['scenario_id']}_{case['rule_name']}_variant_{generated:02d}"
        candidate.scenario_id = scenario_id
        perturb_obstacle(
            candidate,
            case["ego_id"],
            ego_offset,
            ego_longitudinal_offset,
            ego_speed,
            ego_time_shift,
        )
        perturb_obstacle(
            candidate,
            case["target_id"],
            target_offset,
            target_longitudinal_offset,
            target_speed,
            target_time_shift,
        )

        output_path = OUTPUT_ROOT / f"{scenario_id}.xml"
        write_scenario(candidate, planning_problem_set, output_path)
        try:
            valid, tv, other_id = verify_case(scenario_id, case)
        except Exception:
            output_path.unlink(missing_ok=True)
            continue
        if not valid:
            output_path.unlink(missing_ok=True)
            continue

        rows.append(
            {
                "scenario_id": scenario_id,
                "ego_id": case["ego_id"],
                "target_id": case["target_id"],
                "rule_STL": case["rule"],
                "source_scenario_id": case["scenario_id"],
                "tv": tv,
                "other_id": other_id,
                "ego_lateral_offset": ego_offset,
                "target_lateral_offset": target_offset,
                "ego_longitudinal_offset": ego_longitudinal_offset,
                "target_longitudinal_offset": target_longitudinal_offset,
                "ego_velocity_scale": ego_speed,
                "target_velocity_scale": target_speed,
                "ego_time_shift": ego_time_shift,
                "target_time_shift": target_time_shift,
            }
        )
        generated += 1
        print(
            f"generated {scenario_id}: rule={case['rule']}, ego={case['ego_id']}, "
            f"target={case['target_id']}, tv={tv}, other={other_id}",
            flush=True,
        )
        if generated >= count:
            break

    if generated < count:
        raise RuntimeError(
            f"Only generated {generated}/{count} valid variants for {case['rule']}"
        )
    return rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in BASE_CASES:
        rows.extend(generate_case(case, count=10))

    fieldnames = [
        "scenario_id",
        "ego_id",
        "target_id",
        "rule_STL",
        "source_scenario_id",
        "tv",
        "other_id",
        "ego_lateral_offset",
        "target_lateral_offset",
        "ego_longitudinal_offset",
        "target_longitudinal_offset",
        "ego_velocity_scale",
        "target_velocity_scale",
        "ego_time_shift",
        "target_time_shift",
    ]
    with CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} generated cases to {CSV_PATH}")
    print(f"Generated scenarios are under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
