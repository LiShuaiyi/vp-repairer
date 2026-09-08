"""Batch evaluation for dimension-reduced, fixed-reference-path MICP replanning.

Run from the repository root, for example::

    python -m fixed_path_micp_comparison.batch_micp_fixed_path \
        --dataset ind \
        --scenario-dir /path/to/ind_scenarios_2024_repaired \
        --input evaluation/inD_evaluation_rin1_4_filtered.csv \
        --output comparison/inD_fixed_path_micp.csv \
        --monitor-config /path/to/commonroad-stl-monitor/crmonitor/config.yaml

The script deliberately records both the historical MICP timer (solver setup
through ``Solve``) and a core timer that also includes specification building.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.state import CustomState
from crmonitor.common.vehicle import CurvilinearStateManager
from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from fixed_path_micp_comparison.traffic_rule_fixed_path import (
    RG123FixedPath,
    RIN1FixedPath,
    RIN4FixedPath,
)
from fixed_path_micp_comparison.fixed_reference import build_trajectory_reference_lane


OUTPUT_FIELDS = [
    "scenario_id",
    "ego_id",
    "rule",
    "reference_path",
    "num_steps",
    "micp_feasible",
    "replanability",
    "specification_time",
    "solver_setup_time",
    "solve_wall_time",
    "gurobi_runtime",
    "legacy_total_time",
    "core_total_time",
    "num_variables",
    "num_binary_variables",
    "num_constraints",
    "robustness",
    "objective",
    "gurobi_status",
    "monitor_compliant",
    "updated_tv",
    "validation_time",
    "error",
]


def _default_gurobi_license() -> Optional[Path]:
    """Prefer an explicit environment setting, then this workspace's license."""

    configured = os.environ.get("GRB_LICENSE_FILE")
    if configured:
        return Path(configured)
    user_license = Path("/home/shuaiyi/gurobi.lic")
    return user_license if user_license.is_file() else None


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("highd", "ind"), required=True)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gurobi-license",
        type=Path,
        default=_default_gurobi_license(),
        help=(
            "Gurobi license file. It is installed as GRB_LICENSE_FILE before "
            "gurobipy is imported; defaults to the existing environment value "
            "or /home/shuaiyi/gurobi.lic when available."
        ),
    )
    parser.add_argument(
        "--monitor-config",
        type=Path,
        help="crmonitor config.yaml; required for the inD intersection world",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--reference-path",
        choices=("trajectory", "lane"),
        default="trajectory",
        help=(
            "trajectory uses VP's trajectory-aligned CLCS (default); lane uses "
            "the legacy MICP route/lane reference"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="disable Gurobi's detailed output"
    )
    return parser.parse_args()


def _read_cases(path: Path, limit: Optional[int]) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            return
        count = 0
        for row in reader:
            if len(row) < 3 or row[:3] == header[:3] or not row[0].strip():
                continue
            yield {
                "scenario_id": row[0].strip(),
                "ego_id": row[1].strip(),
                "rule": row[2].strip(),
            }
            count += 1
            if limit is not None and count >= limit:
                return


def _scenario_stem(dataset: str, scenario_id: str, ego_id: int) -> str:
    if dataset == "highd":
        return scenario_id.replace("T-1", f"T-{ego_id}")
    return scenario_id.replace("-1_", f"-{ego_id}_")


def _resolve_scenario_path(
    dataset: str, scenario_dir: Path, scenario_id: str, ego_id: int
):
    """Support both the legacy ego-in-filename data and current converters."""

    transformed_stem = _scenario_stem(dataset, scenario_id, ego_id)
    for stem in dict.fromkeys((transformed_stem, scenario_id)):
        path = scenario_dir / f"{stem}.xml"
        if path.is_file():
            return stem, path
    raise FileNotFoundError(
        "Scenario not found; tried "
        f"{scenario_dir / (transformed_stem + '.xml')} and "
        f"{scenario_dir / (scenario_id + '.xml')}"
    )


def _world_config(dataset: str, monitor_config: Optional[Path]):
    if monitor_config is None:
        if dataset == "ind":
            raise ValueError("--monitor-config is required for dataset=ind")
        return None
    config = load_yaml(str(monitor_config))
    if dataset == "ind":
        config["scenario"] = "intersection"
        config["intersection_road_network_param"]["map_type"] = "dataset"
    else:
        config["scenario"] = "interstate"
    return config


def _build_rule_monitor(
    dataset: str,
    scenario_dir: Path,
    scenario_stem: str,
    crscenario,
    ego_id: int,
    rule: str,
):
    repair_config = RepairerConfiguration()
    repair_config.general.path_scenarios = str(scenario_dir)
    repair_config.general.set_path_scenario(scenario_stem)
    repair_config.update()
    repair_config.repair.scenario_type = (
        "intersection" if dataset == "ind" else "interstate"
    )
    if dataset == "ind":
        repair_config.repair.intersection_type = "dataset"
    repair_config.repair.rules = [rule]
    repair_config.repair.ego_id = ego_id
    repair_config.scenario = crscenario
    return STLRuleMonitor(repair_config)


def _build_scenario(
    dataset: str,
    rule: str,
    world,
    ego_vehicle,
    other_vehicle,
    num_steps: int,
    reference_lane,
):
    lanelet_network = world.road_network.lanelet_network
    if dataset == "highd":
        return RG123FixedPath(
            num_steps,
            world,
            ego_vehicle,
            other_vehicle,
            lanelet_network,
            reference_lane,
        )
    if rule == "R_IN1":
        return RIN1FixedPath(
            num_steps, world, ego_vehicle, lanelet_network, reference_lane
        )
    if rule == "R_IN4":
        return RIN4FixedPath(
            num_steps,
            world,
            ego_vehicle,
            other_vehicle,
            lanelet_network,
            reference_lane,
        )
    raise ValueError(f"Unsupported fixed-path MICP rule: {rule}")


def _empty_result(case: Dict[str, str]) -> Dict[str, object]:
    result = {field: "" for field in OUTPUT_FIELDS}
    result.update(case)
    result["micp_feasible"] = "no"
    result["replanability"] = "no"
    return result


def _solution_states(x: np.ndarray, reference_lane):
    states = []
    for time_step in range(x.shape[1]):
        s = float(x[0, time_step])
        position = np.asarray(
            reference_lane.clcs.convert_to_cartesian_coords(s, 0.0), dtype=float
        )
        states.append(
            CustomState(
                time_step=time_step,
                position=position,
                orientation=reference_lane.orientation(s),
                velocity=float(x[1, time_step]),
                acceleration=float(x[2, time_step]),
            )
        )
    return states


def _validate_solution(rule_monitor, ego_id: int, updated_states):
    """Return the first updated violation time using the VP monitor protocol."""

    monitor = copy.copy(rule_monitor)
    world = copy.deepcopy(rule_monitor.world)
    monitor._world = world
    world_ego = world.vehicle_by_id(ego_id)
    for state in updated_states:
        world_ego.states_cr[state.time_step] = state
        ego_shape = world_ego.shape.rotate_translate_local(
            state.position, state.orientation
        )
        world_ego.ccosy_cache = CurvilinearStateManager(world.road_network)
        world_ego.lanelet_assignment[state.time_step] = set(
            world.road_network.lanelet_network.find_lanelet_by_shape(ego_shape)
        )
        world_ego.predicate_cache.cache[state.time_step] = defaultdict()

    rule_robustness, _ = monitor.evaluate_consecutively(
        world, monitor.start_time_step
    )
    if not all(len(values) == len(rule_robustness[0]) for values in rule_robustness):
        return -math.inf
    robustness = np.asarray(rule_robustness)
    if np.any(robustness[:, 0] < 0):
        return -math.inf
    tv_per_rule = np.argmax(robustness < 0, axis=-1)
    if np.all(tv_per_rule + world_ego.start_time == world_ego.start_time):
        return math.inf
    min_tv = np.min(tv_per_rule[tv_per_rule != 0])
    return float(min_tv * world.dt)


def evaluate_case(args, case: Dict[str, str], config) -> Dict[str, object]:
    result = _empty_result(case)
    result["reference_path"] = args.reference_path
    ego_id = int(case["ego_id"])
    rule = case["rule"]

    try:
        scenario_stem, scenario_path = _resolve_scenario_path(
            args.dataset, args.scenario_dir, case["scenario_id"], ego_id
        )
        crscenario, _ = CommonRoadFileReader(str(scenario_path)).open(
            lanelet_assignment=True
        )
        world = (
            World.create_from_scenario(crscenario)
            if config is None
            else World.create_from_scenario(crscenario, config)
        )
        ego_vehicle = world.vehicle_by_id(ego_id)
        start_time_step = int(getattr(ego_vehicle, "start_time", 0))
        if start_time_step != 0:
            raise ValueError(
                "The fixed-path MICP batch currently expects local time to start at 0, "
                f"got {start_time_step}"
            )
        final_time_step = int(
            crscenario.obstacle_by_id(
                ego_id
            ).prediction.trajectory.final_state.time_step
        )
        num_steps = final_time_step + 1
        result["num_steps"] = num_steps

        if args.dataset == "highd" and rule != "R_G1":
            raise ValueError(
                "The legacy highD MICP comparison only has a defensible R_G1 "
                f"mapping; got {rule}"
            )
        rule_monitor = _build_rule_monitor(
            args.dataset,
            args.scenario_dir,
            scenario_stem,
            crscenario,
            ego_id,
            rule,
        )
        other_vehicle = None
        if args.dataset == "highd" or rule == "R_IN4":
            other_id = int(rule_monitor.other_id)
            other_vehicle = world.vehicle_by_id(other_id)

        specification_start = time.perf_counter()
        reference_lane = (
            build_trajectory_reference_lane(
                crscenario.obstacle_by_id(ego_id), ego_vehicle, rule
            )
            if args.reference_path == "trajectory"
            else (ego_vehicle.ref_path_lane or ego_vehicle.get_lane(0))
        )
        if reference_lane is None:
            raise ValueError("Ego vehicle has no usable lane reference")
        benchmark = _build_scenario(
            args.dataset,
            rule,
            world,
            ego_vehicle,
            other_vehicle,
            num_steps,
            reference_lane,
        )
        specification = benchmark.GetSpecification()
        system = benchmark.GetSystem()
        specification_time = time.perf_counter() - specification_start
        result["specification_time"] = specification_time

        initial_lon = ego_vehicle.get_lon_state(0, reference_lane)
        if initial_lon is None:
            raise ValueError("Initial ego state cannot be projected onto reference path")
        x0 = np.array([initial_lon.s, initial_lon.v, initial_lon.a, 0.0])
        # Longitudinal entries [s, v_s, a_s, j_s] selected from the legacy Q.
        Q = np.diag([0.1, 0.5, 0.1, 0.5])
        R = np.eye(1)

        setup_start = time.perf_counter()
        # Import only after main() has selected GRB_LICENSE_FILE. Importing the
        # solver at module load time can make Gurobi cache a different license.
        from stlpy.solvers import GurobiMICPSolver

        solver = GurobiMICPSolver(
            specification,
            system,
            x0,
            num_steps,
            robustness_cost=True,
            verbose=not args.quiet,
        )
        solver.AddQuadraticCost(Q, R)
        solver.AddControlBounds(np.array([-2000.0]), np.array([2000.0]))
        solver.AddStateBounds(
            np.array(
                [getattr(reference_lane, "s_min", 0.0), -np.inf, -np.inf, -np.inf]
            ),
            np.array(
                [
                    getattr(
                        reference_lane,
                        "s_max",
                        float(reference_lane.clcs.length()),
                    ),
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            ),
        )
        if args.time_limit is not None:
            solver.model.setParam("TimeLimit", float(args.time_limit))
        solver.model.update()
        setup_time = time.perf_counter() - setup_start
        result["solver_setup_time"] = setup_time
        result["num_variables"] = int(solver.model.NumVars)
        result["num_binary_variables"] = int(solver.model.NumBinVars)
        result["num_constraints"] = int(solver.model.NumConstrs)

        solve_start = time.perf_counter()
        x, _, robustness, gurobi_runtime = solver.Solve()
        solve_wall_time = time.perf_counter() - solve_start
        result["solve_wall_time"] = solve_wall_time
        result["gurobi_runtime"] = gurobi_runtime
        result["legacy_total_time"] = setup_time + solve_wall_time
        result["core_total_time"] = (
            specification_time + setup_time + solve_wall_time
        )
        result["gurobi_status"] = int(solver.model.Status)
        result["robustness"] = robustness

        if x is not None and math.isfinite(float(robustness)):
            result["micp_feasible"] = "yes"
            result["objective"] = float(solver.model.ObjVal)
            validation_start = time.perf_counter()
            updated_states = _solution_states(x, reference_lane)
            updated_tv = _validate_solution(rule_monitor, ego_id, updated_states)
            result["validation_time"] = time.perf_counter() - validation_start
            result["updated_tv"] = updated_tv
            result["monitor_compliant"] = (
                math.isinf(updated_tv) and updated_tv > 0.0
            )
            if result["monitor_compliant"]:
                result["replanability"] = "yes"
            else:
                result["error"] = (
                    "MICP trajectory remains non-compliant under the rule monitor: "
                    f"updated_tv={updated_tv}"
                )
        else:
            result["error"] = "Gurobi did not return an optimal trajectory"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    args = _parse_args()
    if args.gurobi_license is not None:
        args.gurobi_license = args.gurobi_license.expanduser().resolve()
        if not args.gurobi_license.is_file():
            raise FileNotFoundError(
                f"Gurobi license file not found: {args.gurobi_license}"
            )
        os.environ["GRB_LICENSE_FILE"] = str(args.gurobi_license)
    args.scenario_dir = args.scenario_dir.resolve()
    config = _world_config(args.dataset, args.monitor_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for index, case in enumerate(_read_cases(args.input, args.limit), start=1):
            print(
                f"[{index}] {case['rule']} {case['scenario_id']} ego={case['ego_id']}",
                flush=True,
            )
            result = evaluate_case(args, case, config)
            writer.writerow(result)
            stream.flush()
            print(
                "    "
                f"{result['replanability']}; core={result['core_total_time'] or 'N/A'}; "
                f"binaries={result['num_binary_variables'] or 'N/A'}; "
                f"error={result['error'] or '-'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
