"""Batch runner for the isolated free lateral/longitudinal MICP baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

# Batch experiments must never open interactive visualization windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.state import CustomState
from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from stlpy.solvers import GurobiMICPSolver

from micp_comparison.common import (
    make_monitor,
    read_cases,
    resolve_scenario_path,
    validate_states,
)
from micp_comparison.fewer_binary_solver import FewerBinaryGurobiSolver
from micp_comparison.rules import RULE_CLASSES


FIELDS = [
    "scenario_id", "scenario_path", "ego_id", "rule", "repeat", "encoding",
    "rule_semantics", "num_steps",
    "solver_feasible", "monitor_compliant", "success", "specification_time",
    "solver_setup_time", "solve_time", "core_total_time", "validation_time",
    "gurobi_runtime", "num_variables", "num_binary_variables",
    "num_constraints", "solver_status", "mip_gap", "robustness",
    "updated_tv", "updated_other_id", "updated_diagnostics",
    "candidate_sd_at_tv", "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("highd", "ind"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument("--rule", choices=tuple(RULE_CLASSES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--only-recorded-violations", action="store_true",
        help="Skip VP result rows with an empty original violation time (tv).",
    )
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--threads", type=int,
        help="Maximum Gurobi threads per rule-level worker (unset keeps Gurobi default).",
    )
    parser.add_argument(
        "--robustness-margin", type=float, default=0.01,
        help="Strict positive STL robustness required to avoid boundary-only repairs.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--encoding", choices=("standard", "fewer_binary"),
        default="standard",
        help="STL MICP encoding; standard reproduces Lin2025/stlpy.",
    )
    parser.add_argument(
        "--rule-semantics",
        choices=(
            "lin2025", "vp_compatible", "vp_no_crossing_temporal",
            "vp_no_crossing_rule_only", "vp_witness", "vp_quantified",
        ),
        default="lin2025",
        help="Rule formula family; vp_compatible contains separately labelled sound repairs.",
    )
    parser.add_argument(
        "--gurobi-license", type=Path,
        default=Path(
            "/data_linux/planning-sim/repairer/commonroad-repairer-vp/"
            "autoware-repair-docker/gurobi.lic"
        ),
    )
    parser.add_argument(
        "--monitor-config", type=Path,
        default=Path("/data_linux/Lab/commonroad-stl-monitor/crmonitor/config.yaml"),
    )
    return parser.parse_args()


def world_config(args):
    config = load_yaml(str(args.monitor_config))
    config["scenario"] = "intersection" if args.dataset == "ind" else "interstate"
    if args.dataset == "ind":
        config["intersection_road_network_param"]["map_type"] = "dataset"
    return config


def candidate_states(x, u, lane):
    states = []
    for k in range(x.shape[1]):
        s, d, vs, vd = (float(x[i, k]) for i in (0, 1, 2, 3))
        a_s = float(x[4, k])
        position = np.asarray(lane.clcs.convert_to_cartesian_coords(s, d), dtype=float)
        speed = math.hypot(vs, vd)
        tangent = float(lane.orientation(s))
        # This Frenet MICP has no yaw state or yaw dynamics.  Its rectangular
        # collision and stop-line footprints are therefore lane-aligned;
        # reconstruct the CommonRoad state with the same convention.  Treating
        # atan2(v_d, v_s) as yaw would incorrectly rotate a laterally shifting
        # (or almost stationary) lane-aligned vehicle.
        orientation = tangent
        states.append(CustomState(
            time_step=k,
            position=position,
            orientation=orientation,
            velocity=max(0.0, speed),
            acceleration=a_s,
        ))
    return states


def evaluate(args, case, repeat, config):
    row = {field: "" for field in FIELDS}
    row.update(
        case, repeat=repeat, encoding=args.encoding,
        rule_semantics=args.rule_semantics,
        solver_feasible=False, monitor_compliant=False, success=False,
    )
    try:
        path = resolve_scenario_path(args.dataset, args.scenario_dir, case)
        row["scenario_path"] = str(path)
        scenario, _ = CommonRoadFileReader(str(path)).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario, config)
        ego = world.vehicle_by_id(case["ego_id"])
        monitor = make_monitor(args.dataset, path, scenario, case["ego_id"], case["rule"])
        other = None
        if case["rule"] not in {"R_G3", "R_IN1"}:
            if monitor.other_id is not None and int(monitor.other_id) != int(case["ego_id"]):
                other = world.vehicle_by_id(int(monitor.other_id))
            elif case["rule"] == "R_G2":
                # R_G2's outer implication reports the ego as other_id.  Lin's
                # MICP takes one witness vehicle, so deterministically select
                # the nearest route-relevant vehicle in front at the trigger.
                step = max(0, int(monitor.tv_time_step or 0))
                lane = ego.get_lane(min(step, ego.end_time))
                candidates = []
                for vehicle_id in world.vehicle_ids_for_time_step(step):
                    if int(vehicle_id) == int(case["ego_id"]):
                        continue
                    target = world.vehicle_by_id(vehicle_id)
                    try:
                        rear = target.rear_s(step, lane)
                        front = ego.front_s(step, lane)
                        if (
                            rear is not None and front is not None
                            and rear >= front
                            and ego.lanes_at_state(step).intersection(
                                target.lanes_at_state(step)
                            )
                        ):
                            candidates.append((rear - front, int(vehicle_id)))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        continue
                if candidates:
                    other = world.vehicle_by_id(min(candidates)[1])
            elif case["rule"].startswith("R_IN"):
                raise ValueError(f"{case['rule']} has no related vehicle in the monitor")
        final = scenario.obstacle_by_id(case["ego_id"]).prediction.trajectory.final_state.time_step
        row["num_steps"] = num_steps = int(final) + 1

        started = time.perf_counter()
        rule_class = RULE_CLASSES[case["rule"]]
        rule = rule_class(
            num_steps, world, ego, world.road_network.lanelet_network,
            scenario.dt, other, rule_name=case["rule"],
            trigger_step=monitor.tv_time_step,
            rule_semantics=args.rule_semantics,
        )
        specification = rule.GetSpecification()
        system = rule.GetSystem()
        row["specification_time"] = time.perf_counter() - started

        lane = ego.ref_path_lane or ego.get_lane(0)
        lon, lat = ego.get_lon_state(0, lane), ego.get_lat_state(0, lane)
        x0 = np.array([lon.s, lat.d, lon.v, 0.0, lon.a, 0.0, 0.0, 0.0])
        started = time.perf_counter()
        # stlpy's public T argument is the final time index; internally it
        # allocates T+1 samples.  The scenario already reports a sample count.
        solver_class = (
            FewerBinaryGurobiSolver
            if args.encoding == "fewer_binary"
            else GurobiMICPSolver
        )
        solver = solver_class(
            specification, system, x0, num_steps - 1,
            robustness_cost=True, verbose=not args.quiet,
        )
        solver.AddQuadraticCost(
            np.diag([0.1, 0.1, 0.5, 1.0, 0.1, 0.1, 0.5, 1.0]),
            np.eye(2),
        )
        control_min, control_max = rule.control_bounds()
        solver.AddControlBounds(control_min, control_max)
        state_min, state_max = rule.state_bounds()
        solver.AddStateBounds(state_min, state_max)
        # Keep the free longitudinal/lateral solution inside the coordinate
        # chart.  This is not a fixed-path constraint: both s and d remain
        # decision variables over the full CLCS projection domain.
        projection_domain = np.asarray(lane.clcs.curvilinear_projection_domain())
        chart_min = projection_domain.min(axis=0) + 1e-3
        chart_max = projection_domain.max(axis=0) - 1e-3
        solver.model.addConstr(solver.x[0, :] >= chart_min[0])
        solver.model.addConstr(solver.x[0, :] <= chart_max[0])
        solver.model.addConstr(solver.x[1, :] >= chart_min[1])
        solver.model.addConstr(solver.x[1, :] <= chart_max[1])
        solver.model.addConstr(solver.rho[0] >= args.robustness_margin)
        if args.time_limit is not None:
            solver.model.setParam("TimeLimit", args.time_limit)
        if args.threads is not None:
            solver.model.setParam("Threads", args.threads)
        solver.model.update()
        row["solver_setup_time"] = time.perf_counter() - started
        row["num_variables"] = int(solver.model.NumVars)
        row["num_binary_variables"] = int(solver.model.NumBinVars)
        row["num_constraints"] = int(solver.model.NumConstrs)

        started = time.perf_counter()
        x, u, robustness, grb_runtime = solver.Solve()
        row["solve_time"] = time.perf_counter() - started
        row["gurobi_runtime"] = grb_runtime
        row["solver_status"] = int(solver.model.Status)
        if solver.model.SolCount > 0 and solver.model.IsMIP:
            row["mip_gap"] = float(solver.model.MIPGap)
        row["robustness"] = robustness
        row["core_total_time"] = sum(float(row[name]) for name in (
            "specification_time", "solver_setup_time", "solve_time"
        ))
        if x is not None:
            row["solver_feasible"] = True
            started = time.perf_counter()
            (
                row["updated_tv"], row["updated_other_id"],
                row["updated_diagnostics"],
            ) = validate_states(
                monitor, case["ego_id"], candidate_states(x, u, lane),
                return_other_id=True, return_details=True,
            )
            row["validation_time"] = time.perf_counter() - started
            row["monitor_compliant"] = math.isinf(row["updated_tv"]) and row["updated_tv"] > 0
            row["success"] = row["monitor_compliant"]
            if math.isfinite(row["updated_tv"]) and row["updated_tv"] >= 0:
                index = min(
                    x.shape[1] - 1,
                    int(round(row["updated_tv"] / scenario.dt)),
                )
                row["candidate_sd_at_tv"] = json.dumps({
                    "t": index, "s": float(x[0, index]), "d": float(x[1, index]),
                })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return row


def main():
    args = parse_args()
    if args.gurobi_license:
        os.environ["GRB_LICENSE_FILE"] = str(args.gurobi_license.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = list(read_cases(
        args.input, args.rule, args.limit, args.offset,
        require_recorded_violation=args.only_recorded_violations,
    ))
    config = world_config(args)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for repeat in range(args.repeat):
                print(f"MICP {case['rule']} {case['scenario_id']} repeat={repeat}", flush=True)
                row = evaluate(args, case, repeat, config)
                writer.writerow(row)
                stream.flush()
                print(f"  success={row['success']} core={row['core_total_time'] or 'N/A'} error={row['error'] or '-'}", flush=True)


if __name__ == "__main__":
    main()
