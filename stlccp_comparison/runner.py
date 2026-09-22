"""Batch runner for the STLCCP comparison baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stlccp")

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World

from micp_comparison.common import (
    make_monitor,
    read_cases,
    resolve_scenario_path,
    select_fixed_other_id,
    validate_states,
)
from micp_comparison.rules import RULE_CLASSES
from micp_comparison.runner import candidate_states, world_config

from .decomposition import decompose_formula
from .solver import STLCCPSolver


FIELDS = [
    "scenario_id", "scenario_path", "ego_id", "rule", "repeat",
    "rule_semantics", "num_steps", "smoothing", "initialization", "seed",
    "solver_feasible", "encoded_compliant", "monitor_compliant", "success",
    "specification_time", "decomposition_time", "model_setup_time",
    "solver_setup_time", "solve_time", "core_total_time", "validation_time",
    "qp_solver_time", "gurobi_runtime", "num_variables",
    "num_binary_variables", "num_constraints", "num_equalities", "num_inequalities",
    "num_auxiliary_variables", "num_disjunctive_constraints",
    "num_predicate_occurrences", "ccp_iterations", "ccp_converged",
    "max_final_slack", "solver_status", "objective", "robustness",
    "fixed_other_id", "updated_tv", "updated_other_id", "updated_diagnostics",
    "candidate_sd_at_tv", "phase_history", "error",
]


def default_gurobi_license():
    candidates = (
        Path.home() / "gurobi.lic",
        Path(
            "/data_linux/planning-sim/repairer/commonroad-repairer-vp/"
            "autoware-repair-docker/gurobi.lic"
        ),
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


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
    parser.add_argument("--only-recorded-violations", action="store_true")
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--robustness-margin", type=float, default=0.01)
    parser.add_argument(
        "--rule-semantics",
        choices=(
            "lin2025", "vp_compatible", "vp_no_crossing_temporal",
            "vp_no_crossing_rule_only", "vp_quantified",
        ),
        default="lin2025",
    )
    parser.add_argument(
        "--smoothing",
        choices=("lse", "mellowmin", "lse_mellowmin", "true_min"),
        default="lse_mellowmin",
    )
    parser.add_argument("--lse-k", type=float, default=10.0)
    parser.add_argument("--mellowmin-k", type=float, default=1000.0)
    parser.add_argument("--tau0", type=float, default=5e-3)
    parser.add_argument("--tau-max", type=float, default=1e3)
    parser.add_argument("--tau-rate", type=float, default=2.0)
    parser.add_argument("--slack-tolerance", type=float, default=1e-5)
    parser.add_argument("--cost-tolerance", type=float, default=1e-2)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument(
        "--initialization", choices=("trajectory", "rollout", "random"),
        default="trajectory",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--gurobi-license", type=Path,
        default=default_gurobi_license(),
    )
    parser.add_argument(
        "--monitor-config", type=Path,
        default=Path("/data_linux/Lab/commonroad-stl-monitor/crmonitor/config.yaml"),
    )
    return parser.parse_args()


def smoothing_phases(value):
    return ("lse", "mellowmin") if value == "lse_mellowmin" else (value,)


def _derivative(values, dt):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, float(dt), edge_order=1)


def initial_guess(args, ego, lane, system, num_steps, x0, dt, repeat):
    u = np.zeros((system.m, num_steps), dtype=float)
    x = np.zeros((system.n, num_steps), dtype=float)
    x[:, 0] = x0
    for step in range(num_steps - 1):
        x[:, step + 1] = system.A @ x[:, step]
    if args.initialization == "trajectory":
        try:
            sd = np.asarray([
                lane.clcs.convert_to_curvilinear_coords(*ego.states_cr[step].position)
                for step in range(num_steps)
            ], dtype=float).T
            x[0:2] = sd
            x[2] = _derivative(x[0], dt)
            x[3] = _derivative(x[1], dt)
            x[4] = _derivative(x[2], dt)
            x[5] = _derivative(x[3], dt)
            x[6] = _derivative(x[4], dt)
            x[7] = _derivative(x[5], dt)
            u[0] = _derivative(x[6], dt)
            u[1] = _derivative(x[7], dt)
            x[:, 0] = x0
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    elif args.initialization == "random":
        rng = np.random.default_rng(int(args.seed) + int(repeat))
        x += rng.normal(0.0, 0.1, size=x.shape)
        u += rng.normal(0.0, 0.1, size=u.shape)
        x[:, 0] = x0
    return x, u


def comparison_fixed_other_id(monitor, rule, ego_id):
    """Match the established MICP fallback for ego-centric R_G2 cases.

    Some recorded R_G2 violations have no geometrically preceding vehicle at
    the single reported trigger sample.  The Lin2025 rule implementation has
    an explicit ``other is None`` branch which enforces ``not_abrupt`` over
    the horizon.  Treat that branch as the comparison fallback instead of
    turning these cases into setup errors.
    """
    try:
        return select_fixed_other_id(monitor, rule, ego_id)
    except ValueError as exc:
        if rule == "R_G2" and "no preceding vehicle" in str(exc):
            return None
        raise


def validation_other_id(rule, fixed_other_id):
    """Use the same final validation scope as the saved MICP baseline."""
    # R_G2 is ego-centric with an existentially quantified justification.
    # Its final monitor must be allowed to select any valid witness.  The
    # fixed vehicle is only the Lin2025 optimization witness.
    return None if rule == "R_G2" else fixed_other_id


def evaluate(args, case, repeat, config):
    row = {field: "" for field in FIELDS}
    row.update(
        case,
        repeat=repeat,
        rule_semantics=args.rule_semantics,
        smoothing=args.smoothing,
        initialization=args.initialization,
        seed=int(args.seed) + int(repeat),
        solver_feasible=False,
        encoded_compliant=False,
        monitor_compliant=False,
        success=False,
    )
    try:
        path = resolve_scenario_path(args.dataset, args.scenario_dir, case)
        row["scenario_path"] = str(path)
        scenario, _ = CommonRoadFileReader(str(path)).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario, config)
        ego = world.vehicle_by_id(case["ego_id"])
        monitor = make_monitor(
            args.dataset, path, scenario, case["ego_id"], case["rule"]
        )
        fixed_other_id = comparison_fixed_other_id(
            monitor, case["rule"], case["ego_id"]
        )
        other = None if fixed_other_id is None else world.vehicle_by_id(fixed_other_id)
        row["fixed_other_id"] = "" if fixed_other_id is None else fixed_other_id
        final = scenario.obstacle_by_id(
            case["ego_id"]
        ).prediction.trajectory.final_state.time_step
        row["num_steps"] = num_steps = int(final) + 1

        started = time.perf_counter()
        rule = RULE_CLASSES[case["rule"]](
            num_steps,
            world,
            ego,
            world.road_network.lanelet_network,
            scenario.dt,
            other,
            rule_name=case["rule"],
            trigger_step=monitor.tv_time_step,
            rule_semantics=args.rule_semantics,
        )
        specification = rule.GetSpecification()
        system = rule.GetSystem()
        row["specification_time"] = time.perf_counter() - started

        started = time.perf_counter()
        decomposition = decompose_formula(specification, num_steps)
        row["decomposition_time"] = time.perf_counter() - started
        row["num_auxiliary_variables"] = decomposition.num_auxiliary
        row["num_disjunctive_constraints"] = len(
            decomposition.disjunctive_bounds
        )
        row["num_predicate_occurrences"] = decomposition.num_predicate_occurrences

        lane = ego.ref_path_lane or ego.get_lane(0)
        lon, lat = ego.get_lon_state(0, lane), ego.get_lat_state(0, lane)
        x0 = np.array([lon.s, lat.d, lon.v, 0.0, lon.a, 0.0, 0.0, 0.0])
        projection_domain = np.asarray(lane.clcs.curvilinear_projection_domain())
        chart_min = projection_domain.min(axis=0) + 1e-3
        chart_max = projection_domain.max(axis=0) - 1e-3
        solver = STLCCPSolver(
            specification,
            decomposition,
            system,
            x0,
            num_steps,
            Q=np.diag([0.1, 0.1, 0.5, 1.0, 0.1, 0.1, 0.5, 1.0]),
            R=np.eye(2),
            state_min=rule.state_bounds()[0],
            state_max=rule.state_bounds()[1],
            control_min=rule.control_bounds()[0],
            control_max=rule.control_bounds()[1],
            chart_min=chart_min,
            chart_max=chart_max,
            robustness_margin=args.robustness_margin,
            slack_tolerance=args.slack_tolerance,
        )
        row["model_setup_time"] = solver.model_setup_time
        row["solver_setup_time"] = solver.model_setup_time
        row["num_binary_variables"] = 0
        row.update(solver.size_metrics)

        initial_x, initial_u = initial_guess(
            args, ego, lane, system, num_steps, x0, scenario.dt, repeat
        )
        result = solver.solve(
            phases=smoothing_phases(args.smoothing),
            lse_k=args.lse_k,
            mellowmin_k=args.mellowmin_k,
            initial_x=initial_x,
            initial_u=initial_u,
            tau0=args.tau0,
            tau_max=args.tau_max,
            tau_rate=args.tau_rate,
            max_iterations=args.max_iterations,
            cost_tolerance=args.cost_tolerance,
            time_limit=args.time_limit,
            threads=args.threads,
            verbose=not args.quiet,
        )
        row["solve_time"] = result.solve_wall_time
        row["qp_solver_time"] = result.solver_time
        row["gurobi_runtime"] = result.solver_time
        row["ccp_iterations"] = result.iterations
        row["ccp_converged"] = result.converged
        row["max_final_slack"] = result.max_slack
        row["solver_status"] = result.status
        row["objective"] = result.objective
        row["robustness"] = result.robustness
        row["phase_history"] = json.dumps(result.phase_history, sort_keys=True)
        row["core_total_time"] = sum(float(row[name]) for name in (
            "specification_time", "decomposition_time", "model_setup_time", "solve_time"
        ))

        if result.x is not None:
            row["solver_feasible"] = True
            row["encoded_compliant"] = result.robustness >= args.robustness_margin
            started = time.perf_counter()
            (
                row["updated_tv"],
                row["updated_other_id"],
                row["updated_diagnostics"],
            ) = validate_states(
                monitor,
                case["ego_id"],
                candidate_states(result.x, result.u, lane),
                fixed_other_id=validation_other_id(
                    case["rule"], fixed_other_id
                ),
                return_other_id=True,
                return_details=True,
            )
            row["validation_time"] = time.perf_counter() - started
            row["monitor_compliant"] = (
                math.isinf(row["updated_tv"]) and row["updated_tv"] > 0
            )
            row["success"] = row["monitor_compliant"]
            if math.isfinite(row["updated_tv"]) and row["updated_tv"] >= 0:
                index = min(
                    result.x.shape[1] - 1,
                    int(round(row["updated_tv"] / scenario.dt)),
                )
                row["candidate_sd_at_tv"] = json.dumps({
                    "t": index,
                    "s": float(result.x[0, index]),
                    "d": float(result.x[1, index]),
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
        args.input,
        args.rule,
        args.limit,
        args.offset,
        require_recorded_violation=args.only_recorded_violations,
    ))
    config = world_config(args)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for repeat in range(args.repeat):
                print(
                    f"STLCCP {case['rule']} {case['scenario_id']} repeat={repeat}",
                    flush=True,
                )
                row = evaluate(args, case, repeat, config)
                writer.writerow(row)
                stream.flush()
                print(
                    f"  success={row['success']} iterations={row['ccp_iterations'] or 'N/A'} "
                    f"core={row['core_total_time'] or 'N/A'} error={row['error'] or '-'}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
