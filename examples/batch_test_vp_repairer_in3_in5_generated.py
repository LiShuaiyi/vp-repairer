import argparse
import csv
import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if sys.path[0] != repo_root_str:
    try:
        sys.path.remove(repo_root_str)
    except ValueError:
        pass
    sys.path.insert(0, repo_root_str)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crrepairer.smt.monitor_wrapper as monitor_wrapper
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


VIOLATION_CSV = REPO_ROOT / "evaluation" / "config" / "generated_in3_in5_tests.csv"
RESULT_CSV = REPO_ROOT / "evaluation" / "config" / "vp_repairer_in3_in5_generated_batch_results.csv"
RESULT_PREFIX = "__BATCH_RESULT__="
DEFAULT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
SCENARIO_ROOT = REPO_ROOT / "scenarios" / "generated_in3_in5_tests"
REPAIRER_TYPES = ("vp", "smt")
TIMED_OPERATOR_PATTERN = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)

plt.ioff()


def load_cases(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def normalize_case(case, default_rule: str):
    scenario_id = case.get("scenario_id") or case.get("scenario")
    scenario_path = case.get("scenario_path") or ""
    if scenario_path and not scenario_id:
        scenario_id = Path(scenario_path).stem
    if not scenario_id:
        raise ValueError(f"Missing scenario id in CSV row: {case}")
    rule = case.get("rule_STL") or case.get("rule") or default_rule
    return {
        "scenario_id": scenario_id,
        "scenario_path": scenario_path,
        "ego_id": int(case["ego_id"]),
        "rule": rule,
    }


def _rule_params(rule: str):
    if rule == "R_IN1":
        return None, 20, 2
    if rule == "R_IN3":
        return "dataset", 50, 2
    if rule == "R_IN3_hand_draft":
        return "hand_draft", 50, 2
    if rule == "R_IN4":
        return "dataset", 20, 1
    if rule == "R_IN5":
        return "dataset", 149, 2
    raise ValueError(f"Unsupported intersection rule: {rule}")


def align_bound_to_sampling_period(bound: float, dt: float, mode: str = "ceil"):
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


def format_time_bound(value: float):
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.12g}"


def align_rtamt_bounds_in_rule(rule_str: str, dt: float, mode: str = "ceil"):
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


def patch_rtamt_bound_alignment_for_batch(config):
    if not any(
        rule in config.repair.rules
        for rule in ("R_IN3", "R_IN5")
    ):
        return None

    original_get_config = monitor_wrapper.get_traffic_rule_config
    dt = float(config.scenario.dt)
    rules = set(config.repair.rules)

    def aligned_get_traffic_rule_config(*args, **kwargs):
        traffic_rules_config = original_get_config(*args, **kwargs)
        traffic_rules = traffic_rules_config.get("traffic_rules", {})
        for rule in rules:
            if rule not in traffic_rules:
                continue
            aligned_rule, changes = align_rtamt_bounds_in_rule(
                traffic_rules[rule],
                dt,
                mode="ceil",
            )
            traffic_rules[rule] = aligned_rule
            for original, low, high, low_aligned, high_aligned in changes:
                print(
                    f"* \t<Batch>: aligned {rule} RTAMT bound {original}: "
                    f"[{low},{high}] -> [{low_aligned},{high_aligned}] for dt={dt}",
                    flush=True,
                )
        return traffic_rules_config

    monitor_wrapper.get_traffic_rule_config = aligned_get_traffic_rule_config
    return original_get_config


def build_config(
    scenario_id: str,
    ego_id: int,
    rule: str,
    scenario_path: str | None = None,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int | None = None,
    constraint_mode: int | None = None,
):
    intersection_type, n_r, default_planner = _rule_params(rule)
    config = RepairerConfiguration()
    if scenario_path:
        path = Path(scenario_path)
        config.general.path_scenarios = str(path.parent) + "/"
        config.general.set_path_scenario(path.name)
    else:
        config.general.path_scenarios = str(SCENARIO_ROOT) + "/"
        config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    if intersection_type is not None:
        config.repair.intersection_type = intersection_type
    config.repair.rules = [rule]
    config.repair.ego_id = ego_id
    config.repair.N_r = n_r
    config.repair.planner = planner if planner is not None else default_planner
    if constraint_mode is not None:
        config.repair.constraint_mode = constraint_mode
    elif repairer_type == "vp":
        config.repair.constraint_mode = 1
    else:
        config.repair.constraint_mode = 2
    config.repair.sat_solver_mode = sat_solver_mode
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False
    config.update()
    return config


def disable_batch_visualization(repairer):
    tc_object = getattr(getattr(repairer, "t_solver", None), "tc_object", None)
    if tc_object is not None:
        if hasattr(tc_object, "_visualize"):
            tc_object._visualize = False
        if hasattr(tc_object, "_save_state_lists"):
            tc_object._save_state_lists = False


def empty_result(
    scenario_id: str,
    scenario_path: str,
    ego_id: int,
    rule: str,
    sat_solver_mode: str,
    repairer_type: str,
    planner: int | None,
):
    return {
        "scenario_id": scenario_id,
        "scenario_path": scenario_path,
        "ego_id": ego_id,
        "rule": rule,
        "repairer_type": repairer_type,
        "planner": planner if planner is not None else _rule_params(rule)[2],
        "sat_solver_mode": sat_solver_mode,
        "success": False,
        "iterations": 0,
        "tv": "",
        "tc": "",
        "updated_tv": "",
        "candidate_tvs": "",
        "candidate_diagnostics": "",
        "domain_dict_size": 0,
        "domain_dict_time": 0.0,
        "sat_time": 0.0,
        "clcs_time": 0.0,
        "constraint_extraction_time": 0.0,
        "constraint_conversion_time": 0.0,
        "lp_time": 0.0,
        "trajectory_build_time": 0.0,
        "core_total_time": 0.0,
        "wall_time": 0.0,
        "error": "",
    }


def run_case(
    scenario_id: str,
    ego_id: int,
    rule: str,
    scenario_path: str = "",
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int | None = None,
    constraint_mode: int | None = None,
):
    result = empty_result(
        scenario_id,
        scenario_path,
        ego_id,
        rule,
        sat_solver_mode,
        repairer_type,
        planner,
    )
    case_start_time = time.time()
    try:
        config = build_config(
            scenario_id,
            ego_id,
            rule,
            scenario_path=scenario_path,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
            planner=planner,
            constraint_mode=constraint_mode,
        )
        ego_initial = retrieve_ego_vehicle(config)
        original_get_config = patch_rtamt_bound_alignment_for_batch(config)
        try:
            traffic_rule_monitor = STLRuleMonitor(config)
        finally:
            if original_get_config is not None:
                monitor_wrapper.get_traffic_rule_config = original_get_config
        if traffic_rule_monitor.tv_time_step in (math.inf, -math.inf):
            result["error"] = f"invalid initial tv: {traffic_rule_monitor.tv_time_step}"
            result["wall_time"] = time.time() - case_start_time
            return result

        repairer_cls = VPTrajectoryRepairer if repairer_type == "vp" else SMTTrajectoryRepairer
        repairer = repairer_cls(traffic_rule_monitor, ego_initial, config)
        disable_batch_visualization(repairer)
        repair_start_time = time.time()
        repaired_traj = repairer.repair()
        repair_elapsed = time.time() - repair_start_time

        result["iterations"] = repairer.nr_iter
        result["tv"] = repairer.tv
        result["tc"] = repairer.tc
        result["candidate_tvs"] = repr(getattr(repairer, "candidate_tvs", []))
        result["candidate_diagnostics"] = repr(
            getattr(repairer, "candidate_diagnostics", [])
        )
        result["domain_dict_size"] = len(getattr(repairer, "domain_dict", {}))
        result["domain_dict_time"] = getattr(repairer, "domain_dict_time", 0.0)

        if repairer_type == "vp" and getattr(repairer, "runtime_breakdown", None):
            result["sat_time"] = repairer.runtime_breakdown.get("sat", 0.0)
            result["clcs_time"] = repairer.runtime_breakdown.get("clcs", 0.0)
            result["constraint_extraction_time"] = repairer.runtime_breakdown.get("constraint_extraction", 0.0)
            result["constraint_conversion_time"] = repairer.runtime_breakdown.get("constraint_conversion", 0.0)
            result["lp_time"] = repairer.runtime_breakdown.get("lp", 0.0)
            result["trajectory_build_time"] = repairer.runtime_breakdown.get("trajectory_build", 0.0)
            result["core_total_time"] = repairer.core_runtime
        else:
            result["sat_time"] = getattr(repairer, "sat_reasoning_time", 0.0)
            result["core_total_time"] = repair_elapsed

        if repaired_traj is not None:
            result["success"] = True
            if repairer_type == "vp":
                tv_updated, _ = repairer.calc_tv_updated(
                    repaired_traj.state_list,
                    repairer.tc,
                )
            else:
                tv_updated, _ = repairer.t_solver.tc_object.calc_tv_updated(
                    repaired_traj.state_list,
                    repairer.t_solver.tc_object.tc,
                )
            result["updated_tv"] = tv_updated
            result["success"] = math.isinf(tv_updated) and tv_updated > 0
            if not result["success"]:
                result["error"] = (
                    "repaired trajectory remains non-compliant: "
                    f"updated_tv={tv_updated}"
                )
        else:
            result["error"] = "repair returned None"

    except Exception as exc:
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["wall_time"] = time.time() - case_start_time
    return result


def run_smt_case_with_fallback(
    scenario_id: str,
    ego_id: int,
    rule: str,
    scenario_path: str = "",
    sat_solver_mode: str = "dpll",
):
    attempts = [
        (1, 2),
        (2, 2),
    ]
    results = []
    for planner, constraint_mode in attempts:
        result = run_case(
            scenario_id,
            ego_id,
            rule,
            scenario_path=scenario_path,
            sat_solver_mode=sat_solver_mode,
            repairer_type="smt",
            planner=planner,
            constraint_mode=constraint_mode,
        )
        results.append(result)
        if result.get("success"):
            return result
    successful = [res for res in results if res.get("success")]
    if successful:
        return successful[0]
    return results[-1]


def run_case_isolated(
    scenario_id: str,
    ego_id: int,
    rule: str,
    scenario_path: str = "",
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        scenario_id,
        str(ego_id),
        rule,
        scenario_path,
        sat_solver_mode,
        repairer_type,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        result = empty_result(
            scenario_id,
            scenario_path,
            ego_id,
            rule,
            sat_solver_mode,
            repairer_type,
            None,
        )
        result["error"] = "isolated run timed out after 600s"
        return result
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    result = empty_result(scenario_id, scenario_path, ego_id, rule, sat_solver_mode, repairer_type, None)
    result["error"] = f"isolated run failed to return result (exit={completed.returncode})"
    return result


def write_results(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "scenario_path",
        "ego_id",
        "rule",
        "repairer_type",
        "planner",
        "sat_solver_mode",
        "success",
        "iterations",
        "tv",
        "tc",
        "updated_tv",
        "candidate_tvs",
        "candidate_diagnostics",
        "domain_dict_size",
        "domain_dict_time",
        "sat_time",
        "clcs_time",
        "constraint_extraction_time",
        "constraint_conversion_time",
        "lp_time",
        "trajectory_build_time",
        "core_total_time",
        "wall_time",
        "error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch test VP/SMT repairer for generated IN3/IN5 or scanned inD CSV cases.",
    )
    parser.add_argument("--csv", type=Path, default=VIOLATION_CSV)
    parser.add_argument("--output", type=Path, default=RESULT_CSV)
    parser.add_argument(
        "--default-rule",
        default="R_IN3_hand_draft",
        help="Rule used when the CSV has no rule/rule_STL column.",
    )
    parser.add_argument(
        "--repairers",
        default="vp,smt",
        help="Comma-separated repairers to run: vp,smt. Default: vp,smt",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = [normalize_case(case, args.default_rule) for case in load_cases(args.csv)]
    if args.limit is not None:
        cases = cases[: args.limit]
    repairer_types = tuple(
        item.strip() for item in args.repairers.split(",") if item.strip()
    )
    invalid_repairers = [item for item in repairer_types if item not in REPAIRER_TYPES]
    if invalid_repairers:
        raise ValueError(f"Unsupported repairer(s): {','.join(invalid_repairers)}")
    total_cases = len(cases)
    print(f"Loaded {total_cases} cases from {args.csv}")
    print(f"Using up to {args.max_workers} parallel worker(s)")

    indexed_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_case = {}
        for idx, case in enumerate(cases, start=1):
            scenario_id = case["scenario_id"]
            scenario_path = case["scenario_path"]
            ego_id = case["ego_id"]
            rule = case["rule"]
            for repairer_type in repairer_types:
                sat_solver_mode = "domain_dpll" if repairer_type == "vp" else "dpll"
                print(
                    f"[submit {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, "
                    f"rule={rule}, repairer={repairer_type}"
                )
                future = executor.submit(
                    run_case_isolated,
                    scenario_id,
                    ego_id,
                    rule,
                    scenario_path,
                    sat_solver_mode,
                    repairer_type,
                )
                future_to_case[future] = (idx, scenario_id, ego_id, rule, repairer_type)

        for future in concurrent.futures.as_completed(future_to_case):
            idx, scenario_id, ego_id, rule, repairer_type = future_to_case[future]
            result = future.result()
            indexed_results[(idx, repairer_type)] = result
            checkpoint_results = [
                indexed_results[key]
                for key in sorted(
                    indexed_results,
                    key=lambda item: (item[0], repairer_types.index(item[1])),
                )
            ]
            write_results(checkpoint_results, args.output)
            print(
                f"[done {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, "
                f"rule={rule}, repairer={repairer_type}\n"
                f"  success={result['success']}, iterations={result['iterations']}, "
                f"core_total={float(result['core_total_time'] or 0.0):.6f}s, "
                f"wall={float(result['wall_time'] or 0.0):.6f}s, error={result['error']}"
            )

    results = [
        indexed_results[key]
        for key in sorted(indexed_results, key=lambda item: (item[0], repairer_types.index(item[1])))
    ]
    write_results(results, args.output)
    success_count = sum(1 for result in results if result["success"])
    print(f"Finished batch evaluation: {success_count}/{len(results)} successful")
    print(f"Result CSV written to: {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        scenario_id = sys.argv[2]
        ego_id = int(sys.argv[3])
        rule = sys.argv[4]
        scenario_path = sys.argv[5]
        sat_solver_mode = sys.argv[6]
        repairer_type = sys.argv[7] if len(sys.argv) > 7 else "vp"
        if repairer_type == "smt":
            case_result = run_smt_case_with_fallback(
                scenario_id,
                ego_id,
                rule,
                scenario_path,
                sat_solver_mode,
            )
        else:
            case_result = run_case(
                scenario_id,
                ego_id,
                rule,
                scenario_path,
                sat_solver_mode,
                repairer_type,
            )
        print(f"{RESULT_PREFIX}{json.dumps(case_result)}")
    else:
        main()
