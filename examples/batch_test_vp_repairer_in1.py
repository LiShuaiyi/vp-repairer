import csv
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
VIOLATION_CSV = REPO_ROOT / "evaluation" / "config" / "ind_in1.csv"
RESULT_CSV = REPO_ROOT / "evaluation" / "config" / "vp_repairer_in1_batch_results.csv"
RESULT_PREFIX = "__BATCH_RESULT__="
DEFAULT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
SCENARIO_ROOT = "/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024/"
REPAIRER_TYPES = ("vp", "smt")

plt.ioff()


def load_cases(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_config(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int | None = None,
    constraint_mode: int | None = None,
):
    config = RepairerConfiguration()
    config.general.path_scenarios = SCENARIO_ROOT
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.rules = ["R_IN1"]
    config.repair.ego_id = ego_id
    config.repair.N_r = 20
    if planner is not None:
        config.repair.planner = planner
    elif repairer_type == "vp":
        config.repair.planner = 2
    else:
        config.repair.planner = 2

    if constraint_mode is not None:
        config.repair.constraint_mode = constraint_mode
    elif repairer_type == "vp":
        config.repair.constraint_mode = 1
    else:
        config.repair.constraint_mode = 1
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


def run_case(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int | None = None,
    constraint_mode: int | None = None,
):
    result = {
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "rule": "R_IN1",
        "repairer_type": repairer_type,
        "planner": planner if planner is not None else (2 if repairer_type == "vp" else 1),
        "sat_solver_mode": sat_solver_mode,
        "success": False,
        "iterations": 0,
        "tv": "",
        "tc": "",
        "updated_tv": "",
        "domain_dict_size": 0,
        "domain_dict_time": 0.0,
        "sat_time": 0.0,
        "constraint_extraction_time": 0.0,
        "constraint_conversion_time": 0.0,
        "lp_time": 0.0,
        "trajectory_build_time": 0.0,
        "compliance_check_time": 0.0,
        "core_total_time": 0.0,
        "wall_time": 0.0,
        "error": "",
    }

    case_start_time = time.time()
    try:
        config = build_config(
            scenario_id,
            ego_id,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
            planner=planner,
            constraint_mode=constraint_mode,
        )
        ego_initial = retrieve_ego_vehicle(config)
        traffic_rule_monitor = STLRuleMonitor(config)

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
        result["domain_dict_size"] = len(getattr(repairer, "domain_dict", {}))
        result["domain_dict_time"] = getattr(repairer, "domain_dict_time", 0.0)

        if repairer_type == "vp" and getattr(repairer, "runtime_breakdown", None):
            result["sat_time"] = repairer.runtime_breakdown.get("sat", 0.0)
            result["constraint_extraction_time"] = repairer.runtime_breakdown.get("constraint_extraction", 0.0)
            result["constraint_conversion_time"] = repairer.runtime_breakdown.get("constraint_conversion", 0.0)
            result["lp_time"] = repairer.runtime_breakdown.get("lp", 0.0)
            result["trajectory_build_time"] = repairer.runtime_breakdown.get("trajectory_build", 0.0)
            result["compliance_check_time"] = repairer.runtime_breakdown.get("compliance_check", 0.0)
            result["core_total_time"] = sum(repairer.runtime_breakdown.values())
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
        else:
            result["error"] = "repair returned None"

    except Exception as exc:
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["wall_time"] = time.time() - case_start_time
    return result


def _is_valid_timing_result(result):
    error = (result.get("error") or "").strip()
    if not error:
        return True
    return error == "repair returned None"


def _select_best_smt_result(results):
    successful = [res for res in results if res.get("success")]
    if successful:
        return min(successful, key=lambda res: float(res.get("wall_time", math.inf) or math.inf))

    valid = [res for res in results if _is_valid_timing_result(res)]
    if valid:
        return min(valid, key=lambda res: float(res.get("wall_time", math.inf) or math.inf))

    return results[-1]


def run_smt_case_with_fallback(
    scenario_id: str,
    ego_id: int,
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
            sat_solver_mode=sat_solver_mode,
            repairer_type="smt",
            planner=planner,
            constraint_mode=constraint_mode,
        )
        results.append(result)
        if result.get("success"):
            return result
    return _select_best_smt_result(results)


def run_case_isolated(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "dpll",
    repairer_type: str = "vp",
):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        scenario_id,
        str(ego_id),
        sat_solver_mode,
        repairer_type,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])

    return {
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "rule": "R_IN1",
        "repairer_type": repairer_type,
        "planner": 2 if repairer_type == "vp" else 1,
        "sat_solver_mode": sat_solver_mode,
        "success": False,
        "iterations": 0,
        "tv": "",
        "tc": "",
        "updated_tv": "",
        "domain_dict_size": 0,
        "domain_dict_time": 0.0,
        "sat_time": 0.0,
        "constraint_extraction_time": 0.0,
        "constraint_conversion_time": 0.0,
        "lp_time": 0.0,
        "trajectory_build_time": 0.0,
        "compliance_check_time": 0.0,
        "core_total_time": 0.0,
        "wall_time": 0.0,
        "error": f"isolated run failed to return result (exit={completed.returncode})",
    }


def write_results(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
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
        "domain_dict_size",
        "domain_dict_time",
        "sat_time",
        "constraint_extraction_time",
        "constraint_conversion_time",
        "lp_time",
        "trajectory_build_time",
        "compliance_check_time",
        "core_total_time",
        "wall_time",
        "error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_existing_vp_results(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Existing VP result CSV not found: {csv_path}")

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    return [row for row in rows if row.get("repairer_type") == "vp"]


def main(save_csv: bool = True):
    cases = load_cases(VIOLATION_CSV)
    total_cases = len(cases)
    print(f"Loaded {total_cases} cases from {VIOLATION_CSV}")
    print(f"Using up to {DEFAULT_MAX_WORKERS} parallel worker(s)")
    print(f"Loading existing VP results from {RESULT_CSV}")

    vp_results = load_existing_vp_results(RESULT_CSV)
    if len(vp_results) != total_cases:
        raise ValueError(
            f"Existing VP result count ({len(vp_results)}) does not match case count ({total_cases})"
        )

    indexed_results = {}
    for idx, (case, vp_result) in enumerate(zip(cases, vp_results), start=1):
        scenario_id = case["scenario_id"]
        ego_id = int(case["ego_id"])
        if (
            vp_result.get("scenario_id") != scenario_id
            or int(vp_result.get("ego_id")) != ego_id
        ):
            raise ValueError(
                "Existing VP results do not align with ind_in1.csv order at "
                f"index {idx}: expected ({scenario_id}, {ego_id}), got "
                f"({vp_result.get('scenario_id')}, {vp_result.get('ego_id')})"
            )
        indexed_results[(idx, "vp")] = vp_result

    with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        future_to_case = {}
        for idx, case in enumerate(cases, start=1):
            scenario_id = case["scenario_id"]
            ego_id = int(case["ego_id"])
            repairer_type = "smt"
            print(
                f"[submit {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, "
                f"repairer={repairer_type}"
            )
            future = executor.submit(
                run_case_isolated,
                scenario_id,
                ego_id,
                "dpll",
                repairer_type,
            )
            future_to_case[future] = (idx, scenario_id, ego_id, repairer_type)

        for future in concurrent.futures.as_completed(future_to_case):
            idx, scenario_id, ego_id, repairer_type = future_to_case[future]
            result = future.result()
            indexed_results[(idx, repairer_type)] = result
            print(
                f"[done {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, repairer={repairer_type}\n"
                f"  success={result['success']}, iterations={result['iterations']}, "
                f"domain_dict_time={result['domain_dict_time']:.6f}s, "
                f"core_total={result['core_total_time']:.6f}s, wall={result['wall_time']:.6f}s, "
                f"error={result['error']}"
            )

    results = [
        indexed_results[key]
        for key in sorted(indexed_results, key=lambda item: (item[0], REPAIRER_TYPES.index(item[1])))
    ]
    if save_csv:
        write_results(results, RESULT_CSV)

    success_count = sum(1 for result in results if result["success"])
    print(f"Finished batch evaluation: {success_count}/{len(results)} successful")
    if save_csv:
        print(f"Result CSV written to: {RESULT_CSV}")
    else:
        print("Result CSV not written (--no-save)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        repairer_type = sys.argv[5] if len(sys.argv) > 5 else "vp"
        if repairer_type == "smt":
            case_result = run_smt_case_with_fallback(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        else:
            case_result = run_case(sys.argv[2], int(sys.argv[3]), sys.argv[4], repairer_type)
        print(f"{RESULT_PREFIX}{json.dumps(case_result)}")
    else:
        main(save_csv="--no-save" not in sys.argv)
