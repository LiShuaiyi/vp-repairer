import csv
import concurrent.futures
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
VIOLATION_CSV = REPO_ROOT / "evaluation" / "config" / "highd_rg1_rg3.csv"
RESULT_CSV = REPO_ROOT / "evaluation" / "config" / "vp_repairer_rg1_rg3_batch_results.csv"
RESULT_PREFIX = "__BATCH_RESULT__="
DEFAULT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
REPAIRER_TYPES = ("vp", "smt")


def load_cases(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def build_config(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
):
    config = RepairerConfiguration()
    config.general.path_scenarios = "/data_linux/Lab/highD-cr-scenarios/highD-repair/"
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.rules = ["R_G1", "R_G3"]
    config.repair.ego_id = ego_id
    config.repair.planner = 1
    config.repair.constraint_mode = 1 if repairer_type == "vp" else 2
    config.repair.sat_solver_mode = sat_solver_mode
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False
    return config


def run_case(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
):
    result = {
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "sat_solver_mode": sat_solver_mode,
        "repairer_type": repairer_type,
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
        "tc_search_time": 0.0,
        "reach_set_time": 0.0,
        "optimization_time": 0.0,
        "core_total_time": 0.0,
        "wall_time": 0.0,
        "error": "",
    }

    case_start_time = time.time()
    try:
        print(
            f"  using VPTrajectoryRepairer from "
            f"{inspect.getfile(VPTrajectoryRepairer)}"
        )
        config = build_config(
            scenario_id,
            ego_id,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
        )
        ego_initial = retrieve_ego_vehicle(config)
        traffic_rule_monitor = STLRuleMonitor(config)
        repairer_cls = VPTrajectoryRepairer if repairer_type == "vp" else SMTTrajectoryRepairer
        repairer = repairer_cls(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()

        result["iterations"] = repairer.nr_iter
        result["tv"] = repairer.tv
        result["tc"] = repairer.tc
        result["domain_dict_size"] = len(getattr(repairer, "domain_dict", {}))
        result["domain_dict_time"] = getattr(repairer, "domain_dict_time", 0.0)

        if getattr(repairer, "runtime_breakdown", None):
            result["sat_time"] = repairer.runtime_breakdown.get("sat", 0.0)
            result["constraint_extraction_time"] = repairer.runtime_breakdown.get(
                "constraint_extraction", 0.0
            )
            result["constraint_conversion_time"] = repairer.runtime_breakdown.get(
                "constraint_conversion", 0.0
            )
            result["lp_time"] = repairer.runtime_breakdown.get("lp", 0.0)
            result["trajectory_build_time"] = repairer.runtime_breakdown.get(
                "trajectory_build", 0.0
            )
            result["compliance_check_time"] = repairer.runtime_breakdown.get(
                "compliance_check", 0.0
            )
            result["core_total_time"] = sum(repairer.runtime_breakdown.values())
        else:
            result["sat_time"] = getattr(repairer, "sat_reasoning_time", 0.0)
            result["tc_search_time"] = getattr(repairer.t_solver, "tc_search_time", 0.0)
            result["reach_set_time"] = getattr(repairer.t_solver, "reach_set_time", 0.0)
            result["optimization_time"] = getattr(repairer.t_solver, "opti_plan_time", 0.0)
            result["core_total_time"] = (
                result["sat_time"]
                + result["tc_search_time"]
                + result["reach_set_time"]
                + result["optimization_time"]
            )

        if repaired_traj is not None:
            result["success"] = True
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


def run_case_isolated(
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str = "domain_dpll",
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

    result = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX):])
            break

    if result is None:
        result = {
            "scenario_id": scenario_id,
            "ego_id": ego_id,
            "sat_solver_mode": sat_solver_mode,
            "repairer_type": repairer_type,
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
            "tc_search_time": 0.0,
            "reach_set_time": 0.0,
            "optimization_time": 0.0,
            "core_total_time": 0.0,
            "wall_time": 0.0,
            "error": f"isolated run failed to return result (exit={completed.returncode})",
        }
    return result


def write_results(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "ego_id",
        "sat_solver_mode",
        "repairer_type",
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
        "tc_search_time",
        "reach_set_time",
        "optimization_time",
        "core_total_time",
        "wall_time",
        "error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    cases = load_cases(VIOLATION_CSV)
    total_cases = len(cases)

    print(f"Loaded {total_cases} cases from {VIOLATION_CSV}")
    print(f"Using up to {DEFAULT_MAX_WORKERS} parallel worker(s)")

    indexed_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        future_to_case = {}
        for idx, case in enumerate(cases, start=1):
            scenario_id = case["scenario_id"]
            ego_id = int(case["ego_id"])
            for repairer_type in REPAIRER_TYPES:
                print(
                    f"[submit {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, "
                    f"repairer={repairer_type}"
                )
                future = executor.submit(
                    run_case_isolated,
                    scenario_id,
                    ego_id,
                    "domain_dpll",
                    repairer_type,
                )
                future_to_case[future] = (idx, scenario_id, ego_id, repairer_type)

        for future in concurrent.futures.as_completed(future_to_case):
            idx, scenario_id, ego_id, repairer_type = future_to_case[future]
            result = future.result()
            indexed_results[(idx, repairer_type)] = result
            print(
                f"[done {idx}/{total_cases}] scenario={scenario_id}, ego_id={ego_id}, repairer={repairer_type}, "
                f"success={result['success']}, iterations={result['iterations']}, "
                f"wall_time={result['wall_time']:.3f}s"
            )

    results = [
        indexed_results[key]
        for key in sorted(indexed_results, key=lambda item: (item[0], REPAIRER_TYPES.index(item[1])))
    ]
    write_results(results, RESULT_CSV)

    success_count = sum(1 for result in results if result["success"])
    print(f"Wrote {len(results)} results to {RESULT_CSV}")
    print(f"Successful repairs: {success_count}/{len(results)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        scenario_id = sys.argv[2]
        ego_id = int(sys.argv[3])
        sat_solver_mode = sys.argv[4] if len(sys.argv) > 4 else "domain_dpll"
        repairer_type = sys.argv[5] if len(sys.argv) > 5 else "vp"
        result = run_case(
            scenario_id,
            ego_id,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
        )
        print(f"{RESULT_PREFIX}{json.dumps(result, default=str)}")
    else:
        main()
