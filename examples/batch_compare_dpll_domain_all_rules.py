import csv
import math
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = REPO_ROOT / "evaluation" / "config" / "vp_compare_dpll_domain_all_rules.csv"
HIGH_D_ROOT = "/data_linux/Lab/highD-cr-scenarios/highD-repair/"
IND_ROOT = "/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024/"
SAT_SOLVER_MODES = ("dpll", "domain_dpll")

plt.ioff()

RULE_SPECS = [
    {
        "rule_group": "R_G1",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "highd_rg1.csv",
        "scenario_root": HIGH_D_ROOT,
        "rules": ["R_G1"],
        "planner": 1,
        "constraint_mode": 1,
    },
    {
        "rule_group": "R_G3",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "highd_rg3.csv",
        "scenario_root": HIGH_D_ROOT,
        "rules": ["R_G3"],
        "planner": 1,
        "constraint_mode": 1,
    },
    {
        "rule_group": "R_G1_R_G3",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "highd_rg1_rg3.csv",
        "scenario_root": HIGH_D_ROOT,
        "rules": ["R_G1", "R_G3"],
        "planner": 2,
        "constraint_mode": 1,
    },
    {
        "rule_group": "R_IN1",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "ind_in1.csv",
        "scenario_root": IND_ROOT,
        "rules": ["R_IN1"],
        "planner": 2,
        "constraint_mode": 1,
        "scenario_type": "intersection",
    },
    {
        "rule_group": "R_IN4",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "ind_in4.csv",
        "scenario_root": IND_ROOT,
        "rules": ["R_IN4"],
        "planner": 1,
        "constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
    },
]


def load_cases(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_config(spec, scenario_id: str, ego_id: int, sat_solver_mode: str):
    config = RepairerConfiguration()
    config.general.path_scenarios = spec["scenario_root"]
    config.general.set_path_scenario(scenario_id)
    config.update()

    config.repair.rules = list(spec["rules"])
    config.repair.ego_id = ego_id
    config.repair.planner = spec["planner"]
    config.repair.constraint_mode = spec["constraint_mode"]
    config.repair.sat_solver_mode = sat_solver_mode
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False

    if "scenario_type" in spec:
        config.repair.scenario_type = spec["scenario_type"]
    if "intersection_type" in spec:
        config.repair.intersection_type = spec["intersection_type"]
    if spec["rule_group"].startswith("R_IN"):
        config.repair.N_r = 20

    config.update()
    return config


def disable_batch_visualization(repairer):
    tc_object = getattr(getattr(repairer, "t_solver", None), "tc_object", None)
    if tc_object is not None:
        if hasattr(tc_object, "_visualize"):
            tc_object._visualize = False
        if hasattr(tc_object, "_save_state_lists"):
            tc_object._save_state_lists = False


def run_case(spec, scenario_id: str, ego_id: int, sat_solver_mode: str):
    result = {
        "rule_group": spec["rule_group"],
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "planner": spec["planner"],
        "constraint_mode": spec["constraint_mode"],
        "sat_solver_mode": sat_solver_mode,
        "success": False,
        "iterations": 0,
        "tv": "",
        "tc": "",
        "updated_tv": "",
        "core_total_time": 0.0,
        "repair_time": 0.0,
        "wall_time": 0.0,
        "error": "",
    }

    case_start_time = time.time()
    try:
        config = build_config(spec, scenario_id, ego_id, sat_solver_mode)
        ego_initial = retrieve_ego_vehicle(config)
        traffic_rule_monitor = STLRuleMonitor(config)

        if traffic_rule_monitor.tv_time_step in (math.inf, -math.inf):
            result["error"] = f"invalid initial tv: {traffic_rule_monitor.tv_time_step}"
            result["wall_time"] = time.time() - case_start_time
            return result

        repairer = VPTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        disable_batch_visualization(repairer)

        repair_start = time.time()
        repaired_traj = repairer.repair()
        result["repair_time"] = time.time() - repair_start

        result["iterations"] = repairer.nr_iter
        result["tv"] = repairer.tv
        result["tc"] = repairer.tc
        if getattr(repairer, "runtime_breakdown", None):
            result["core_total_time"] = sum(repairer.runtime_breakdown.values())

        if repaired_traj is not None:
            result["success"] = True
            tv_updated, _ = repairer.calc_tv_updated(
                repaired_traj.state_list,
                repairer.tc,
            )
            result["updated_tv"] = tv_updated
        else:
            result["error"] = "repair returned None"

    except Exception as exc:
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["wall_time"] = time.time() - case_start_time
    return result


def write_results(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule_group",
        "scenario_id",
        "ego_id",
        "planner",
        "constraint_mode",
        "sat_solver_mode",
        "success",
        "iterations",
        "tv",
        "tc",
        "updated_tv",
        "core_total_time",
        "repair_time",
        "wall_time",
        "error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    results = []
    total_runs = 0
    for spec in RULE_SPECS:
        cases = load_cases(spec["csv_path"])
        total_runs += len(cases) * len(SAT_SOLVER_MODES)

    run_idx = 0
    for spec in RULE_SPECS:
        cases = load_cases(spec["csv_path"])
        print(f"Loaded {len(cases)} cases for {spec['rule_group']} from {spec['csv_path']}")
        for case in cases:
            scenario_id = case["scenario_id"]
            ego_id = int(case["ego_id"])
            for sat_solver_mode in SAT_SOLVER_MODES:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] rule={spec['rule_group']}, scenario={scenario_id}, "
                    f"ego_id={ego_id}, sat_solver={sat_solver_mode}"
                )
                result = run_case(spec, scenario_id, ego_id, sat_solver_mode)
                results.append(result)
                print(
                    f"  success={result['success']}, iterations={result['iterations']}, "
                    f"core_total={result['core_total_time']:.6f}s, "
                    f"repair_time={result['repair_time']:.6f}s, wall={result['wall_time']:.6f}s, "
                    f"error={result['error']}"
                )

    write_results(results, RESULT_CSV)
    success_count = sum(1 for result in results if result["success"])
    print(f"Finished comparison: {success_count}/{len(results)} successful")
    print(f"Result CSV written to: {RESULT_CSV}")


if __name__ == "__main__":
    main()
