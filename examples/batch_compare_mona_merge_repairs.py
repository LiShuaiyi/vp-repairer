import argparse
import concurrent.futures
import csv
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration, ScenarioType
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT / "evaluation" / "config" / "mona_merge_rg1_rg3_filtered.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "evaluation" / "config" / "mona_merge_vp_vs_smt_results.csv"
)
DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/mona/scenarios")
RESULT_PREFIX = "__BATCH_RESULT__="
DEFAULT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
REPAIRER_TYPES = ("vp", "smt")
SMT_COMBINATIONS = ((1, 1), (1, 2), (2, 1), (2, 2))
FAILED_SMT_PREFERENCE = (1, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare VP and SMT repair results on filtered MONA merge violation cases."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input case CSV. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output result CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help=f"Directory containing MONA scenarios. Default: {DEFAULT_SCENARIO_DIR}",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Only evaluate the first N filtered cases from the input CSV.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel worker count. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=5,
        help="Flush accumulated results every N finished repair runs. Default: 5",
    )
    parser.add_argument(
        "--repairers",
        nargs="+",
        choices=REPAIRER_TYPES,
        default=list(REPAIRER_TYPES),
        help="Which repairers to run. Default: vp smt",
    )
    parser.add_argument(
        "--reuse-vp-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing previously saved VP results. If provided, VP rows "
            "are copied from this file instead of rerunning VP."
        ),
    )
    parser.add_argument(
        "--smt-stop-on-first-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When trying multiple SMT planner/constraint combinations, stop after "
            "the first successful one. Default: true"
        ),
    )
    return parser.parse_args()


def load_cases(csv_path: Path, max_cases: int = None):
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        cases = [row for row in reader if case_has_valid_tv(row)]
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def _parse_tv_values(rule_to_tv: str):
    values = []
    for item in (rule_to_tv or "").split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        _, value = item.split(":", 1)
        value = value.strip()
        if value.lower() in {"inf", "+inf", "-inf"}:
            values.append(math.inf if not value.startswith("-") else -math.inf)
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def case_has_valid_tv(case_row) -> bool:
    tv_values = _parse_tv_values(case_row.get("rule_to_tv", ""))
    if not tv_values:
        tv_values = _parse_tv_values(case_row.get("raw_rule_to_tv", ""))
    if not tv_values:
        return False
    return any(math.isfinite(tv) for tv in tv_values)


def parse_rules(case_row):
    violated_rules = (case_row.get("violated_rules") or "").strip()
    if violated_rules:
        return [rule for rule in violated_rules.split(";") if rule]
    raw_rules = (case_row.get("raw_violated_rules") or "").strip()
    if raw_rules:
        return [rule for rule in raw_rules.split(";") if rule]
    return ["R_G1", "R_G3"]


def build_config(
    scenario_name: str,
    ego_id: int,
    rules,
    scenario_dir: Path,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int = 1,
    constraint_mode: int = None,
):
    config = RepairerConfiguration()
    config.general.path_scenarios = str(scenario_dir)
    config.general.set_path_scenario(scenario_name)
    config.update()
    config.repair.rules = list(rules)
    config.repair.ego_id = ego_id
    config.repair.planner = planner
    if constraint_mode is None:
        constraint_mode = 1 if repairer_type == "vp" else 2
    config.repair.constraint_mode = constraint_mode
    config.repair.sat_solver_mode = sat_solver_mode
    config.repair.scenario_type = ScenarioType.INTERSTATE
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.repair.multiproc = False
    config.debug.show_plots = False
    return config


def run_case(
    case_row,
    scenario_dir: Path,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int = 1,
    constraint_mode: int = None,
):
    scenario_name = case_row["scenario_name"]
    ego_id = int(case_row["ego_id"])
    rules = parse_rules(case_row)

    result = {
        "scenario_name": scenario_name,
        "ego_id": ego_id,
        "violated_rules": ";".join(rules),
        "rule_to_tv": case_row.get("rule_to_tv", ""),
        "lanelet_assigned_steps": case_row.get("lanelet_assigned_steps", ""),
        "raw_violated_rules": case_row.get("raw_violated_rules", ""),
        "raw_rule_to_tv": case_row.get("raw_rule_to_tv", ""),
        "sat_solver_mode": sat_solver_mode,
        "repairer_type": repairer_type,
        "planner": planner,
        "constraint_mode": (
            constraint_mode
            if constraint_mode is not None
            else (1 if repairer_type == "vp" else 2)
        ),
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
            scenario_name,
            ego_id,
            rules,
            scenario_dir=scenario_dir,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
            planner=planner,
            constraint_mode=constraint_mode,
        )
        ego_initial = retrieve_ego_vehicle(config)
        traffic_rule_monitor = STLRuleMonitor(config)
        repairer_cls = (
            VPTrajectoryRepairer if repairer_type == "vp" else SMTTrajectoryRepairer
        )
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
            result["tc_search_time"] = getattr(
                getattr(repairer, "t_solver", None), "tc_search_time", 0.0
            )
            result["reach_set_time"] = getattr(
                getattr(repairer, "t_solver", None), "reach_set_time", 0.0
            )
            result["optimization_time"] = getattr(
                getattr(repairer, "t_solver", None), "opti_plan_time", 0.0
            )
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
    case_row,
    scenario_dir: Path,
    sat_solver_mode: str = "domain_dpll",
    repairer_type: str = "vp",
    planner: int = 1,
    constraint_mode: int = None,
):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        json.dumps(case_row),
        str(scenario_dir),
        sat_solver_mode,
        repairer_type,
        str(planner),
        str(constraint_mode if constraint_mode is not None else ""),
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
            "scenario_name": case_row.get("scenario_name", ""),
            "ego_id": case_row.get("ego_id", ""),
            "violated_rules": case_row.get("violated_rules", ""),
            "rule_to_tv": case_row.get("rule_to_tv", ""),
            "lanelet_assigned_steps": case_row.get("lanelet_assigned_steps", ""),
            "raw_violated_rules": case_row.get("raw_violated_rules", ""),
            "raw_rule_to_tv": case_row.get("raw_rule_to_tv", ""),
            "sat_solver_mode": sat_solver_mode,
            "repairer_type": repairer_type,
            "planner": planner,
            "constraint_mode": constraint_mode,
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
            "error": (
                f"isolated run failed to return result (exit={completed.returncode})"
            ),
        }
    return result


def _normalize_previous_result(row):
    normalized = dict(row)
    normalized.setdefault("planner", "")
    normalized.setdefault("constraint_mode", "")
    normalized.setdefault("selection_reason", "reused_previous_result")
    return normalized


def load_previous_vp_results(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader if row.get("repairer_type") == "vp"]
    result_by_case = {}
    for row in rows:
        key = (row.get("scenario_name"), str(row.get("ego_id")))
        result_by_case[key] = _normalize_previous_result(row)
    return result_by_case


def select_smt_result(results):
    for planner, constraint_mode in SMT_COMBINATIONS:
        for result in results:
            if (
                result.get("planner") == planner
                and result.get("constraint_mode") == constraint_mode
                and result.get("success") is True
            ):
                result["selection_reason"] = "first_successful_smt_configuration"
                return result

    preferred_planner, preferred_constraint_mode = FAILED_SMT_PREFERENCE
    for result in results:
        if (
            result.get("planner") == preferred_planner
            and result.get("constraint_mode") == preferred_constraint_mode
        ):
            result["selection_reason"] = "preferred_failed_smt_configuration"
            return result

    fallback = results[0]
    fallback["selection_reason"] = "fallback_failed_smt_configuration"
    return fallback


def run_smt_case_multi_config_isolated(
    case_row,
    scenario_dir: Path,
    sat_solver_mode: str = "domain_dpll",
    stop_on_first_success: bool = True,
):
    all_results = []
    for planner, constraint_mode in SMT_COMBINATIONS:
        result = run_case_isolated(
            case_row,
            scenario_dir,
            sat_solver_mode=sat_solver_mode,
            repairer_type="smt",
            planner=planner,
            constraint_mode=constraint_mode,
        )
        all_results.append(result)
        if stop_on_first_success and result.get("success") is True:
            break

    selected = select_smt_result(all_results)
    selected["attempted_smt_configurations"] = ";".join(
        f"p{result.get('planner')}_c{result.get('constraint_mode')}:{'ok' if result.get('success') else 'fail'}"
        for result in all_results
    )
    return selected


def write_results(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_name",
        "ego_id",
        "violated_rules",
        "rule_to_tv",
        "lanelet_assigned_steps",
        "raw_violated_rules",
        "raw_rule_to_tv",
        "sat_solver_mode",
        "repairer_type",
        "planner",
        "constraint_mode",
        "selection_reason",
        "attempted_smt_configurations",
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
    args = parse_args()
    cases = load_cases(args.input_csv, args.max_cases)
    total_cases = len(cases)

    print(f"Loaded {total_cases} cases from {args.input_csv}")
    print(f"Using up to {args.max_workers} parallel worker(s)")

    indexed_results = {}
    completed_runs_since_flush = 0
    reused_vp_results = {}
    if args.reuse_vp_csv is not None and args.reuse_vp_csv.exists():
        reused_vp_results = load_previous_vp_results(args.reuse_vp_csv)
        print(f"Loaded {len(reused_vp_results)} reusable VP result(s) from {args.reuse_vp_csv}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_case = {}
        for idx, case in enumerate(cases, start=1):
            scenario_name = case["scenario_name"]
            ego_id = int(case["ego_id"])
            for repairer_type in args.repairers:
                if repairer_type == "vp" and reused_vp_results:
                    reused = reused_vp_results.get((scenario_name, str(ego_id)))
                    if reused is not None:
                        indexed_results[(idx, repairer_type)] = reused
                        continue
                print(
                    f"[submit {idx}/{total_cases}] scenario={scenario_name}, "
                    f"ego_id={ego_id}, repairer={repairer_type}"
                )
                if repairer_type == "smt":
                    future = executor.submit(
                        run_smt_case_multi_config_isolated,
                        case,
                        args.scenario_dir,
                        "domain_dpll",
                        args.smt_stop_on_first_success,
                    )
                else:
                    future = executor.submit(
                        run_case_isolated,
                        case,
                        args.scenario_dir,
                        "domain_dpll",
                        repairer_type,
                        1,
                        1,
                    )
                future_to_case[future] = (idx, scenario_name, ego_id, repairer_type)

        for future in concurrent.futures.as_completed(future_to_case):
            idx, scenario_name, ego_id, repairer_type = future_to_case[future]
            result = future.result()
            indexed_results[(idx, repairer_type)] = result
            completed_runs_since_flush += 1
            print(
                f"[done {idx}/{total_cases}] scenario={scenario_name}, "
                f"ego_id={ego_id}, repairer={repairer_type}, "
                f"success={result['success']}, iterations={result['iterations']}, "
                f"wall_time={result['wall_time']:.3f}s"
            )

            if completed_runs_since_flush >= max(args.write_every, 1):
                partial_results = [
                    indexed_results[key]
                    for key in sorted(
                        indexed_results,
                        key=lambda item: (item[0], args.repairers.index(item[1])),
                    )
                ]
                write_results(partial_results, args.output_csv)
                print(
                    f"Flushed {len(partial_results)} result(s) to {args.output_csv}"
                )
                completed_runs_since_flush = 0

    results = [
        indexed_results[key]
        for key in sorted(
            indexed_results,
            key=lambda item: (item[0], args.repairers.index(item[1])),
        )
    ]
    write_results(results, args.output_csv)

    success_count = sum(1 for result in results if result["success"])
    print(f"Wrote {len(results)} results to {args.output_csv}")
    print(f"Successful repairs: {success_count}/{len(results)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        case_row = json.loads(sys.argv[2])
        scenario_dir = Path(sys.argv[3])
        sat_solver_mode = sys.argv[4] if len(sys.argv) > 4 else "domain_dpll"
        repairer_type = sys.argv[5] if len(sys.argv) > 5 else "vp"
        planner = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else 1
        constraint_mode = (
            int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None
        )
        result = run_case(
            case_row,
            scenario_dir=scenario_dir,
            sat_solver_mode=sat_solver_mode,
            repairer_type=repairer_type,
            planner=planner,
            constraint_mode=constraint_mode,
        )
        print(f"{RESULT_PREFIX}{json.dumps(result, default=str)}")
    else:
        main()
