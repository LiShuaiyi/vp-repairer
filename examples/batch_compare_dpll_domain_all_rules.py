import argparse
import csv
import math
import re
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crrepairer.smt.monitor_wrapper as monitor_wrapper
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = REPO_ROOT / "evaluation" / "config" / "vp_compare_dpll_domain_all_rules.csv"
HIGH_D_ROOT = "/data_linux/Lab/highD-cr-scenarios/highD-repair/"
IND_ROOT = "/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024/"
SAT_SOLVER_MODES = ("dpll", "domain_dpll")
TIMED_OPERATOR_PATTERN = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)

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
        "rule_group": "R_G2",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "highd_rg2.csv",
        "scenario_root": HIGH_D_ROOT,
        "rules": ["R_G2"],
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
        "rule_group": "R_IN3",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "ind_in3_original.csv",
        "scenario_root": IND_ROOT,
        "rules": ["R_IN3_hand_draft"],
        "planner": 2,
        "constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "hand_draft",
        "N_r": 50,
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
    {
        "rule_group": "R_IN5",
        "csv_path": REPO_ROOT / "evaluation" / "config" / "ind_in5_original.csv",
        "scenario_root": IND_ROOT,
        "rules": ["R_IN5"],
        "planner": 2,
        "constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
        "N_r": 149,
    },
]


def load_cases(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def normalize_case(case):
    scenario_id = case.get("scenario_id") or case.get("scenario")
    scenario_path = case.get("scenario_path") or ""
    if scenario_path and not scenario_id:
        scenario_id = Path(scenario_path).stem
    if not scenario_id:
        raise ValueError(f"Missing scenario id in CSV row: {case}")
    return scenario_id, scenario_path, int(case["ego_id"])


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
    if "R_IN5" not in config.repair.rules:
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
    spec,
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str,
    scenario_path: str = "",
):
    config = RepairerConfiguration()
    if scenario_path:
        path = Path(scenario_path)
        config.general.path_scenarios = str(path.parent) + "/"
        config.general.set_path_scenario(path.name)
    else:
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
        config.repair.N_r = spec.get("N_r", 20)

    config.update()
    return config


def disable_batch_visualization(repairer):
    tc_object = getattr(getattr(repairer, "t_solver", None), "tc_object", None)
    if tc_object is not None:
        if hasattr(tc_object, "_visualize"):
            tc_object._visualize = False
        if hasattr(tc_object, "_save_state_lists"):
            tc_object._save_state_lists = False


def plain_dpll_unsupported_vp_predicate(repairer):
    if repairer.sat_solver.solver_mode != "dpll":
        return None
    if repairer.config.repair.constraint_mode != 1:
        return None
    if not any(
        rule in repairer.config.repair.rules
        for rule in ("R_IN1", "R_IN4", "R_IN3_hand_draft", "R_IN5")
    ):
        return None

    selected_props = [prop for prop in (getattr(repairer, "_sel_prop", []) or []) if prop is not None]
    if not selected_props:
        return "empty"

    for prop in selected_props:
        if prop is None:
            continue
        if "R_IN1" in repairer.config.repair.rules:
            if "stop_line" not in prop.name:
                return prop
            continue
        if "in_intersection_conflict_area" not in prop.name:
            return prop
    return None


def reject_plain_dpll_unsupported_vp_predicates(repairer):
    original_repair_with_vp = repairer._repair_with_velocity_planning

    def wrapped_repair_with_vp(*args, **kwargs):
        unsupported_prop = plain_dpll_unsupported_vp_predicate(repairer)
        if unsupported_prop is not None:
            if unsupported_prop == "empty":
                raise RuntimeError("plain DPLL selected no VP-repairable predicates")
            raise RuntimeError(
                "plain DPLL selected unsupported VP predicate: "
                f"{unsupported_prop.name}"
            )
        return original_repair_with_vp(*args, **kwargs)

    repairer._repair_with_velocity_planning = wrapped_repair_with_vp


def include_dpll_solve_time_in_batch_breakdown(repairer):
    original_solve = repairer.sat_solver.solve

    def timed_solve(*args, **kwargs):
        solve_start = time.time()
        result = original_solve(*args, **kwargs)
        solve_elapsed = time.time() - solve_start
        runtime_breakdown = getattr(repairer, "runtime_breakdown", None)
        if runtime_breakdown is not None and "sat" in runtime_breakdown:
            runtime_breakdown["sat"] += solve_elapsed
            repairer.sat_reasoning_time += solve_elapsed
        return result

    repairer.sat_solver.solve = timed_solve


def run_case(
    spec,
    scenario_id: str,
    ego_id: int,
    sat_solver_mode: str,
    scenario_path: str = "",
):
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
        config = build_config(
            spec,
            scenario_id,
            ego_id,
            sat_solver_mode,
            scenario_path=scenario_path,
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

        repairer = VPTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        disable_batch_visualization(repairer)
        include_dpll_solve_time_in_batch_breakdown(repairer)
        reject_plain_dpll_unsupported_vp_predicates(repairer)

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


def append_results(results, csv_path: Path):
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
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def normalize_rule_group_name(rule_group: str):
    normalized = rule_group.strip().upper()
    aliases = {
        "RG1": "R_G1",
        "RG2": "R_G2",
        "RG3": "R_G3",
        "RG1RG3": "R_G1_R_G3",
        "RG1_RG3": "R_G1_R_G3",
        "RIN1": "R_IN1",
        "IN1": "R_IN1",
        "RIN3": "R_IN3",
        "IRIN3": "R_IN3",
        "IN3": "R_IN3",
        "RIN4": "R_IN4",
        "IN4": "R_IN4",
        "RIN5": "R_IN5",
        "IN5": "R_IN5",
    }
    return aliases.get(normalized, normalized)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare dpll and domain_dpll for selected traffic rules.",
    )
    parser.add_argument(
        "--rules",
        default="",
        help=(
            "Comma-separated rule groups to run, e.g. RG2,RIN3,RIN5. "
            "Default: run all configured groups."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to the output CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_CSV,
        help=f"Output CSV path. Default: {RESULT_CSV}",
    )
    return parser.parse_args()


def selected_rule_specs(rule_groups: str):
    if not rule_groups:
        return RULE_SPECS
    selected = {
        normalize_rule_group_name(rule_group)
        for rule_group in rule_groups.split(",")
        if rule_group.strip()
    }
    specs = [spec for spec in RULE_SPECS if spec["rule_group"] in selected]
    missing = selected - {spec["rule_group"] for spec in specs}
    if missing:
        raise ValueError(f"Unknown rule group(s): {','.join(sorted(missing))}")
    return specs


def main():
    args = parse_args()
    rule_specs = selected_rule_specs(args.rules)
    results = []
    total_runs = 0
    for spec in rule_specs:
        cases = load_cases(spec["csv_path"])
        total_runs += len(cases) * len(SAT_SOLVER_MODES)

    run_idx = 0
    for spec in rule_specs:
        cases = load_cases(spec["csv_path"])
        print(f"Loaded {len(cases)} cases for {spec['rule_group']} from {spec['csv_path']}")
        for case in cases:
            scenario_id, scenario_path, ego_id = normalize_case(case)
            for sat_solver_mode in SAT_SOLVER_MODES:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] rule={spec['rule_group']}, scenario={scenario_id}, "
                    f"ego_id={ego_id}, sat_solver={sat_solver_mode}"
                )
                result = run_case(
                    spec,
                    scenario_id,
                    ego_id,
                    sat_solver_mode,
                    scenario_path=scenario_path,
                )
                results.append(result)
                print(
                    f"  success={result['success']}, iterations={result['iterations']}, "
                    f"core_total={result['core_total_time']:.6f}s, "
                    f"repair_time={result['repair_time']:.6f}s, wall={result['wall_time']:.6f}s, "
                    f"error={result['error']}"
                )

    if args.append:
        append_results(results, args.output)
        write_mode = "appended to"
    else:
        write_results(results, args.output)
        write_mode = "written to"
    success_count = sum(1 for result in results if result["success"])
    print(f"Finished comparison: {success_count}/{len(results)} successful")
    print(f"Result CSV {write_mode}: {args.output}")


if __name__ == "__main__":
    main()
