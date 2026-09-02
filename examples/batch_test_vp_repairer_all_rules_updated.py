#!/usr/bin/env python3
"""Run VP/SMT timing batches for all paper rule groups.

The four VP timing components are mutually exclusive and exclude monitor creation
and post-repair compliance validation:

* predicate_value_estimate_time: DomainDPLL predicate/domain construction
* sat_solve_time: SAT solve plus model extraction
* constraint_extract_time: rule extraction plus coordinate conversion
* vp_planning_time: trajectory CLCS, LP, and repaired trajectory construction

SMT component fields are zero; only its repair-method total is reported.  SMT uses
the same multi-configuration fallback idea as the existing batch runners.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from batch_test_vp_repairer_in3_in5_generated import (
    patch_rtamt_bound_alignment_for_batch,
)
import crrepairer.smt.monitor_wrapper as monitor_wrapper


RESULT_PREFIX = "__UPDATED_BATCH_RESULT__="
HIGH_D_ROOT = Path("/data_linux/Lab/highD-cr-scenarios/highD-repair")
MONA_ROOT = Path("/data_linux/mona/scenarios")
IND_ROOT = Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024")
INTERSECTION_RESULT_INDEX = (
    REPO_ROOT / "evaluation/config/vp_smt_repairer_intersection_final_batch_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/vp_repairer_updated_batches")
ISOLATED_WORK_DIR = Path("/tmp/vp_repairer_updated_work")
DEFAULT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
REPAIRER_TYPES = ("vp", "smt")
VP_SAT_SOLVER_MODES = ("dpll", "domain_dpll")
VP_SAT_SOLVER_MODE_ENV = "CRREPAIR_VP_SAT_SOLVER_MODE"
VP_EXTEND_ACCELERATION_REFERENCE_ENV = (
    "CRREPAIR_VP_EXTEND_ACCELERATION_REFERENCE_PATH"
)
IN3_RULE_VARIANT_ENV = "CRREPAIR_BATCH_IN3_RULE_VARIANT"
IN3_RULE_VARIANTS = ("full", "hand_draft")
BATCH_CASE_OUTPUT_ROOT_ENV = "CRREPAIR_BATCH_CASE_OUTPUT_ROOT"
# Exercise both baseline planner modes and both constraint implementations.
# The reachability constraints remain the primary configuration; manual
# constraints are the semantic fallback when reach extraction cannot represent
# a candidate.  On hosts without an MIQP license planner 2 transparently falls
# back to QP, but retaining it keeps the batch policy portable.
SMT_CONFIGURATIONS = ((1, 2), (2, 2), (1, 1), (2, 1))
SMT_FAILED_PREFERENCE = (2, 1)


RULE_SPECS = {
    "rg1": {
        "rule_label": "R_G1",
        "rules": ["R_G1"],
        "csv_paths": [REPO_ROOT / "evaluation/config/highd_rg1.csv"],
        "scenario_root": HIGH_D_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "smt_sat_solver_mode": "domain_dpll",
    },
    # The paper's interstate timing figures merge these 93 MONA RG1 cases
    # with the 100 highD RG1 cases.  Keep this as an explicitly selected
    # auxiliary group so the default eight-rule batch remains unchanged.
    "rg1_mona": {
        "rule_label": "R_G1",
        "rules": ["R_G1"],
        "csv_paths": [REPO_ROOT / "evaluation/config/mona_merge_RG1_results.csv"],
        "scenario_root": MONA_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "smt_sat_solver_mode": "domain_dpll",
        "repairer_type_filter": "vp",
    },
    "rg2": {
        "rule_label": "R_G2",
        "rules": ["R_G2"],
        "csv_paths": [REPO_ROOT / "evaluation/config/highd_rg2.csv"],
        "scenario_root": HIGH_D_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "smt_sat_solver_mode": "domain_dpll",
        "populate_acceleration": True,
    },
    "rg3": {
        "rule_label": "R_G3",
        "rules": ["R_G3"],
        "csv_paths": [REPO_ROOT / "evaluation/config/highd_rg3.csv"],
        "scenario_root": HIGH_D_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "smt_sat_solver_mode": "domain_dpll",
    },
    "rg1_rg3": {
        "rule_label": "R_G1_R_G3",
        "rules": ["R_G1", "R_G3"],
        "csv_paths": [REPO_ROOT / "evaluation/config/highd_rg1_rg3.csv"],
        "scenario_root": HIGH_D_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "smt_sat_solver_mode": "domain_dpll",
    },
    "in1": {
        "rule_label": "R_IN1",
        "rules": ["R_IN1"],
        "csv_paths": [REPO_ROOT / "evaluation/config/ind_in1.csv"],
        "scenario_root": IND_ROOT,
        "vp_planner": 2,
        "vp_constraint_mode": 1,
        "scenario_type": "intersection",
        "N_r": 20,
    },
    "in3": {
        "rule_label": "R_IN3",
        "rules": ["R_IN3"],
        "csv_paths": [
            REPO_ROOT / "evaluation/config/ind_in3_original.csv",
            REPO_ROOT / "evaluation/config/ind_in3_generated.csv",
        ],
        "scenario_root": IND_ROOT,
        "vp_planner": 2,
        "vp_constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
        "N_r": 50,
    },
    "in4": {
        "rule_label": "R_IN4",
        "rules": ["R_IN4"],
        "csv_paths": [REPO_ROOT / "evaluation/config/ind_in4.csv"],
        "scenario_root": IND_ROOT,
        "vp_planner": 1,
        "vp_constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
        "N_r": 20,
    },
    "in5": {
        "rule_label": "R_IN5",
        "rules": ["R_IN5"],
        "csv_paths": [
            REPO_ROOT / "evaluation/config/ind_in5_original.csv",
            REPO_ROOT / "evaluation/config/ind_in5_generated.csv",
        ],
        "scenario_root": IND_ROOT,
        "vp_planner": 2,
        "vp_constraint_mode": 1,
        "scenario_type": "intersection",
        "intersection_type": "dataset",
        "N_r": 149,
    },
}


def configure_in3_rule_variant(variant: str):
    """Select the full or configured hand-draft IN3 formula for this batch.

    Isolated case workers inherit the environment variable, so the parent and
    every child construct the monitor with exactly the same rule variant.
    """
    variant = str(variant).strip().lower()
    if variant not in IN3_RULE_VARIANTS:
        raise ValueError(
            f"Unsupported IN3 rule variant {variant!r}; "
            f"expected one of {IN3_RULE_VARIANTS}."
        )
    rule = "R_IN3" if variant == "full" else "R_IN3_hand_draft"
    RULE_SPECS["in3"]["rule_label"] = rule
    RULE_SPECS["in3"]["rules"] = [rule]


configure_in3_rule_variant(os.environ.get(IN3_RULE_VARIANT_ENV, "full"))


FIELDNAMES = [
    "scenario_id",
    "scenario_path",
    "ego_id",
    "rule",
    "repairer_type",
    "planner",
    "constraint_mode",
    "sat_solver_mode",
    "extend_acceleration_reference_path",
    "attempted_smt_configurations",
    "success",
    "iterations",
    "successful_repair_mode",
    "deceleration_iterations",
    "acceleration_iterations",
    "tv",
    "tc",
    "updated_tv",
    "domain_dict_size",
    "predicate_value_estimate_time",
    "vp_planning_time",
    "constraint_extract_time",
    "sat_solve_time",
    # Legacy component columns retained for the existing plotting workflow.
    "domain_dict_time",
    "sat_time",
    "clcs_time",
    "constraint_extraction_time",
    "constraint_conversion_time",
    "lp_time",
    "trajectory_build_time",
    "tc_search_time",
    "reach_set_time",
    "optimization_time",
    "core_total_time",
    "wall_time",
    "error",
]


def output_name(group: str) -> str:
    return f"vp_repairer_{group}_batch_result_updated.csv"


def normalize_case(row, spec):
    scenario_id = (
        row.get("scenario_id") or row.get("scenario") or row.get("scenario_name")
    )
    scenario_path = row.get("scenario_path") or ""
    if not scenario_id and scenario_path:
        scenario_id = Path(scenario_path).stem
    if not scenario_id:
        raise ValueError(f"Missing scenario id: {row}")
    if scenario_path:
        path = Path(scenario_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        scenario_path = str(path.resolve())
    return {
        "scenario_id": scenario_id,
        "scenario_path": scenario_path,
        "ego_id": int(row["ego_id"]),
        "rule": spec["rule_label"],
    }


def load_intersection_path_index():
    """Recover the exact old/new-converter source selected for each IN case."""
    path_index = {}
    with INTERSECTION_RESULT_INDEX.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            scenario_path = row.get("scenario_path", "")
            if not scenario_path:
                continue
            key = (row["scenario_id"], int(row["ego_id"]), row["rule"])
            previous = path_index.setdefault(key, scenario_path)
            if previous != scenario_path:
                raise ValueError(
                    f"Conflicting scenario paths for {key}: {previous}, {scenario_path}"
                )
    return path_index


def load_group_cases(group: str):
    spec = RULE_SPECS[group]
    cases = []
    for csv_path in spec["csv_paths"]:
        with csv_path.open(newline="") as csv_file:
            rows = csv.DictReader(csv_file)
            repairer_type_filter = spec.get("repairer_type_filter")
            cases.extend(
                normalize_case(row, spec)
                for row in rows
                if not repairer_type_filter
                or row.get("repairer_type") == repairer_type_filter
            )
    if group.startswith("in"):
        path_index = load_intersection_path_index()
        for case in cases:
            key = (case["scenario_id"], case["ego_id"], case["rule"])
            if key not in path_index and group == "in3":
                alternate_rule = (
                    "R_IN3_hand_draft"
                    if case["rule"] == "R_IN3"
                    else "R_IN3"
                )
                key = (case["scenario_id"], case["ego_id"], alternate_rule)
            if key not in path_index:
                raise ValueError(f"Missing scenario-path index for {group} case {key}")
            case["scenario_path"] = path_index[key]
    return cases


def solver_mode(group, repairer_type):
    if repairer_type == "vp":
        mode = os.environ.get(VP_SAT_SOLVER_MODE_ENV, "domain_dpll")
        if mode not in VP_SAT_SOLVER_MODES:
            raise ValueError(
                f"Unsupported VP SAT solver mode {mode!r}; "
                f"expected one of {VP_SAT_SOLVER_MODES}"
            )
        return mode
    return RULE_SPECS[group].get("smt_sat_solver_mode", "dpll")


def extend_acceleration_reference_path_enabled():
    value = os.environ.get(VP_EXTEND_ACCELERATION_REFERENCE_ENV, "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def empty_result(group, case, repairer_type, planner, constraint_mode):
    result = {field: "" for field in FIELDNAMES}
    result.update(
        {
            "scenario_id": case["scenario_id"],
            "scenario_path": case["scenario_path"],
            "ego_id": case["ego_id"],
            "rule": case["rule"],
            "repairer_type": repairer_type,
            "planner": planner,
            "constraint_mode": constraint_mode,
            "sat_solver_mode": solver_mode(group, repairer_type),
            "extend_acceleration_reference_path": (
                extend_acceleration_reference_path_enabled()
                if repairer_type == "vp"
                else ""
            ),
            "attempted_smt_configurations": "",
            "success": False,
            "iterations": 0,
            "successful_repair_mode": "",
            "deceleration_iterations": 0,
            "acceleration_iterations": 0,
            "domain_dict_size": 0,
            "predicate_value_estimate_time": 0.0,
            "vp_planning_time": 0.0,
            "constraint_extract_time": 0.0,
            "sat_solve_time": 0.0,
            "domain_dict_time": 0.0,
            "sat_time": 0.0,
            "clcs_time": 0.0,
            "constraint_extraction_time": 0.0,
            "constraint_conversion_time": 0.0,
            "lp_time": 0.0,
            "trajectory_build_time": 0.0,
            "tc_search_time": 0.0,
            "reach_set_time": 0.0,
            "optimization_time": 0.0,
            "core_total_time": 0.0,
            "wall_time": 0.0,
            "error": "",
        }
    )
    return result


def build_config(group, case, repairer_type, planner, constraint_mode):
    spec = RULE_SPECS[group]
    config = RepairerConfiguration()
    # Batch execution must not write figures/logs into the repository.  These
    # paths are outside every repair-method timer and visualization is disabled.
    output_root = Path(
        os.environ.get(
            BATCH_CASE_OUTPUT_ROOT_ENV,
            "/tmp/vp_repairer_updated_output",
        )
    )
    config.general.path_output = str(output_root) + "/"
    config.general.path_logs = str(output_root / "logs") + "/"
    config.general.path_figures = str(output_root / "figures") + "/"
    if case["scenario_path"]:
        scenario_path = Path(case["scenario_path"])
        config.general.path_scenarios = str(scenario_path.parent) + "/"
        config.general.set_path_scenario(scenario_path.name)
    else:
        config.general.path_scenarios = str(spec["scenario_root"]) + "/"
        config.general.set_path_scenario(case["scenario_id"])
    config.update()
    config.repair.rules = list(spec["rules"])
    config.repair.ego_id = case["ego_id"]
    config.repair.planner = planner
    config.repair.constraint_mode = constraint_mode
    config.repair.sat_solver_mode = solver_mode(group, repairer_type)
    config.repair.extend_acceleration_reference_path = (
        extend_acceleration_reference_path_enabled()
    )
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False
    if "scenario_type" in spec:
        config.repair.scenario_type = spec["scenario_type"]
    if "intersection_type" in spec:
        config.repair.intersection_type = spec["intersection_type"]
    if "N_r" in spec:
        config.repair.N_r = spec["N_r"]
    config.update()
    return config


def populate_acceleration(ego_vehicle, dt):
    states = ego_vehicle.prediction.trajectory.state_list
    for state, next_state in zip(states, states[1:]):
        state.acceleration = (next_state.velocity - state.velocity) / dt


def disable_visualization(repairer):
    tc_object = getattr(getattr(repairer, "t_solver", None), "tc_object", None)
    if tc_object is None:
        return
    if hasattr(tc_object, "_visualize"):
        tc_object._visualize = False
    if hasattr(tc_object, "_save_state_lists"):
        tc_object._save_state_lists = False


def is_positive_infinity(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isinf(value) and value > 0.0


def collect_vp_timing(result, repairer):
    breakdown = getattr(repairer, "runtime_breakdown", {}) or {}
    domain_time = float(getattr(repairer, "domain_dict_time", 0.0) or 0.0)
    domain_breakdown = getattr(repairer, "domain_dict_breakdown", {}) or {}
    breakdown_by_mode = getattr(
        repairer, "domain_dict_breakdown_by_mode", {}
    ) or {}
    if breakdown_by_mode:
        # Deceleration and acceleration use distinct trajectory contexts.  If
        # both phases run, both CLCS setup costs are physically inside the
        # accumulated domain time and belong to VP planning in the paper.
        domain_clcs_time = sum(
            float(item.get("build_trajectory_clcs", 0.0) or 0.0)
            for item in breakdown_by_mode.values()
        )
    else:
        domain_clcs_time = float(
            domain_breakdown.get("build_trajectory_clcs", 0.0) or 0.0
        )
    # The shared trajectory CLCS is first requested while constructing RG
    # predicate domains, so its build time is physically included in
    # domain_dict_time even though it belongs to velocity planning.  Move it
    # out of predicate estimation and merge every CLCS contribution into the
    # LP/planning component used by the paper plots.
    domain_clcs_time = min(max(domain_clcs_time, 0.0), domain_time)
    predicate_time = domain_time - domain_clcs_time
    sat_time = float(breakdown.get("sat", 0.0) or 0.0)
    runtime_clcs_time = float(breakdown.get("clcs", 0.0) or 0.0)
    clcs_time = domain_clcs_time + runtime_clcs_time
    extraction_time = float(breakdown.get("constraint_extraction", 0.0) or 0.0)
    conversion_time = float(breakdown.get("constraint_conversion", 0.0) or 0.0)
    raw_lp_time = float(breakdown.get("lp", 0.0) or 0.0)
    lp_time = clcs_time + raw_lp_time
    trajectory_time = float(breakdown.get("trajectory_build", 0.0) or 0.0)

    result["predicate_value_estimate_time"] = predicate_time
    result["sat_solve_time"] = sat_time
    result["constraint_extract_time"] = extraction_time + conversion_time
    result["vp_planning_time"] = lp_time + trajectory_time

    result["domain_dict_time"] = predicate_time
    result["sat_time"] = sat_time
    result["clcs_time"] = clcs_time
    result["constraint_extraction_time"] = extraction_time
    result["constraint_conversion_time"] = conversion_time
    # lp_time is the paper's velocity-planning component and intentionally
    # includes CLCS preprocessing/construction. clcs_time remains as a
    # diagnostic subset and must not be added to lp_time a second time.
    result["lp_time"] = lp_time
    result["trajectory_build_time"] = trajectory_time
    result["core_total_time"] = sum(
        float(result[key])
        for key in (
            "predicate_value_estimate_time",
            "sat_solve_time",
            "constraint_extract_time",
            "vp_planning_time",
        )
    )


def collect_smt_total(result, repairer):
    # Match SMTTrajectoryRepairer's own reported method total.  Repairer setup,
    # initial monitoring, and the external strict compliance check are excluded.
    sat_time = float(getattr(repairer, "sat_reasoning_time", 0.0) or 0.0)
    t_solver = getattr(repairer, "t_solver", None)
    t_solver_total = float(getattr(t_solver, "total_runtime", 0.0) or 0.0)
    result["core_total_time"] = sat_time + t_solver_total
    # The paper comparison only decomposes the proposed VP method.  All SMT
    # component and legacy timing fields deliberately remain zero.


def run_single_configuration(group, case, repairer_type, planner, constraint_mode):
    result = empty_result(group, case, repairer_type, planner, constraint_mode)
    wall_start = time.time()
    repairer = None
    try:
        config = build_config(
            group, case, repairer_type, planner, constraint_mode
        )
        ego_vehicle = retrieve_ego_vehicle(config)
        if RULE_SPECS[group].get("populate_acceleration"):
            populate_acceleration(ego_vehicle, config.scenario.dt)

        original_get_config = patch_rtamt_bound_alignment_for_batch(config)
        try:
            rule_monitor = STLRuleMonitor(config)
        finally:
            if original_get_config is not None:
                monitor_wrapper.get_traffic_rule_config = original_get_config

        if rule_monitor.tv_time_step in (-math.inf, math.inf):
            result["error"] = f"invalid initial tv: {rule_monitor.tv_time_step}"
            return result

        repairer_cls = (
            VPTrajectoryRepairer if repairer_type == "vp" else SMTTrajectoryRepairer
        )
        repairer = repairer_cls(rule_monitor, ego_vehicle, config)
        disable_visualization(repairer)
        repaired_trajectory = repairer.repair()

        result["iterations"] = int(getattr(repairer, "nr_iter", 0) or 0)
        result["successful_repair_mode"] = (
            getattr(repairer, "successful_repair_mode", "")
            if repairer_type == "vp"
            else ""
        )
        phase_iterations = getattr(repairer, "phase_iterations", {}) or {}
        result["deceleration_iterations"] = int(
            phase_iterations.get("deceleration", 0) or 0
        )
        result["acceleration_iterations"] = int(
            phase_iterations.get("acceleration", 0) or 0
        )
        result["tv"] = getattr(repairer, "tv", "")
        result["tc"] = getattr(repairer, "tc", "")
        result["domain_dict_size"] = len(getattr(repairer, "domain_dict", {}))
        if repairer_type == "vp":
            collect_vp_timing(result, repairer)
        else:
            collect_smt_total(result, repairer)

        if repaired_trajectory is None:
            result["error"] = "repair returned None"
            return result

        # This validation is intentionally outside every method timer.
        if repairer_type == "vp":
            updated_tv, _ = repairer.calc_tv_updated(
                repaired_trajectory.state_list, repairer.tc
            )
        else:
            updated_tv, _ = repairer.t_solver.tc_object.calc_tv_updated(
                repaired_trajectory.state_list,
                repairer.t_solver.tc_object.tc,
            )
        result["updated_tv"] = updated_tv
        result["success"] = is_positive_infinity(updated_tv)
        if not result["success"]:
            result["error"] = (
                "repaired trajectory remains non-compliant: "
                f"updated_tv={updated_tv}"
            )
        return result
    except Exception as exc:
        traceback.print_exc()
        if repairer is not None:
            result["iterations"] = int(getattr(repairer, "nr_iter", 0) or 0)
            result["successful_repair_mode"] = getattr(
                repairer, "successful_repair_mode", ""
            )
            phase_iterations = getattr(repairer, "phase_iterations", {}) or {}
            result["deceleration_iterations"] = int(
                phase_iterations.get("deceleration", 0) or 0
            )
            result["acceleration_iterations"] = int(
                phase_iterations.get("acceleration", 0) or 0
            )
            result["tv"] = getattr(repairer, "tv", "")
            result["tc"] = getattr(repairer, "tc", "")
            if repairer_type == "vp":
                collect_vp_timing(result, repairer)
            else:
                collect_smt_total(result, repairer)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["wall_time"] = time.time() - wall_start


def run_smt_with_fallback(group, case):
    attempts = []
    for planner, constraint_mode in SMT_CONFIGURATIONS:
        attempt = run_single_configuration(
            group, case, "smt", planner, constraint_mode
        )
        attempts.append(attempt)
        if attempt["success"]:
            break

    selected = next((attempt for attempt in attempts if attempt["success"]), None)
    if selected is None:
        selected = next(
            (
                attempt
                for attempt in attempts
                if (attempt["planner"], attempt["constraint_mode"])
                == SMT_FAILED_PREFERENCE
            ),
            attempts[-1],
        )
    selected = dict(selected)
    selected["attempted_smt_configurations"] = ";".join(
        f"p{attempt['planner']}_c{attempt['constraint_mode']}:"
        f"{'ok' if attempt['success'] else 'fail'}"
        for attempt in attempts
    )
    selected["iterations"] = sum(int(attempt["iterations"] or 0) for attempt in attempts)
    selected["core_total_time"] = sum(
        float(attempt["core_total_time"] or 0.0) for attempt in attempts
    )
    selected["wall_time"] = sum(
        float(attempt["wall_time"] or 0.0) for attempt in attempts
    )
    return selected


def run_case(group, case, repairer_type):
    spec = RULE_SPECS[group]
    if repairer_type == "vp":
        return run_single_configuration(
            group,
            case,
            "vp",
            spec["vp_planner"],
            spec["vp_constraint_mode"],
        )
    return run_smt_with_fallback(group, case)


def run_case_isolated(group, case, repairer_type, timeout):
    ISOLATED_WORK_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        group,
        repairer_type,
        json.dumps(case),
    ]
    with tempfile.TemporaryDirectory(
        prefix=f"{group}_{repairer_type}_",
        dir=ISOLATED_WORK_DIR,
    ) as work_dir:
        child_env = os.environ.copy()
        child_env[BATCH_CASE_OUTPUT_ROOT_ENV] = str(Path(work_dir) / "output")
        try:
            completed = subprocess.run(
                cmd,
                # Legacy SMT/reachability components write auxiliary files to
                # relative paths.  A per-case cwd and output root prevent
                # concurrent cases from overwriting one another; the temporary
                # directory is removed when this call finishes.
                cwd=work_dir,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            spec = RULE_SPECS[group]
            result = empty_result(
                group,
                case,
                repairer_type,
                spec["vp_planner"] if repairer_type == "vp" else 1,
                spec["vp_constraint_mode"] if repairer_type == "vp" else 2,
            )
            result["error"] = f"isolated run timed out after {timeout}s"
            return result

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])

    result = empty_result(group, case, repairer_type, "", "")
    result["error"] = (
        f"isolated run returned no result (exit={completed.returncode})"
    )
    return result


def write_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def run_group(group, args):
    cases = load_group_cases(group)
    if args.limit is not None:
        cases = cases[: args.limit]
    repairers = tuple(item.strip() for item in args.repairers.split(",") if item.strip())
    output_path = args.output_dir / output_name(group)
    total_runs = len(cases) * len(repairers)
    print(
        f"[{group}] loaded {len(cases)} cases; {total_runs} runs; "
        f"workers={args.max_workers}",
        flush=True,
    )

    indexed_results = {}
    completed_count = 0
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for index, case in enumerate(cases):
            for repairer_type in repairers:
                future = executor.submit(
                    run_case_isolated,
                    group,
                    case,
                    repairer_type,
                    args.timeout,
                )
                future_map[future] = (index, repairer_type)

        for future in concurrent.futures.as_completed(future_map):
            index, repairer_type = future_map[future]
            result = future.result()
            indexed_results[(index, repairer_type)] = result
            completed_count += 1
            ordered = [
                indexed_results[key]
                for key in sorted(
                    indexed_results,
                    key=lambda key: (key[0], repairers.index(key[1])),
                )
            ]
            write_results(ordered, output_path)
            elapsed = time.time() - start
            rate = completed_count / elapsed if elapsed > 0 else 0.0
            eta = (total_runs - completed_count) / rate if rate > 0 else math.inf
            print(
                f"[{group}] {completed_count:4d}/{total_runs} "
                f"({100.0 * completed_count / total_runs:5.1f}%) "
                f"success={sum(bool(row['success']) for row in ordered)} "
                f"errors={sum(bool(row['error']) and row['error'] != 'repair returned None' and not str(row['error']).startswith('repaired trajectory remains') for row in ordered)} "
                f"elapsed={elapsed:7.1f}s eta={eta:7.1f}s",
                flush=True,
            )

    results = [
        indexed_results[key]
        for key in sorted(
            indexed_results,
            key=lambda key: (key[0], repairers.index(key[1])),
        )
    ]
    write_results(results, output_path)
    print(f"[{group}] wrote {len(results)} rows to {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        default=",".join(
            group for group in RULE_SPECS if group != "rg1_mona"
        ),
        help="Comma-separated groups: " + ",".join(RULE_SPECS),
    )
    parser.add_argument("--repairers", default="vp,smt")
    parser.add_argument(
        "--vp-sat-solver-mode",
        choices=VP_SAT_SOLVER_MODES,
        default=os.environ.get(VP_SAT_SOLVER_MODE_ENV, "domain_dpll"),
        help=(
            "SAT solver used by the VP repairer. The default remains "
            "domain_dpll; dpll enumerates the original CNF using standard "
            "failed-model blocking and the same deceleration-then-acceleration "
            "VP phases, but without predicate domains or unsupported-candidate "
            "pre-rejection."
        ),
    )
    parser.add_argument(
        "--extend-acceleration-reference-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve and route-extend the original trajectory in the "
            "acceleration branch. Use --no-extend-acceleration-reference-path "
            "to reuse the route-lane CLCS directly."
        ),
    )
    parser.add_argument(
        "--in3-rule-variant",
        choices=IN3_RULE_VARIANTS,
        default=os.environ.get(IN3_RULE_VARIANT_ENV, "full"),
        help=(
            "IN3 formula used by every repairer and isolated worker: the "
            "current complete rule or the configured hand-draft variant."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def main():
    args = parse_args()
    # Isolated case subprocesses inherit this setting.  Keeping the selection
    # in the unified runner guarantees identical cases, repair configuration,
    # strict success checks, and timing columns for the DPLL ablation.
    os.environ[VP_SAT_SOLVER_MODE_ENV] = args.vp_sat_solver_mode
    os.environ[VP_EXTEND_ACCELERATION_REFERENCE_ENV] = (
        "1" if args.extend_acceleration_reference_path else "0"
    )
    os.environ[IN3_RULE_VARIANT_ENV] = args.in3_rule_variant
    configure_in3_rule_variant(args.in3_rule_variant)
    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    invalid_groups = [group for group in groups if group not in RULE_SPECS]
    if invalid_groups:
        raise ValueError(f"Unknown groups: {invalid_groups}")
    invalid_repairers = [
        item for item in args.repairers.split(",") if item not in REPAIRER_TYPES
    ]
    if invalid_repairers:
        raise ValueError(f"Unknown repairers: {invalid_repairers}")
    for group in groups:
        run_group(group, args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        case_result = run_case(sys.argv[2], json.loads(sys.argv[4]), sys.argv[3])
        print(RESULT_PREFIX + json.dumps(case_result, default=str))
    else:
        main()
