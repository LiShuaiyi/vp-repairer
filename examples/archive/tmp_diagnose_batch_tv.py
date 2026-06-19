import contextlib
import csv
import io
import math
import sys
import time
import traceback
from pathlib import Path

from examples.batch_compare_dpll_domain_all_rules import (
    RULE_SPECS,
    SAT_SOLVER_MODES,
    build_config,
    load_cases,
)
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.repair import retrieve_ego_vehicle


MAX_CASES_PER_RULE = 100
RESULT_CSV = Path("/tmp/commonroad_repairer_tv_diagnostic.csv")


def tv_class(value):
    try:
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
        if value == 0:
            return "zero"
        if value < 0:
            return "finite_negative"
        return "finite_positive"
    except Exception:
        return f"non_numeric:{type(value).__name__}"


def compact_lines(text):
    keywords = (
        "Traceback",
        "Exception",
        "Error",
        "failed",
        "Failed",
        "invalid initial tv",
        "VP repair failed for current SAT model",
        "repair returned None",
    )
    lines = []
    for line in text.splitlines():
        if any(keyword in line for keyword in keywords):
            lines.append(line.strip())
    return " | ".join(lines[-8:])


def run_one(spec, case, sat_solver_mode):
    scenario_id = case["scenario_id"]
    ego_id = int(case["ego_id"])
    row = {
        "rule_group": spec["rule_group"],
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "sat_solver_mode": sat_solver_mode,
        "monitor_ok": False,
        "repair_ok": False,
        "repaired": False,
        "tv": "",
        "tv_class": "",
        "rule_to_tv": "",
        "rule_to_tv_classes": "",
        "iterations": 0,
        "error_type": "",
        "error": "",
        "captured_failure_lines": "",
        "potential_masked_bug": False,
        "wall_time": 0.0,
    }
    start = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            config = build_config(spec, scenario_id, ego_id, sat_solver_mode)
            ego_initial = retrieve_ego_vehicle(config)
            monitor = STLRuleMonitor(config)
        row["monitor_ok"] = True
        row["tv"] = repr(monitor.tv_time_step)
        row["tv_class"] = tv_class(monitor.tv_time_step)
        row["rule_to_tv"] = repr(getattr(monitor, "rule_to_tv", {}))
        row["rule_to_tv_classes"] = repr(
            {
                rule: tv_class(tv)
                for rule, tv in getattr(monitor, "rule_to_tv", {}).items()
            }
        )

        repairer = VPTrajectoryRepairer(monitor, ego_initial, config)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            repaired_traj = repairer.repair()
        row["repair_ok"] = True
        row["repaired"] = repaired_traj is not None
        row["iterations"] = repairer.nr_iter
        if repaired_traj is None:
            row["error"] = "repair returned None"

    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error"] = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=8)
        buf.write("\n")
        buf.write(tb)

    captured = buf.getvalue()
    row["captured_failure_lines"] = compact_lines(captured)
    if (
        (not row["repaired"])
        and (
            "VP repair failed for current SAT model" in captured
            or "Traceback" in captured
            or row["error_type"]
        )
    ):
        row["potential_masked_bug"] = True
    row["wall_time"] = time.time() - start
    return row


def main():
    rows = []
    total = sum(min(MAX_CASES_PER_RULE, len(load_cases(spec["csv_path"]))) for spec in RULE_SPECS)
    total *= len(SAT_SOLVER_MODES)
    idx = 0
    print(f"Running TV diagnostic: max {MAX_CASES_PER_RULE} cases per rule group, total runs={total}")
    for spec in RULE_SPECS:
        cases = load_cases(spec["csv_path"])[:MAX_CASES_PER_RULE]
        print(f"Rule {spec['rule_group']}: {len(cases)} case(s)")
        for case in cases:
            for sat_solver_mode in SAT_SOLVER_MODES:
                idx += 1
                row = run_one(spec, case, sat_solver_mode)
                rows.append(row)
                interesting = (
                    row["tv_class"] in ("zero", "finite_negative")
                    or row["error_type"]
                    or row["potential_masked_bug"]
                )
                print(
                    f"[{idx}/{total}] {row['rule_group']} {row['scenario_id']} ego={row['ego_id']} "
                    f"{sat_solver_mode}: tv={row['tv']} ({row['tv_class']}), "
                    f"repaired={row['repaired']}, err={row['error']}"
                    + ("  <CHECK>" if interesting else "")
                )

    fieldnames = list(rows[0].keys()) if rows else []
    with RESULT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    for key in ("tv_class", "rule_group", "error_type"):
        counts = {}
        for row in rows:
            value = row[key] or "<empty>"
            counts[value] = counts.get(value, 0) + 1
        print(f"{key}: {counts}")
    print(f"repair returned None: {sum(1 for row in rows if row['error'] == 'repair returned None')}")
    print(f"potential masked bug rows: {sum(1 for row in rows if row['potential_masked_bug'])}")
    print(f"Result CSV: {RESULT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
