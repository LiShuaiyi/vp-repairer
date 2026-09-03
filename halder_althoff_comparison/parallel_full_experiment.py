"""Run all single-rule RG/IN Halder cases in parallel with resumable outputs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import signal
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace


SOURCE_FILES = (
    ("vp_repairer_rg1_batch_result_updated.csv", "interstate", "R_G1"),
    ("vp_repairer_rg2_batch_result_updated.csv", "interstate", "R_G2"),
    ("vp_repairer_rg3_batch_result_updated.csv", "interstate", "R_G3"),
    ("vp_repairer_in1_batch_result_updated.csv", "intersection", "R_IN1"),
    ("vp_repairer_in3_batch_result_updated.csv", "intersection", "R_IN3_hand_draft"),
    ("vp_repairer_in3_full_batch_result_updated.csv", "intersection", "R_IN3"),
    ("vp_repairer_in4_batch_result_updated.csv", "intersection", "R_IN4"),
    ("vp_repairer_in5_batch_result_updated.csv", "intersection", "R_IN5"),
)

FIELDS = (
    "case_index", "scenario_id", "scenario_path", "ego_id", "rule",
    "scenario_type", "dv", "tc_s", "selected_target_id", "status",
    "selected_tc_target_compliant", "monitor_initialization_time_s",
    "preprocessing_time_s", "search_time_s", "ha_core_time_s",
    "validation_time_s", "wall_time_s", "expanded_nodes", "generated_nodes",
    "rule_evaluations", "rule_evaluation_time_s", "vp_core_time_s",
    "ha_over_vp", "search_state_key", "violation_costs", "error",
)


class CaseTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise CaseTimeout("search-time limit reached")


def build_manifest(source_dir: Path, highd_root: Path):
    cases = {}
    for filename, scenario_type, expected_rule in SOURCE_FILES:
        with (source_dir / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("repairer_type") != "vp":
                    continue
                if row.get("sat_solver_mode") != "domain_dpll":
                    continue
                rule = row["rule"]
                if rule != expected_rule:
                    continue
                scenario_path = row.get("scenario_path", "")
                if not scenario_path:
                    scenario_path = str(highd_root / f"{row['scenario_id']}.xml")
                key = (row["scenario_id"], int(row["ego_id"]), rule)
                cases.setdefault(
                    key,
                    {
                        "scenario_id": row["scenario_id"],
                        "scenario_path": scenario_path,
                        "ego_id": int(row["ego_id"]),
                        "rule": rule,
                        "scenario_type": scenario_type,
                        "intersection_type": "dataset",
                        "dv": 1.0,
                        "tc_s": float(row.get("tc") or 0.0),
                        "vp_core_time_s": float(row["core_total_time"]),
                    },
                )
    result = []
    for index, case in enumerate(
        sorted(cases.values(), key=lambda item: (item["rule"], item["scenario_id"], item["ego_id"]))
    ):
        case["case_index"] = index
        result.append(case)
    return result


def _run_case(case, result_dir, timeout_s, max_expansions):
    # Imports happen once per persistent worker, not in the coordinating process.
    from halder_althoff_comparison.author_reference_planner import plan_with_author_engine
    from halder_althoff_comparison.commonroad_runner import (
        _monitor,
        _states_on_reference,
        extract_problem,
        validate,
        validate_rule_group,
    )

    result_dir = Path(result_dir)
    stem = f"{case['case_index']:04d}_{case['rule']}_{case['scenario_id']}_{case['ego_id']}"
    log_path = result_dir / "logs" / f"{stem}.log"
    json_path = result_dir / "cases" / f"{stem}.json"
    base = {field: "" for field in FIELDS}
    base.update(case)
    base["search_state_key"] = "t_s_v_rule_sufficient_memories"
    started = time.perf_counter()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            planner_rule = "R_G1" if case["rule"] == "R_G1_MONA" else case["rule"]
            monitor_started = time.perf_counter()
            monitor = _monitor(
                Path(case["scenario_path"]),
                case["ego_id"],
                planner_rule,
                case["scenario_type"],
                case["intersection_type"],
            )
            monitor_time = time.perf_counter() - monitor_started
            preprocessing_started = time.perf_counter()
            cfg, env, rules, _s0, _v0, reference_lane = extract_problem(
                monitor,
                planner_rule,
                case["dv"],
                -10.0,
                5.0,
                max_expansions,
            )
            preprocessing_time = time.perf_counter() - preprocessing_started
            ego = monitor.world.vehicle_by_id(case["ego_id"])
            planning_problem = SimpleNamespace(initial_state=ego.states_cr[0])
            signal.alarm(max(1, int(math.ceil(timeout_s))))
            try:
                result = plan_with_author_engine(
                    monitor.world.scenario,
                    planning_problem,
                    reference_lane,
                    cfg,
                    env,
                    rules,
                    history_aware=True,
                )
            finally:
                signal.alarm(0)
            repaired_states = _states_on_reference(result, reference_lane)
            if planner_rule == "R_G1_R_G3":
                compliant, validation_time, validation = validate_rule_group(
                    monitor, case["ego_id"], repaired_states, tc=case["tc_s"]
                )
                target_id = validation["targets"]
            else:
                target_id = monitor.rule_to_other_id.get(planner_rule)
                compliant, validation_time, validation = validate(
                    monitor,
                    case["ego_id"],
                    repaired_states,
                    tc=case["tc_s"],
                    target_id=target_id,
                )
        core_time = preprocessing_time + result.runtime_s
        base.update(
            {
                "status": "ok",
                "selected_target_id": target_id,
                "selected_tc_target_compliant": compliant,
                "monitor_initialization_time_s": monitor_time,
                "preprocessing_time_s": preprocessing_time,
                "search_time_s": result.runtime_s,
                "ha_core_time_s": core_time,
                "validation_time_s": validation_time,
                "wall_time_s": time.perf_counter() - started,
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "rule_evaluations": result.rule_evaluations,
                "rule_evaluation_time_s": result.rule_evaluation_time_s,
                "ha_over_vp": core_time / case["vp_core_time_s"],
                "violation_costs": dict(zip(result.rule_names, result.violation_costs)),
                "validation": validation,
            }
        )
    except CaseTimeout as exc:
        base.update(status="search_timeout", wall_time_s=time.perf_counter() - started, error=str(exc))
    except Exception as exc:
        base.update(
            status="error",
            wall_time_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    json_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return base


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_results(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["case_index"])):
            writer.writerow({field: _csv_value(row.get(field, "")) for field in FIELDS})


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize(rows):
    summary = {"total": len(rows), "by_rule": {}}
    for rule in sorted({row["rule"] for row in rows}):
        group = [row for row in rows if row["rule"] == rule]
        valid = [row for row in group if row["status"] == "ok"]
        cores = [float(row["ha_core_time_s"]) for row in valid]
        searches = [float(row["search_time_s"]) for row in valid]
        ratios = [float(row["ha_over_vp"]) for row in valid]
        summary["by_rule"][rule] = {
            "total": len(group),
            "completed": len(valid),
            "compliant": sum(row["selected_tc_target_compliant"] is True for row in valid),
            "timeouts": sum(row["status"] == "search_timeout" for row in group),
            "errors": sum(row["status"] == "error" for row in group),
            "core_mean_s": statistics.mean(cores) if cores else None,
            "core_median_s": statistics.median(cores) if cores else None,
            "core_p95_s": percentile(cores, 0.95),
            "search_median_s": statistics.median(searches) if searches else None,
            "ha_over_vp_median": statistics.median(ratios) if ratios else None,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--max-expansions", type=int, default=10_000)
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=sorted({item[2] for item in SOURCE_FILES}),
        help="run only the selected rules (default: all)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("evaluation/config/vp_temporal_full"),
    )
    parser.add_argument(
        "--highd-root",
        type=Path,
        default=Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("halder_althoff_comparison/full_results_history_aware"),
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "logs").mkdir(exist_ok=True)
    (args.result_dir / "cases").mkdir(exist_ok=True)
    cases = build_manifest(args.source_dir, args.highd_root)
    if args.rules:
        selected_rules = set(args.rules)
        cases = [case for case in cases if case["rule"] in selected_rules]
    manifest_path = args.result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")

    completed = {}
    if not args.fresh:
        for path in (args.result_dir / "cases").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            completed[int(row["case_index"])] = row
    pending = [case for case in cases if case["case_index"] not in completed]
    print(f"cases={len(cases)} resumed={len(completed)} pending={len(pending)} workers={args.workers}", flush=True)

    rows = list(completed.values())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_case,
                case,
                str(args.result_dir),
                args.timeout_s,
                args.max_expansions,
            ): case
            for case in pending
        }
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if count % 10 == 0 or row["status"] != "ok":
                print(
                    f"finished={len(completed) + count}/{len(cases)} "
                    f"latest={row['rule']}:{row['scenario_id']} status={row['status']}",
                    flush=True,
                )
            write_results(rows, args.result_dir / "results.csv")

    summary = summarize(rows)
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
