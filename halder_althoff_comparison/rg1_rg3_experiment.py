"""Run the combined R_G1 + R_G3 Halder baseline as a separate experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .parallel_full_experiment import FIELDS, _run_case, write_results


def build_manifest(source_csv: Path, highd_root: Path):
    cases = {}
    with source_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("repairer_type") != "vp":
                continue
            if row.get("sat_solver_mode") != "domain_dpll":
                continue
            if row.get("rule") != "R_G1_R_G3":
                continue
            key = (row["scenario_id"], int(row["ego_id"]))
            cases.setdefault(
                key,
                {
                    "scenario_id": row["scenario_id"],
                    "scenario_path": row.get("scenario_path")
                    or str(highd_root / f"{row['scenario_id']}.xml"),
                    "ego_id": int(row["ego_id"]),
                    "rule": "R_G1_R_G3",
                    "scenario_type": "interstate",
                    "intersection_type": "dataset",
                    "dv": 1.0,
                    "tc_s": float(row.get("tc") or 0.0),
                    "vp_core_time_s": float(row["core_total_time"]),
                },
            )
    result = []
    for index, case in enumerate(
        sorted(cases.values(), key=lambda item: (item["scenario_id"], item["ego_id"]))
    ):
        case["case_index"] = index
        result.append(case)
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def write_final_csv(rows, output: Path):
    summary_fields = (
        "total_cases", "completed_cases", "timeout_cases", "error_cases",
        "compliant_cases", "search_mean_s", "search_median_s", "search_p95_s",
        "core_mean_s", "core_median_s", "core_p95_s", "ha_over_vp_median",
    )
    fields = ("row_type", "repair_success", "outcome") + FIELDS + summary_fields
    completed = [row for row in rows if row["status"] == "ok"]
    compliant = [row for row in completed if row["selected_tc_target_compliant"] is True]
    searches = [float(row["search_time_s"]) for row in completed]
    cores = [float(row["ha_core_time_s"]) for row in completed]
    ratios = [float(row["ha_over_vp"]) for row in completed]
    summary_rule = next(iter({row["rule"] for row in rows}))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["case_index"])):
            success = row["status"] == "ok" and row["selected_tc_target_compliant"] is True
            if success:
                outcome = "success"
            elif row["status"] == "search_timeout":
                outcome = "search_timeout"
            elif row["status"] == "error":
                outcome = "setup_error"
            else:
                outcome = "validation_failed"
            output_row = {
                field: (
                    json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                )
                for field in fields
            }
            output_row.update(
                row_type="case", repair_success=str(success).lower(), outcome=outcome
            )
            writer.writerow(output_row)
        writer.writerow(
            {
                "row_type": "rule_summary",
                "rule": summary_rule,
                "total_cases": len(rows),
                "completed_cases": len(completed),
                "timeout_cases": sum(row["status"] == "search_timeout" for row in rows),
                "error_cases": sum(row["status"] == "error" for row in rows),
                "compliant_cases": len(compliant),
                "search_mean_s": statistics.mean(searches) if searches else "",
                "search_median_s": statistics.median(searches) if searches else "",
                "search_p95_s": percentile(searches, 0.95) if searches else "",
                "core_mean_s": statistics.mean(cores) if cores else "",
                "core_median_s": statistics.median(cores) if cores else "",
                "core_p95_s": percentile(cores, 0.95) if cores else "",
                "ha_over_vp_median": statistics.median(ratios) if ratios else "",
            }
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--max-expansions", type=int, default=10_000)
    parser.add_argument(
        "--source-csv", type=Path,
        default=Path("evaluation/config/vp_temporal_full/vp_repairer_rg1_rg3_batch_result_updated.csv"),
    )
    parser.add_argument(
        "--highd-root", type=Path,
        default=Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    )
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("halder_althoff_comparison/rg1_rg3_results_3s_10k"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("halder_althoff_comparison/halder_rg1_rg3_timing_results_2026-09-03.csv"),
    )
    args = parser.parse_args(argv)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "logs").mkdir(exist_ok=True)
    (args.result_dir / "cases").mkdir(exist_ok=True)
    cases = build_manifest(args.source_csv, args.highd_root)
    (args.result_dir / "manifest.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )
    print(f"cases={len(cases)} workers={args.workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_case, case, str(args.result_dir),
                args.timeout_s, args.max_expansions,
            ): case
            for case in cases
        }
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if count % 10 == 0 or row["status"] != "ok":
                print(
                    f"finished={count}/{len(cases)} status={row['status']}",
                    flush=True,
                )
            write_results(rows, args.result_dir / "results.csv")
    write_final_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
