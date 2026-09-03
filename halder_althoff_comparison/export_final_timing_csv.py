"""Export cases and per-rule summaries into one final Halder timing CSV."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_RESULTS = ROOT / "full_results_history_aware_3s_10k" / "results.csv"
FIXED_RESULTS = ROOT / "audit_rg2_in1_fixed_3s_10k" / "results.csv"
COMBINED_RESULTS = ROOT / "rg1_rg3_results_3s_10k" / "results.csv"
MONA_RESULTS = ROOT / "mona_rg1_results_3s_10k" / "results.csv"
OUTPUT = ROOT / "halder_full_timing_results_2026-09-03.csv"
REPLACED_RULES = {"R_G2", "R_IN1"}

CASE_FIELDS = (
    "case_index",
    "scenario_id",
    "ego_id",
    "rule",
    "scenario_type",
    "dv",
    "tc_s",
    "selected_target_id",
    "status",
    "selected_tc_target_compliant",
    "repair_success",
    "outcome",
    "monitor_initialization_time_s",
    "preprocessing_time_s",
    "search_time_s",
    "ha_core_time_s",
    "validation_time_s",
    "wall_time_s",
    "expanded_nodes",
    "generated_nodes",
    "rule_evaluations",
    "rule_evaluation_time_s",
    "vp_core_time_s",
    "ha_over_vp",
    "error",
)

SUMMARY_FIELDS = (
    "total_cases",
    "completed_cases",
    "timeout_cases",
    "error_cases",
    "compliant_cases",
    "compliance_rate_completed",
    "preprocessing_mean_s",
    "search_mean_s",
    "search_median_s",
    "search_p95_s",
    "core_mean_s",
    "core_median_s",
    "core_p95_s",
    "validation_mean_s",
    "wall_mean_s",
    "expanded_nodes_mean",
    "generated_nodes_mean",
    "rule_evaluations_mean",
    "rule_evaluation_time_mean_s",
    "vp_core_mean_s",
    "ha_over_vp_median",
)

OUTPUT_FIELDS = ("row_type",) + CASE_FIELDS + SUMMARY_FIELDS


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, field):
    return [float(row[field]) for row in rows if row.get(field) not in {None, ""}]


def percentile(values, fraction):
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def mean(values):
    return statistics.mean(values) if values else ""


def median(values):
    return statistics.median(values) if values else ""


def main():
    old = [row for row in read_rows(OLD_RESULTS) if row["rule"] not in REPLACED_RULES]
    fixed = read_rows(FIXED_RESULTS)
    combined = read_rows(COMBINED_RESULTS)
    mona = read_rows(MONA_RESULTS)
    rows = sorted(
        old + fixed + combined + mona,
        key=lambda row: (row["rule"], row["scenario_id"], int(row["ego_id"])),
    )
    for case_index, row in enumerate(rows):
        row["case_index"] = str(case_index)

    summaries = []
    for rule in sorted({row["rule"] for row in rows}):
        group = [row for row in rows if row["rule"] == rule]
        completed = [row for row in group if row["status"] == "ok"]
        compliant = [
            row for row in completed
            if row["selected_tc_target_compliant"].lower() == "true"
        ]
        search = numeric(completed, "search_time_s")
        core = numeric(completed, "ha_core_time_s")
        summaries.append(
            {
                "rule": rule,
                "total_cases": len(group),
                "completed_cases": len(completed),
                "timeout_cases": sum(r["status"] == "search_timeout" for r in group),
                "error_cases": sum(r["status"] == "error" for r in group),
                "compliant_cases": len(compliant),
                "compliance_rate_completed": len(compliant) / len(completed) if completed else "",
                "preprocessing_mean_s": mean(numeric(completed, "preprocessing_time_s")),
                "search_mean_s": mean(search),
                "search_median_s": median(search),
                "search_p95_s": percentile(search, 0.95),
                "core_mean_s": mean(core),
                "core_median_s": median(core),
                "core_p95_s": percentile(core, 0.95),
                "validation_mean_s": mean(numeric(completed, "validation_time_s")),
                "wall_mean_s": mean(numeric(completed, "wall_time_s")),
                "expanded_nodes_mean": mean(numeric(completed, "expanded_nodes")),
                "generated_nodes_mean": mean(numeric(completed, "generated_nodes")),
                "rule_evaluations_mean": mean(numeric(completed, "rule_evaluations")),
                "rule_evaluation_time_mean_s": mean(numeric(completed, "rule_evaluation_time_s")),
                "vp_core_mean_s": mean(numeric(completed, "vp_core_time_s")),
                "ha_over_vp_median": median(numeric(completed, "ha_over_vp")),
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            success = (
                row["status"] == "ok"
                and row["selected_tc_target_compliant"].lower() == "true"
            )
            if success:
                outcome = "success"
            elif row["status"] == "search_timeout":
                outcome = "search_timeout"
            elif row["status"] == "error":
                outcome = "setup_error"
            else:
                outcome = "validation_failed"
            output = {field: row.get(field, "") for field in CASE_FIELDS}
            output.update(
                row_type="case",
                repair_success=str(success).lower(),
                outcome=outcome,
            )
            writer.writerow(output)
        for summary in summaries:
            writer.writerow({"row_type": "rule_summary", **summary})

    print(f"wrote {len(rows)} cases and {len(summaries)} summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
