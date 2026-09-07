#!/usr/bin/env python3
"""Join strict sampling results to DomainDPLL VP results and report metrics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from .specs import GROUPS, parse_groups
except ImportError:  # direct script execution
    from specs import GROUPS, parse_groups


HERE = Path(__file__).resolve().parent
CASE_FIELDS = [
    "group", "scenario_id", "ego_id", "rule", "vp_success", "sampling_success",
    "vp_core_time_s", "sampling_core_time_s", "runtime_ratio_sampling_over_vp",
    "sampling_candidates_checked", "sampling_error",
]
SUMMARY_FIELDS = [
    "group", "rule", "cases", "vp_successes", "vp_success_rate",
    "sampling_successes", "sampling_success_rate", "vp_mean_time_s",
    "vp_median_time_s", "vp_p95_time_s", "sampling_mean_time_s",
    "sampling_median_time_s", "sampling_p95_time_s", "mean_runtime_ratio",
]


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def load_vp(path):
    selected = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("repairer_type") != "vp" or row.get("sat_solver_mode") != "domain_dpll":
                continue
            key = row["scenario_id"], int(row["ego_id"]), row["rule"]
            # Some legacy multi-rule CSVs contain the same case twice because
            # two source violation lists were merged. benchmark.read_cases()
            # keeps the first occurrence, so the join must make the identical
            # deterministic choice.
            selected.setdefault(key, row)
    return selected


def load_sampling(path):
    selected = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["scenario_id"], int(row["ego_id"]), row["rule"]
            if key in selected:
                raise ValueError(f"Duplicate sampling row {key} in {path}; summarize repeats separately")
            selected[key] = row
    return selected


def metric_row(group, rows):
    vp_times = [float(r["vp_core_time_s"]) for r in rows if r["vp_core_time_s"] != ""]
    sampling_times = [float(r["sampling_core_time_s"]) for r in rows if r["sampling_core_time_s"] != ""]
    ratios = [float(r["runtime_ratio_sampling_over_vp"]) for r in rows if r["runtime_ratio_sampling_over_vp"] != ""]
    vp_successes = sum(truth(r["vp_success"]) for r in rows)
    sampling_successes = sum(truth(r["sampling_success"]) for r in rows)
    return {
        "group": group, "rule": GROUPS[group]["rule"], "cases": len(rows),
        "vp_successes": vp_successes, "vp_success_rate": vp_successes / len(rows) if rows else math.nan,
        "sampling_successes": sampling_successes,
        "sampling_success_rate": sampling_successes / len(rows) if rows else math.nan,
        "vp_mean_time_s": statistics.fmean(vp_times) if vp_times else math.nan,
        "vp_median_time_s": statistics.median(vp_times) if vp_times else math.nan,
        "vp_p95_time_s": percentile(vp_times, .95),
        "sampling_mean_time_s": statistics.fmean(sampling_times) if sampling_times else math.nan,
        "sampling_median_time_s": statistics.median(sampling_times) if sampling_times else math.nan,
        "sampling_p95_time_s": percentile(sampling_times, .95),
        "mean_runtime_ratio": statistics.fmean(ratios) if ratios else math.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/latest")
    args = parser.parse_args()
    case_rows, summary_rows, sampling_rows = [], [], []
    for group in parse_groups(args.groups):
        vp = load_vp(GROUPS[group]["input"])
        sampling_path = args.results_dir / "sampling" / f"{group}.csv"
        if not sampling_path.is_file():
            raise FileNotFoundError(sampling_path)
        sampling = load_sampling(sampling_path)
        sampling_rows.extend(sampling.values())
        extra = sorted(sampling.keys() - vp.keys())
        if extra:
            raise ValueError(f"{group}: {len(extra)} sampling rows are absent from the VP cohort")
        group_rows = []
        # Using the sampling keys also makes --limit/--offset smoke runs
        # summarize the exact tested subset. A complete run still joins every
        # VP case because its sampling CSV contains every cohort key.
        for key, sample_row in sampling.items():
            vp_row = vp[key]
            vp_time, sample_time = number(vp_row.get("core_total_time")), number(sample_row.get("core_total_time"))
            out = {
                "group": group, "scenario_id": key[0], "ego_id": key[1], "rule": key[2],
                "vp_success": truth(vp_row.get("success")),
                "sampling_success": truth(sample_row.get("success")),
                "vp_core_time_s": "" if vp_time is None else vp_time,
                "sampling_core_time_s": "" if sample_time is None else sample_time,
                "runtime_ratio_sampling_over_vp": "" if vp_time in (None, 0) or sample_time is None else sample_time / vp_time,
                "sampling_candidates_checked": sample_row.get("rule_candidates_checked", ""),
                "sampling_error": (sample_row.get("error", "").splitlines() or [""])[0],
            }
            case_rows.append(out)
            group_rows.append(out)
        summary_rows.append(metric_row(group, group_rows))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "case_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS); writer.writeheader(); writer.writerows(case_rows)
    with (args.results_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS); writer.writeheader(); writer.writerows(summary_rows)
    total = metric_row("rg1", case_rows)
    sampling_times = [number(row.get("core_total_time")) for row in sampling_rows]
    sampling_times = [value for value in sampling_times if value is not None]
    vp_times = [number(row.get("vp_core_time_s")) for row in case_rows]
    vp_times = [value for value in vp_times if value is not None]
    limits = sorted({row.get("rule_candidate_limit", "") for row in sampling_rows})
    errors = Counter(
        ((row.get("error", "").split(":", 1)[0]) if row.get("error") else "none")
        for row in sampling_rows
    )
    exhausted = sum(
        not truth(row.get("success"))
        and row.get("rule_candidates_checked") == row.get("rule_candidate_limit")
        and row.get("rule_candidate_limit") not in (None, "", "0")
        for row in sampling_rows
    )
    lines = [
        "# VP repair vs. 2-D sampling replan", "",
        "Strict success means the returned trajectory is collision/kinematics feasible and the independent full STL monitor reports `updated_tv = +inf`.", "",
        "Planning time excludes scenario/monitor setup and the redundant final validation. Sampling time includes rule checks performed while searching candidates.", "",
        f"Candidate-check budget per case/planner invocation: {', '.join(limits)}. Runs used parallel rule cohorts; each planner invocation itself is single-process.", "",
        "| Cohort | Rule | N | VP success | Sampling success | VP mean (ms) | Sampling mean (ms) | Sampling/VP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['rule']} | {row['cases']} | {row['vp_success_rate']:.1%} | "
            f"{row['sampling_success_rate']:.1%} | {1000*row['vp_mean_time_s']:.2f} | "
            f"{1000*row['sampling_mean_time_s']:.2f} | {row['mean_runtime_ratio']:.1f}x |"
        )
    lines += [
        "",
        f"Overall matched cases: {len(case_rows)}; VP success {total['vp_success_rate']:.1%}; sampling success {total['sampling_success_rate']:.1%}.",
        f"Overall core-time mean (available timings): VP {1000*statistics.fmean(vp_times):.2f} ms ({len(vp_times)} cases), sampling {1000*statistics.fmean(sampling_times):.2f} ms ({len(sampling_times)} cases), ratio of means {statistics.fmean(sampling_times)/statistics.fmean(vp_times):.1f}x.",
        f"Sampling failures that exhausted the explicit candidate budget: {exhausted}. Error categories (counted as method failures): {dict(errors)}.",
        "",
    ]
    (args.results_dir / "REPORT.md").write_text("\n".join(lines))
    print(args.results_dir / "REPORT.md")


if __name__ == "__main__":
    main()
