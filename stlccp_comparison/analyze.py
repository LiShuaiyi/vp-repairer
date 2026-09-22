"""Paired accuracy/runtime summary for STLCCP versus VP batch results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from micp_comparison.analyze import load_method, load_vp, summarize


def stlccp_summary(method, vp):
    rows = summarize(method, vp)
    return [
        {
            ("stlccp_" + key.removeprefix("micp_") if key.startswith("micp_") else key): value
            for key, value in row.items()
        }
        for row in rows
    ]


def paired_method_summary(stlccp, micp):
    """Compare STLCCP and MICP on their shared case keys."""
    rows = []
    shared = sorted(set(stlccp).intersection(micp))
    rules = sorted({key[2] for key in shared})
    for rule in (*rules, "ALL"):
        keys = shared if rule == "ALL" else [key for key in shared if key[2] == rule]
        both_success = [
            key for key in keys
            if stlccp[key]["success"] and micp[key]["success"]
        ]
        timed = [
            key for key in keys
            if stlccp[key]["time"] and micp[key]["time"]
        ]
        timed_both_success = [
            key for key in both_success
            if stlccp[key]["time"] and micp[key]["time"]
        ]

        def median_time(method, selected):
            values = [method[key]["time"] for key in selected if method[key]["time"] is not None]
            return statistics.median(values) if values else None

        def geomean_ratio(selected):
            ratios = [stlccp[key]["time"] / micp[key]["time"] for key in selected]
            return math.exp(statistics.mean(map(math.log, ratios))) if ratios else None

        rows.append({
            "rule": rule,
            "n": len(keys),
            "stlccp_feasible": sum(stlccp[key]["feasible"] for key in keys),
            "micp_feasible": sum(micp[key]["feasible"] for key in keys),
            "stlccp_success": sum(stlccp[key]["success"] for key in keys),
            "micp_success": sum(micp[key]["success"] for key in keys),
            "both_success": len(both_success),
            "stlccp_only_success": sum(
                stlccp[key]["success"] and not micp[key]["success"] for key in keys
            ),
            "micp_only_success": sum(
                micp[key]["success"] and not stlccp[key]["success"] for key in keys
            ),
            "neither_success": sum(
                not stlccp[key]["success"] and not micp[key]["success"] for key in keys
            ),
            "timed_n": len(timed),
            "stlccp_time_median": median_time(stlccp, timed),
            "micp_time_median": median_time(micp, timed),
            "stlccp_over_micp_geomean": geomean_ratio(timed),
            "both_success_timed_n": len(timed_both_success),
            "stlccp_time_median_both_success": median_time(stlccp, timed_both_success),
            "micp_time_median_both_success": median_time(micp, timed_both_success),
            "stlccp_over_micp_geomean_both_success": geomean_ratio(timed_both_success),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--vp", nargs="+", type=Path, required=True)
    parser.add_argument("--micp", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vp = load_vp(args.vp)
    stlccp = load_method(args.results)
    report = {"stlccp_vs_vp": stlccp_summary(stlccp, vp)}
    if args.micp:
        micp = load_method(args.micp)
        report["micp_vs_vp"] = summarize(micp, vp)
        report["stlccp_vs_micp"] = paired_method_summary(stlccp, micp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
