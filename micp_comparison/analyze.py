"""Paired accuracy/runtime summary for MICP versus VP batch CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes", "bingo"}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def load_method(paths):
    grouped = defaultdict(list)
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["scenario_id"].removesuffix(".xml"), int(row["ego_id"]), row["rule"]
                grouped[key].append(row)
    result = {}
    for key, rows in grouped.items():
        times = [finite(row.get("core_total_time")) for row in rows]
        times = [value for value in times if value is not None]
        result[key] = {
            "success": any(truth(row.get("success")) for row in rows),
            "feasible": any(truth(row.get("solver_feasible")) for row in rows),
            "time": statistics.median(times) if times else None,
        }
    return result


def load_vp(paths):
    result = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("repairer_type") not in (None, "", "vp"):
                    continue
                if row.get("sat_solver_mode") not in (None, "", "domain_dpll"):
                    continue
                rule = row.get("rule") or row.get("rule_STL")
                key = row["scenario_id"].removesuffix(".xml"), int(row["ego_id"]), rule
                result[key] = {"success": truth(row.get("success", row.get("repairability"))),
                               "time": finite(row.get("core_total_time", row.get("total_time")))}
    return result


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return center - radius, center + radius


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def summarize(method, vp):
    rows = []
    for rule in sorted({key[2] for key in method}):
        keys = [key for key in method if key[2] == rule]
        success = sum(method[key]["success"] for key in keys)
        feasible = sum(method[key]["feasible"] for key in keys)
        vp_success = sum(vp.get(key, {}).get("success", False) for key in keys)
        paired = [key for key in keys if method[key]["success"] and vp.get(key, {}).get("success")
                  and method[key]["time"] and vp[key]["time"]]
        ratios = [method[key]["time"] / vp[key]["time"] for key in paired]
        low, high = wilson(success, len(keys))
        rows.append({
            "rule": rule, "n": len(keys), "micp_feasible": feasible,
            "micp_success": success, "micp_accuracy": success / len(keys) if keys else None,
            "micp_accuracy_ci_low": low, "micp_accuracy_ci_high": high,
            "vp_success_same_cases": vp_success, "paired_success_n": len(paired),
            "micp_time_median": median(method[key]["time"] for key in keys if method[key]["time"] is not None),
            "micp_time_median_paired": median(method[key]["time"] for key in paired),
            "vp_time_median_same_cases": median(vp[key]["time"] for key in keys if key in vp and vp[key]["time"] is not None),
            "vp_time_median_paired": median(vp[key]["time"] for key in paired),
            "micp_over_vp_geomean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
        })
    keys = list(method)
    paired = [key for key in keys if method[key]["success"] and vp.get(key, {}).get("success")
              and method[key]["time"] and vp[key]["time"]]
    success = sum(method[key]["success"] for key in keys)
    low, high = wilson(success, len(keys))
    ratios = [method[key]["time"] / vp[key]["time"] for key in paired]
    rows.append({
        "rule": "ALL", "n": len(keys),
        "micp_feasible": sum(method[key]["feasible"] for key in keys),
        "micp_success": success, "micp_accuracy": success / len(keys),
        "micp_accuracy_ci_low": low, "micp_accuracy_ci_high": high,
        "vp_success_same_cases": sum(vp.get(key, {}).get("success", False) for key in keys),
        "paired_success_n": len(paired),
        "micp_time_median": median(method[key]["time"] for key in keys if method[key]["time"] is not None),
        "micp_time_median_paired": median(method[key]["time"] for key in paired),
        "vp_time_median_same_cases": median(vp[key]["time"] for key in keys if key in vp and vp[key]["time"] is not None),
        "vp_time_median_paired": median(vp[key]["time"] for key in paired),
        "micp_over_vp_geomean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--vp", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(load_method(args.results), load_vp(args.vp))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
