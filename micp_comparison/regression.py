"""Deterministic stratified regression against completed legacy MICP results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from micp_comparison.runner import FIELDS, evaluate, world_config


RULE_FILES = {
    "R_G1": ("rg1.csv", "highd"),
    "R_G2": ("rg2.csv", "highd"),
    "R_G3": ("rg3.csv", "highd"),
    "R_G1_R_G3": ("rg1_rg3.csv", "highd"),
    "R_IN1": ("in1.csv", "ind"),
    "R_IN3": ("in3.csv", "ind"),
    "R_IN4": ("in4.csv", "ind"),
    "R_IN5": ("in5.csv", "ind"),
}

EXTRA_FIELDS = [
    "cohort", "old_solver_feasible", "old_monitor_compliant", "old_success",
    "old_core_total_time", "old_num_binary_variables",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cohort", type=int, default=5)
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Run every row in each legacy result instead of cohort sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--encoding", choices=("standard", "fewer_binary"),
        default="standard",
    )
    parser.add_argument(
        "--rule-semantics",
        choices=(
            "lin2025", "vp_compatible", "vp_no_crossing_temporal",
            "vp_no_crossing_rule_only", "vp_quantified",
        ),
        default="lin2025",
    )
    parser.add_argument(
        "--rules", nargs="+", choices=tuple(RULE_FILES),
        help="Run only the selected rules (default: all rules).",
    )
    parser.add_argument(
        "--monitor-config", type=Path,
        default=Path("/data_linux/Lab/commonroad-stl-monitor/crmonitor/config.yaml"),
    )
    parser.add_argument(
        "--gurobi-license", type=Path,
        default=Path(
            "/data_linux/planning-sim/repairer/commonroad-repairer-vp/"
            "autoware-repair-docker/gurobi.lic"
        ),
    )
    return parser.parse_args()


def sample_rows(path, count, seed, all_cases=False):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if all_cases:
        selected, seen = [], set()
        for row in rows:
            key = (row.get("scenario_id"), row.get("ego_id"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(("all_cases", row))
        return selected
    cohorts = {
        "old_false_positive": [
            row for row in rows
            if row.get("solver_feasible") == "True"
            and row.get("monitor_compliant") == "False"
        ],
        "old_success": [row for row in rows if row.get("success") == "True"],
    }
    selected = []
    for index, (name, candidates) in enumerate(cohorts.items()):
        rng = random.Random(seed + index)
        candidates = list(candidates)
        rng.shuffle(candidates)
        for row in candidates[:count]:
            selected.append((name, row))
    return selected


def run_rule(payload):
    rule, dataset, old_file, output_file, options = payload
    args = SimpleNamespace(
        dataset=dataset,
        scenario_dir=None,
        quiet=True,
        time_limit=options["time_limit"],
        threads=options["threads"],
        encoding=options["encoding"],
        rule_semantics=options["rule_semantics"],
        robustness_margin=0.01,
        monitor_config=Path(options["monitor_config"]),
        gurobi_license=Path(options["gurobi_license"]),
    )
    os.environ["GRB_LICENSE_FILE"] = str(args.gurobi_license)
    selected = sample_rows(
        Path(old_file), options["per_cohort"], options["seed"],
        options.get("all_cases", False),
    )
    config = world_config(args)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_fields = EXTRA_FIELDS + FIELDS
    with output_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        stream.flush()
        for cohort, old in selected:
            case = {
                "scenario_id": old["scenario_id"],
                "scenario_path": old["scenario_path"],
                "ego_id": int(old["ego_id"]),
                "rule": rule,
            }
            print(f"REGRESSION {rule} {cohort} {case['scenario_id']}", flush=True)
            new = evaluate(args, case, 0, config)
            enriched = {
                "cohort": cohort,
                "old_solver_feasible": old.get("solver_feasible", ""),
                "old_monitor_compliant": old.get("monitor_compliant", ""),
                "old_success": old.get("success", ""),
                "old_core_total_time": old.get("core_total_time", ""),
                "old_num_binary_variables": old.get("num_binary_variables", ""),
                **new,
            }
            writer.writerow(enriched)
            stream.flush()
            print(
                f"DONE {rule} cohort={cohort} feasible={new['solver_feasible']} "
                f"monitor={new['monitor_compliant']} time={new['core_total_time'] or 'N/A'}",
                flush=True,
            )
    return rule, len(selected), str(output_file)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "per_cohort": args.per_cohort,
        "seed": args.seed,
        "all_cases": args.all_cases,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "encoding": args.encoding,
        "rule_semantics": args.rule_semantics,
        "monitor_config": str(args.monitor_config),
        "gurobi_license": str(args.gurobi_license),
    }
    tasks = []
    selected_rules = set(args.rules or RULE_FILES)
    for rule, (filename, dataset) in RULE_FILES.items():
        if rule not in selected_rules:
            continue
        tasks.append((
            rule, dataset, args.old_dir / filename,
            args.output_dir / filename, options,
        ))
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_rule, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"RULE COMPLETE {result[0]} cases={result[1]}", flush=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({"options": options, "completed": completed}, indent=2)
    )


if __name__ == "__main__":
    main()
