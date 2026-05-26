#!/usr/bin/env python3
"""Find ego ids that violate R_G2 in highD CommonRoad scenarios."""

import argparse
import csv
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/highD-cr-scenarios/highD-repair")
DEFAULT_OUTPUT = REPO_ROOT / "output" / "highd_rg2_minus1_violations.csv"
DEFAULT_FAILED_OUTPUT = REPO_ROOT / "output" / "highd_rg2_minus1_failed.csv"
RULE = "R_G2"


def first_violation_index(values, ignore_initial_violation):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if ignore_initial_violation and values[0] < 0:
        return None
    negative_indices = np.flatnonzero(values < 0)
    if negative_indices.size == 0:
        return None
    return int(negative_indices[0])


def evaluate_rg2(world, ego_id, ignore_initial_violation, show_parser_warnings):
    if show_parser_warnings:
        evaluator = RuleEvaluator.create_from_config(
            world,
            int(ego_id),
            rule=RULE,
            use_boolean=True,
        )
        values = evaluator.evaluate()
    else:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            evaluator = RuleEvaluator.create_from_config(
                world,
                int(ego_id),
                rule=RULE,
                use_boolean=True,
            )
            values = evaluator.evaluate()
    violation_index = first_violation_index(values, ignore_initial_violation)
    if violation_index is None:
        return None
    return {
        "first_violation_index": violation_index,
        "min_value": float(np.min(values)),
        "num_steps": int(len(values)),
    }


def scan_scenario(scenario_path, args):
    scenario, _ = CommonRoadFileReader(str(scenario_path)).open(
        lanelet_assignment=True,
    )
    world = World.create_from_scenario(scenario)
    rows = []
    checked = 0

    for obstacle in scenario.dynamic_obstacles:
        ego_id = int(obstacle.obstacle_id)
        checked += 1
        try:
            violation = evaluate_rg2(
                world,
                ego_id,
                args.ignore_initial_violation,
                args.show_parser_warnings,
            )
        except Exception as exc:
            if args.verbose_errors:
                print(
                    f"  skip ego={ego_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            continue
        if violation is None:
            continue

        row = {
            "scenario_id": str(scenario.scenario_id),
            "scenario_path": str(scenario_path),
            "ego_id": ego_id,
            "rule_STL": RULE,
            "first_violation_index": violation["first_violation_index"],
            "min_value": violation["min_value"],
            "num_steps": violation["num_steps"],
        }
        rows.append(row)
        print(
            f"  violation scenario={scenario.scenario_id} ego={ego_id} "
            f"first_index={violation['first_violation_index']} "
            f"min={violation['min_value']:.12g}",
            flush=True,
        )

    return rows, checked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan highD CommonRoad scenarios ending in -1.xml for R_G2 violations.",
    )
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--pattern",
        default="*-1.xml",
        help="Scenario filename glob. Default scans only files ending in -1.xml.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED_OUTPUT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument(
        "--ignore-initial-violation",
        action="store_true",
        help=(
            "Match evaluation/highD_evaluator_STL.py behavior by ignoring cases "
            "already violating at the first robustness sample."
        ),
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print per-ego evaluation errors instead of silently skipping them.",
    )
    parser.add_argument(
        "--show-parser-warnings",
        action="store_true",
        help="Do not suppress ANTLR/RTAMT parser warnings from rule evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")

    scenario_paths = sorted(args.scenario_dir.glob(args.pattern))
    total_matching = len(scenario_paths)
    scenario_paths = scenario_paths[args.start_index - 1 :]
    if args.limit is not None:
        scenario_paths = scenario_paths[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "scenario_path",
        "ego_id",
        "rule_STL",
        "first_violation_index",
        "min_value",
        "num_steps",
    ]
    failed_fieldnames = ["scenario_id", "scenario_path", "error"]

    total_checked = 0
    total_rows = 0
    failed_count = 0

    with args.output.open("w", newline="") as output_file, args.failed_output.open(
        "w",
        newline="",
    ) as failed_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        failed_writer = csv.DictWriter(failed_file, fieldnames=failed_fieldnames)
        failed_writer.writeheader()

        for local_idx, scenario_path in enumerate(scenario_paths, start=1):
            global_idx = args.start_index + local_idx - 1
            print(f"[{global_idx}/{total_matching}] {scenario_path.name}", flush=True)
            try:
                rows, checked = scan_scenario(scenario_path, args)
            except Exception as exc:
                failed_count += 1
                failed_writer.writerow(
                    {
                        "scenario_id": scenario_path.stem,
                        "scenario_path": str(scenario_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                failed_file.flush()
                continue

            total_checked += checked
            total_rows += len(rows)
            writer.writerows(rows)
            output_file.flush()

            if args.flush_every > 0 and local_idx % args.flush_every == 0:
                failed_file.flush()
                print(
                    f"  progress: scenarios={local_idx}, egos={total_checked}, "
                    f"violations={total_rows}, failed={failed_count}",
                    flush=True,
                )

    print(f"Scanned scenarios this run: {len(scenario_paths)}", flush=True)
    print(f"Ego vehicles checked: {total_checked}", flush=True)
    print(f"R_G2 violation rows: {total_rows}", flush=True)
    print(f"Failed scenarios: {failed_count}", flush=True)
    print(f"Output CSV: {args.output}", flush=True)
    print(f"Failed CSV: {args.failed_output}", flush=True)


if __name__ == "__main__":
    main()
