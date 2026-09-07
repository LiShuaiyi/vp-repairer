#!/usr/bin/env python3
"""Run all sampling cohorts, with isolated logs/results and safe resume."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

try:
    from .specs import GROUPS, parse_groups
except ImportError:  # direct script execution
    from specs import GROUPS, parse_groups


HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/data_linux/conda-envs/repairverse310_gpu/bin/python")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="all", help="Comma list, or all")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/latest")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel rule cohorts")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-rule-candidates", type=int, default=0)
    parser.add_argument(
        "--in1-strategy",
        choices=("paper_example", "paper_batch", "stop_position"),
        default="paper_batch",
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Do not reconstruct missing generated inD windows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_for(args, group):
    spec = GROUPS[group]
    output = args.results_dir / "sampling" / f"{group}.csv"
    cmd = [
        str(args.python), str(HERE / "benchmark.py"),
        "--dataset", spec["dataset"], "--input", str(spec["input"]),
        "--output", str(output), "--scenario-dir", str(spec["scenario_dir"]),
        "--rule", spec["rule"], "--mode", "rule_filtered",
        "--max-rule-candidates", str(args.max_rule_candidates),
        "--in1-strategy", args.in1_strategy,
        "--offset", str(args.offset), "--repeat", str(args.repeat),
    ]
    generated = HERE / "generated_scenarios" / group
    if generated.is_dir():
        cmd += ["--scenario-dir", str(generated)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return output, cmd


def run_group(args, group):
    output, cmd = command_for(args, group)
    log = args.results_dir / "logs" / f"{group}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        return group, "skipped", output
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-sampling-comparison")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    with log.open("w") as stream:
        completed = subprocess.run(cmd, cwd=HERE.parent, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return group, "completed", output


def main():
    args = parse_args()
    groups = parse_groups(args.groups)
    if not args.python.is_file():
        raise FileNotFoundError(f"Python environment not found: {args.python}")
    for group in groups:
        spec = GROUPS[group]
        if not spec["input"].is_file():
            raise FileNotFoundError(spec["input"])
        if not spec["scenario_dir"].is_dir():
            raise FileNotFoundError(spec["scenario_dir"])
    intersection_groups = [group for group in groups if GROUPS[group]["dataset"] == "ind"]
    if intersection_groups and not args.skip_prepare:
        prepare = [
            str(args.python), str(HERE / "prepare_scenarios.py"),
            "--groups", ",".join(intersection_groups),
            "--workers", str(min(args.jobs, len(intersection_groups))),
        ]
        subprocess.run(prepare, cwd=HERE.parent, check=True)
    workers = max(1, min(args.jobs, len(groups)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_group, args, group): group for group in groups}
        for future in concurrent.futures.as_completed(futures):
            group, status, output = future.result()
            print(f"{group}: {status} -> {output}", flush=True)
    summarize = [
        str(args.python), str(HERE / "summarize.py"),
        "--groups", ",".join(groups), "--results-dir", str(args.results_dir),
    ]
    subprocess.run(summarize, cwd=HERE.parent, check=True)


if __name__ == "__main__":
    main()
