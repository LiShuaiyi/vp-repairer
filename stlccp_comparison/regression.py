"""Run STLCCP over the VP-aligned rule cohorts, optionally in parallel."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stlccp")

from micp_comparison.common import read_cases

from .runner import FIELDS, default_gurobi_license, evaluate, world_config


COHORTS = {
    "rg1": ("vp_repairer_rg1_batch_result_updated.csv", "highd", "R_G1", "highd"),
    "rg1_mona": ("vp_repairer_rg1_mona_batch_result_updated.csv", "highd", "R_G1", "mona"),
    "rg1_rg3": ("vp_repairer_rg1_rg3_batch_result_updated.csv", "highd", "R_G1_R_G3", "highd"),
    "rg2": ("vp_repairer_rg2_batch_result_updated.csv", "highd", "R_G2", "highd"),
    "rg3": ("vp_repairer_rg3_batch_result_updated.csv", "highd", "R_G3", "highd"),
    "in1": ("vp_repairer_in1_batch_result_updated.csv", "ind", "R_IN1", "ind_2024"),
    "in3_hand_draft": ("vp_repairer_in3_batch_result_updated.csv", "ind", "R_IN3_hand_draft", "ind_converter_2026"),
    "in3": ("vp_repairer_in3_full_batch_result_updated.csv", "ind", "R_IN3", "variants/in3"),
    "in4": ("vp_repairer_in4_batch_result_updated.csv", "ind", "R_IN4", "ind_converter_2026"),
    "in5": ("vp_repairer_in5_batch_result_updated.csv", "ind", "R_IN5", "variants/in5"),
}

# The established 704-case MICP population uses the hand-draft IN3 cohort.
# The separate full-semantics IN3 manifest is available as an explicit
# optional cohort but is not silently added to that population.
DEFAULT_COHORTS = tuple(name for name in COHORTS if name != "in3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("evaluation/config/vp_temporal_full"),
    )
    parser.add_argument(
        "--scenario-root", type=Path,
        default=Path("scenarios/experiment_scenarios"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=tuple(COHORTS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--robustness-margin", type=float, default=0.01)
    parser.add_argument(
        "--smoothing",
        choices=("lse", "mellowmin", "lse_mellowmin", "true_min"),
        default="lse_mellowmin",
    )
    parser.add_argument("--lse-k", type=float, default=10.0)
    parser.add_argument("--mellowmin-k", type=float, default=1000.0)
    parser.add_argument("--tau0", type=float, default=5e-3)
    parser.add_argument("--tau-max", type=float, default=1e3)
    parser.add_argument("--tau-rate", type=float, default=2.0)
    parser.add_argument("--slack-tolerance", type=float, default=1e-5)
    parser.add_argument("--cost-tolerance", type=float, default=1e-2)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--initialization", choices=("trajectory", "rollout", "random"), default="trajectory")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--rule-semantics",
        choices=(
            "lin2025", "vp_compatible", "vp_no_crossing_temporal",
            "vp_no_crossing_rule_only", "vp_quantified",
        ),
        default="lin2025",
    )
    parser.add_argument("--gurobi-license", type=Path, default=default_gurobi_license())
    parser.add_argument(
        "--monitor-config", type=Path,
        default=Path("/data_linux/Lab/commonroad-stl-monitor/crmonitor/config.yaml"),
    )
    return parser.parse_args()


def run_cohort(payload):
    name, filename, dataset, forced_rule, scenario_subdir, options = payload
    args = SimpleNamespace(
        dataset=dataset,
        input=Path(options["input_dir"]) / filename,
        output=Path(options["output_dir"]) / f"{name}.csv",
        scenario_dir=Path(options["scenario_root"]) / scenario_subdir,
        rule=forced_rule,
        limit=options["limit"],
        offset=0,
        repeat=options["repeat"],
        only_recorded_violations=True,
        time_limit=options["time_limit"],
        threads=options["threads"],
        robustness_margin=options["robustness_margin"],
        rule_semantics=options["rule_semantics"],
        smoothing=options["smoothing"],
        lse_k=options["lse_k"],
        mellowmin_k=options["mellowmin_k"],
        tau0=options["tau0"],
        tau_max=options["tau_max"],
        tau_rate=options["tau_rate"],
        slack_tolerance=options["slack_tolerance"],
        cost_tolerance=options["cost_tolerance"],
        max_iterations=options["max_iterations"],
        initialization=options["initialization"],
        seed=options["seed"],
        quiet=True,
        gurobi_license=Path(options["gurobi_license"]),
        monitor_config=Path(options["monitor_config"]),
    )
    os.environ["GRB_LICENSE_FILE"] = str(args.gurobi_license.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = list(read_cases(
        args.input,
        forced_rule,
        args.limit,
        require_recorded_violation=True,
    ))
    config = world_config(args)
    completed = 0
    successful = 0
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for repeat in range(args.repeat):
                print(
                    f"STLCCP REGRESSION {name} {case['scenario_id']} repeat={repeat}",
                    flush=True,
                )
                row = evaluate(args, case, repeat, config)
                writer.writerow(row)
                stream.flush()
                completed += 1
                successful += str(row["success"]).lower() == "true"
    return {
        "cohort": name,
        "rows": completed,
        "successful": successful,
        "output": str(args.output),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"cohorts", "workers"}
    }
    selected = args.cohorts or list(DEFAULT_COHORTS)
    tasks = [
        (name, *COHORTS[name], options)
        for name in selected
    ]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cohort, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"COHORT COMPLETE {result['cohort']} rows={result['rows']} "
                f"successful={result['successful']}",
                flush=True,
            )
    manifest = {
        "options": options,
        "cohorts": sorted(completed, key=lambda value: value["cohort"]),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
