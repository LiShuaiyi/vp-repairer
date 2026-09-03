"""Run the Halder R_G1 baseline on the 93-case MONA merge set."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .parallel_full_experiment import _run_case, write_results
from .rg1_rg3_experiment import write_final_csv


def build_manifest(source_csv: Path, scenario_root: Path):
    cases = {}
    with source_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("repairer_type") != "vp":
                continue
            if row.get("sat_solver_mode") != "domain_dpll":
                continue
            key = (row["scenario_id"], int(row["ego_id"]))
            scenario_name = row["scenario_id"]
            if not scenario_name.endswith(".xml"):
                scenario_name += ".xml"
            cases.setdefault(
                key,
                {
                    "scenario_id": row["scenario_id"],
                    "scenario_path": row.get("scenario_path")
                    or str(scenario_root / scenario_name),
                    "ego_id": int(row["ego_id"]),
                    # Keep MONA distinct from the 100-case highD R_G1 group in
                    # exported results; the worker maps this label to R_G1.
                    "rule": "R_G1_MONA",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--max-expansions", type=int, default=10_000)
    parser.add_argument(
        "--source-csv", type=Path,
        default=Path("evaluation/config/vp_temporal_full/vp_repairer_rg1_mona_batch_result_updated.csv"),
    )
    parser.add_argument(
        "--scenario-root", type=Path, default=Path("/data_linux/mona/scenarios")
    )
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("halder_althoff_comparison/mona_rg1_results_3s_10k"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("halder_althoff_comparison/halder_mona_rg1_timing_results_2026-09-03.csv"),
    )
    args = parser.parse_args(argv)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "logs").mkdir(exist_ok=True)
    (args.result_dir / "cases").mkdir(exist_ok=True)
    cases = build_manifest(args.source_csv, args.scenario_root)
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
