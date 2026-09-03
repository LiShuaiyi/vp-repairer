"""Run a manifest of extracted cases and write comparison-ready CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .io import load_problem, write_result
from .planner import MinimumViolationPlanner


FIELDS = (
    "case_id",
    "input",
    "success",
    "compliant_on_lattice_encoding",
    "preprocessing_time_s",
    "search_time_s",
    "core_total_time_s",
    "expanded_nodes",
    "generated_nodes",
    "violation_costs",
    "result_json",
    "error",
)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with case_id,input columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--result-dir", type=Path, help="optional directory for full trajectory JSON"
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    manifest_root = args.manifest.resolve().parent
    if args.result_dir:
        args.result_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for index, item in enumerate(csv.DictReader(source)):
            if args.limit is not None and index >= args.limit:
                break
            case_id = item.get("case_id") or str(index)
            input_path = Path(item["input"])
            if not input_path.is_absolute():
                input_path = manifest_root / input_path
            row = {field: "" for field in FIELDS}
            row.update({"case_id": case_id, "input": str(input_path)})
            total_start = time.perf_counter()
            try:
                preprocess_start = time.perf_counter()
                cfg, env, rules, s0, v0, data = load_problem(input_path)
                preprocess_time = time.perf_counter() - preprocess_start
                result = MinimumViolationPlanner(cfg, env, rules).plan(s0, v0)
                row.update(
                    {
                        "success": True,
                        "compliant_on_lattice_encoding": result.compliant,
                        "preprocessing_time_s": preprocess_time,
                        "search_time_s": result.runtime_s,
                        "expanded_nodes": result.expanded_nodes,
                        "generated_nodes": result.generated_nodes,
                        "violation_costs": json.dumps(
                            dict(zip(result.rule_names, result.violation_costs)),
                            sort_keys=True,
                        ),
                    }
                )
                if args.result_dir:
                    result_path = args.result_dir / f"{case_id}.json"
                    write_result(result_path, result, data.get("metadata"))
                    row["result_json"] = str(result_path)
            except Exception as exc:
                row["success"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["core_total_time_s"] = time.perf_counter() - total_start
            writer.writerow(row)
            target.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
