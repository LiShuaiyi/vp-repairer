#!/usr/bin/env python3
"""Reconstruct missing inD converter windows inside this experiment folder."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

try:
    from .specs import GROUPS, parse_groups
except ImportError:
    from specs import GROUPS, parse_groups


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EXAMPLES = REPO_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import generate_ind_converter_time_shifts as converter


def missing_cases(group):
    spec = GROUPS[group]
    seen, cases = set(), []
    with spec["input"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("repairer_type") != "vp" or row.get("sat_solver_mode") != "domain_dpll":
                continue
            key = row["scenario_id"], int(row["ego_id"]), row["rule"]
            if key in seen:
                continue
            seen.add(key)
            declared = Path(row.get("scenario_path", ""))
            generated = HERE / "generated_scenarios" / group / f"{key[0]}.xml"
            fallback = spec["scenario_dir"] / f"{key[0]}.xml"
            if declared.is_file() or generated.is_file() or fallback.is_file():
                continue
            cases.append({
                "group": group, "rule": key[2], "scenario_id": key[0], "ego_id": key[1]
            })
    return cases


def prepare_group(group):
    sources = missing_cases(group)
    if not sources:
        return group, 0
    api = converter.import_converter(converter.DEFAULT_CONVERTER_ROOT)
    args = SimpleNamespace(
        input_dir=converter.DEFAULT_INPUT_DIR,
        output_dir=HERE / "generated_scenarios",
        raw_window_length=100,
        downsample=5,
    )
    factory = converter.build_factory(api, args)
    index = converter.load_recording_index(args.input_dir)
    locations = {int(k): v for k, v in factory._config["locations"].items()}
    candidates = []
    for source in sources:
        item = converter.resolve_source(source, index, locations, args.downsample)
        item.update(
            offset_seconds=0.0,
            offset_steps=0,
            shifted_converted_start=item["converted_start"],
            raw_start=item["converted_start"] * args.downsample,
            raw_end=(item["converted_start"] + item["converted_length"]) * args.downsample - 1,
            status="candidate",
            error="",
        )
        candidates.append(item)
    converter.generate_candidates(api, factory, candidates, args)
    failures = [item for item in candidates if item["status"] != "generated_pending_screen"]
    if failures:
        raise RuntimeError(f"{group}: failed to generate {len(failures)}/{len(candidates)} windows")
    for item in candidates:
        if item["generated_scenario_id"] != item["scenario_id"]:
            raise ValueError(
                f"Expected {item['scenario_id']}, converter produced {item['generated_scenario_id']}"
            )
    converter.write_manifest(candidates, HERE / "generated_scenarios" / f"{group}_manifest.csv")
    return group, len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="in1,in3,in4,in5")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    groups = [group for group in parse_groups(args.groups) if GROUPS[group]["dataset"] == "ind"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(groups)))) as pool:
        for group, count in pool.map(prepare_group, groups):
            print(f"{group}: generated {count} missing scenarios", flush=True)


if __name__ == "__main__":
    main()
