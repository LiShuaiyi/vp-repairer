#!/usr/bin/env python3
"""Generate genuine inD converter windows shifted by at most one second.

The script never edits an existing CommonRoad XML trajectory.  It resolves each
selected source case back to its inD recording/track, slices the original CSV
recording with commonroad-dataset-converter's window classes, and uses the
converter's own scenario, obstacle, signal, and file writer implementation.
"""

import argparse
import concurrent.futures
import contextlib
import csv
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from commonroad.common.file_writer import FileFormat
from pandas.errors import SettingWithCopyWarning


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERTER_ROOT = Path("/data_linux/Lab/commonroad-dataset-converter")
DEFAULT_INPUT_DIR = Path(
    "/data_linux/Lab/highD-cr-scenarios/13_inD/inD-dataset-v1.1/data"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/ind_converter_time_shifts")
DEFAULT_OFFSETS = tuple(step / 5.0 for step in range(-5, 6) if step)
GROUP_TO_RULE = {
    "in1": "R_IN1",
    "in3": "R_IN3_hand_draft",
    "in4": "R_IN4",
    "in5": "R_IN5",
}
SCENARIO_RE = re.compile(
    r"^(?P<map>.+)_(?P<configuration>\d+)_T-(?P<end>\d+)$"
)


def parse_csv_list(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_offsets(value):
    offsets = tuple(float(item) for item in parse_csv_list(value))
    allowed = set(DEFAULT_OFFSETS)
    invalid = [value for value in offsets if value not in allowed]
    if invalid:
        raise ValueError(
            f"Offsets must be non-zero 0.2 s multiples within ±1.0 s: {invalid}"
        )
    return offsets


def import_converter(converter_root):
    converter_root = converter_root.resolve()
    if str(converter_root) not in sys.path:
        sys.path.insert(0, str(converter_root))

    from crdata.conversion.tabular.job_consumer import TabularJobConsumer
    from crdata.conversion.tabular.job_producer import TabularJob
    from crdata.conversion.tabular.planning_problem import (
        FixedEgoPlanningProblemCreator,
    )
    from crdata.conversion.tabular.windowing import (
        DownsamplingWindowWrapper,
        FixedTimeRangeWindowGenerator,
    )
    from crdata.datasets.inD import IndConverterFactory

    return {
        "consumer": TabularJobConsumer,
        "job": TabularJob,
        "planning_problem": FixedEgoPlanningProblemCreator,
        "downsample": DownsamplingWindowWrapper,
        "fixed_window": FixedTimeRangeWindowGenerator,
        "factory": IndConverterFactory,
    }


def load_source_cases(groups):
    sources = {}
    for group in groups:
        path = (
            REPO_ROOT
            / "evaluation/config/vp_temporal_full"
            / f"vp_repairer_{group}_batch_result_updated.csv"
        )
        with path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        selected = []
        seen = set()
        for row in rows:
            if row.get("repairer_type") != "vp":
                continue
            if row.get("sat_solver_mode") != "domain_dpll":
                continue
            scenario_id = row.get("scenario_id") or row.get("scenario") or ""
            # Hand-generated safe/speed variants are intentionally excluded.
            if SCENARIO_RE.fullmatch(scenario_id) is None:
                continue
            key = (scenario_id, int(row["ego_id"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "group": group,
                    "rule": GROUP_TO_RULE[group],
                    "scenario_id": scenario_id,
                    "ego_id": int(row["ego_id"]),
                }
            )
        sources[group] = selected
    return sources


def load_recording_index(input_dir):
    index = defaultdict(list)
    for meta_path in sorted(input_dir.glob("*_recordingMeta.csv")):
        with meta_path.open(newline="") as csv_file:
            row = next(csv.DictReader(csv_file))
        recording_id = int(row["recordingId"])
        location_id = int(row["locationId"])
        tracks_meta_path = meta_path.with_name(f"{recording_id:02d}_tracksMeta.csv")
        index[location_id].append(
            {
                "recording_id": recording_id,
                "recording_meta_path": meta_path,
                "tracks_meta_path": tracks_meta_path,
            }
        )
    return index


def resolve_source(source, recording_index, location_names, downsample):
    match = SCENARIO_RE.fullmatch(source["scenario_id"])
    if match is None:
        raise ValueError(f"Unsupported source scenario id: {source['scenario_id']}")

    map_name = match.group("map")
    configuration = match.group("configuration")
    converted_end = int(match.group("end"))
    location_ids = [key for key, value in location_names.items() if value == map_name]
    candidates = []
    for location_id in location_ids:
        for recording in recording_index[location_id]:
            prefix = str(recording["recording_id"])
            if not configuration.startswith(prefix):
                continue
            suffix = configuration[len(prefix) :]
            if not suffix:
                continue
            converted_start = int(suffix)
            if converted_start > converted_end:
                continue
            candidates.append((len(prefix), recording, converted_start))
    if not candidates:
        raise ValueError(f"Cannot resolve recording for {source['scenario_id']}")

    _, recording, converted_start = max(candidates, key=lambda item: item[0])
    converted_length = converted_end - converted_start + 1
    if converted_length <= 1:
        raise ValueError(f"Invalid source window length: {source['scenario_id']}")

    track_id = source["ego_id"] - 10000
    with recording["tracks_meta_path"].open(newline="") as csv_file:
        track_rows = {
            int(row["trackId"]): row for row in csv.DictReader(csv_file)
        }
    if track_id not in track_rows:
        raise ValueError(
            f"Raw track {track_id} missing for {source['scenario_id']}"
        )
    track_meta = track_rows[track_id]
    return {
        **source,
        "recording_id": recording["recording_id"],
        "track_id": track_id,
        "converted_start": converted_start,
        "converted_end": converted_end,
        "converted_length": converted_length,
        "raw_track_start": int(track_meta["initialFrame"]),
        "raw_track_end": int(track_meta["finalFrame"]),
        "downsample": downsample,
    }


def choose_candidates(sources, offsets, sources_per_rule, variants_per_source, seed):
    rng = random.Random(seed)
    candidates = []
    for group, group_sources in sources.items():
        group_sources = list(group_sources)
        rng.shuffle(group_sources)
        if sources_per_rule is not None:
            group_sources = group_sources[:sources_per_rule]
        for source in group_sources:
            source_offsets = list(offsets)
            rng.shuffle(source_offsets)
            source_offsets = source_offsets[:variants_per_source]
            for offset_seconds in sorted(source_offsets):
                offset_steps = int(round(offset_seconds * 5.0))
                converted_start = source["converted_start"] + offset_steps
                raw_start = converted_start * source["downsample"]
                raw_end = (
                    raw_start
                    + source["converted_length"] * source["downsample"]
                    - 1
                )
                status = "candidate"
                error = ""
                if raw_start < source["raw_track_start"]:
                    status = "ego_not_observed_for_full_window"
                    error = "shifted window starts before ego raw track"
                elif raw_end > source["raw_track_end"]:
                    status = "ego_not_observed_for_full_window"
                    error = "shifted window ends after ego raw track"
                candidates.append(
                    {
                        **source,
                        "offset_seconds": offset_seconds,
                        "offset_steps": offset_steps,
                        "shifted_converted_start": converted_start,
                        "raw_start": raw_start,
                        "raw_end": raw_end,
                        "status": status,
                        "error": error,
                    }
                )
    return candidates


def build_factory(api, args):
    return api["factory"](
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_time_steps=args.raw_window_length,
        num_planning_problems=1,
        keep_ego=True,
        obstacles_start_at_zero=True,
        downsample=args.downsample,
        num_processes=1,
        file_writer_type=FileFormat.XML,
        max_scenarios=None,
        samples_per_recording=None,
        start_time_step=None,
        end_time_step=None,
        ego_vehicle_id=None,
    )


def screen_violation(group, rule, scenario_path, scenario_id, ego_id):
    # Monitor construction is intentionally isolated and quiet: it emits many
    # predicate diagnostics which are not useful during a batch scan.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ind-time-shifts")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(
        sink
    ), contextlib.redirect_stderr(sink):
        import batch_test_vp_repairer_all_rules_updated as batch
        import crrepairer.smt.monitor_wrapper as monitor_wrapper
        from crrepairer.smt.monitor_wrapper import STLRuleMonitor
        from crrepairer.utils.repair import retrieve_ego_vehicle

        if group == "in3":
            batch.configure_in3_rule_variant("hand_draft")
        case = {
            "scenario_id": scenario_id,
            "scenario_path": str(scenario_path),
            "ego_id": ego_id,
            "rule": rule,
        }
        spec = batch.RULE_SPECS[group]
        config = batch.build_config(
            group,
            case,
            "vp",
            spec["vp_planner"],
            spec["vp_constraint_mode"],
        )
        ego_vehicle = retrieve_ego_vehicle(config)
        original_get_config = batch.patch_rtamt_bound_alignment_for_batch(config)
        try:
            monitor = STLRuleMonitor(config)
        finally:
            if original_get_config is not None:
                monitor_wrapper.get_traffic_rule_config = original_get_config

        tv = float(monitor.tv_time_step)
        initial_step = int(ego_vehicle.initial_state.time_step)
    return {
        "tv": tv,
        "initial_step": initial_step,
        "is_eligible_violation": math.isfinite(tv) and tv > initial_step,
    }


def screen_candidate(candidate):
    try:
        screening = screen_violation(
            candidate["group"],
            candidate["rule"],
            candidate["generated_scenario_path"],
            candidate["generated_scenario_id"],
            candidate["ego_id"],
        )
        return screening, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def generate_candidates(api, factory, candidates, args):
    by_recording = defaultdict(list)
    for candidate in candidates:
        if candidate["status"] == "candidate":
            by_recording[candidate["recording_id"]].append(candidate)

    prototype_creator = factory.build_scenario_prototype_creator()
    recording_generator = factory.build_recording_generator()
    processed = 0
    total = sum(len(items) for items in by_recording.values())
    for recording, recording_meta in recording_generator:
        recording_id = int(recording_meta.recording_meta["recordingId"])
        if recording_id not in by_recording:
            continue
        for candidate in by_recording[recording_id]:
            processed += 1
            group_dir = args.output_dir / candidate["group"]
            group_dir.mkdir(parents=True, exist_ok=True)
            try:
                window_generator = api["fixed_window"](
                    candidate["raw_start"], candidate["raw_end"]
                )
                window_generator = api["downsample"](
                    window_generator, args.downsample
                )
                window, window_meta = next(
                    iter(window_generator(recording, recording_meta))
                )
                scenario = prototype_creator(window, window_meta)
                planning_problem_set = api["planning_problem"](
                    candidate["track_id"], True
                )(window, scenario)
                if not planning_problem_set.planning_problem_dict:
                    raise ValueError("converter could not create the fixed-ego problem")
                job = api["job"](
                    window.vehicle_states,
                    window.vehicle_meta,
                    scenario,
                    planning_problem_set,
                )
                consumer = api["consumer"](
                    group_dir,
                    FileFormat.XML,
                    True,
                    create_turning_indicator=True,
                )
                consumer(job)
                generated_id = str(scenario.scenario_id)
                generated_path = group_dir / f"{generated_id}.xml"
                candidate["generated_scenario_id"] = generated_id
                candidate["generated_scenario_path"] = str(generated_path)
                if not generated_path.is_file():
                    raise FileNotFoundError(generated_path)

                candidate["status"] = "generated_pending_screen"
            except Exception as exc:
                candidate["status"] = "generation_or_screen_error"
                candidate["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{processed:3d}/{total}] {candidate['group']} "
                f"source={candidate['scenario_id']} "
                f"offset={candidate['offset_seconds']:+.1f}s "
                f"status={candidate['status']}",
                flush=True,
            )


def screen_candidates(candidates, workers):
    pending = [
        candidate
        for candidate in candidates
        if candidate["status"] == "generated_pending_screen"
    ]
    if not pending:
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(screen_candidate, candidate): candidate
            for candidate in pending
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            candidate = futures[future]
            screening, error = future.result()
            if error:
                candidate["status"] = "screen_error"
                candidate["error"] = error
            else:
                candidate.update(screening)
                candidate["status"] = (
                    "eligible_violation"
                    if screening["is_eligible_violation"]
                    else "not_eligible_violation"
                )
            print(
                f"[screen {index:3d}/{len(pending)}] {candidate['group']} "
                f"offset={candidate['offset_seconds']:+.1f}s "
                f"status={candidate['status']}",
                flush=True,
            )


def write_manifest(candidates, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "rule",
        "scenario_id",
        "ego_id",
        "recording_id",
        "track_id",
        "converted_start",
        "converted_end",
        "offset_seconds",
        "offset_steps",
        "shifted_converted_start",
        "raw_start",
        "raw_end",
        "generated_scenario_id",
        "generated_scenario_path",
        "tv",
        "initial_step",
        "is_eligible_violation",
        "status",
        "error",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter-root", type=Path, default=DEFAULT_CONVERTER_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "manifest.csv",
    )
    parser.add_argument("--eligible-manifest", type=Path, default=None)
    parser.add_argument(
        "--groups", type=parse_csv_list, default=tuple(GROUP_TO_RULE)
    )
    parser.add_argument(
        "--offsets",
        type=parse_offsets,
        default=DEFAULT_OFFSETS,
        help="Comma-separated offsets from -1.0 to +1.0 s in 0.2 s steps.",
    )
    parser.add_argument("--sources-per-rule", type=int, default=20)
    parser.add_argument("--variants-per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--raw-window-length", type=int, default=100)
    parser.add_argument("--screen-workers", type=int, default=4)
    parser.add_argument("--no-screen", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    invalid_groups = [group for group in args.groups if group not in GROUP_TO_RULE]
    if invalid_groups:
        raise ValueError(f"Unsupported groups: {invalid_groups}")
    if args.sources_per_rule <= 0 or args.variants_per_source <= 0:
        raise ValueError("Source and variant counts must be positive")
    api = import_converter(args.converter_root)
    factory = build_factory(api, args)
    source_groups = load_source_cases(args.groups)
    recording_index = load_recording_index(args.input_dir)
    location_names = {
        int(key): value for key, value in factory._config["locations"].items()
    }
    resolved = {
        group: [
            resolve_source(source, recording_index, location_names, args.downsample)
            for source in sources
        ]
        for group, sources in source_groups.items()
    }
    candidates = choose_candidates(
        resolved,
        args.offsets,
        args.sources_per_rule,
        args.variants_per_source,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_candidates(api, factory, candidates, args)
    if args.no_screen:
        for candidate in candidates:
            if candidate["status"] == "generated_pending_screen":
                candidate["status"] = "generated_not_screened"
    else:
        screen_candidates(candidates, args.screen_workers)
    write_manifest(candidates, args.manifest)
    eligible_manifest = (
        args.eligible_manifest
        if args.eligible_manifest is not None
        else args.output_dir / "eligible_violations.csv"
    )
    write_manifest(
        [
            candidate
            for candidate in candidates
            if candidate["status"] == "eligible_violation"
        ],
        eligible_manifest,
    )
    counts = defaultdict(int)
    for candidate in candidates:
        counts[candidate["status"]] += 1
    print(f"Manifest: {args.manifest}")
    print(f"Eligible violations: {eligible_manifest}")
    print("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
