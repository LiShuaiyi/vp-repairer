import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_mona_rg1_rg3 import (
    DEFAULT_SCENARIO_DIR,
    _filter_violations_by_tv,
    _lanelet_assigned_steps,
    detect_violations,
    is_known_invalid_ego_error,
    iter_candidate_ego_ids,
    iter_scenario_files,
    load_scenario,
    write_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing MONAMerge filtered violation CSV with more rows "
            "using the same filtering rules."
        )
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=Path("evaluation/config/mona_merge_rg1_rg3_filtered.csv"),
        help="Existing filtered MONAMerge CSV to extend.",
    )
    parser.add_argument(
        "--extra-count",
        type=int,
        default=100,
        help="How many new rows to add on top of the base CSV. Default: 100",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help="Directory containing MONA scenarios.",
    )
    parser.add_argument(
        "--scenario-pattern",
        type=str,
        default="DEU_MONAMerge-2_*.xml",
        help="Glob pattern for MONAMerge scenarios.",
    )
    parser.add_argument(
        "--next-csv",
        type=Path,
        default=Path("evaluation/config/mona_merge_rg1_rg3_filtered_next100.csv"),
        help="Where to write only the newly added rows.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("evaluation/config/mona_merge_rg1_rg3_filtered_200.csv"),
        help="Where to write the combined base + new rows.",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=20,
        help="Flush progress every N new rows. Default: 20",
    )
    parser.add_argument(
        "--start-after-scenario",
        type=str,
        default=None,
        help=(
            "Only scan scenarios strictly after this file name in sorted order. "
            "Defaults to the last scenario_name found in base-csv."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.base_csv.open(newline="") as csv_file:
        base_rows = list(csv.DictReader(csv_file))

    existing_keys = {(row["scenario_name"], str(row["ego_id"])) for row in base_rows}
    new_rows = []
    scenario_files = iter_scenario_files(args.scenario_dir, args.scenario_pattern)
    start_after_scenario = args.start_after_scenario
    if start_after_scenario is None and base_rows:
        start_after_scenario = base_rows[-1]["scenario_name"]
    if start_after_scenario is not None:
        scenario_files = [
            path for path in scenario_files if path.name > start_after_scenario
        ]

    print(
        f"Loaded {len(base_rows)} existing rows from {args.base_csv}; "
        f"scanning {len(scenario_files)} scenarios for {args.extra_count} more rows"
    )
    if start_after_scenario is not None:
        print(f"Starting strictly after scenario {start_after_scenario}")

    for idx, scenario_path in enumerate(scenario_files, start=1):
        scenario_name = scenario_path.name
        try:
            scenario, planning_problem_set = load_scenario(scenario_path)
        except Exception as exc:
            print(
                f"[{idx}/{len(scenario_files)}] {scenario_name}: "
                f"load failed with {type(exc).__name__}: {exc}"
            )
            continue

        added_here = 0
        for ego_id in iter_candidate_ego_ids(scenario):
            key = (scenario_name, str(ego_id))
            if key in existing_keys:
                continue

            lanelet_assigned_steps = _lanelet_assigned_steps(
                scenario.obstacle_by_id(ego_id)
            )
            try:
                violated_rules, rule_to_tv = detect_violations(
                    scenario,
                    planning_problem_set,
                    scenario_path,
                    ego_id,
                    verbose=False,
                )
            except Exception as exc:
                if is_known_invalid_ego_error(exc):
                    continue
                print(
                    f"  ego {ego_id}: failed with {type(exc).__name__}: {exc}"
                )
                continue

            filtered_rules, filtered_rule_to_tv = _filter_violations_by_tv(
                violated_rules, rule_to_tv, 6
            )
            if violated_rules and lanelet_assigned_steps != 20:
                filtered_rules = []
                filtered_rule_to_tv = {}
            if not filtered_rules:
                continue

            row = {
                "scenario_name": scenario_name,
                "ego_id": ego_id,
                "violated_rules": ";".join(filtered_rules),
                "rule_to_tv": ";".join(
                    f"{rule}:{filtered_rule_to_tv.get(rule)}"
                    for rule in filtered_rules
                ),
                "lanelet_assigned_steps": lanelet_assigned_steps,
                "raw_violated_rules": ";".join(violated_rules),
                "raw_rule_to_tv": ";".join(
                    f"{rule}:{rule_to_tv.get(rule)}" for rule in violated_rules
                ),
            }
            new_rows.append(row)
            existing_keys.add(key)
            added_here += 1
            print(
                f"  added {len(new_rows)}/{args.extra_count}: "
                f"{scenario_name} ego {ego_id} -> {row['violated_rules']} "
                f"({row['rule_to_tv']})"
            )

            if len(new_rows) % max(args.write_every, 1) == 0:
                write_results(new_rows, args.next_csv)
                write_results(base_rows + new_rows, args.combined_csv)
                print(
                    f"  flushed progress: next={len(new_rows)}, "
                    f"combined={len(base_rows) + len(new_rows)}"
                )

            if len(new_rows) >= args.extra_count:
                break

        if added_here:
            print(
                f"[{idx}/{len(scenario_files)}] {scenario_name}: "
                f"added {added_here} new row(s)"
            )
        if len(new_rows) >= args.extra_count:
            break

    write_results(new_rows, args.next_csv)
    write_results(base_rows + new_rows, args.combined_csv)
    print(f"Wrote {len(new_rows)} new rows to {args.next_csv}")
    print(
        f"Wrote {len(base_rows) + len(new_rows)} total rows to "
        f"{args.combined_csv}"
    )


if __name__ == "__main__":
    main()
