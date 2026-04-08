import argparse
import csv
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import iterparse


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/mona/scenarios")
DEFAULT_OUTPUT_CSV = Path("evaluation/config/mona_rg1_rg3_screening.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen MONA scenarios for RG1/RG3 interstate-monitor compatibility."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help=f"Directory containing MONA XML scenarios. Default: {DEFAULT_SCENARIO_DIR}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Where to save the screening result. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only screen the first N scenarios after sorting by file name.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N scenarios. Default: 1000",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def iter_scenarios(scenario_dir: Path, limit: int = None):
    scenarios = sorted(scenario_dir.glob("*.xml"))
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def analyze_scenario(path: Path):
    lanelet_types = Counter()
    scenario_tags = []
    intersection_count = 0

    for _, elem in iterparse(path, events=("end",)):
        tag = local_name(elem.tag)
        if tag == "laneletType" and elem.text:
            lanelet_types[elem.text.strip()] += 1
        elif tag == "scenarioTags":
            scenario_tags = [
                local_name(child.tag)
                for child in list(elem)
            ]
        elif tag == "intersection":
            intersection_count += 1
        elem.clear()

    reasons = []
    if intersection_count > 0:
        reasons.append("has_intersection_elements")
    if lanelet_types.get("intersection", 0) > 0:
        reasons.append("has_intersection_lanelets")
    if lanelet_types.get("urban", 0) > 0:
        reasons.append("has_urban_lanelets")
    if "interstate" not in lanelet_types and "interstate" not in scenario_tags:
        reasons.append("no_interstate_annotation")

    compatible = not reasons
    return {
        "scenario_name": path.name,
        "compatible_for_rg1_rg3_monitor": compatible,
        "intersection_count": intersection_count,
        "scenario_tags": ";".join(sorted(scenario_tags)),
        "lanelet_types": ";".join(
            f"{lanelet_type}:{count}"
            for lanelet_type, count in sorted(lanelet_types.items())
        ),
        "reasons": ";".join(reasons),
    }


def create_writer(csv_file):
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "scenario_name",
            "compatible_for_rg1_rg3_monitor",
            "intersection_count",
            "scenario_tags",
            "lanelet_types",
            "reasons",
        ],
    )
    writer.writeheader()
    return writer


def main():
    args = parse_args()
    scenario_paths = iter_scenarios(args.scenario_dir, args.limit_scenarios)
    if not scenario_paths:
        raise FileNotFoundError(f"No XML scenarios found in {args.scenario_dir}")

    reason_counter = Counter()
    compatible_count = 0
    total_count = len(scenario_paths)

    print(f"Screening {total_count} scenario(s) from {args.scenario_dir}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as csv_file:
        writer = create_writer(csv_file)
        for index, path in enumerate(scenario_paths, start=1):
            row = analyze_scenario(path)
            writer.writerow(row)

            if row["compatible_for_rg1_rg3_monitor"]:
                compatible_count += 1
            else:
                for reason in filter(None, row["reasons"].split(";")):
                    reason_counter[reason] += 1

            if (
                index <= 5
                or index % args.progress_every == 0
                or index == total_count
            ):
                print(
                    f"[{index}/{total_count}] {path.name}: "
                    f"{'compatible' if row['compatible_for_rg1_rg3_monitor'] else row['reasons']}",
                    flush=True,
                )
    print()
    print(f"Compatible scenarios: {compatible_count}/{total_count}")
    if reason_counter:
        print(
            "Top incompatibility reasons: "
            + ", ".join(
                f"{reason}={count}"
                for reason, count in reason_counter.most_common(10)
            )
        )
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
