import argparse
import collections
import csv
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable, List

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import (
    create_ego_vehicle_param,
    get_evaluation_config,
)
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.predicates.velocity import (
    PredBrSpeedLimit,
    PredFovSpeedLimit,
    PredLaneSpeedLimit,
    PredTypeSpeedLimit,
)

from commonroad.common.file_reader import CommonRoadFileReader

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration, ScenarioType
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg


DEFAULT_SCENARIO_DIR = Path("/data_linux/Lab/mona/scenarios")
DEFAULT_OUTPUT_CSV = Path("evaluation/config/mona_rg1_rg3_violations.csv")
VEHICLE_OBSTACLE_TYPES = {
    "CAR",
    "TRUCK",
    "BUS",
    "MOTORCYCLE",
    "TAXI",
}
MIN_TRAJECTORY_STATES = 5
MIN_TEMPORAL_OVERLAP_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan CommonRoad scenarios and record vehicles violating RG1 or RG3."
        )
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help=f"Directory containing CommonRoad XML scenarios. Default: {DEFAULT_SCENARIO_DIR}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Where to save the detected violations. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only scan the first N scenarios after sorting by file name.",
    )
    parser.add_argument(
        "--scenario-pattern",
        type=str,
        default="*.xml",
        help="Glob pattern for scenario files inside scenario-dir. Default: *.xml",
    )
    parser.add_argument(
        "--limit-egos",
        type=int,
        default=None,
        help="Only test the first N candidate ego vehicles in each scenario.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep STL monitor stdout visible while scanning.",
    )
    parser.add_argument(
        "--debug-rg3",
        action="store_true",
        help="Print detailed RG3 predicate/constraint values while scanning.",
    )
    parser.add_argument(
        "--debug-ego-id",
        type=int,
        default=None,
        help="Only print RG3 debug details for this ego id.",
    )
    parser.add_argument(
        "--debug-max-steps",
        type=int,
        default=20,
        help="Maximum number of time steps to print for RG3 debug output.",
    )
    parser.add_argument(
        "--min-valid-tv",
        type=int,
        default=6,
        help=(
            "Only keep violations whose finite time-to-violation is at least this "
            "value. Default: 6"
        ),
    )
    parser.add_argument(
        "--require-lanelet-steps",
        type=int,
        default=20,
        help=(
            "Only keep violations for egos whose rebuilt lanelet_assignment has exactly "
            "this many non-empty steps. Default: 20"
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help=(
            "Stop scanning as soon as this many filtered violating ego vehicles "
            "have been collected."
        ),
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=10,
        help=(
            "Flush filtered results to the output CSV every N newly collected rows. "
            "Default: 10"
        ),
    )
    return parser.parse_args()


def iter_scenario_files(
    scenario_dir: Path, scenario_pattern: str = "*.xml", limit: int = None
) -> List[Path]:
    scenario_files = sorted(scenario_dir.glob(scenario_pattern))
    if limit is not None:
        scenario_files = scenario_files[:limit]
    return scenario_files


def load_scenario(path: Path):
    scenario, planning_problem_set = CommonRoadFileReader(str(path)).open(
        lanelet_assignment=True
    )
    rebuild_lanelet_assignments(scenario)
    return scenario, planning_problem_set


def _iter_obstacle_states(obstacle):
    initial_state = getattr(obstacle, "initial_state", None)
    if initial_state is not None:
        yield initial_state

    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)
    state_list = getattr(trajectory, "state_list", None)
    if state_list is not None:
        for state in state_list:
            yield state


def rebuild_lanelet_assignments(scenario) -> None:
    lanelet_network = scenario.lanelet_network
    for obstacle in scenario.dynamic_obstacles:
        obstacle_shape = getattr(obstacle, "obstacle_shape", None)
        if obstacle_shape is None:
            continue

        lanelet_assignment = {}
        for state in _iter_obstacle_states(obstacle):
            position = getattr(state, "position", None)
            orientation = getattr(state, "orientation", None)
            time_step = getattr(state, "time_step", None)
            if position is None or orientation is None or time_step is None:
                continue

            try:
                obstacle_at_state = obstacle_shape.rotate_translate_local(
                    position, orientation
                )
                assigned_lanelets = lanelet_network.find_lanelet_by_shape(
                    obstacle_at_state
                )
            except Exception:
                assigned_lanelets = []

            if not assigned_lanelets:
                try:
                    assigned_lanelets = lanelet_network.find_lanelet_by_position(
                        [position]
                    )[0]
                except Exception:
                    assigned_lanelets = []

            lanelet_assignment[time_step] = set(assigned_lanelets)

        obstacle.lanelet_assignment = lanelet_assignment


def iter_candidate_ego_ids(scenario, limit: int = None) -> Iterable[int]:
    ego_ids = []
    for obstacle in scenario.dynamic_obstacles:
        obstacle_type = getattr(obstacle.obstacle_type, "name", str(obstacle.obstacle_type))
        if obstacle_type in VEHICLE_OBSTACLE_TYPES and get_invalid_ego_reason(
            obstacle, scenario
        ) is None:
            ego_ids.append(obstacle.obstacle_id)
    ego_ids.sort()
    if limit is not None:
        ego_ids = ego_ids[:limit]
    return ego_ids


def _trajectory_state_count(obstacle) -> int:
    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)
    state_list = getattr(trajectory, "state_list", None)
    return len(state_list) if state_list is not None else 0


def _get_time_bounds(obstacle):
    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)
    if prediction is None or trajectory is None:
        return None, None

    initial_time_step = getattr(trajectory, "initial_time_step", None)
    final_state = getattr(trajectory, "final_state", None)
    final_time_step = getattr(final_state, "time_step", None)
    return initial_time_step, final_time_step


def _lanelet_assigned_steps(obstacle) -> int:
    lanelet_assignment = getattr(obstacle, "lanelet_assignment", None)
    if not lanelet_assignment:
        return 0
    return sum(1 for lanelets in lanelet_assignment.values() if lanelets)


def _has_temporal_overlap_with_other_vehicle(obstacle, scenario) -> bool:
    ego_start, ego_end = _get_time_bounds(obstacle)
    if ego_start is None or ego_end is None:
        return False

    for other in scenario.dynamic_obstacles:
        if other.obstacle_id == obstacle.obstacle_id:
            continue
        other_type = getattr(other.obstacle_type, "name", str(other.obstacle_type))
        if other_type not in VEHICLE_OBSTACLE_TYPES:
            continue

        other_start, other_end = _get_time_bounds(other)
        if other_start is None or other_end is None:
            continue

        overlap_start = max(ego_start, other_start)
        overlap_end = min(ego_end, other_end)
        if overlap_end - overlap_start + 1 >= MIN_TEMPORAL_OVERLAP_STEPS:
            return True

    return False


def get_invalid_ego_reason(obstacle, scenario):
    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)
    if prediction is None or trajectory is None:
        return "missing_prediction"

    if _trajectory_state_count(obstacle) < MIN_TRAJECTORY_STATES:
        return "trajectory_too_short"

    initial_time_step, final_time_step = _get_time_bounds(obstacle)
    if initial_time_step is None or final_time_step is None:
        return "missing_time_bounds"
    if final_time_step <= initial_time_step:
        return "invalid_time_range"

    if not _has_temporal_overlap_with_other_vehicle(obstacle, scenario):
        return "no_temporal_overlap"

    return None


def collect_skipped_obstacles(scenario) -> List[str]:
    skipped = []
    for obstacle in scenario.dynamic_obstacles:
        obstacle_type = getattr(obstacle.obstacle_type, "name", str(obstacle.obstacle_type))
        if obstacle_type not in VEHICLE_OBSTACLE_TYPES:
            continue
        reason = get_invalid_ego_reason(obstacle, scenario)
        if reason is None:
            continue
        skipped.append(f"{obstacle.obstacle_id}:{reason}")
    return skipped


def collect_lanelet_diagnostics(scenario, ego_ids: List[int]) -> List[str]:
    diagnostics = []
    for ego_id in ego_ids:
        obstacle = scenario.obstacle_by_id(ego_id)
        assigned_steps = _lanelet_assigned_steps(obstacle)
        if assigned_steps == 0:
            diagnostics.append(f"{ego_id}:no_lanelet_assignment")
        elif assigned_steps < MIN_TRAJECTORY_STATES:
            diagnostics.append(f"{ego_id}:sparse_lanelet_assignment={assigned_steps}")
    return diagnostics


def build_config(scenario, planning_problem_set, scenario_path: Path, ego_id: int) -> RepairerConfiguration:
    config = RepairerConfiguration()
    config.scenario = scenario
    config.planning_problem_set = planning_problem_set
    planning_problem_dict = getattr(planning_problem_set, "planning_problem_dict", {})
    if planning_problem_dict:
        config.planning_problem = next(iter(planning_problem_dict.values()))
    config.general.path_scenarios = str(scenario_path.parent)
    config.general.set_path_scenario(scenario_path.name)
    config.repair.rules = ["R_G1", "R_G3"]
    config.repair.ego_id = ego_id
    config.repair.scenario_type = ScenarioType.INTERSTATE
    config.repair.multiproc = False
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = False
    return config


def detect_violations(
    scenario,
    planning_problem_set,
    scenario_path: Path,
    ego_id: int,
    verbose: bool = False,
):
    config = build_config(scenario, planning_problem_set, scenario_path, ego_id)
    if verbose:
        monitor = STLRuleMonitor(config)
    else:
        with redirect_stdout(io.StringIO()):
            monitor = STLRuleMonitor(config)

    violated_rules = list(getattr(monitor, "_violated_rules", []))
    return violated_rules, getattr(monitor, "rule_to_tv", {})


def _is_valid_tv(tv_value, min_valid_tv: int) -> bool:
    if tv_value is None:
        return False
    if isinstance(tv_value, float) and math.isinf(tv_value):
        return False
    try:
        return int(tv_value) >= min_valid_tv
    except (TypeError, ValueError, OverflowError):
        return False


def _filter_violations_by_tv(violated_rules, rule_to_tv, min_valid_tv: int):
    filtered_rules = []
    filtered_rule_to_tv = {}
    for rule in violated_rules:
        tv_value = rule_to_tv.get(rule)
        if _is_valid_tv(tv_value, min_valid_tv):
            filtered_rules.append(rule)
            filtered_rule_to_tv[rule] = int(tv_value)
    return filtered_rules, filtered_rule_to_tv


def _format_debug_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.4f}"
    return str(value)


def collect_rg3_debug_rows(
    scenario,
    planning_problem_set,
    scenario_path: Path,
    ego_id: int,
    max_steps: int,
):
    config = build_config(scenario, planning_problem_set, scenario_path, ego_id)
    config.repair.rules = ["R_G3"]

    world_config = get_world_config()
    traffic_rules_config = get_traffic_rule_config()
    traffic_rule_params = traffic_rules_config["traffic_rules_param"]
    world_config["scenario"] = traffic_rule_params["mpr_scenario"] = (
        config.repair.scenario_type
    )
    Cfg["common"]["scenario"] = config.repair.scenario_type
    traffic_rule_params["use_mpr"] = config.repair.use_mpr

    world = STLRuleMonitor._create_world_for_config(config, world_config)
    ego_vehicle = world.vehicle_by_id(config.repair.ego_id)
    ego_vehicle.vehicle_param = create_ego_vehicle_param(
        get_evaluation_config().get("ego_vehicle_param"), world.dt
    )

    evaluator = PropositionRuleEvaluator.create_from_config(
        world,
        config.repair.ego_id,
        "R_G3",
        traffic_rules_config=traffic_rules_config,
    )
    lane_pred = PredLaneSpeedLimit(traffic_rule_params)
    type_pred = PredTypeSpeedLimit(traffic_rule_params)
    fov_pred = PredFovSpeedLimit(traffic_rule_params)
    brake_pred = PredBrSpeedLimit(traffic_rule_params)

    rows = []
    for _ in range(
        evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1
    ):
        rule_rob = evaluator.update()
        time_step = evaluator._last_evaluation_time_step
        if len(rows) >= max_steps:
            continue

        speed = ego_vehicle.states_cr[time_step].velocity
        rows.append(
            {
                "t": time_step,
                "speed": speed,
                "lane_limit": lane_pred.get_speed_limit(world, time_step, [ego_id]),
                "type_limit": type_pred.get_speed_limit(world, time_step, [ego_id]),
                "fov_limit": fov_pred.get_speed_limit(world, time_step, [ego_id]),
                "brake_limit": brake_pred.get_speed_limit(world, time_step, [ego_id]),
                "lane_rob": lane_pred.evaluate_robustness(world, time_step, [ego_id]),
                "type_rob": type_pred.evaluate_robustness(world, time_step, [ego_id]),
                "fov_rob": fov_pred.evaluate_robustness(world, time_step, [ego_id]),
                "brake_rob": brake_pred.evaluate_robustness(world, time_step, [ego_id]),
                "rule_rob": rule_rob,
            }
        )

    return rows


def print_rg3_debug(
    scenario,
    planning_problem_set,
    scenario_path: Path,
    ego_id: int,
    max_steps: int,
) -> None:
    rows = collect_rg3_debug_rows(
        scenario, planning_problem_set, scenario_path, ego_id, max_steps
    )
    print(
        "  RG3 formula: keeps_lane_speed_limit(a0) and "
        "keeps_type_speed_limit(a0) and keeps_fov_speed_limit(a0) and "
        "keeps_brake_speed_limit(a0)"
    )
    if not rows:
        print("  RG3 debug: no evaluation rows")
        return

    for row in rows:
        print(
            "  RG3 t={t} speed={speed} "
            "lane_limit={lane_limit} type_limit={type_limit} "
            "fov_limit={fov_limit} brake_limit={brake_limit} "
            "lane_rob={lane_rob} type_rob={type_rob} "
            "fov_rob={fov_rob} brake_rob={brake_rob} rule_rob={rule_rob}".format(
                t=row["t"],
                speed=_format_debug_value(row["speed"]),
                lane_limit=_format_debug_value(row["lane_limit"]),
                type_limit=_format_debug_value(row["type_limit"]),
                fov_limit=_format_debug_value(row["fov_limit"]),
                brake_limit=_format_debug_value(row["brake_limit"]),
                lane_rob=_format_debug_value(row["lane_rob"]),
                type_rob=_format_debug_value(row["type_rob"]),
                fov_rob=_format_debug_value(row["fov_rob"]),
                brake_rob=_format_debug_value(row["brake_rob"]),
                rule_rob=_format_debug_value(row["rule_rob"]),
            )
        )


def is_known_invalid_ego_error(exc: Exception) -> bool:
    return isinstance(exc, IndexError) and "list index out of range" in str(exc)


def diagnose_monitor_failure(config: RepairerConfiguration) -> str:
    world_config = get_world_config()
    traffic_rules_config = get_traffic_rule_config()

    world_config["scenario"] = traffic_rules_config["traffic_rules_param"][
        "mpr_scenario"
    ] = config.repair.scenario_type
    Cfg["common"]["scenario"] = config.repair.scenario_type
    traffic_rules_config["traffic_rules_param"]["use_mpr"] = config.repair.use_mpr

    world = World.create_from_scenario(config.scenario, config=world_config)
    ego_vehicle = world.vehicle_by_id(config.repair.ego_id)
    ego_vehicle.vehicle_param = create_ego_vehicle_param(
        get_evaluation_config().get("ego_vehicle_param"), world.dt
    )

    reasons = []
    for rule in config.repair.rules:
        evaluator = PropositionRuleEvaluator.create_from_config(
            world,
            config.repair.ego_id,
            rule,
            traffic_rules_config=traffic_rules_config,
        )

        rule_rob = []
        other_ids = []
        non_empty_other_id_steps = 0
        non_empty_prop_steps = 0

        try:
            for _ in range(evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1):
                rule_rob.append(evaluator.update())
                other_ids.append(evaluator.other_ids)
                if evaluator.other_ids not in (None, (), []):
                    non_empty_other_id_steps += 1

                _, other_id_props, _, _ = evaluator.get_propositions_all()
                if other_id_props:
                    non_empty_prop_steps += 1
        except Exception as exc:
            reasons.append(f"{rule}:evaluator_error:{type(exc).__name__}")
            continue

        if not rule_rob:
            reasons.append(f"{rule}:no_rule_samples")
            continue

        first_other = other_ids[0] if other_ids else None
        if rule_rob[0] < 0 and first_other in (None, (), []):
            reasons.append(f"{rule}:immediate_violation_empty_other_ids")
            continue

        if non_empty_other_id_steps == 0:
            reasons.append(f"{rule}:no_related_vehicle_ids")
            continue

        if non_empty_prop_steps == 0:
            reasons.append(f"{rule}:no_proposition_matches")
            continue

        reasons.append(f"{rule}:unknown_monitor_mismatch")

    return ";".join(reasons)


def summarize_skip_reasons(skipped_obstacles: List[str]) -> str:
    counter = collections.Counter(
        item.split(":", 1)[1] for item in skipped_obstacles if ":" in item
    )
    return ", ".join(f"{reason}={count}" for reason, count in sorted(counter.items()))


def write_results(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scenario_name",
                "ego_id",
                "violated_rules",
                "rule_to_tv",
                "lanelet_assigned_steps",
                "raw_violated_rules",
                "raw_rule_to_tv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def maybe_flush_results(
    rows: List[dict],
    output_csv: Path,
    pending_count: int,
    write_every: int,
) -> int:
    if pending_count < write_every:
        return pending_count
    write_results(rows, output_csv)
    print(
        f"Flushed {len(rows)} filtered result(s) to {output_csv}",
        flush=True,
    )
    return 0


def main() -> None:
    args = parse_args()
    scenario_files = iter_scenario_files(
        args.scenario_dir, args.scenario_pattern, args.limit_scenarios
    )
    if not scenario_files:
        raise FileNotFoundError(f"No XML scenarios found in {args.scenario_dir}")

    results = []
    failed_cases = 0
    skipped_invalid_egos = 0
    filtered_out_cases = 0
    pending_writes = 0

    print(f"Scanning {len(scenario_files)} scenario(s) from {args.scenario_dir}")
    stop_early = False
    for scenario_index, scenario_path in enumerate(scenario_files, start=1):
        print(f"[{scenario_index}/{len(scenario_files)}] {scenario_path.name}")
        scenario, planning_problem_set = load_scenario(scenario_path)
        ego_ids = list(iter_candidate_ego_ids(scenario, args.limit_egos))
        skipped_obstacles = collect_skipped_obstacles(scenario)
        lanelet_diagnostics = collect_lanelet_diagnostics(scenario, ego_ids)
        print(f"  testing {len(ego_ids)} candidate ego vehicle(s)")
        if skipped_obstacles:
            print(
                f"  skipped {len(skipped_obstacles)} vehicle(s) without usable ego trajectories"
            )
            print(f"  skip summary: {summarize_skip_reasons(skipped_obstacles)}")
        if lanelet_diagnostics:
            print(
                f"  candidate lanelet diagnostics: {len(lanelet_diagnostics)} vehicle(s) "
                f"have missing or sparse lanelet assignments"
            )

        for ego_id in ego_ids:
            lanelet_assigned_steps = _lanelet_assigned_steps(scenario.obstacle_by_id(ego_id))
            try:
                violated_rules, rule_to_tv = detect_violations(
                    scenario,
                    planning_problem_set,
                    scenario_path,
                    ego_id,
                    verbose=args.verbose,
                )
            except Exception as exc:
                if is_known_invalid_ego_error(exc):
                    skipped_invalid_egos += 1
                    config = build_config(
                        scenario, planning_problem_set, scenario_path, ego_id
                    )
                    diagnostic = diagnose_monitor_failure(config)
                    print(
                        f"  ego {ego_id}: skipped because monitor found no usable "
                        f"rule-evaluation horizon ({diagnostic})"
                    )
                    continue
                failed_cases += 1
                print(f"  ego {ego_id}: failed with {type(exc).__name__}: {exc}")
                continue

            if args.debug_rg3 and (
                args.debug_ego_id is None or args.debug_ego_id == ego_id
            ):
                try:
                    print_rg3_debug(
                        scenario,
                        planning_problem_set,
                        scenario_path,
                        ego_id,
                        args.debug_max_steps,
                    )
                except Exception as exc:
                    print(
                        f"  ego {ego_id}: RG3 debug failed with "
                        f"{type(exc).__name__}: {exc}"
                    )

            filtered_rules, filtered_rule_to_tv = _filter_violations_by_tv(
                violated_rules, rule_to_tv, args.min_valid_tv
            )

            if violated_rules and lanelet_assigned_steps != args.require_lanelet_steps:
                filtered_rules = []
                filtered_rule_to_tv = {}

            if violated_rules and not filtered_rules:
                filtered_out_cases += 1
                continue

            if filtered_rules:
                row = {
                    "scenario_name": scenario_path.name,
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
                results.append(row)
                pending_writes += 1
                print(
                    f"  ego {ego_id}: violated {row['violated_rules']} "
                    f"({row['rule_to_tv']})"
                )
                pending_writes = maybe_flush_results(
                    results,
                    args.output_csv,
                    pending_writes,
                    max(args.write_every, 1),
                )
                if args.max_results is not None and len(results) >= args.max_results:
                    print(
                        f"Reached max-results={args.max_results}; stopping early."
                    )
                    stop_early = True
                    break
        if stop_early:
            break

    write_results(results, args.output_csv)
    print()
    print(f"Detected {len(results)} violating ego vehicle(s).")
    print(f"Filtered out violating ego checks: {filtered_out_cases}")
    print(f"Skipped invalid ego checks: {skipped_invalid_egos}")
    print(f"Failed ego checks: {failed_cases}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
