"""Run the isolated lattice baseline on a real CommonRoad RG/IN1 case."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from commonroad.scenario.state import CustomState
from commonroad.scenario.lanelet import LaneletType
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.vehicle import CurvilinearStateManager
from crmonitor.predicates.velocity import (
    PredBrSpeedLimit,
    PredFovSpeedLimit,
    PredLaneSpeedLimit,
    PredTypeSpeedLimit,
)

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration

from .environment import Environment
from .model import LatticeConfig
from .planner import MinimumViolationPlanner
from .rules import build_rulebook


INTERSECTION_RULES = {"R_IN3", "R_IN3_hand_draft", "R_IN4", "R_IN5"}
RULE_GROUPS = {"R_G1_R_G3": ("R_G1", "R_G3")}
SUPPORTED_RULES = {
    "R_G1", "R_G2", "R_G3", "R_IN1", *INTERSECTION_RULES, *RULE_GROUPS
}


def expanded_rules(rule):
    return RULE_GROUPS.get(rule, (rule,))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--ego-id", type=int, required=True)
    parser.add_argument("--rule", choices=sorted(SUPPORTED_RULES), required=True)
    parser.add_argument("--scenario-type", choices=("interstate", "intersection"))
    parser.add_argument("--intersection-type", default="dataset")
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--a-min", type=float, default=-10.0)
    parser.add_argument("--a-max", type=float, default=5.0)
    parser.add_argument("--max-expansions", type=int, default=1_000_000)
    parser.add_argument(
        "--tc", type=float, default=0.0,
        help="selected repair cut-off time in seconds (default: 0)",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _monitor(scenario_path, ego_id, rule, scenario_type, intersection_type):
    config = RepairerConfiguration()
    config.general.path_scenarios = str(scenario_path.parent) + "/"
    config.general.set_path_scenario(scenario_path.name)
    config.update()
    config.repair.rules = list(expanded_rules(rule))
    config.repair.ego_id = ego_id
    if scenario_type:
        config.repair.scenario_type = scenario_type
    if scenario_type == "intersection":
        config.repair.intersection_type = intersection_type
    config.debug.show_plots = False
    config.update()
    # This experiment defines an explicitly named R_IN5-0.6 variant so the
    # temporal bound is representable at dt=0.2 s.  The cached shared config is
    # restored immediately after monitor construction; no package file or VP
    # configuration is modified.
    if rule == "R_IN5":
        traffic_config = get_traffic_rule_config()
        original_formula = traffic_config["traffic_rules"]["R_IN5"]
        traffic_config["traffic_rules"]["R_IN5"] = original_formula.replace(
            "eventually[0,0.52s]", "eventually[0,0.6s]"
        )
        try:
            return STLRuleMonitor(config)
        finally:
            traffic_config["traffic_rules"]["R_IN5"] = original_formula
    return STLRuleMonitor(config)


def _lead_signals(world, ego, reference_lane, count):
    leads = []
    vehicle_ids = set()
    for k in range(count):
        vehicle_ids.update(world.vehicle_ids_for_time_step(k))
    for vehicle_id in sorted(vehicle_ids):
        if vehicle_id == ego.id:
            continue
        vehicle = world.vehicle_by_id(vehicle_id)
        rear_s, velocity, acceleration, same_lane = [], [], [], []
        for k in range(count):
            if k < vehicle.start_time or k > vehicle.end_time:
                rear_s.append(1e9)
                velocity.append(0.0)
                acceleration.append(0.0)
                same_lane.append(False)
                continue
            try:
                lon = vehicle.get_lon_state(k, reference_lane)
                rear = vehicle.rear_s(k, reference_lane)
                lanes = vehicle.lanes_at_state(k)
                ego_lanes = ego.lanes_at_state(min(k, ego.end_time))
            except Exception:
                lon, rear, lanes, ego_lanes = None, None, set(), set()
            valid = lon is not None and rear is not None
            rear_s.append(float(rear) if valid else 1e9)
            velocity.append(float(lon.v) if valid else 0.0)
            # MONA's projected StateLongitudinal omits acceleration in some
            # scenarios. R_G1 does not use it, while R_G2 can fall back to the
            # original CommonRoad state when available.
            state_cr = vehicle.states_cr.get(k)
            raw_acceleration = getattr(
                lon,
                "a",
                getattr(state_cr, "acceleration", 0.0),
            )
            acceleration.append(float(raw_acceleration) if valid else 0.0)
            same_lane.append(bool(valid and lanes.intersection(ego_lanes)))
        leads.append(
            {
                "id": str(vehicle_id),
                "rear_s": rear_s,
                "velocity": velocity,
                "acceleration": acceleration,
                "same_lane": same_lane,
            }
        )
    return leads


def _speed_limits(world, ego_id, count):
    params = get_traffic_rule_config()["traffic_rules_param"]
    predicates = {
        "lane": PredLaneSpeedLimit(params),
        "type": PredTypeSpeedLimit(params),
        "fov": PredFovSpeedLimit(params),
        "braking": PredBrSpeedLimit(params),
    }
    limits = {name: [] for name in predicates}
    for k in range(count):
        for name, predicate in predicates.items():
            try:
                value = predicate.get_speed_limit(world, k, [ego_id])
            except Exception:
                value = None
            limits[name].append(None if value is None else float(value))
    return limits


def _stop_line(reference_lane, world, initial_s):
    network = world.road_network.lanelet_network
    candidates = []
    for lanelet_id in getattr(reference_lane, "contained_lanelets", ()):
        lanelet = network.find_lanelet_by_id(lanelet_id)
        if lanelet.stop_line is None:
            continue
        start = np.asarray(lanelet.stop_line.start, dtype=float)
        end = np.asarray(lanelet.stop_line.end, dtype=float)
        projected = []
        for point in (start, end):
            try:
                projected.append(
                    float(reference_lane.clcs.convert_to_curvilinear_coords(*point)[0])
                )
            except Exception:
                continue
        if projected and min(projected) >= initial_s - 1.0:
            # Matches PredStopLineInFront, which uses the smaller projection of
            # the two stop-line endpoints.
            candidates.append(min(projected))
    if not candidates:
        raise ValueError("no forward stop line could be projected onto the reference path")
    return min(candidates)


def _reference_bounds(reference_lane):
    domain = np.asarray(reference_lane.clcs.curvilinear_projection_domain(), dtype=float)
    return float(np.min(domain[:, 0])), float(np.max(domain[:, 0]))


def _collect_coordinates(geometry, output):
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            _collect_coordinates(child, output)
    elif hasattr(geometry, "coords"):
        output.extend(geometry.coords)


def _intersection_signal(monitor, ego, reference_lane, count, rule):
    other_id = int(monitor.other_id)
    target = monitor.world.vehicle_by_id(other_id)
    network = monitor.world.road_network.lanelet_network

    def intersection_lanelets(vehicle):
        result = []
        for lanelet_id in vehicle.lanelets_dir:
            lanelet = network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                result.append(lanelet)
        return result

    ego_lanelets = intersection_lanelets(ego)
    target_lanelets = intersection_lanelets(target)
    if not ego_lanelets or not target_lanelets:
        raise ValueError("selected ego/target routes have no intersection lanelets")
    ego_region = shapely.unary_union(
        [lanelet.polygon.shapely_object for lanelet in ego_lanelets]
    )
    target_region = shapely.unary_union(
        [lanelet.polygon.shapely_object for lanelet in target_lanelets]
    )
    conflict = ego_region.intersection(target_region)
    if conflict.is_empty:
        raise ValueError("selected ego/target routes have no geometric conflict")
    enlarged = conflict.buffer(0.5 * float(ego.shape.length))
    reference = np.asarray(reference_lane.clcs.reference_path(), dtype=float)
    path_conflict = shapely.LineString(reference).intersection(enlarged)
    coordinates = []
    _collect_coordinates(path_conflict, coordinates)
    projected = []
    for x, y in coordinates:
        try:
            projected.append(
                float(reference_lane.clcs.convert_to_curvilinear_coords(x, y)[0])
            )
        except Exception:
            continue
    if not projected:
        raise ValueError("conflict area does not intersect the fixed reference path")

    target_in_conflict = []
    for k in range(count):
        if k < target.start_time or k > target.end_time:
            target_in_conflict.append(False)
            continue
        state = target.states_cr[k]
        shape = target.shape.rotate_translate_local(state.position, state.orientation)
        target_in_conflict.append(bool(shape.shapely_object.intersects(conflict)))
    clearance = 1.0 if rule in {"R_IN3", "R_IN3_hand_draft"} else 0.6
    return {
        "conflict_interval": [min(projected), max(projected)],
        # The monitor selected this target as the earliest violating quantified
        # witness. Its route-level priority antecedent is therefore active for
        # this paired repair instance.
        "priority_active": True,
        "target_in_conflict": target_in_conflict,
        "ego_lookahead_s": 1.0,
        "target_clearance_s": clearance,
        # The monitor tests the oriented ego shape against lanelet/conflict
        # polygons with a 1 mm predicate margin.  The path projection of the
        # two-dimensional conflict polygon is not exact along a curved vehicle
        # footprint; use the same 0.6 m clearance horizon as the IN formula so
        # a longitudinal boundary cell cannot still overlap in Cartesian space.
        "clearance": 0.6 if rule == "R_IN4" else 0.0,
    }


def extract_problem(monitor, rule, requested_dv, a_min, a_max, max_expansions):
    world = monitor.world
    ego_id = monitor._vehicle_id
    ego = world.vehicle_by_id(ego_id)
    reference_lane = ego.ref_path_lane or ego.get_lane(0)
    if reference_lane is None:
        raise ValueError("ego has no fixed reference lane")
    if ego.start_time != 0:
        raise ValueError(f"adapter requires ego start_time=0, got {ego.start_time}")
    count = ego.end_time + 1
    initial = ego.get_lon_state(0, reference_lane)
    if initial is None:
        raise ValueError("initial ego state cannot be projected")
    initial_v = max(0.0, float(initial.v))
    # The released Halder planner does not require the measured initial speed
    # to lie on the regular velocity grid.  Its special cubic first edge joins
    # that initial state to every reachable (s, v) grid point.  Shrinking dv to
    # fit the initial speed changes the problem size by orders of magnitude.
    dv = requested_dv
    dt = float(world.dt)
    ds = 0.5 * dv * dt
    s_min, s_max = _reference_bounds(reference_lane)
    original_velocities = []
    original_positions = []
    for k in range(count):
        lon = ego.get_lon_state(k, reference_lane)
        if lon is not None:
            original_velocities.append(max(0.0, float(lon.v)))
            original_positions.append(float(lon.s))
    v_max = max(original_velocities + [initial_v + max(0.0, a_max) * count * dt])
    s_max = min(s_max, max(original_positions) + max(10.0, initial_v * count * dt))

    data = {
        "speed_limits": {},
        "lead_vehicles": [],
        "original_trajectory": {
            "s": original_positions,
            "v": original_velocities,
        },
    }
    active_rules = expanded_rules(rule)
    if any(name in {"R_G1", "R_G2"} for name in active_rules):
        data["lead_vehicles"] = _lead_signals(world, ego, reference_lane, count)
    if "R_G3" in active_rules:
        data["speed_limits"] = _speed_limits(world, ego.id, count)
    if rule in INTERSECTION_RULES:
        data["intersection_rules"] = {
            rule: _intersection_signal(monitor, ego, reference_lane, count, rule)
        }

    cfg = LatticeConfig(
        dt=dt,
        ds=ds,
        dv=dv,
        horizon_steps=count - 1,
        s_min=min(s_min, float(initial.s)),
        s_max=s_max,
        v_min=0.0,
        v_max=math.ceil(v_max / dv) * dv,
        a_min=a_min,
        a_max=a_max,
        max_expansions=max_expansions,
    )
    parameters = {}
    if rule == "R_IN1":
        parameters[rule] = {
            "stop_line_s": _stop_line(reference_lane, world, float(initial.s)),
            "ego_length": float(ego.shape.length),
            "standstill_tolerance": 0.01,
            "approach_distance": 1.0,
            "stop_duration_s": 3.0,
            "history_s": 3.0,
        }
    for active_rule in active_rules:
        if active_rule in {"R_G1", "R_G2"}:
            parameters[active_rule] = {"ego_length": float(ego.shape.length)}
    rules = build_rulebook(
        ["vehicle_constraints", *active_rules, "trajectory_tracking", "acceleration_comfort"],
        parameters,
        cfg,
    )
    return cfg, Environment(data, count, dt), rules, float(initial.s), initial_v, reference_lane


def _states_on_reference(result, reference_lane):
    states = []
    for lattice_state in result.states:
        position = np.asarray(
            reference_lane.clcs.convert_to_cartesian_coords(lattice_state.s, 0.0),
            dtype=float,
        )
        states.append(
            CustomState(
                time_step=lattice_state.k,
                position=position,
                orientation=float(reference_lane.orientation(lattice_state.s)),
                velocity=lattice_state.v,
                acceleration=lattice_state.a,
            )
        )
    return states


def validate(monitor, ego_id, states, tc=0.0, target_id=None):
    """Recheck the rule for the target selected by this repair iteration.

    The monitor's universal quantifier may select a different worst vehicle
    after ego has been repaired.  That is a new repair obligation, not failure
    of the current iteration.  We therefore retain the original rule and
    target, evaluate its complete temporal trace from the scenario start (so
    past/future operators keep their history), and judge only samples at/after
    the selected cut-off time ``tc``.
    """
    validation_start = time.perf_counter()
    checked = copy.copy(monitor)
    world = copy.deepcopy(monitor.world)
    checked._world = world
    ego = world.vehicle_by_id(ego_id)
    for state in states:
        ego.states_cr[state.time_step] = state
        shape = ego.shape.rotate_translate_local(state.position, state.orientation)
        ego.ccosy_cache = CurvilinearStateManager(world.road_network)
        ego.lanelet_assignment[state.time_step] = set(
            world.road_network.lanelet_network.find_lanelet_by_shape(shape)
        )
        ego.predicate_cache.cache[state.time_step] = defaultdict()

    rule_idx = checked.min_rule_idx
    rule_name = checked._rules[rule_idx]
    if target_id is None:
        target_id = checked.rule_to_other_id.get(rule_name)
    if target_id is None:
        target_id = ego_id
    target_id = int(target_id)
    first_checked_step = max(
        checked.start_time_step,
        checked.start_time_step + int(round(float(tc) / float(world.dt))),
    )

    evaluator = checked._rule_eval[rule_idx]
    evaluator.reset(ego_id, world, checked.start_time_step)
    target_trace = []
    selected_trace = []
    while evaluator.current_time < evaluator.ego_vehicle.end_time:
        aggregate = float(evaluator.update())
        values = evaluator.all_values_all_ids
        value = values.get(target_id)
        if value is None and target_id == ego_id:
            value = aggregate
        if value is not None:
            value = float(value)
        target_trace.append(
            {
                "time_step": int(evaluator.current_time),
                "robustness": value,
            }
        )
        if evaluator.current_time >= first_checked_step and value is not None:
            selected_trace.append(value)

    # Predicate geometry and coordinate projection introduce round-off at an
    # exactly neutral boundary (observed around 1e-15).  Treat that as zero,
    # consistently with the monitor's own epsilon-based predicate checks.
    compliant = bool(
        selected_trace and all(value >= -1.0e-9 for value in selected_trace)
    )
    details = {
        "criterion": "selected_tc_target",
        "rule": rule_name,
        "tc_s": float(tc),
        "first_checked_time_step": first_checked_step,
        "target_id": target_id,
        "robustness": target_trace,
    }
    return compliant, time.perf_counter() - validation_start, details


def validate_rule_group(monitor, ego_id, states, tc=0.0):
    """Recheck every rule in a combined repair on the same repaired world."""
    validation_start = time.perf_counter()
    checked = copy.copy(monitor)
    world = copy.deepcopy(monitor.world)
    checked._world = world
    ego = world.vehicle_by_id(ego_id)
    for state in states:
        ego.states_cr[state.time_step] = state
        shape = ego.shape.rotate_translate_local(state.position, state.orientation)
        ego.ccosy_cache = CurvilinearStateManager(world.road_network)
        ego.lanelet_assignment[state.time_step] = set(
            world.road_network.lanelet_network.find_lanelet_by_shape(shape)
        )
        ego.predicate_cache.cache[state.time_step] = defaultdict()

    first_checked_step = max(
        checked.start_time_step,
        checked.start_time_step + int(round(float(tc) / float(world.dt))),
    )
    targets = {}
    rule_details = []
    group_compliant = True
    for rule_idx, rule_name in enumerate(checked._rules):
        evaluator = checked._rule_eval[rule_idx]
        evaluator.reset(ego_id, world, checked.start_time_step)
        target_id = checked.rule_to_other_id.get(rule_name)
        if target_id is None:
            target_id = ego_id
        target_id = int(target_id)
        targets[rule_name] = target_id
        trace = []
        selected_trace = []
        while evaluator.current_time < evaluator.ego_vehicle.end_time:
            aggregate = float(evaluator.update())
            value = evaluator.all_values_all_ids.get(target_id)
            if value is None and target_id == ego_id:
                value = aggregate
            if value is not None:
                value = float(value)
            trace.append(
                {"time_step": int(evaluator.current_time), "robustness": value}
            )
            if evaluator.current_time >= first_checked_step and value is not None:
                selected_trace.append(value)
        compliant = bool(
            selected_trace
            and all(value >= -1.0e-9 for value in selected_trace)
        )
        group_compliant = group_compliant and compliant
        rule_details.append(
            {
                "rule": rule_name,
                "target_id": target_id,
                "compliant": compliant,
                "robustness": trace,
            }
        )

    details = {
        "criterion": "selected_tc_target_all_rules",
        "tc_s": float(tc),
        "first_checked_time_step": first_checked_step,
        "targets": targets,
        "rules": rule_details,
    }
    return group_compliant, time.perf_counter() - validation_start, details


def run_case(args):
    overall_start = time.perf_counter()
    monitor_start = time.perf_counter()
    monitor = _monitor(
        args.scenario.resolve(), args.ego_id, args.rule,
        args.scenario_type, args.intersection_type,
    )
    monitor_initialization = time.perf_counter() - monitor_start
    preprocessing_start = time.perf_counter()
    cfg, env, rules, s0, v0, reference_lane = extract_problem(
        monitor, args.rule, args.dv, args.a_min, args.a_max, args.max_expansions
    )
    preprocessing = time.perf_counter() - preprocessing_start
    result = MinimumViolationPlanner(cfg, env, rules).plan(s0, v0)
    states = _states_on_reference(result, reference_lane)
    monitor_compliant, validation_time, robustness = validate(
        monitor, args.ego_id, states, tc=args.tc
    )
    payload = result.as_dict()
    payload.update(
        {
            "scenario": str(args.scenario.resolve()),
            "ego_id": args.ego_id,
            "rule": args.rule,
            "monitor_initialization_time_s": monitor_initialization,
            "preprocessing_time_s": preprocessing,
            "search_time_s": result.runtime_s,
            "core_total_time_s": preprocessing + result.runtime_s,
            "wall_time_s": time.perf_counter() - overall_start,
            "monitor_validation_time_s": validation_time,
            "monitor_compliant": monitor_compliant,
            "selected_tc_target_compliant": monitor_compliant,
            "selected_tc_target_validation": robustness,
            "lattice": {
                "dt": cfg.dt,
                "ds": cfg.ds,
                "dv": cfg.dv,
                "horizon_steps": cfg.horizon_steps,
                "v_max": cfg.v_max,
                "a_min": cfg.a_min,
                "a_max": cfg.a_max,
            },
        }
    )
    return payload


def main(argv=None):
    args = _parser().parse_args(argv)
    payload = run_case(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
