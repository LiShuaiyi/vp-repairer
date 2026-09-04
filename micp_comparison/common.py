"""I/O and exact-monitor validation shared by the isolated MICP experiment."""

from __future__ import annotations

import copy
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from crmonitor.common.vehicle import CurvilinearStateManager
from crmonitor.evaluation.visitor import EvaluationMonitorTreeVisitor

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
import crrepairer.smt.monitor_wrapper as monitor_wrapper


RULE_GROUPS = {
    "R_G1": ("R_G1",),
    "R_G2": ("R_G2",),
    "R_G3": ("R_G3",),
    "R_G1_R_G3": ("R_G1", "R_G3"),
    "R_IN1": ("R_IN1",),
    "R_IN3": ("R_IN3",),
    "R_IN3_hand_draft": ("R_IN3_hand_draft",),
    "R_IN4": ("R_IN4",),
    "R_IN5": ("R_IN5",),
}

TIMED = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)


class FixedVehicleEvaluationVisitor(EvaluationMonitorTreeVisitor):
    """Evaluate quantified rules against one preselected vehicle binding.

    The regular monitor visitor enumerates every vehicle at every time step.
    That is appropriate for whole-scene monitoring, but it changes the target
    of a pairwise repair after optimization.  This visitor keeps the world
    intact for predicate evaluation while restricting each quantifier to the
    vehicle ID selected from the original violation.
    """

    def __init__(self, fixed_other_ids, **kwargs):
        super().__init__(**kwargs)
        self._fixed_other_ids = tuple(int(value) for value in fixed_other_ids)

    def _visit_quant_node(self, node, *ctx):
        world, mpr_world, time_step, bound_ids = ctx[:4]
        depth = len(bound_ids) - 1  # a0 (ego) is already bound by walk().
        if depth >= len(self._fixed_other_ids):
            raise ValueError("Missing fixed vehicle ID for nested rule quantifier")
        vehicle_id = self._fixed_other_ids[depth]
        if vehicle_id in bound_ids or vehicle_id not in world.vehicle_ids_for_time_step(time_step):
            return [], []
        ids = bound_ids + (vehicle_id,)
        value = node.monitors[vehicle_id].visit(
            self, world, mpr_world, time_step, ids, *ctx[2:]
        )
        return [value], [ids]


def align_rule_bounds(text, dt):
    def replace(match):
        values = []
        for name in ("low", "high"):
            value = float(match.group(name))
            steps = round(value / dt)
            if not math.isclose(value / dt, steps, abs_tol=1e-9):
                steps = math.ceil(value / dt)
            aligned = steps * dt
            values.append(str(int(aligned)) if math.isclose(aligned, round(aligned)) else f"{aligned:.12g}")
        return f"{match.group('op')}[{values[0]},{values[1]}{match.group('unit')}]"
    return TIMED.sub(replace, text)


def read_cases(
    path: Path, forced_rule=None, limit=None, offset=0,
    require_recorded_violation=False,
):
    """Read either a case manifest or a VP batch result without duplicates."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        seen = set()
        count = 0
        for row in reader:
            if row.get("repairer_type") not in (None, "", "vp"):
                continue
            if row.get("sat_solver_mode") not in (None, "", "domain_dpll"):
                continue
            # VP full result tables retain explicit skip rows for trajectories
            # which are already compliant.  A repair-method comparison must
            # use only rows on which a violation was actually recorded.
            if require_recorded_violation and "tv" in row and not row.get("tv"):
                continue
            scenario_id = row.get("scenario_id") or row.get("scenario")
            ego_id = row.get("ego_id")
            rule = forced_rule or row.get("rule") or row.get("rule_STL")
            if not scenario_id or not ego_id or not rule:
                continue
            scenario_id = scenario_id.removesuffix(".xml")
            key = (scenario_id, int(ego_id), rule)
            if key in seen:
                continue
            seen.add(key)
            if len(seen) <= offset:
                continue
            yield {
                "scenario_id": scenario_id,
                "scenario_path": row.get("scenario_path", ""),
                "ego_id": int(ego_id),
                "rule": rule,
            }
            count += 1
            if limit is not None and count >= limit:
                return


def resolve_scenario_path(dataset, scenario_dir, case):
    explicit = case.get("scenario_path")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if path.is_file():
            return path
    if scenario_dir is None:
        raise FileNotFoundError("No valid scenario_path and no --scenario-dir")
    candidates = [case["scenario_id"]]
    if dataset == "highd":
        candidates.insert(0, case["scenario_id"].replace("T-1", f"T-{case['ego_id']}"))
    else:
        candidates.insert(0, case["scenario_id"].replace("-1_", f"-{case['ego_id']}_"))
    for stem in dict.fromkeys(candidates):
        path = scenario_dir / f"{stem}.xml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Scenario not found for {case['scenario_id']}")


def make_monitor(dataset, scenario_path, scenario, ego_id, rule):
    config = RepairerConfiguration()
    config.general.path_scenarios = str(scenario_path.parent)
    config.general.set_path_scenario(scenario_path.stem)
    config.update()
    config.repair.scenario_type = "intersection" if dataset == "ind" else "interstate"
    if dataset == "ind":
        config.repair.intersection_type = "dataset"
    config.repair.rules = list(RULE_GROUPS[rule])
    config.repair.ego_id = ego_id
    config.scenario = scenario
    if rule not in {"R_IN3", "R_IN5"}:
        return STLRuleMonitor(config)
    original = monitor_wrapper.get_traffic_rule_config
    def aligned_config(*args, **kwargs):
        result = original(*args, **kwargs)
        for name in RULE_GROUPS[rule]:
            result["traffic_rules"][name] = align_rule_bounds(
                result["traffic_rules"][name], float(scenario.dt)
            )
        return result
    monitor_wrapper.get_traffic_rule_config = aligned_config
    try:
        return STLRuleMonitor(config)
    finally:
        monitor_wrapper.get_traffic_rule_config = original


def select_fixed_other_id(rule_monitor, rule, ego_id):
    """Return the vehicle binding associated with the original violation."""
    if rule in {"R_G3", "R_IN1"}:
        return None
    quantified_rule = "R_G1" if rule == "R_G1_R_G3" else rule
    vehicle_id = rule_monitor.rule_to_other_id.get(quantified_rule)
    if vehicle_id is not None and int(vehicle_id) != int(ego_id):
        return int(vehicle_id)

    if rule != "R_G2":
        raise ValueError(f"{rule} has no related vehicle in the original violation")

    # R_G2's outer implication is ego-centric, so the monitor wrapper can
    # report a0 rather than the existential a1.  Bind the nearest relevant
    # preceding vehicle once at the original trigger and retain it thereafter.
    world = rule_monitor.world
    ego = world.vehicle_by_id(ego_id)
    trigger = rule_monitor.rule_to_tv.get("R_G2", rule_monitor.tv_time_step)
    step = 0 if not math.isfinite(trigger) else max(0, int(trigger))
    step = min(step, ego.end_time)
    lane = ego.get_lane(step)
    candidates = []
    for candidate_id in world.vehicle_ids_for_time_step(step):
        if int(candidate_id) == int(ego_id):
            continue
        target = world.vehicle_by_id(candidate_id)
        try:
            rear = target.rear_s(step, lane)
            front = ego.front_s(step, lane)
            if (
                rear is not None and front is not None and rear >= front
                and ego.lanes_at_state(step).intersection(target.lanes_at_state(step))
            ):
                candidates.append((rear - front, int(candidate_id)))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    if not candidates:
        raise ValueError("R_G2 has no preceding vehicle at the original trigger")
    return min(candidates)[1]


def validate_states(
    rule_monitor, ego_id, states, fixed_other_id=None,
    return_other_id=False, return_details=False,
):
    """Validate a candidate and return its fixed-binding time to violation.

    Pairwise rules must pass the same ``fixed_other_id`` used to construct the
    optimization problem.  This prevents final validation from silently
    switching to another vehicle in the scene.  Single-vehicle rules leave it
    unset.  ``return_other_id`` is retained for diagnostics.
    """
    monitor = copy.copy(rule_monitor)
    world = copy.deepcopy(rule_monitor.world)
    monitor._world = world
    if fixed_other_id is not None:
        for evaluator in monitor._rule_eval:
            previous = evaluator._eval_visitor
            evaluator._eval_visitor = FixedVehicleEvaluationVisitor(
                (fixed_other_id,),
                use_boolean=previous.use_boolean,
                output_type=previous.output_type,
            )
    ego = world.vehicle_by_id(ego_id)
    for state in states:
        ego.states_cr[state.time_step] = state
        shape = ego.shape.rotate_translate_local(state.position, state.orientation)
        ego.ccosy_cache = CurvilinearStateManager(world.road_network)
        ego.lanelet_assignment[state.time_step] = set(
            world.road_network.lanelet_network.find_lanelet_by_shape(shape)
        )
        # A predicate may cache on its first vehicle while depending on the ego
        # as another argument (notably conflict_area(a1, a0)). Replacing only
        # the ego trajectory therefore invalidates caches of every vehicle at
        # this time step, not just ego.predicate_cache.
        for vehicle_id in world.vehicle_ids_for_time_step(state.time_step):
            world.vehicle_by_id(vehicle_id).predicate_cache.cache[
                state.time_step
            ] = defaultdict()
    robustness, other_ids = monitor.evaluate_consecutively(
        world, monitor.start_time_step
    )

    def diagnostics(time_index, raw_other_id):
        if not return_details or raw_other_id in (None, (), []):
            return ""
        other_id = raw_other_id[0] if isinstance(raw_other_id, (tuple, list)) else raw_other_id
        try:
            from crmonitor.common.config import get_traffic_rule_config
            from crmonitor.predicates.acceleration import PredCausesBrakingIntersection
            from crmonitor.predicates.position import (
                PredInIntersectionConflictArea,
                PredOnLaneletWithTypeIntersection,
            )
            params = get_traffic_rule_config()["traffic_rules_param"]
            conflict = PredInIntersectionConflictArea(params)
            on_intersection = PredOnLaneletWithTypeIntersection(params)
            braking = PredCausesBrakingIntersection(params)
            samples = []
            final_time = min(
                world.vehicle_by_id(ego_id).end_time,
                time_index + int(getattr(monitor, "future_time_step", 0)) + 1,
            )
            for step in range(max(0, time_index - 1), final_time + 1):
                ego_vehicle = world.vehicle_by_id(ego_id)
                other_vehicle = world.vehicle_by_id(other_id)
                assignment = set(ego_vehicle.lanelet_assignment[step])
                conflict_ids = assignment.intersection(
                    other_vehicle.ref_path_lane.contained_lanelets
                ).difference(ego_vehicle.lanelets_dir)
                lon = ego_vehicle.get_lon_state(step, ego_vehicle.ref_path_lane)
                lat = ego_vehicle.get_lat_state(step, ego_vehicle.ref_path_lane)
                samples.append({
                    "t": step,
                    "s": None if lon is None else lon.s,
                    "d": None if lat is None else lat.d,
                    "conflict_lanelet_ids": sorted(conflict_ids),
                    "ego_in_conflict": conflict.evaluate_boolean(
                        world, step, [ego_id, other_id]
                    ),
                    "other_in_conflict": conflict.evaluate_boolean(
                        world, step, [other_id, ego_id]
                    ),
                    "causes_braking": braking.evaluate_boolean(
                        world, step, [ego_id, other_id]
                    ),
                    "ego_on_intersection": on_intersection.evaluate_boolean(
                        world, step, [ego_id]
                    ),
                })
            return json.dumps(samples, sort_keys=True)
        except (AttributeError, KeyError, TypeError, ValueError):
            return ""

    def result(tv, other_id=None, time_index=None):
        values = [tv]
        if return_other_id:
            values.append(other_id)
        if return_details:
            values.append(diagnostics(time_index, other_id))
        return tuple(values) if len(values) > 1 else tv

    lengths = [len(values) for values in robustness]
    if not lengths or any(length != lengths[0] for length in lengths):
        return result(-math.inf)
    values = np.asarray(robustness)
    if np.any(values[:, 0] < 0):
        rule_index = int(np.flatnonzero(values[:, 0] < 0)[0])
        selected = other_ids[rule_index][0] if other_ids[rule_index] else None
        return result(-math.inf, selected, 0)
    tv_per_rule = np.argmax(values < 0, axis=-1)
    if np.all(tv_per_rule == 0):
        return result(math.inf)
    positive = np.flatnonzero(tv_per_rule != 0)
    rule_index = int(positive[np.argmin(tv_per_rule[positive])])
    time_index = int(tv_per_rule[rule_index])
    selected = (
        other_ids[rule_index][time_index]
        if time_index < len(other_ids[rule_index])
        else None
    )
    return result(float(time_index * world.dt), selected, time_index)
