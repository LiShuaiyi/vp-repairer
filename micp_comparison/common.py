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


def validate_states(
    rule_monitor, ego_id, states, return_other_id=False, return_details=False,
):
    """Validate a candidate with the same protocol used by VP repair.

    ``return_other_id`` exposes the quantified vehicle selected at the first
    violation.  It is diagnostic only and does not affect compliance.
    """
    monitor = copy.copy(rule_monitor)
    world = copy.deepcopy(rule_monitor.world)
    monitor._world = world
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
