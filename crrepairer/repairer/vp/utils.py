"""General helpers for velocity-planning repair."""

import copy
import math
import os

import numpy as np

# from crrepairer.cut_off.utils import update_ego_vehicle
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from typing import List
from collections import defaultdict
from commonroad.scenario.state import CustomState


class VPUtils:
    """Small shared utilities used by the VP repair loop and example checks."""

    def _assign_proposition(self, propositions, model):
        """Select violated propositions from the SAT model for constraint extraction."""
        self._prop_full = propositions
        self._sel_prop = []
        for prop in propositions:
            if prop is not None and prop.alphabet in model:
                variable = prop.alphabet[-1]
                desired_value = 0 if prop.alphabet.startswith("~") else 1
                domain = getattr(self, "domain_dict", {}).get(variable)
                fixed_literal = (
                    variable in getattr(self, "_hard_domain_vars", set())
                    and domain == {desired_value}
                )
                entailed_negative_same_lane = (
                    prop.alphabet.startswith("~")
                    and "lane" in prop.name
                    and "same" in prop.name
                    and domain == {0}
                )
                if fixed_literal or entailed_negative_same_lane:
                    # This literal is a fixed fact already enforced by SAT.
                    # It needs no VP constraint and, for a negative literal,
                    # must not accidentally invoke the positive extractor.
                    continue
                if (prop.ttv_value < 0 and prop.alphabet[0] != "~") or (
                    prop.ttv_value > 0 and prop.alphabet[0] == "~"
                ):
                    self._sel_prop.append(prop)
                    print(
                        f"* \t<VPRepairer>: selected propositions: "
                        f"{prop.alphabet[-1]} {prop.name} = {prop.ttv_value}"
                    )

    def calc_tv_updated(self, updated_states, cut_off_time=None):
        """Re-evaluate the repaired trajectory and return the updated violation time."""
        monitor = copy.copy(self.rule_monitor)
        world = copy.deepcopy(self.rule_monitor.world)
        monitor._world = world

        world_ego = world.vehicle_by_id(self.ego_vehicle.obstacle_id)
        self.update_ego_vehicle(world.road_network, world_ego, updated_states, 0, world.dt)

        rule_rob, other_ids = monitor.evaluate_consecutively(world, monitor.start_time_step)
        if not all(len(arr) == len(rule_rob[0]) for arr in rule_rob):
            return -math.inf, None

        rule_rob = np.array(rule_rob)
        if np.any(rule_rob[:, 0] < 0):
            self._collect_candidate_predicate_debug(world, -math.inf)
            rule_idx = np.where(rule_rob[:, 0] < 0)[0][0]
            if other_ids[rule_idx][0] == ():
                return -math.inf, None
            return -math.inf, other_ids[rule_idx][0][0]

        tv_per_rule = np.argmax(rule_rob < 0, axis=-1)
        if np.all(tv_per_rule + world_ego.start_time == world_ego.start_time):
            self._collect_candidate_predicate_debug(world, math.inf)
            return math.inf, None

        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        self._collect_candidate_predicate_debug(world, min_tv * world.dt)
        if rule_idx == monitor.min_rule_idx:
            if other_ids[rule_idx][min_tv] == ():
                return min_tv * world.dt, self.ego_vehicle.obstacle_id
            return min_tv * world.dt, other_ids[rule_idx][min_tv][0]

        print("Violated rule changed.")
        return min_tv * world.dt, None

    @staticmethod
    def _debug_scalar(value):
        """Convert monitor scalar-like values to compact serializable values."""
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return repr(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 9)

    def _collect_candidate_predicate_debug(self, world, candidate_tv):
        """Collect a small robustness trace around TV when explicitly requested."""
        if not os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
            return

        diagnostic = {
            "candidate_tv": self._debug_scalar(candidate_tv),
            "rules": [],
        }
        try:
            if math.isfinite(candidate_tv):
                center = int(round(float(candidate_tv) / float(world.dt)))
            else:
                center = 0
            future_step = int(self.rule_monitor.future_time_step)
            shifted_center = center + future_step
            wanted = {
                max(0, center - 1),
                center,
                center + 1,
                max(0, shifted_center - 1),
                shifted_center,
                shifted_center + 1,
            }
            diagnostic["tv_step"] = center
            diagnostic["future_time_step"] = future_step

            for rule_idx, evaluator in enumerate(self.rule_monitor._rule_eval):
                evaluator.reset(
                    world.vehicle_by_id(self.rule_monitor.vehicle_id),
                    world,
                    self.rule_monitor.start_time_step,
                )
                self.rule_monitor.switch_to_robustness(evaluator)
                candidate_steps = {}
                relative_step = 0
                while evaluator.current_time < evaluator.ego_vehicle.end_time:
                    rule_value = evaluator.update()
                    if relative_step in wanted:
                        _, relevant_props, _, _ = (
                            self.rule_monitor._safe_get_propositions_all(evaluator)
                        )
                        predicates = evaluator.get_predicates() or {}
                        candidate_steps[str(relative_step)] = {
                            "rule": self._debug_scalar(rule_value),
                            "propositions": {
                                str(key): self._debug_scalar(value)
                                for key, value in relevant_props.items()
                            },
                            "predicates": {
                                str(key): self._debug_scalar(value)
                                for key, value in predicates.items()
                            },
                        }
                    relative_step += 1

                original_steps = {}
                for step in sorted(wanted):
                    if step >= len(self.rule_monitor.rob_predicate[rule_idx]):
                        continue
                    pred_values = self.rule_monitor.rob_predicate[rule_idx][step]
                    names = self.rule_monitor.abstraction_names[rule_idx][step]
                    values = self.rule_monitor.rob_abstraction[rule_idx][step]
                    original_steps[str(step)] = {
                        "rule": self._debug_scalar(
                            self.rule_monitor.rob_rule[rule_idx][step]
                        ),
                        "propositions": {
                            str(name): self._debug_scalar(value)
                            for name, value in zip(names, values)
                            if name
                        },
                        "predicates": {
                            str(key): self._debug_scalar(value)
                            for key, value in pred_values.items()
                        },
                    }
                diagnostic["rules"].append(
                    {
                        "rule": self.rule_monitor._rules[rule_idx],
                        "original": original_steps,
                        "candidate": candidate_steps,
                    }
                )
        except Exception as exc:
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        self._last_candidate_predicate_debug = diagnostic
    
    def update_ego_vehicle(
            self,
            road_network: RoadNetwork,
            ego_vehicle: Vehicle,
            updated_ego_states: List[CustomState],
            cut_off_time: int,
            dt,
        ):
        """
        Update the ego vehicle based on the new given trajectory
        """
        for state in updated_ego_states[cut_off_time:]:
            ego_vehicle.states_cr[state.time_step] = state
            ego_shape = ego_vehicle.shape.rotate_translate_local(
                state.position, state.orientation
            )
            ego_vehicle.ccosy_cache = CurvilinearStateManager(road_network)
            # use the shape lanelet assignment
            ego_vehicle.lanelet_assignment[state.time_step] = set(
                road_network.lanelet_network.find_lanelet_by_shape(ego_shape)
            )
            ego_vehicle.predicate_cache.cache[state.time_step] = defaultdict()
        pass
