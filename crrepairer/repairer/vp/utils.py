"""General helpers for velocity-planning repair."""

import copy
import math

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
            rule_idx = np.where(rule_rob[:, 0] < 0)[0][0]
            if other_ids[rule_idx][0] == ():
                return -math.inf, None
            return -math.inf, other_ids[rule_idx][0][0]

        tv_per_rule = np.argmax(rule_rob < 0, axis=-1)
        if np.all(tv_per_rule + world_ego.start_time == world_ego.start_time):
            return math.inf, None

        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if rule_idx == monitor.min_rule_idx:
            if other_ids[rule_idx][min_tv] == ():
                return min_tv * world.dt, self.ego_vehicle.obstacle_id
            return min_tv * world.dt, other_ids[rule_idx][min_tv][0]

        print("Violated rule changed.")
        return min_tv * world.dt, None
    
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
