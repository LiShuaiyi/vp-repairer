from typing import Union, List, Any, Tuple
from collections import defaultdict
import math
import functools
from abc import ABC
import enum

import numpy as np
from commonroad.scenario.obstacle import State, DynamicObstacle

from crrepairer.cut_off.base import CutOffBase
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.cut_off.utils import update_ego_vehicle, visualize_state_list, int_round
from crrepairer.cut_off.simulation import (CutOffAction,
                                           SimulationLateral,
                                           SimulationLong,
                                           check_elements_state_list)


class TCSearchMode(str, enum.Enum):
    LINEAR = "linear search"
    BINARY = "binary search"


class TC(CutOffBase, ABC):
    """
    Time-To-Compliance.
    """

    def __init__(self,
                 ego_vehicle: DynamicObstacle,
                 rule_monitor: STLRuleMonitor):
        super().__init__(ego_vehicle, rule_monitor.world)
        self.rule_monitor = rule_monitor
        self._world_ego = self.world.vehicle_by_id(ego_vehicle.obstacle_id)
        self._tv_time_step = rule_monitor.tv_time_step
        self._other_id = rule_monitor.other_id
        self._visualize = False
        self._compliant_maneuver = None
        self._tc = -math.inf
        self._tc_dict = defaultdict(float)
        self._mid = None
        self._search_mode = TCSearchMode.LINEAR

        # simulators
        self._sim_lon = SimulationLong(None,
                                       self.ego_vehicle,
                                       None,
                                       dt=rule_monitor.world.dt)
        self._sim_lat = SimulationLateral(None,
                                          self.ego_vehicle,
                                          None,
                                          rule_monitor.world.vehicle_by_id(ego_vehicle.obstacle_id),
                                          dt=rule_monitor.world.dt)

    @property
    def simulation_lateral(self) -> Union[SimulationLateral]:
        return self._sim_lat

    @property
    def simulation_longitudinal(self) -> Union[SimulationLong]:
        return self._sim_lon

    @property
    def tv(self):
        return int_round(self._tv_time_step * self.dT, 1)

    @property
    def tc(self):
        if self._tc == -math.inf:
            return self._tc
        return int_round(self._tc, 1)

    @property
    def tc_time_step(self) -> Union[int, float]:
        if self._tc == -math.inf:
            return self._tc
        return int(self._tc / self.dT)

    @property
    def tv_time_step(self) -> Union[int, float]:
        return self._tv_time_step

    @property
    def compliant_maneuver(self) -> CutOffAction:
        return self._compliant_maneuver

    def calc_tv_updated(self, updated_states: List[State], cut_off_time: int) -> Tuple[float, Any]:
        # detect violation time using STL monitor
        # self.rule_monitor.evaluate_initially()
        self.rule_monitor.world.time_step = 0
        update_ego_vehicle(self.world.road_network,
                           self._world_ego,
                           updated_states,
                           0,
                           self.dT)
        rule_rob, other_ids = self.rule_monitor.evaluate_consecutively(self.world, cut_off_time)
        if np.any(rule_rob[:, 0] < 0):
            rule_idx = np.where(rule_rob[:, 0] < 0)[0][0]
            if other_ids[rule_idx][0] is ():
                return -math.inf, None
            return -math.inf, other_ids[rule_idx][0][0]
        tv_per_rule = np.argmax(rule_rob < 0, axis=-1)
        if np.all(tv_per_rule == 0):
            return math.inf, None # no violation
        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if rule_idx == self.rule_monitor._violated_rule_idx:
            if other_ids[rule_idx][min_tv] is ():
                return min_tv * self.dT, self.ego_vehicle.obstacle_id
            return min_tv * self.dT, other_ids[rule_idx][min_tv][0]
        else:
            print("Violated rule changed.")

    def generate(self, cut_off_maneuvers: List[CutOffAction]):
        """
        Computes the Time-to-Compliance (with traffic rules).
        :param cut_off_maneuvers: the given maneuvers of ego vehicle
        :return: TC, corresponding maneuver
        """
        if not cut_off_maneuvers:
            return -math.inf
        if self.tv == -math.inf:
            raise ValueError("<TC>: the trajectory is not repairable since it already disobeys the rules")
        elif self.tv == math.inf:
            self._tc = math.inf
        else:
            ttm = dict()
            for maneuver in cut_off_maneuvers:
                if maneuver not in self._tc_dict:
                    if self._search_mode == TCSearchMode.BINARY:
                        ttm[maneuver] = self.search_ttm_binary(maneuver)
                    else:
                        ttm[maneuver] = self.search_ttm_linear(maneuver)
                    self._tc_dict[maneuver] = ttm[maneuver]
                else:
                    ttm[maneuver] = self._tc_dict[maneuver]

            self._tc = max(ttm.values())
            self._compliant_maneuver = max(ttm, key=ttm.get)
        return self._tc

    @functools.lru_cache(128)
    def search_ttm_binary(self, maneuver: CutOffAction):
        ttm = - math.inf
        low = 0
        high = int(int_round(self.tv / self.dT))
        while low < high:
            self._mid = int(int_round(low + high) / 2)
            if maneuver in [CutOffAction.BRAKE, CutOffAction.KICKDOWN, CutOffAction.STEADYSPEED]:
                self._sim_lon.update_action(maneuver, self._mid)
                state_list = self._sim_lon.simulate_state_list(self._mid)
            elif maneuver in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT]:
                self._sim_lat.update_action(maneuver, self._mid)
                state_list = self._sim_lat.simulate_state_list(self._mid)
            else:
                raise ValueError("<TC>: given compliant maneuver {} is not supported".format(maneuver))
            if state_list is None:
                flag_collision = True
                tv = -math.inf
            else:
                if self._visualize:
                    visualize_state_list(self._collision_checker, state_list, self.scenario,
                                         self._sim_lat.vehicle_dynamics.shape)
                # flag_collision = self._detect_collision(state_list)  # bool value
                check_elements_state_list(state_list, self.dT)
                try:
                    tv, _ = self.calc_tv_updated(state_list, self._mid)  # which should be tv instead of ttm
                except:
                    tv = -math.inf
            # if violation-free and collision-free
            if tv == math.inf:  # and not flag_collision:
                low = self._mid + 1
            else:
                high = self._mid

        if low != 0:
            ttm = (low - 1) * self.dT
        return ttm

    @functools.lru_cache(128)
    def search_ttm_linear(self, maneuver: CutOffAction):
        ts = int(int_round(self.tv / self.dT))
        while ts > 0:
            tv = self.singleton_search(maneuver, ts)
            if tv == math.inf:
                break
            else:
                ts -= 1
        if ts == 0:
            ttm = - math.inf
        else:
            ttm = ts * self.dT
        return ttm

    def singleton_search(self, maneuver: CutOffAction, start_time: int):
        if maneuver in [CutOffAction.BRAKE, CutOffAction.KICKDOWN, CutOffAction.STEADYSPEED]:
            self._sim_lon.update_action(maneuver, start_time)
            state_list = self._sim_lon.simulate_state_list(start_time)
        elif maneuver in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT]:
            self._sim_lat.update_action(maneuver, start_time)
            state_list = self._sim_lat.simulate_state_list(start_time)
        else:
            raise ValueError(": given compliant maneuver {} is not supported".format(maneuver))
        if state_list is None:
            tv = -math.inf
        else:
            if self._visualize:
                visualize_state_list(self._collision_checker, state_list, self.scenario,
                                     self._sim_lat.vehicle_dynamics.shape)
            check_elements_state_list(state_list, self.dT)
            try:
                tv, _ = self.calc_tv_updated(state_list, self._mid)  # which should be tv instead of ttm
            except:
                tv = -math.inf
        return tv
