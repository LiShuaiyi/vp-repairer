from typing import Iterable, Union, List, Any, Tuple
import math
from abc import ABC

import numpy as np
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world_state import WorldState

import matplotlib.pyplot as plt
from cut_off.base import CutOffBase
from lazy_smt.monitor import STLRuleMonitor
from cut_off.utils import update_ego_vehicle, visualize_state_list
from cut_off.simulation import CutOffAction, SimulationLateral, SimulationLong


class TC(CutOffBase, ABC):
    """
    Time-To-Compliance.
    """
    def __init__(self,
                 rule_monitor: STLRuleMonitor,
                 dT: float = 0.1):
        super().__init__(rule_monitor.world_state, dT)
        self.rule_monitor = rule_monitor
        self._tv = rule_monitor.tv * self.dT  # time step -> time
        self._other_id = rule_monitor.other_id
        self._visualize = True
        self._compliant_maneuver = None
        self._tc = None
        self._simulation_lateral = None

    @property
    def simulation_lateral(self) -> Union[SimulationLong, SimulationLateral]:
        return self._simulation_lateral

    @property
    def tv(self):
        return self._tv

    @property
    def tc(self):
        if self._tc is None:
            raise ValueError("<TC> the tc is not evaluated yet.")
        return self._tc

    @property
    def tc_time_step(self) -> int:
        return int(self._tc/self.dT)

    @property
    def tv_time_step(self) -> int:
        return int(self._tv / self.dT)

    @property
    def compliant_maneuver(self) -> CutOffAction:
        return self._compliant_maneuver

    def _calc_tv_updated(self, updated_states: List[State] = None,
                         start_time_step: int = None) -> Tuple[float, Any]:
        # detect violation time using STL monitor
        # self.rule_monitor.evaluate_initially()
        self.rule_monitor.world_state.time_step = start_time_step
        update_ego_vehicle(self.world_state.road_network,
                           self.world_state.ego_vehicle,
                           updated_states,
                           start_time_step,
                           self.dT)
        self.rule_monitor.evaluate_consecutively()
        evaluated_robustness, evaluated_ids = self.rule_monitor.query_rule_rob_all()
        if evaluated_robustness[0] < 0:
            return -math.inf, evaluated_ids[0][0]  # all violated
        tv = np.argmax(evaluated_robustness < 0)
        if tv == 0:
            return math.inf, None  # no violation
        return tv * self.dT, evaluated_ids[tv][0]

    def generate(self, cut_off_maneuvers: List[CutOffAction]):
        """
        Computes the Time-to-Compliance (with traffic rules).
        :param cut_off_maneuvers: the given maneuvers of ego vehicle
        :return: TC, corresponding maneuver
        """
        if self._tv == -math.inf:
            raise ValueError("<TC>: the trajectory is not repairable since it already disobeys the rules")
        elif self._tv == math.inf:
            self._tc = math.inf
        else:
            ttm = dict()
            for maneuver in cut_off_maneuvers:
                ttm[maneuver] = self.search_ttm(maneuver)
            self._tc = max(ttm.values())
            self._compliant_maneuver = max(ttm, key=ttm.get)
        return self._tc

    def search_ttm(self, maneuver: CutOffAction):
        ttm = - math.inf
        low = 0
        high = int(self._tv / self.dT)
        while low < high:
            mid = int((low + high)/2)
            if maneuver in [CutOffAction.BRAKE, CutOffAction.KICKDOWN, CutOffAction.STEADYSPEED]:
                SL = SimulationLong(maneuver,
                                    self.ego_vehicle,
                                    mid)
            elif maneuver in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT]:
                SL = SimulationLateral(maneuver,
                                       self.ego_vehicle,
                                       mid,
                                       self.world_state)
                self._simulation_lateral = SL
            else:
                raise ValueError("<TTCC>: given compliant maneuver {} is not supported".format(maneuver))

            state_list = SL.simulate_state_list()

            if self._visualize:
                visualize_state_list(self._collision_checker, state_list, self.scenario,
                                         SL.vehicle_dynamics.shape)
            flag_collision = self._detect_collision(state_list)  # bool value
            tv, _ = self._calc_tv_updated(state_list, mid) # which should be tv instead of ttm
            # if violation-free and collision-free
            if tv == math.inf and not flag_collision:
                low = mid + 1
            else:
                high = mid

        if low != 0:
            ttm = (low - 1) * self.dT
        return ttm




