from typing import Iterable, Union, List
import math
from abc import ABC

import numpy as np
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world_state import WorldState

from cut_off.base import CutOffBase
from encoding.monitor import RuleMonitor
from cut_off.utils import update_ego_vehicle, visualize_state_list
from cut_off.simulation import CutOffAction, SimulationLateral, SimulationLong


class TTCC(CutOffBase, ABC):
    """
    Time-To-Compliance.
    """
    def __init__(self,
                 rule_monitor: RuleMonitor,
                 dT: float = 0.1):
        super().__init__(rule_monitor.world_state, dT)
        self.rule_monitor = rule_monitor
        self._check_initial = True
        self._ttv = self._calc_ttv()
        self._visualize = False

    @property
    def ttv(self) -> float:
        return self._ttv

    def _calc_ttv(self, updated_states: List[State] = None,
                 start_time_step: int = None) -> float:
        # detect violation time using STL monitor
        # self.rule_monitor.evaluate_initially()
        if self._check_initial:
            self._check_initial = False
        else:
            self.rule_monitor.world_state.time_step = start_time_step
            update_ego_vehicle(self.world_state.road_network,
                               self.world_state.ego_vehicle,
                               updated_states,
                               start_time_step,
                               self.dT)
            self.rule_monitor.evaluate_consecutively()
        evaluated_robustness = self.rule_monitor.query_rule_rob_all()
        if evaluated_robustness[0] < 0:
            return -math.inf  # all violated
        ttv = np.argmax(evaluated_robustness < 0)
        if ttv == 0:
            return math.inf  # no violation
        return ttv * self.dT

    def generate(self, maneuver: CutOffAction, ):
        """
        Generates the TTCC regarding traffic rule violations.
        """
        ttcc = - math.inf
        low = 0
        if self._ttv == math.inf:
            return math.inf
        high = int(self._ttv / self.dT)
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
            else:
                raise ValueError("<TTCC>: given compliant maneuver {} is not supported".format(maneuver))
            state_list = SL.simulate_state_list()

            if self._visualize:
                visualize_state_list(state_list, self.scenario, SL.vehicle_dynamics.shape)

            flag_collision = self._detect_collision(state_list)  # bool value
            ttv = self._calc_ttv(state_list, mid)
            # if violation-free and collision-free
            if ttv == math.inf and not flag_collision:
                low = mid + 1
            else:
                high = mid

        if low != 0:
            ttcc = (low - 1) * self.dT
        return ttcc





