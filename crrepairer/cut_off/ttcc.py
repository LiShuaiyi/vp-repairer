from typing import Iterable, Union, List
import math

import numpy as np
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world_state import WorldState
from cut_off.base import CutOffBase
from cut_off.monitor import RuleMonitor
from cut_off.utils import update_ego_vehicle
from cut_off.simulation import CutOffAction, SimulationLateral, SimulationLong


class TTCC(CutOffBase):
    """
    Time-To-Compliance.
    """
    def __init__(self, world_state: WorldState,
                 scenario: Scenario,
                 ego_vehicle_cr: DynamicObstacle,
                 rules: Union[str, Iterable[str]],
                 dT: float = 0.1):
        super().__init__(world_state, scenario, ego_vehicle_cr, dT)
        self._rule_monitor = RuleMonitor(world_state, rules)
        self._check_initial = True
        self._ttv = self._calc_ttv()

    @property
    def rule_monitor(self) -> RuleMonitor:
        return self._rule_monitor

    @property
    def ttv(self) -> float:
        return self._ttv

    def _calc_ttv(self, updated_states: List[State] = None,
                 start_time_step: int = None) -> float:
        # detect violation time using STL monitor
        if self._check_initial:
            self.rule_monitor.evaluate_initially()
            self._check_initial = False
        else:
            self.rule_monitor.world_state.time_step = start_time_step
            update_ego_vehicle(self.world_state,
                               updated_states,
                               start_time_step)
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
            ttc = self._calc_ttc(state_list)
            ttv = self._calc_ttv(state_list, mid)
            if ttv == math.inf and ttc == math.inf:
                low = mid + 1
            else:
                high = mid
        if low != 0:
            ttcc = (low - 1) * self.dT
        return ttcc





