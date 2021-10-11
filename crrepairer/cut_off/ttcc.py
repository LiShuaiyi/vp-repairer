from typing import Iterable, Union, List
import math

import numpy as np
from commonroad.scenario.obstacle import State
from crmonitor.common.world_state import WorldState
from cut_off.base import CutOffBase
from cut_off.monitor import RuleMonitor
from cut_off.utils import update_ego_vehicle


class TTCC(CutOffBase):
    """
    Time-To-Compliance.
    """
    def __init__(self, world_state: WorldState,
                 rules: Union[str, Iterable[str]],
                 dT: float = 0.1):
        super().__init__(world_state, dT)
        self._rule_monitor = RuleMonitor(world_state, rules)
        self._check_initial = False

    @property
    def rule_monitor(self):
        return self._rule_monitor

    def calc_ttv(self):
        """
        Time-To-Violation.
        """
        # check the initially-planned trajectory
        self._check_initial = True
        return self._robustness_violation_time()

    def _robustness_violation_time(self, updated_states: List[State]=None,
                                   start_time_step: int = None):
        # detect violation time using STL monitor
        if self._check_initial:
            self.rule_monitor.evaluate_initially()
        else:
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

    def generate(self):
        pass




