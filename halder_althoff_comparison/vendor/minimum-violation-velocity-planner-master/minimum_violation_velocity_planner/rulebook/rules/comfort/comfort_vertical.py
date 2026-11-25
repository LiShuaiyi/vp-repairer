from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.common.utils import getIntervalOverlap
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class ComfortVertical(Rule):
    def __init__(self, params: Dict):
        super().__init__("COMFORT_VERTICAL", "The ego shall not exceed the vertical comfort limit.")

        self.params = params.get('comfort').get('vertical_comfort')
        self.active = self.params.get('active')
        self.s_bump_start = self.params.get('s_bump_start')
        self.s_bump_end = self.params.get('s_bump_end')
        self.h_min = self.params.get('h_min')
        self.h_max = self.params.get('h_max')
        self.height_or_depth = self.params.get('height_or_depth')
        self.v_max_h_min = self.params.get('v_max_h_min')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        if self.active:
            if getIntervalOverlap((self.s_bump_start, self.s_bump_end), (state1.s, state2.s)) > 0.0:
                if self.h_min < self.height_or_depth < self.h_max:
                    v_max = -self.v_max_h_min / (self.h_max - self.h_min) * (self.height_or_depth - self.h_min) + self.v_max_h_min
                    return max(0.0, state2.v - v_max), 0.0, None
        return 0.0, 0.0, None
