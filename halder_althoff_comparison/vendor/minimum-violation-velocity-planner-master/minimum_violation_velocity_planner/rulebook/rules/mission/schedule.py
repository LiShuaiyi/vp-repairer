from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Schedule(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, params: Dict):
        super().__init__("SCHEDULE", "The ego shall be in a certain position-interval in a certain time-interval.")
        self.scenario = scenario
        self.planning_problem = planning_problem

        self.params = params.get('mission').get('schedule')
        self.t_min = self.params.get('t_min')
        self.t_max = self.params.get('t_max')
        self.s_min = self.params.get('s_min')
        self.s_max = self.params.get('s_max')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=0.0) -> Tuple[float, float, Union[None, bool, int, float]]:
        station_violation = 0.0
        t, s = state2.t, state2.s
        if self.t_min <= t <= self.t_max and not (self.s_min <= s <= self.s_max):
            station_violation = max(self.s_min - s, s - self.s_max)

        if self.t_max == t:
            return max(0.0, station_violation, transfer_value), 0.0, 0.0
        else:
            return 0.0, 0.0, max(0.0, station_violation, transfer_value)
