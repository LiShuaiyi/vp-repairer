from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class GlobalAccelerationLimit(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, params: Dict):
        super().__init__("GLOBAL_ACCELERATION_LIMIT", "The ego shall not exceed the global acceleration limits.")

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.params = params.get('mission').get('global_acceleration_limit')
        self.a_min = self.params.get('a_min')
        self.a_max = self.params.get('a_max')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:

        a_max_violation = abs(state2.a) - self.a_max

        return max(0.0, a_max_violation), 0.0, None
