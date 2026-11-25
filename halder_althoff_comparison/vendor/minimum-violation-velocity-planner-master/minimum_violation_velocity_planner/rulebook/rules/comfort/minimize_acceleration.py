from typing import Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class MinimizeAcceleration(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem):
        """It's better to drive with constant speed then to de- or accelerate."""
        super().__init__("MINIMIZE_ACCELERATION", "The ego shall minimize the longitudinal acceleration.")
        self.scenario = scenario
        self.planning_problem = planning_problem

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        return state2.a ** 2, 0.0, None
