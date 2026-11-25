from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class BusStop(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, params: Dict):
        super().__init__("BUS_STOP", "The ego shall stop at bus stops.")
        self.scenario = scenario
        self.planning_problem = planning_problem

        self.params = params.get('mission').get('bus_stop')
        self.s_min = self.params.get('s_min')
        self.s_max = self.params.get('s_max')
        self.v_max = self.params.get('v_max')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # Violation Metric TODO: Does this metric makes sense? Also take interval overlap ? Or simply stop at position!!! (v_max at station value), Or maybe similar to stop line
        # Maybe take goal area as bus stop ??
        if self.s_min < state2.s < self.s_max:
            # Determine the level of ego velocity, which is over v_max multiplied with the traveled distance
            return max(0.0, (state2.v - self.v_max) * abs(state2.s - state1.s)), 0.0, None
        return 0.0, 0.0, None
