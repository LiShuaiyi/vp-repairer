from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class AbruptBraking(Rule):
    def __init__(self, params: Dict):
        super().__init__("ABRUPT_BRAKING", "The ego shall not break abruptly without reason.")

        self.params = params.get('traffic_rules').get('abrupt_braking')
        self.delta_a = self.params.get('delta_a')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        return max(0.0, -(state2.a - self.delta_a)), 0.0, None
