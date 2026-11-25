from typing import Dict, Tuple, Union

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class ComfortAccelerationLimitLong(Rule):
    def __init__(self, params: Dict):
        super().__init__("COMFORT_ACCELERATION_LONG", "The ego shall not exceed the longitudinal comfort acceleration limits.")

        self.params = params.get('comfort').get('acceleration_longitudinal')
        self.a_max = self.params.get('a_max')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        a_max_violation = abs(state2.a) - self.a_max
        return max(0.0, a_max_violation), 0.0, None
