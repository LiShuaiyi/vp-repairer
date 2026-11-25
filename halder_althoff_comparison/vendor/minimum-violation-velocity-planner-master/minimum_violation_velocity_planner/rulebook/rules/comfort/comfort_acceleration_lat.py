from typing import Dict, Tuple, Union

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from minimum_violation_velocity_planner.common.utils import getCurvatureAtStation
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class ComfortAccelerationLimitLat(Rule):
    def __init__(self, ccosy: CurvilinearCoordinateSystem, params: Dict):
        super().__init__("COMFORT_ACCELERATION_LAT", "The ego shall not exceed the lateral comfort acceleration limits.")

        self.ccosy = ccosy

        self.params = params.get('comfort').get('acceleration_lateral')
        self.a_max = self.params.get('a_max')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        a_lat = abs(state2.v ** 2 * getCurvatureAtStation(self.ccosy, state2.s))

        # Approximate the area under the a-t curve, which is over a_max/ below a_min
        a_max_violation = a_lat - self.a_max

        return max(0.0, a_max_violation), 0.0, None
