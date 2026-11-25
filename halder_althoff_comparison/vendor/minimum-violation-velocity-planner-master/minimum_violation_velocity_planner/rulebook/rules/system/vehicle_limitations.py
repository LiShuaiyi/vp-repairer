from math import sqrt
from typing import Tuple, Union

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getCurvatureAtStation
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule


from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class VehicleLimitations(Rule):
    def __init__(self, ccosy: CurvilinearCoordinateSystem, vehicle_parameters: VehicleParameters):
        super().__init__("VEHICLE_LIMITATIONS", "The ego shall not exceed its physical limitations.", is_path_independent=True)

        self.ccosy = ccosy
        self.vehicle_parameters = vehicle_parameters

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # Calculate a_lat
        a_lat = state2.v ** 2 * abs(getCurvatureAtStation(self.ccosy, state2.s))
        a_current = sqrt(a_lat**2 + state2.a**2)

        if not (self.vehicle_parameters.longitudinal.v_min <= state2.v <= self.vehicle_parameters.longitudinal.v_max) or \
                not(a_current <= self.vehicle_parameters.longitudinal.a_max):
            return 1.0, 0.0, None

        return 0.0, 0.0, None
