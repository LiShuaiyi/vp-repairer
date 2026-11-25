from typing import Dict, Tuple, Union, List

from commonroad.scenario.scenario import Scenario
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import violationIntegrationLinearInterpolation, violationIntegrationMaxValue, \
    violationIntegrationCurrentValue, globalHeuristic
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Degradation(Rule):
    def __init__(self, vehicle_parameters: VehicleParameters, params: Dict, integration_strategy: int, dt: float, time_horizon: List):
        super().__init__("DEGRADATION", "The ego shall slow down if internal errors occur.", is_path_independent=True)

        self.params = params.get('system').get('degradation')
        self.active_level = self.params.get('active_level')
        self.activation_time = self.params.get('activation_time')
        self.v_max_level_1 = self.params.get('v_max_level_1')
        self.v_max_level_2 = self.params.get('v_max_level_2')
        self.v_max_level_3 = self.params.get('v_max_level_3')

        if not (0 <= self.active_level <= 3):
            raise ValueError("Degradation level of " + str(self.active_level) + " is not supported!")

        # TODO: Get this directly from vehicle model
        self.min_acceleration = -vehicle_parameters.longitudinal.a_max - params.get('vehicle_model').get('a_min_extension')

        self.dt = dt
        self.time_horizon = time_horizon[1]

        # Set the integration function
        if integration_strategy == 0:
            self._integration_fct = violationIntegrationLinearInterpolation
        elif integration_strategy == 1:
            self._integration_fct = violationIntegrationMaxValue
        else:
            self._integration_fct = violationIntegrationCurrentValue

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # Case 0: No internal error or activation time not yet reached
        if self.active_level == 0 or state2.t < self.activation_time:
            return 0.0, 0.0, None
        # Case 1: Minor internal error
        elif self.active_level == 1:
            v_max = self.v_max_level_1
        # Case 2: Major internal error
        elif self.active_level == 2:
            v_max = self.v_max_level_2
        # Case 3: Super Gau
        else:
            v_max= self.v_max_level_3

        v_max_violation = max(0.0, state2.v - v_max)
        heuristic = globalHeuristic(v_max_violation, state2.t, self.time_horizon, self.dt, self.min_acceleration,
                                    self._integration_fct)
        return v_max_violation, heuristic, None
