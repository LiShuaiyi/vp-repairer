from typing import Dict, Union, Tuple, List

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import globalHeuristic, violationIntegrationLinearInterpolation, \
    violationIntegrationMaxValue, violationIntegrationCurrentValue
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario, Underground

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class RoadConditions(Rule):
    def __init__(self, scenario: Scenario, vehicle_parameters: VehicleParameters, params: Dict, integration_strategy: int, dt: float, time_horizon: List):
        super().__init__("ROAD_CONDITIONS", "The ego shall adapt its velocity according to the road conditions.", is_path_independent=True)

        self.environment = scenario.location.environment

        self.params = params.get('traffic_rules').get('road_conditions')
        self.v_max_clean = self.params.get('v_max_clean')
        self.v_max_wet = self.params.get('v_max_wet')
        self.v_max_dirty = self.params.get('v_max_dirty')
        self.v_max_damaged = self.params.get('v_max_damaged')
        self.v_max_snow = self.params.get('v_max_snow')
        self.v_max_ice = self.params.get('v_max_ice')

        self.v_max = None
        if self.environment:
            self.v_max = self.determineMaxVelocity()

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
        if self.v_max is None:
            return 0.0,  0.0, None
        else:
            v_max_violation = max(0.0, state2.v - self.v_max)
            heuristic = globalHeuristic(v_max_violation, state2.t, self.time_horizon, self.dt, self.min_acceleration,
                                        self._integration_fct)
            return v_max_violation, heuristic, None

    def determineMaxVelocity(self) -> Union[None, float]:
        active_v_max_constraints = []

        # Road Conditions
        if self.environment.underground is Underground.CLEAN:
            active_v_max_constraints.append(self.v_max_clean)
        elif self.environment.underground is Underground.WET:
            active_v_max_constraints.append(self.v_max_wet)
        elif self.environment.underground is Underground.DIRTY:
            active_v_max_constraints.append(self.v_max_dirty)
        elif self.environment.underground is Underground.DAMAGED:
            active_v_max_constraints.append(self.v_max_damaged)
        elif self.environment.underground is Underground.SNOW:
            active_v_max_constraints.append(self.v_max_snow)
        elif self.environment.underground is Underground.ICE:
            active_v_max_constraints.append(self.v_max_ice)

        if len(active_v_max_constraints) == 0:
            return None
        else:
            return min(active_v_max_constraints)
