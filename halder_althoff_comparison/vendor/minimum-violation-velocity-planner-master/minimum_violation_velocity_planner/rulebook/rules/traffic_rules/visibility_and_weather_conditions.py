from typing import Dict, Union, Tuple, List
from vehiclemodels.vehicle_parameters import VehicleParameters
from minimum_violation_velocity_planner.common.utils import globalHeuristic, violationIntegrationLinearInterpolation, \
    violationIntegrationMaxValue, violationIntegrationCurrentValue
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario, TimeOfDay, Weather
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class VisibilityAndWeatherConditions(Rule):
    def __init__(self, scenario: Scenario, vehicle_parameters: VehicleParameters, params: Dict, integration_strategy: int, dt: float, time_horizon: List):
        super().__init__("VISIBILITY_AND_WEATHER_CONDITIONS", "The ego shall adapt its velocity according to the visibility and weather conditions.", is_path_independent=True)

        self.environment = scenario.location.environment

        self.params_time_of_day = params.get('traffic_rules').get('visibility_and_weather_conditions').get('time_of_day')
        self.v_max_day = self.params_time_of_day.get('v_max_day')
        self.v_max_night = self.params_time_of_day.get('v_max_night')

        self.params_weather = params.get('traffic_rules').get('visibility_and_weather_conditions').get('weather')
        self.v_max_sunny = self.params_weather.get('v_max_sunny')
        self.v_max_light_rain = self.params_weather.get('v_max_light_rain')
        self.v_max_heavy_rain = self.params_weather.get('v_max_heavy_rain')
        self.v_max_fog = self.params_weather.get('v_max_fog')
        self.v_max_snow = self.params_weather.get('v_max_snow')
        self.v_max_hail = self.params_weather.get('v_max_hail')

        # TODO: Get this directly from vehicle model
        self.min_acceleration = -vehicle_parameters.longitudinal.a_max - params.get('vehicle_model').get('a_min_extension')

        self.v_max = None
        if self.environment:
            self.v_max = self.determineMaxVelocity()

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
            heuristic = globalHeuristic(v_max_violation, state2.t, self.time_horizon, self.dt, self.min_acceleration, self._integration_fct)
            return v_max_violation, heuristic, None

    def determineMaxVelocity(self) -> Union[None, float]:
        active_v_max_constraints = []

        # Time of Day
        if self.environment.time_of_day is TimeOfDay.DAY:
            active_v_max_constraints.append(self.v_max_day)
        elif self.environment.time_of_day is TimeOfDay.NIGHT:
            active_v_max_constraints.append(self.v_max_night)

        # Weather
        if self.environment.weather is Weather.SUNNY:
            active_v_max_constraints.append(self.v_max_sunny)
        elif self.environment.weather is Weather.LIGHT_RAIN:
            active_v_max_constraints.append(self.v_max_light_rain)
        elif self.environment.weather is Weather.HEAVY_RAIN:
            active_v_max_constraints.append(self.v_max_heavy_rain)
        elif self.environment.weather is Weather.FOG:
            active_v_max_constraints.append(self.v_max_fog)
        elif self.environment.weather is Weather.SNOW:
            active_v_max_constraints.append(self.v_max_snow)
        elif self.environment.weather is Weather.HAIL:
            active_v_max_constraints.append(self.v_max_hail)

        if len(active_v_max_constraints) == 0:
            return None
        else:
            return min(active_v_max_constraints)
