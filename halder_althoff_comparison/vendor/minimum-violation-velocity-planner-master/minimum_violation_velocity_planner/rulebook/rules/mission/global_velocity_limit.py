from typing import Dict, Tuple, Union, List

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import violationIntegrationLinearInterpolation, violationIntegrationMaxValue, \
    violationIntegrationCurrentValue, globalHeuristic
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class GlobalVelocityLimit(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, vehicle_parameters: VehicleParameters,
                 params: Dict, integration_strategy: int, dt: float, time_horizon: List):
        super().__init__("GLOBAL_VELOCITY_LIMIT", "The egos velocity shall be under the global velocity limit.", is_path_independent=True)

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.vehicle_parameters = vehicle_parameters
        self.params = params.get('mission').get('global_velocity_limit')
        self.v_max = self.params.get('v_max')

        # TODO: Get this directly from vehicle model
        self.min_acceleration = -self.vehicle_parameters.longitudinal.a_max - params.get('vehicle_model').get('a_min_extension')

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
        v_max_violation = max(0.0, state2.v - self.v_max)
        heuristic = globalHeuristic(v_max_violation, state2.t, self.time_horizon, self.dt, self.min_acceleration,
                                    self._integration_fct)
        return v_max_violation, heuristic, None
