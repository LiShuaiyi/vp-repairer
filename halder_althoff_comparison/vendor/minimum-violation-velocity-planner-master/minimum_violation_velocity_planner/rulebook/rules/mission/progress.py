from typing import Dict

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import determineTimeHorizon
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Progress(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                 vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("PROGRESS", "The ego shall increase its progress towards its goal.")
        self.scenario = scenario
        self.planning_problem = planning_problem

        self.a_max = vehicle_parameters.longitudinal.a_max + params.get('vehicle_model').get('a_max_extension')
        self.v_0 = planning_problem.initial_state.velocity
        self.s_0 = ccosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0],
                                                       planning_problem.initial_state.position[1])[0]

        t_rounding_prec = params.get('general').get('time_rounding_precision')
        self.dt = round(params.get('general').get('grid_delta_t'), t_rounding_prec)
        self.time_horizon = determineTimeHorizon(scenario, params.get('general').get('max_time_horizon'),
                                                  self.dt, t_rounding_prec)

        self.n_t = int(self.time_horizon / self.dt) + 1

        self.s_max_local = self.determineLocalSMaxValues()

    def determineTotalViolation(self, state1: LongitudinalState, state2: LongitudinalState, score: float) -> float:
        return self.determineInstanceViolation(state1, state2)

    def determineInstanceViolation(self, state1: LongitudinalState, state2: LongitudinalState, *args) -> float:
        # We only have one instance
        return self.determineInstantaneousViolation(state1, state2)

    def determineInstantaneousViolation(self, state1: LongitudinalState, state2: LongitudinalState, *args) -> float:
        # Violation Metric
        # Progress: Ratio of driven distance to possible driven distance until this time step
        progress = state2.s / self.s_max_local[round(state2.t, 9)]
        return 1.0 - progress

    def determineLocalSMaxValues(self) -> Dict:
        s_max_values = {}
        for time_step in range(self.n_t + 1):
            t = time_step * self.dt
            s_max_values[round(t, 9)] = 1 / 2 * self.a_max * t ** 2 + self.v_0 * t + self.s_0
        return s_max_values
