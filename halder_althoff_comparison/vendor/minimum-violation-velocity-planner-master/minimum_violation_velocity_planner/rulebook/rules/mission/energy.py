from math import cos, sin
from typing import Dict, Tuple, Union

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Energy(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("ENERGY", "The ego shall consume as little energy as possible.")
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.mass = vehicle_parameters.m

        self.params = params.get('mission').get('energy')
        self.c_w = self.params.get('c_w')
        self.A = self.params.get('A')
        self.rho = self.params.get('rho')
        self.f_roll = self.params.get('f_roll')
        self.alpha = self.params.get('alpha')

        self.p_max = self.params.get('max_power') * 1000

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        velocity = abs(state2.v)

        # Air resistance
        F_air = self.c_w * self.A * self.rho/2 * velocity ** 2  #TODO Parts can be made static in cosntructor!

        # TODO: Is this always positive?
        # Rolling resistance
        F_roll = self.f_roll * self.mass * 9.81 * cos(self.alpha)  # TODO Can be made static in constuctor

        # Slope resistance
        F_slope = self.mass * 9.81 * sin(self.alpha)

        # Acceleration resistance
        if velocity >= 0.0:         # Forward driving
            F_acc = self.mass * max(0.0, state2.a)
        else:                       # Backwards driving
            F_acc = self.mass * abs(min(0.0, state2.a))

        # Required power
        power_required = (F_air + F_roll + F_slope + F_acc) * velocity
        if power_required < 0.0:
            raise Warning("Required power is negative. Negative weights are NOT allowed!")

        p_violation = max(0.0, power_required - self.p_max)

        return p_violation, 0.0, None
