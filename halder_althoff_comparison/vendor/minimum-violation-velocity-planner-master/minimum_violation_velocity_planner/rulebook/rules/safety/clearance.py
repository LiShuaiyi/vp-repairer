from typing import Dict

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getTimeStepFromTimeValue, createEgoCollisionObject, \
    createObstacleCollisionObject, getObstacleState
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class Clearance(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, vehicle_parameters: VehicleParameters,
                 params: Dict):
        super().__init__("CLEARANCE", "The ego shall keep a static safety distance to all other objects.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.ego_params = vehicle_parameters
        self.params = params.get('safety').get('clearance')

    def determineTotalViolation(self, state1: LongitudinalState, state2: LongitudinalState,
                                score: float) -> float:

        return score + self.determineInstanceViolation(state1, state2)

    def determineInstanceViolation(self, state1: LongitudinalState, state2: LongitudinalState, *args) -> float:
        instant_violations = []

        ego_bb = createEgoCollisionObject(state2.s, self.ccosy, self.ego_params,
                                          extension_minkovski=self.params.get('clearance_distance'))

        for obstacle in self.scenario.obstacles:
            time_step = getTimeStepFromTimeValue(state2.t, self.scenario)
            obstacle_state = getObstacleState(obstacle, time_step)
            if obstacle_state is not None:
                obst_bb = createObstacleCollisionObject(obstacle, obstacle_state)
                instant_violations.append(self.determineInstantaneousViolation(state1, state2, ego_bb, obst_bb))
        return max(instant_violations) if len(instant_violations) > 0 else 0.0

    def determineInstantaneousViolation(self, state1: LongitudinalState, state2: LongitudinalState, *args) -> float:
        ego_bb, obst_bb = args[0], args[1]

        # Collision Check
        if ego_bb.collide(obst_bb):
            return 1.0
        return 0.0
