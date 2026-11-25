import functools
from typing import Dict, Tuple, Union, List

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import violationIntegrationLinearInterpolation, violationIntegrationMaxValue, \
    violationIntegrationCurrentValue, globalHeuristic
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState



class StopAfterCrash(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, std_ego_coll_boxes: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict, integration_strategy: int, dt: float, time_horizon: List):
        super().__init__("STOP_AFTER_CRASH", "The ego shall remain stopped after a crash.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.std_ego_coll_boxes = std_ego_coll_boxes
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.vehicle_parameters = vehicle_parameters
        self.params = params

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
        if transfer_value == 1 or self.crash(state2):
            v_max_violation = abs(state2.v) - 0.0
            heuristic = globalHeuristic(v_max_violation, state2.t, self.time_horizon, self.dt, self.min_acceleration, self._integration_fct)
            return v_max_violation, heuristic, 1
        return 0.0, 0.0, 0

    def crash(self, state: LongitudinalState) -> bool:
        for obstacle in self.scenario.obstacles:
            ego_obstacle_collision = self.checkEgoObstacleCollision((state.t, state.s, obstacle))
            # If ego and obstacle collide
            if ego_obstacle_collision:
                return True
        return False

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle = key[0], key[1], key[2]
        time_step = self.time_step_map[time]
        obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle.obstacle_id))
        ego_bb = self.std_ego_coll_boxes.get(station)
        if obst_bb and ego_bb.collide(obst_bb):
            return True
        return False
