import functools
from typing import Dict, Tuple, Union

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getObstacleState, createObstacleShape
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad.scenario.scenario import Scenario
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class NarrowPassing(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, ego_shapes_map: Dict,
                 obst_ref_path_projections: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("NARROW_PASSING", "The ego shall drive slower, the closer it is to an obstacle to its left"
                                           " or right.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.ego_shapes_map = ego_shapes_map
        self.obst_ref_path_projections = obst_ref_path_projections

        self.ego_params = vehicle_parameters
        self.params = params.get('safety').get('narrow_passing')
        self.distance_threshold = self.params.get('distance_threshold')
        self.v_max_at_distance_threshold = self.params.get('v_max_at_distance_threshold')
        self.distance_threshold = self.params.get('distance_threshold')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]

        for obstacle in self.scenario.obstacles:
            is_passing_and_is_too_close, distance_ego_object = self.checkEgoObstaclePassingAndCloseness((state2.t, state1.s, state2.s, obstacle))
            if is_passing_and_is_too_close:
                v_max = self.v_max_at_distance_threshold / self.distance_threshold * distance_ego_object
                instant_violations.append(max(0.0, (state2.v - v_max)))
        return max(instant_violations), 0.0, None

    @functools.lru_cache(maxsize=None)
    def checkEgoObstaclePassingAndCloseness(self, key):
        time, station_min, station_max, obstacle = key[0], key[1], key[2], key[3]

        time_step = self.time_step_map[time]

        s_d_obstacle = self.obst_ref_path_projections[(time_step, obstacle.obstacle_id)]

        # If ego and obstacle are passing each other
        if s_d_obstacle is not None and station_min <= s_d_obstacle[0] <= station_max:
            obst_shape = createObstacleShape(obstacle, getObstacleState(obstacle, time_step))
            distance_ego_object = abs(self.ego_shapes_map[station_max].shapely_object.distance(obst_shape.shapely_object))
            if distance_ego_object < self.distance_threshold:
                return True, distance_ego_object
        return False, None
