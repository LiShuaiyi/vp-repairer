import functools
from typing import Dict, Tuple, Union

from commonroad.scenario.obstacle import ObstacleType
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getObstacleState, getRelevantObstacles, createObstacleShape
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class KeepDistanceToBicycles(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, ego_shapes_map: Dict,
                 obst_ref_path_projections: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("KEEP_DISTANCE_TO_BICYCLES", "The ego shall keep a safety distance towards bicycles.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.ego_shapes_map = ego_shapes_map
        self.obst_ref_path_projections = obst_ref_path_projections

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('keep_distance_to_bicycles')
        self.safety_distance = self.params.get('safety_distance')

        self.obstacle_types = {ObstacleType.BICYCLE}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]

        for obstacle in self.relevant_obstacles:
            is_passing_and_is_too_close, obst_state = self.checkEgoObstaclePassingAndCloseness((state2.t, state1.s, state2.s, obstacle))
            if is_passing_and_is_too_close:
                instant_violations.append(max(0.0, (state2.v - obst_state.velocity)))
        return max(instant_violations), 0.0, None

    @functools.lru_cache(maxsize=None)
    def checkEgoObstaclePassingAndCloseness(self, key):
        time, station_min, station_max, obstacle = key[0], key[1], key[2], key[3]

        time_step = self.time_step_map[time]
        s_d_obstacle = self.obst_ref_path_projections[(time_step, obstacle.obstacle_id)]

        # If ego and obstacle are passing each other
        if s_d_obstacle is not None and station_min <= s_d_obstacle[0] <= station_max:
            obst_state = getObstacleState(obstacle, time_step)
            obst_shape = createObstacleShape(obstacle, obst_state)
            distance_ego_object = abs(self.ego_shapes_map[station_max].shapely_object.distance(obst_shape.shapely_object))
            if distance_ego_object < self.safety_distance:
                return True, obst_state
        return False, None
