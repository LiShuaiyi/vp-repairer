import functools
import math
from typing import Dict, Tuple, Union

from commonroad.geometry.shape import Shape
from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getObstacleState, createObstacleShape, \
    getOrientationAtStation, getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class DrivingFasterThanLeftTraffic(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict,
                 ego_lane_station_map: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("DRIVING_FASTER_THAN_LEFT_TRAFFIC",
                         "The ego shall not drive faster than vehicles on left lanes on highways.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy

        self.time_step_map = time_step_map
        self.ego_lane_station_map = ego_lane_station_map

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('driving_faster_than_left_traffic')
        self.obstacle_consideration_radius = self.params.get('obstacle_consideration_radius')
        self.corridor_width = self.params.get("corridor_width")

        self.obstacle_types = {ObstacleType.CAR,
                               ObstacleType.TRUCK,
                               ObstacleType.BUS,
                               ObstacleType.PRIORITY_VEHICLE,
                               ObstacleType.MOTORCYCLE,
                               ObstacleType.TAXI}

        self.lanelet_types = {LaneletType.HIGHWAY,
                              LaneletType.INTERSTATE}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        if self.isEgoOnHighway(state2.s):
            instant_violations = [0.0]
            for obstacle in self.relevant_obstacles:
                obst_is_relevant, obstacle_state = self.checkObstacleRelevantAndLeftOfEgo((state2.t, state2.s, obstacle))
                if obst_is_relevant:
                    instant_violations.append(max(0.0, (state2.v - obstacle_state.velocity)))
            return max(instant_violations), 0.0, None
        return 0.0, 0.0, None

    def euclideanDistance(self, ego_station: float, obst_state: State) -> float:
        ego_pos = self.ccosy.convert_to_cartesian_coords(ego_station, 0.0)
        return math.sqrt((ego_pos[0] + obst_state.position[0]) ** 2 + (ego_pos[1] + obst_state.position[1]) ** 2)

    def deltaOrientation(self, ego_station: float, obs_orientation: float) -> float:
        ego_orientation = getOrientationAtStation(self.ccosy, ego_station)
        return math.pi - abs(abs(ego_orientation - obs_orientation) - math.pi)

    def obstacleIsLeftOfEgoCorridor(self, obst_shape: Shape) -> bool:
        obst_vertices = obst_shape.vertices
        obstacle_ccosy_points = []
        for point in obst_vertices:
            if self.ccosy.cartesian_point_inside_projection_domain(point[0], point[1]):
                obstacle_ccosy_points.append(self.ccosy.convert_to_curvilinear_coords(point[0], point[1]).tolist())
            else:
                return False

        obst_min_d = min([x[1] for x in obstacle_ccosy_points])

        # If obstacle does not intersect with corridor and is left of it
        if obst_min_d > (self.corridor_width / 2):
            return True
        else:
            return False

    @functools.lru_cache(maxsize=None)
    def isEgoOnHighway(self, station):
        for lanelet_id in self.ego_lane_station_map[station]:
            # If one of the lanelets is of the respective types
            lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            nof_same_lanelet_types = len(self.lanelet_types.intersection(lanelet.lanelet_type))
            if nof_same_lanelet_types > 0:
                return True
        return False

    @functools.lru_cache(maxsize=None)
    def checkObstacleRelevantAndLeftOfEgo(self, key):
        time, station, obstacle = key[0], key[1], key[2]
        time_step = self.time_step_map[time]
        obstacle_state = getObstacleState(obstacle, time_step)

        # If obstacle is in consideration range and headings are similar (= same driving direction)
        if obstacle_state and self.euclideanDistance(station, obstacle_state) < self.obstacle_consideration_radius and \
                abs(self.deltaOrientation(station, obstacle_state.orientation)) < (math.pi / 2):
            obst_shape = createObstacleShape(obstacle, obstacle_state)
            if self.obstacleIsLeftOfEgoCorridor(obst_shape):
                return True, obstacle_state
        return False, None
