import functools
from typing import Dict, Tuple, Union

from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.trajectory import State
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from minimum_violation_velocity_planner.common.utils import getObstacleState, getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from vehiclemodels.vehicle_parameters import VehicleParameters


class StopAtCrosswalk(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, ego_coll_boxes_crosswalk: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("STOP_AT_CROSSWALK", "The ego shall let pedestrians cross at crosswalks.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.ego_coll_boxes_crosswalk = ego_coll_boxes_crosswalk
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('stop_at_crosswalk')
        self.clearance_radius = self.params.get('clearance_radius')

        self.obstacle_types = {ObstacleType.PEDESTRIAN}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]

        for obstacle in self.relevant_obstacles:
            ego_obstacle_collision = self.checkEgoObstacleCollision((state2.t, state2.s, obstacle))
            # If ego and obstacle collide
            if ego_obstacle_collision:
                instant_violations.append(1.0)
        return max(instant_violations), 0.0, None

    def isOnCrosswalk(self, state: State) -> bool:
        lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([state.position])[0]

        for lanelet_id in lanelet_ids:
            if LaneletType.CROSSWALK in self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id).lanelet_type:
                return True
        return False

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle = key[0], key[1], key[2]

        time_step = self.time_step_map[time]
        obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle.obstacle_id))
        ego_bb = self.ego_coll_boxes_crosswalk.get(station)
        if obst_bb and ego_bb.collide(obst_bb):
            obstacle_state = getObstacleState(obstacle, time_step)
            if self.isOnCrosswalk(obstacle_state):
                return True
        return False
