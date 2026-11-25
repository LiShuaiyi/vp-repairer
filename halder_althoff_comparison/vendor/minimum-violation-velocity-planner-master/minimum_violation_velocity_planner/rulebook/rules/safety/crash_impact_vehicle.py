import functools
from typing import Dict, Tuple, Union

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getProjectedObstacleVelocity, getObstacleState, getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class CrashImpactVehicle(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, std_ego_coll_boxes: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("CRASH_IMPACT_VEHICLE", "The ego shall not collide with fast and heavy vehicles.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.std_ego_coll_boxes = std_ego_coll_boxes
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.ego_params = vehicle_parameters
        self.params = params.get('safety').get('crash_impact')
        self.area_mass_ratio = self.params.get('area_mass_ratio')

        self.obstacle_types = {ObstacleType.CAR,
                               ObstacleType.TRUCK,
                               ObstacleType.BUS,
                               ObstacleType.PRIORITY_VEHICLE,
                               ObstacleType.TRAIN,
                               ObstacleType.MOTORCYCLE,
                               ObstacleType.TAXI}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]

        for obstacle in self.relevant_obstacles:
            ego_obstacle_collision, obstacle_state = self.checkEgoObstacleCollision((state2.t, state2.s, obstacle))

            # If ego and obstacle collide
            if ego_obstacle_collision:
                m1 = self.ego_params.m
                v1 = state2.v

                m2 = obstacle.obstacle_shape.shapely_object.area * self.area_mass_ratio
                v2 = getProjectedObstacleVelocity(state2.s, self.ccosy, obstacle_state)

                # Stoßgleichung TODO: Is this always correct?
                v1_prime = 2 * ((m1 * v1) + (m2 * v2)) / (m1 + m2) - v1  # Ego velocity after crash
                # v2_prime = 2 * ((m1 * v1) + (m2 * v2)) / (m1 + m2) - v2

                instant_violations.append(abs(state2.v - v1_prime))
        return max(instant_violations), 0.0, None

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle = key[0], key[1], key[2]
        time_step = self.time_step_map[time]
        obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle.obstacle_id))
        ego_bb = self.std_ego_coll_boxes.get(station)
        if obst_bb and ego_bb.collide(obst_bb):
            obstacle_state = getObstacleState(obstacle, time_step)
            return True, obstacle_state
        return False, None
