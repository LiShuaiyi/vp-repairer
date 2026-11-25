import functools
from typing import Dict, Tuple, Union

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getProjectedObstacleVelocity, \
    getObstacleState, getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState



class CollisionAvoidanceVehicles(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, std_ego_coll_boxes: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters):
        super().__init__("COLLISION_AVOIDANCE_VEHICLE", "The ego shall not collide with an other vehicle.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.std_ego_coll_boxes = std_ego_coll_boxes
        self.std_obst_coll_boxes = std_obst_coll_boxes
        self.ego_params = vehicle_parameters

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
                v_rel = state2.v - getProjectedObstacleVelocity(state2.s, self.ccosy, obstacle_state)  # TODO: Vermutlich braucht man die relativ geschwindiukeit garnicht und generell gitl, je langsamer bei der kollision desto besser, aber was ist mit AUffahrunfall vo hinteN?
                                                                                                       # TODO das würde auch heißen, dass generell kollisionen mit niedrigen geschwinidgkeiten besser sind als bei hohen z.B: auf der AUtobahn    . ALso würde ein Max() reichen
                instant_violations.append(abs(v_rel))

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
