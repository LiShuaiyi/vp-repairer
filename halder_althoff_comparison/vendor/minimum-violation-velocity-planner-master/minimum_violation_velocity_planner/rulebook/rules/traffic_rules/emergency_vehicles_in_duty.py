import functools
from typing import Dict, Tuple, Union

from commonroad.scenario.obstacle import ObstacleType
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class EmergencyVehiclesInDuty(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict, ego_coll_boxes_emergency_vehicles: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("EMERGENCY_VEHICLES_IN_DUTY", "The ego shall clear the road asap if active emergency vehicles approach.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.ego_coll_boxes_emergency_vehicles = ego_coll_boxes_emergency_vehicles
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('emergency_vehicles_in_duty')

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, {ObstacleType.PRIORITY_VEHICLE})

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]
        for obstacle in self.relevant_obstacles:
            ego_obstacle_collision = self.checkEgoObstacleCollision((state2.t, state2.s, obstacle))
            # If ego and obstacle collide
            if ego_obstacle_collision:
                instant_violations.append(1.0)
                break
        return max(instant_violations), 0.0, None

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle = key[0], key[1], key[2]

        time_step = self.time_step_map[time]
        obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle.obstacle_id))
        ego_bb = self.ego_coll_boxes_emergency_vehicles.get(station)
        if obst_bb and ego_bb.collide(obst_bb):
            obstacle_signale_state = obstacle.signal_state_at_time_step(time_step)
            if obstacle_signale_state and obstacle_signale_state.flashing_blue_lights:
                return True
        return False
