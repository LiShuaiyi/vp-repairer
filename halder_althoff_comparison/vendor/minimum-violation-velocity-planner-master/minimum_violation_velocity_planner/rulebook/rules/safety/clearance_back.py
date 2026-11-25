import functools
from typing import Dict, Tuple, Union

from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class ClearanceBack(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map:Dict, ego_coll_boxes_clearance_back:Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("CLEARANCE_BACK", "The ego shall keep clearance distance to the back.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map
        self.ego_coll_boxes_clearance_back = ego_coll_boxes_clearance_back
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.ego_params = vehicle_parameters
        self.params = params.get('safety').get('clearance_back')
        self.distance = self.params.get('distance')
        self.v_min = self.params.get('v_min')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]

        for obstacle in self.scenario.obstacles:
            # If ego and obstacle collide
            if state2.v > self.v_min:
                ego_obstacle_collision = self.checkEgoObstacleCollision((state2.t, state2.s, obstacle.obstacle_id))
                if ego_obstacle_collision:
                    instant_violations.append(1.0)

        return max(instant_violations), 0.0, None

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle_id = key[0], key[1], key[2]

        time_step = self.time_step_map[time]
        obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle_id))
        ego_bb = self.ego_coll_boxes_clearance_back.get(station)
        if obst_bb and ego_bb.collide(obst_bb):
            return True
        return False
