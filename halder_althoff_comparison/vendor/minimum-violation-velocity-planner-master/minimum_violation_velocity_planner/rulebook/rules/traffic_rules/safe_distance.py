import functools
from typing import Dict, Tuple, Union

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getObstacleState, getRelevantObstacles, \
    getOrientationAtStation, deltaAngles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState




class SafeDistance(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict,
                 vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("SAFE_DISTANCE", "The ego shall keep a safe distance to preceding vehicles.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('safe_distance')
        self.a_min_leading_vehicle = self.params.get("a_min_leading_vehicle")
        self.t_react = self.params.get("t_react")
        self.corridor_width = self.params.get("corridor_width")
        self.delta_orientation = self.params.get('delta_orientation')

        self.a_min_follow = -self.ego_params.longitudinal.a_max
        self.a_min_lead = self.a_min_leading_vehicle

        self.obstacle_types = {ObstacleType.CAR,
                               ObstacleType.TRUCK,
                               ObstacleType.BUS,
                               ObstacleType.PRIORITY_VEHICLE,
                               ObstacleType.TRAIN,
                               ObstacleType.MOTORCYCLE,
                               ObstacleType.TAXI,
                               ObstacleType.BICYCLE,
                               ObstacleType.PEDESTRIAN}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]
        for obstacle in self.relevant_obstacles:
            obst_in_corridor_and_in_front, obst_state, d_ego_obst = self.checkIfObstacleIsInCorridorAndInFront((state2.t, state2.s, obstacle))
            # If obstacle is in corridor and in front of ego
            if obst_in_corridor_and_in_front:
                d_safe = self.safeDistance(state2.v, obst_state.velocity, self.a_min_follow, self.a_min_lead, self.t_react)
                instant_violations.append(max(0.0, (d_safe - d_ego_obst)))
        return max(instant_violations), 0.0, None

    def safeDistance(self, v_follow: float, v_lead: float, a_min_follow: float, a_min_lead: float,
                     t_react_follow: float) -> float:
        """
        Calculates safe distance analytically

        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param t_react_follow: reaction time of following vehicle
        """

        d_safe = (v_lead ** 2) / (-2 * abs(a_min_lead)) - (v_follow ** 2) / (-2 * abs(a_min_follow)) \
                 + v_follow * t_react_follow
        return d_safe

    @functools.lru_cache(maxsize=None)
    def isObstacleInCorridor(self, key: Tuple) -> Tuple[bool, Union[None, State], Union[None, float],  Union[None, float]]:
        obstacle, time_step = key[0], key[1]

        obstacle_state = getObstacleState(obstacle, time_step)
        if obstacle_state is not None:

            obst_position = obstacle_state.position
            if self.ccosy.cartesian_point_inside_projection_domain(obst_position[0], obst_position[1]):
                s_d_obst = self.ccosy.convert_to_curvilinear_coords(obst_position[0], obst_position[1])
                # Check if center point is in corridor
                if abs(s_d_obst[1]) <= self.corridor_width/2:
                    orientation_ref_path = getOrientationAtStation(self.ccosy, s_d_obst[0])
                    delta_orientation = abs(deltaAngles(orientation_ref_path, obstacle_state.orientation))
                    # Check if heading is in allowed range
                    if delta_orientation <= self.delta_orientation:
                        return True, obstacle_state, s_d_obst[0], s_d_obst[0] - obstacle.obstacle_shape.length/2

        return False, None, None, None

    @functools.lru_cache(maxsize=None)
    def checkIfObstacleIsInCorridorAndInFront(self, key):
        time, station, obstacle = key[0], key[1], key[2]

        time_step = self.time_step_map[time]
        is_obst_in_corridor, obst_state, s_obst, s_obst_back = self.isObstacleInCorridor((obstacle, time_step))

        # If obstacle is in corridor and in front of ego
        if is_obst_in_corridor and s_obst >= station:
            d_ego_obst = max(0.0, s_obst_back - (station + self.ego_params.l/2))
            return True, obst_state, d_ego_obst
        return False, None, None
