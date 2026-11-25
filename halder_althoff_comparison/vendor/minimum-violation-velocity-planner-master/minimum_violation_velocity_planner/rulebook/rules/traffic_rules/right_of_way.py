import functools
from typing import Dict, List, Tuple, Union

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getRelevantObstacles
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType, Obstacle

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class RightOfWay(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, route_lanelets: List[int], time_step_map: Dict, std_ego_coll_boxes: Dict,
                 std_obst_coll_boxes: Dict, vehicle_parameters: VehicleParameters, params: Dict, time_horizon: List):
        super().__init__("RIGHT_OF_WAY", "The ego shall yield to vehicles coming from the right or when left turning.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.route_lanelets = set(route_lanelets)
        self.time_step_map = time_step_map
        self.std_ego_coll_boxes = std_ego_coll_boxes
        self.std_obst_coll_boxes = std_obst_coll_boxes

        self.ego_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('priority')
        self.time_behind = self.params.get('time_behind')
        self.time_ahead = self.params.get('time_ahead')

        self.time_horizon = time_horizon[1]

        self.obstacle_types = {ObstacleType.CAR,
                               ObstacleType.TRUCK,
                               ObstacleType.BUS,
                               ObstacleType.PRIORITY_VEHICLE,
                               ObstacleType.TRAIN,
                               ObstacleType.MOTORCYCLE,
                               ObstacleType.TAXI,
                               ObstacleType.BICYCLE}

        self.relevant_obstacles = getRelevantObstacles(self.scenario.obstacles, self.obstacle_types)
        self.higher_priority_obstacles = self.getHigherPriorityObstacles()

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        instant_violations = [0.0]
        for obstacle in self.higher_priority_obstacles:
            ego_obstacle_collision = self.checkEgoObstacleCollision((state2.t, state2.s, obstacle))
            # If ego and obstacle collide
            if ego_obstacle_collision:
                instant_violations.append(1.0)
                break # TODO This could also used in other rules, rigth?
        return max(instant_violations), 0.0, None

    def getHigherPriorityObstacles(self) -> List[Obstacle]:

        higher_priority_obstacles = []

        # If there are no traffic lights in the scenario
        # TODO: Make this check more sophisticated, and check state and position of traffic light
        if len(self.scenario.lanelet_network.traffic_lights) == 0:
            for intersection in self.scenario.lanelet_network.intersections:

                # Get incoming of ego in this intersection
                ego_incoming = None
                obst_incomings = set()
                for incoming in intersection.incomings:
                    # Check if lanelets of the ego route are in incoming
                    if incoming.incoming_lanelets & self.route_lanelets:
                        ego_incoming = incoming

                    # Get obstacle incomings in this junction
                    for obstacle in self.relevant_obstacles:
                        # For the initial state
                        obst_lanelet_ids = set(self.scenario.lanelet_network.find_lanelet_by_position([obstacle.initial_state.position])[0])
                        if incoming.incoming_lanelets & obst_lanelet_ids:
                            obst_incomings.update([(obstacle, incoming)])
                            continue

                        # For the trajectory state
                        for state in obstacle.prediction.trajectory.state_list:
                            obst_lanelet_ids = set(self.scenario.lanelet_network.find_lanelet_by_position([state.position])[0])
                            if incoming.incoming_lanelets & obst_lanelet_ids:
                                obst_incomings.update([(obstacle, incoming)])
                                break

                if ego_incoming:
                    # 1. Check Right-Before-Left Rule
                    for obst_incoming in obst_incomings:
                        # If obstacle incoming is on the right of the ego incoming
                        if ego_incoming.left_of == obst_incoming[1].incoming_id:
                            higher_priority_obstacles.append(obst_incoming[0])

                    # 2. Check Left Turning Rule
                    # First check, if ego turns left
                    ego_left_outgoings = set()
                    if ego_incoming.successors_left:
                        for successor_left_id in ego_incoming.successors_left:
                            ego_left_outgoings.update(self.scenario.lanelet_network.find_lanelet_by_id(successor_left_id).successor)

                    if ego_left_outgoings & self.route_lanelets:
                        for obst_incoming in obst_incomings:
                            # If obstacle incoming is neither left nor right of ego incoming nor is the same as ego
                            if not(ego_incoming.left_of == obst_incoming[1].incoming_id) and \
                                    not(obst_incoming[1].left_of == ego_incoming.incoming_id) and \
                                    not(ego_incoming.incoming_id == obst_incoming[1].incoming_id):
                                higher_priority_obstacles.append(obst_incoming[0])

        return higher_priority_obstacles

    @functools.lru_cache(maxsize=None)
    def checkEgoObstacleCollision(self, key):
        time, station, obstacle = key[0], key[1], key[2]

        # Loop over all object states in time-interval [t0-time_behind, t0+time_ahead]
        time_step_min = self.time_step_map[max(0.0, time - self.time_behind)]
        time_step_max = self.time_step_map[min(self.time_horizon, time + self.time_ahead)]
        ego_bb = self.std_ego_coll_boxes.get(station)
        for time_step in range(time_step_min, time_step_max + 1):
            obst_bb = self.std_obst_coll_boxes.get((time_step, obstacle.obstacle_id))
            if obst_bb and ego_bb.collide(obst_bb):
                return True
        return False
