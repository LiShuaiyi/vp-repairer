import time
from typing import List, Dict, Tuple, Set

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState
from minimum_violation_velocity_planner.common.node import Node
from minimum_violation_velocity_planner.common.utils import getStandardObstacleCollisionBoxes, getTimeStepMap, \
    getEgoLaneStationMap, getEgoCollisionBoxes, getEgoShapes, violationIntegrationLinearInterpolation, \
    violationIntegrationMaxValue, violationIntegrationCurrentValue, getdObstacleRefPathProjection
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from minimum_violation_velocity_planner.rulebook.rules.safety.collision_avoidance_vehicle import CollisionAvoidanceVehicles
from minimum_violation_velocity_planner.rulebook.rules.safety.collision_avoidance_vru import CollisionAvoidanceVRU
from minimum_violation_velocity_planner.rulebook.rules.safety.collision_avoidance_static_env import \
    CollisionAvoidanceStaticEnvironment
from minimum_violation_velocity_planner.rulebook.rules.safety.crash_impact_vehicle import CrashImpactVehicle
from minimum_violation_velocity_planner.rulebook.rules.safety.stop_after_crash import StopAfterCrash
from minimum_violation_velocity_planner.rulebook.rules.safety.clearance_front import ClearanceFront
from minimum_violation_velocity_planner.rulebook.rules.safety.clearance_back import ClearanceBack
from minimum_violation_velocity_planner.rulebook.rules.safety.narrow_passing import NarrowPassing

from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.safe_distance import SafeDistance
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.maximum_lane_velocity_limit import \
    MaximumLaneVelocityLimit
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.minimum_lane_velocity_limit import \
    MinimumLaneVelocityLimit
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.visibility_and_weather_conditions import \
    VisibilityAndWeatherConditions
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.road_conditions import RoadConditions
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.emergency_vehicles_in_duty import EmergencyVehiclesInDuty
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.stopping_on_highway import StoppingOnHighway
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.reversing_on_highway import ReversingOnHighway
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.driving_faster_than_left_traffic import \
    DrivingFasterThanLeftTraffic
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.preserve_traffic_flow import PreserveTrafficFlow
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.abrupt_braking import AbruptBraking
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.stop_at_stop_sign import StopAtStopSign
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.stop_at_traffic_light import StopAtTrafficLight
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.stop_at_crosswalk import StopAtCrosswalk
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.keep_distance_to_bicycles import KeepDistanceToBicycles
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.passing_of_buses import PassingOfBuses
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.blocking_of_crossings import BlockingOfCrossings
from minimum_violation_velocity_planner.rulebook.rules.traffic_rules.right_of_way import RightOfWay

from minimum_violation_velocity_planner.rulebook.rules.comfort.minimize_jerk import MinimizeJerk
from minimum_violation_velocity_planner.rulebook.rules.comfort.minimize_acceleration import MinimizeAcceleration
from minimum_violation_velocity_planner.rulebook.rules.comfort.comfort_acceleration_long import ComfortAccelerationLimitLong
from minimum_violation_velocity_planner.rulebook.rules.comfort.comfort_acceleration_lat import ComfortAccelerationLimitLat
from minimum_violation_velocity_planner.rulebook.rules.comfort.noise_emission import NoiseEmission
from minimum_violation_velocity_planner.rulebook.rules.comfort.comfort_vertical import ComfortVertical

from minimum_violation_velocity_planner.rulebook.rules.mission.global_velocity_limit import GlobalVelocityLimit
from minimum_violation_velocity_planner.rulebook.rules.mission.global_desired_velocity import GlobalDesiredVelocity
from minimum_violation_velocity_planner.rulebook.rules.mission.global_acceleration_limit import GlobalAccelerationLimit
from minimum_violation_velocity_planner.rulebook.rules.mission.reach_goal import ReachGoal
from minimum_violation_velocity_planner.rulebook.rules.mission.schedule import Schedule
from minimum_violation_velocity_planner.rulebook.rules.mission.busstop import BusStop
from minimum_violation_velocity_planner.rulebook.rules.mission.energy import Energy

from minimum_violation_velocity_planner.rulebook.rules.system.vehicle_limitations import VehicleLimitations
from minimum_violation_velocity_planner.rulebook.rules.system.degradation import Degradation


class Rulebook:
    """
    Rulebook
    """

    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                 route_lanelets: List[int], vehicle_parameters: VehicleParameters, params: Dict, grid: tuple,
                 integration_strategy: int):
        """
        """

        self.dt, dv, ds, da, time_horizon, station_horizon, velocity_horizon = grid
        state_rounding_precision = params.get('general').get('state_rounding_precision')

        self.max_violation = 999999.0 # TODO Unify this with velocity planner class

        # Set the integration function
        if integration_strategy == 0:
            self._integration_fct = violationIntegrationLinearInterpolation
        elif integration_strategy == 1:
            self._integration_fct = violationIntegrationMaxValue
        else:
            self._integration_fct = violationIntegrationCurrentValue

        time_step_map = getTimeStepMap(scenario, self.dt, time_horizon, state_rounding_precision)
        ego_lane_station_map = getEgoLaneStationMap(scenario, planning_problem, ccosy, ds, station_horizon, state_rounding_precision)
        std_ego_coll_boxes = getEgoCollisionBoxes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision)

        ego_coll_boxes_clearance_back = getEgoCollisionBoxes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision, extension_left=params.get('safety').get('clearance_back').get('distance'), extension_right=-4.508) # TODO PFUSCH!!
        ego_coll_boxes_clearance_front = getEgoCollisionBoxes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision, extension_right=params.get('safety').get('clearance_back').get('distance'), extension_left=-4.508)
        ego_coll_boxes_emergency_vehicles = getEgoCollisionBoxes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision, extension_right=params.get('traffic_rules').get('emergency_vehicles_in_duty').get('clearance_front'), extension_left=params.get('traffic_rules').get('emergency_vehicles_in_duty').get('clearance_back'), extension_bottom=params.get('traffic_rules').get('emergency_vehicles_in_duty').get('clearance_left_right'), extension_top=params.get('traffic_rules').get('emergency_vehicles_in_duty').get('clearance_left_right'))
        ego_coll_boxes_crosswalk = getEgoCollisionBoxes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision, extension_left=params.get('traffic_rules').get('stop_at_crosswalk').get('clearance_radius'), extension_top=params.get('traffic_rules').get('stop_at_crosswalk').get('clearance_radius'), extension_bottom=params.get('traffic_rules').get('stop_at_crosswalk').get('clearance_radius'), extension_right=params.get('traffic_rules').get('stop_at_crosswalk').get('clearance_radius'))

        ego_shapes_map = getEgoShapes(planning_problem, ccosy, vehicle_parameters, ds, station_horizon, state_rounding_precision)

        std_obst_coll_boxes = getStandardObstacleCollisionBoxes(scenario, time_step_map, self.dt, time_horizon, state_rounding_precision)
        obst_ref_path_projections = getdObstacleRefPathProjection(scenario, time_step_map, self.dt, time_horizon, state_rounding_precision, ccosy)

        traffic_sign_interpreter = TrafficSigInterpreter(SupportedTrafficSignCountry(scenario.scenario_id.country_id),
                                                         scenario.lanelet_network)

        # Instantiate all rules
        self._available_rules = self.initializeRules([
            StopAfterCrash(scenario, ccosy, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            CollisionAvoidanceVehicles(scenario, ccosy, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters),
            CollisionAvoidanceStaticEnvironment(scenario, ccosy, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters),
            CollisionAvoidanceVRU(scenario, ccosy, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters),
            CrashImpactVehicle(scenario, ccosy, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters, params),
            # Clearance(scenario, ccosy, vehicle_parameters, params),
            ClearanceFront(scenario, ccosy, time_step_map, ego_coll_boxes_clearance_front, std_obst_coll_boxes, vehicle_parameters, params),
            ClearanceBack(scenario, ccosy, time_step_map, ego_coll_boxes_clearance_back, std_obst_coll_boxes, vehicle_parameters, params),
            NarrowPassing(scenario, ccosy, time_step_map, ego_shapes_map, obst_ref_path_projections, vehicle_parameters, params),

            SafeDistance(scenario, ccosy, time_step_map, vehicle_parameters, params),
            MaximumLaneVelocityLimit(scenario, ccosy, ego_lane_station_map, traffic_sign_interpreter),
            MinimumLaneVelocityLimit(scenario, ccosy, ego_lane_station_map, traffic_sign_interpreter),
            VisibilityAndWeatherConditions(scenario, vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            RoadConditions(scenario,vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            EmergencyVehiclesInDuty(scenario, ccosy, time_step_map, ego_coll_boxes_emergency_vehicles, std_obst_coll_boxes, vehicle_parameters, params),

            MinimizeJerk(scenario, planning_problem),
            MinimizeAcceleration(scenario, planning_problem),
            ComfortAccelerationLimitLong(params),
            ComfortAccelerationLimitLat(ccosy, params),
            NoiseEmission(vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            ComfortVertical(params),

            StoppingOnHighway(scenario, ccosy, ego_lane_station_map, params),
            ReversingOnHighway(scenario, ccosy, ego_lane_station_map, params),
            DrivingFasterThanLeftTraffic(scenario, ccosy, time_step_map, ego_lane_station_map, vehicle_parameters, params),
            PreserveTrafficFlow(scenario, ccosy, ego_lane_station_map, traffic_sign_interpreter, params),
            AbruptBraking(params),
            StopAtStopSign(scenario, ccosy, vehicle_parameters, params),
            StopAtTrafficLight(scenario, ccosy, time_step_map, vehicle_parameters, params),
            StopAtCrosswalk(scenario, ccosy, time_step_map, ego_coll_boxes_crosswalk, std_obst_coll_boxes, vehicle_parameters, params),
            KeepDistanceToBicycles(scenario, ccosy, time_step_map, ego_shapes_map, obst_ref_path_projections, vehicle_parameters, params),
            PassingOfBuses(scenario, ccosy, time_step_map, ego_shapes_map, obst_ref_path_projections, vehicle_parameters, params),
            BlockingOfCrossings(scenario, ccosy, time_step_map, vehicle_parameters, params, time_horizon),
            RightOfWay(scenario, ccosy, route_lanelets, time_step_map, std_ego_coll_boxes, std_obst_coll_boxes, vehicle_parameters, params, time_horizon),

            #Progress(scenario, planning_problem, ccosy, vehicle_parameters, params),
            ReachGoal(scenario, planning_problem, ccosy, time_step_map, time_horizon),
            Schedule(scenario, planning_problem, params),
            BusStop(scenario, planning_problem, params),
            Energy(scenario, planning_problem, vehicle_parameters, params),
            GlobalVelocityLimit(scenario, planning_problem, vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            GlobalDesiredVelocity(scenario, planning_problem, vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
            GlobalAccelerationLimit(scenario, planning_problem, params),

            VehicleLimitations(ccosy, vehicle_parameters),
            Degradation(vehicle_parameters, params, integration_strategy, self.dt, time_horizon),
        ])
        self._active_rules = []
        self._nof_available_rules = len(self._available_rules)
        self._path_independent_rules_indices = set()
        self._nof_used_rules = 0

        self._rule_eval_times = [0.0] * len(self._available_rules)

        self._evaluation_times_total = []

    def getActiveRules(self) -> List[Rule]:
        return self._active_rules

    def nofActiveRules(self) -> int:
        return len(self._active_rules)

    def getPathIndependentRulesIndices(self) -> Set[int]:
        path_independent_rule_indices = set()
        for i, rule in enumerate(self._active_rules):
            if rule._is_path_independent:
                path_independent_rule_indices.add(i)
        return path_independent_rule_indices

    def setOrder(self, rule_keys_in_order: List[str], verbose=False):

        # Check if key does exist
        for key in rule_keys_in_order:
            if key not in set(self._available_rules):
                raise KeyError('Attention: ' + key + ' does not specify an available rule type!')
        # Check for duplicates
        if len(rule_keys_in_order) != len(set(rule_keys_in_order)):
            raise ValueError('Attention: Certain rules keys are duplicated in the list!')

        # Set active rules
        self._active_rules = [self._available_rules[key] for key in rule_keys_in_order]

        self._nof_used_rules = len(self._active_rules)
        if verbose:
            print('Nof used rules: ' + str(self._nof_used_rules) + '/' + str(self._nof_available_rules))

        # Set indices of path independent rules
        self._path_independent_rules_indices = self.getPathIndependentRulesIndices()
        print()

    def printAvailableRules(self):
        dash = '-' * 120

        print(dash)
        print('{:<40s}{:>4s}'.format('Key', 'Statement'))
        print(dash)

        for key in self._available_rules:
            print('{:<40s}{:>4s}'.format(key, self._available_rules[key].statement))
        print(dash, '\n')

    def initializeRules(self, rules: List[Rule]) -> dict:
        return dict((rule.tag, rule) for rule in rules)

    def evaluateRules(self, node1: Node, node2: Node, node_2_new_state: LongitudinalState) -> Tuple[List, List, List, List]:
        start_time = time.time()

        # TODO: Very ugly!!! Maybe create for each child node a respective parent node and add to graph or so
        # If this is the initial node, we first have to determine the violations since they are dependent on node2!!
        if node1.id[0] == 0:
            n1_state = node1.state
            # Calculate the acceleration of first state, since we use a different vehicle model for first time step
            n1_state.a = -(2 * (2 * self.dt * n1_state.v - 3 * (node_2_new_state.s - n1_state.s) + self.dt * node_2_new_state.v)) / self.dt ** 2
            n1_g_violation, n1_i_violation, _, n1_transfer_values = self.evaluateInitialRules(n1_state)
        else:
            n1_state = node1.state
            n1_g_violation = node1.g_violation
            n1_i_violation = node1.i_violation
            n1_transfer_values = node1.transfer_values

        n2_i_violation, n2_h_violation = None, None
        if node2.i_violation is not None:
            n2_i_violation = node2.i_violation
            n2_h_violation = node2.h_violation

        new_n2_g_violation = []
        new_n2_i_violation = []
        new_n2_h_violation = []
        new_n2_transfer_values = []

        for i, rule in enumerate(self._active_rules):
            rule_time_start = time.time()

            # If rule is not dependent of path (= only dependent on state), check if violation already calculated
            if i in self._path_independent_rules_indices:
                if n2_i_violation is not None:
                    new_i_violation = n2_i_violation[i]
                    new_h_violation = n2_h_violation[i]
                    new_transfer_value = 0.0
                # If violation was not yet calculated -> evaluate it
                else:
                    new_i_violation, new_h_violation, new_transfer_value = rule.determineViolation(n1_state, node_2_new_state, None)
            # if not
            else:
                new_i_violation, new_h_violation, new_transfer_value = rule.determineViolation(n1_state, node_2_new_state, n1_transfer_values[i])
            new_n2_i_violation.append(new_i_violation)
            new_n2_h_violation.append(new_h_violation)
            new_n2_transfer_values.append(new_transfer_value)

            # Integration
            new_n2_g_violation.append(n1_g_violation[i] + self._integration_fct(n1_i_violation[i], new_i_violation, self.dt))

            self._rule_eval_times[i] += time.time() - rule_time_start

        self._evaluation_times_total.append(time.time() - start_time)

        return new_n2_g_violation, new_n2_i_violation, new_n2_h_violation, new_n2_transfer_values

    def evaluateInitialRules(self, node1_state: LongitudinalState) -> Tuple[List, List, List, List]:
        start_time = time.time()

        new_n2_g_violation = [0.0] * len(self._active_rules)
        new_n2_i_violation = []
        new_n2_h_violation = []
        new_n2_transfer_values = []

        for i, rule in enumerate(self._active_rules):
            rule_time_start = time.time()

            new_i_violation, new_h_violation, new_transfer_value = rule.determineViolation(node1_state, node1_state)

            new_n2_i_violation.append(new_i_violation)
            new_n2_h_violation.append(new_h_violation)
            new_n2_transfer_values.append(new_transfer_value)

            self._rule_eval_times[i] += time.time() - rule_time_start

        self._evaluation_times_total.append(time.time() - start_time)
        return new_n2_g_violation, new_n2_i_violation, new_n2_h_violation, new_n2_transfer_values

    def evaluateRuleAtLevel(self, node1: Node, node2: Node, node_2_new_state: LongitudinalState, rule_level: int) -> Tuple[float, float, float, float]:
        # TODO: Very ugly!!! Maybe create for each child node a respective parent node and add to graph or so
        # If this is the initial node, we first have to determine the violations since they are dependent on node2!!
        if node1.id[0] == 0:
            n1_state = node1.state
            # Calculate the acceleration of first state, since we use a different vehicle model for first time step
            n1_state.a = -(2 * (2 * self.dt * n1_state.v - 3 * (node_2_new_state.s - n1_state.s) + self.dt * node_2_new_state.v)) / self.dt ** 2
            n1_g_violation, n1_i_violation, _, n1_transfer_values = self.evaluateInitialRules(n1_state)
            n1_g_violation_score = n1_g_violation[rule_level]
            n1_i_violation_score = n1_i_violation[rule_level]
            n1_transfer_value = n1_transfer_values[rule_level]
        else:
            n1_state = node1.state
            n1_g_violation_score = node1.g_violation[rule_level]
            n1_i_violation_score = node1.i_violation[rule_level]
            n1_transfer_value = node1.transfer_values[rule_level]

        n2_i_violation_score = node2.i_violation[rule_level]
        n2_h_violation_score = node2.h_violation[rule_level]

        rule_time_start = time.time()

        # If rule is not dependent of path (= only dependent on state), check if violation already calculated
        if rule_level in self._path_independent_rules_indices:
            if n2_i_violation_score != self.max_violation:
                new_n2_i_violation_score = n2_i_violation_score
                new_n2_h_violation_score = n2_h_violation_score
                new_n2_transfer_value = 0.0
            # If violation was not yet calculated -> evaluate it
            else:
                new_n2_i_violation_score, new_n2_h_violation_score, new_n2_transfer_value = self._active_rules[rule_level].determineViolation(n1_state, node_2_new_state, None)
        # if not
        else:
            new_n2_i_violation_score, new_n2_h_violation_score, new_n2_transfer_value = self._active_rules[rule_level].determineViolation(n1_state, node_2_new_state, n1_transfer_value)

        # Integration
        new_n2_g_violation_score = n1_g_violation_score + self._integration_fct(n1_i_violation_score, new_n2_i_violation_score, self.dt)

        self._rule_eval_times[rule_level] += time.time() - rule_time_start

        return new_n2_g_violation_score, new_n2_i_violation_score, new_n2_h_violation_score, new_n2_transfer_value

    def evaluateTrajectory(self, trajectory: List[LongitudinalState]) -> List:
        initial_state = trajectory[0]
        initial_node = Node(initial_state.t, initial_state.s, initial_state.v, initial_state)
        ini_g_violation, ini_i_violation, ini_h_violation, ini_transfer_values = self.evaluateInitialRules(initial_state)
        initial_node.update(g_violation=ini_g_violation, h_violation=ini_h_violation, i_violation=ini_i_violation, transfer_values=ini_transfer_values)

        nodes = [initial_node]
        violations = [initial_node.g_violation]

        for i in range(1, len(trajectory)):
            parent = nodes[-1]
            state = trajectory[i]
            child = Node(state.t, state.s, state.v, state)
            new_g_violation, new_i_violation, new_h_violation, new_transfer_values = self.evaluateRules(parent, child, state)
            child.update(state=state, predecessor=parent, g_violation=new_g_violation, i_violation=new_i_violation, h_violation=new_h_violation, transfer_values=new_transfer_values)
            nodes.append(child)
            violations.append(child.g_violation)
        return nodes, violations

    def getTotalRuleEvaluationTime(self):
        return sum(self._rule_eval_times)

    def printEvaluationTimesPerRule(self):
        dash = '-' * 51

        print(dash)
        for i, rule in enumerate(self._active_rules):
            print('{:<40s}{:>4f}{:>4s}'.format(rule.tag, self._rule_eval_times[i], '[s]'))
        print(dash)

    def getEvaluationTime(self) -> Tuple[float, float, float]:
        min_time = min(self._evaluation_times_total) * 1000.0
        mean_time = sum(self._evaluation_times_total) / len(self._evaluation_times_total) * 1000.0
        max_time = max(self._evaluation_times_total) * 1000.0
        return round(min_time, 3), round(mean_time, 3), round(max_time, 3)

    def numberOfEvaluations(self) -> int:
        return len(self._evaluation_times_total)
