from collections import defaultdict
from typing import Dict, Tuple, List, Union
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

# replace the utilities in computing the reachable set
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route

from commonroad_dc.geometry.util import chaikins_corner_cutting, compute_curvature_from_polyline,\
    compute_polyline_length, resample_polyline, compute_orientation_from_polyline, compute_pathlength_from_polyline

# commonroad-io
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle

# commonroad-curvilinear-coordinatesystem
import commonroad_dc.pycrccosy as pycrccosy
import commonroad_dc.pycrcc as pycrcc
import commonroad_dc

from optimizer.configuration import RepairingConfigurationVehicle
from optimization.utils import plot_reference_path

def set_up(settings: Dict, scenario: Scenario, planning_problem: PlanningProblem, initial_trajectory: Trajectory):
    # update goal state
    planning_problem.goal = update_goal_state(initial_trajectory)
    # instantiate a route planner with the scenario and the planning problem
    route_planner = RoutePlanner(scenario, planning_problem,
        backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    vehicle_configuration = create_optimization_configuration_vehicle(
        scenario,
        route_planner,
        planning_problem,
        initial_trajectory,
        settings['vehicle_settings'])
    return vehicle_configuration

def create_optimization_configuration_vehicle(
        scenario: Scenario,
        route_planner: RoutePlanner,
        planning_problem: PlanningProblem,
        initial_trajectory: Trajectory,
        settings: Dict,):
    assert (planning_problem.planning_problem_id in settings), \
        'Cannot find settings for planning problem {}'.format(planning_problem.planning_problem_id)

    # planning_problem.initial_state = initial_trajectory.state_list[0]
    vehicle_settings = settings[planning_problem.planning_problem_id]
    configuration = RepairingConfigurationVehicle()

    reference_path, lanelets_leading_to_goal, initial_lanelet_id = find_reference_path_and_lanelets_leading_to_goal(
        route_planner, planning_problem, settings)
    configuration.initial_lanelet_id = initial_lanelet_id

    configuration.lanelet_network = create_lanelet_network(scenario.lanelet_network, lanelets_leading_to_goal)
    configuration.reference_path = np.array(reference_path)

    configuration.vehicle_id = planning_problem.planning_problem_id
    configuration.min_speed_x = vehicle_settings['min_speed_x']
    configuration.max_speed_x = vehicle_settings['max_speed_x']
    configuration.min_speed_y = vehicle_settings['min_speed_y']
    configuration.max_speed_y = vehicle_settings['max_speed_y']

    configuration.a_max_x = vehicle_settings['a_max_x']
    configuration.a_min_x = vehicle_settings['a_min_x']
    configuration.a_max_y = vehicle_settings['a_max_y']
    configuration.a_min_y = vehicle_settings['a_min_y']
    configuration.a_max = vehicle_settings['a_max']

    configuration.j_min_x = vehicle_settings['j_min_x']
    configuration.j_max_x = vehicle_settings['j_max_x']
    configuration.j_min_y = vehicle_settings['j_min_y']
    configuration.j_max_y = vehicle_settings['j_max_y']

    configuration.initial_time_idx = planning_problem.initial_state.time_step

    configuration.length = vehicle_settings['length']
    configuration.width = vehicle_settings['width']
    radius, wheelbase = compute_approximating_circle_radius(configuration.length,
                                                            configuration.width)
    configuration.radius = radius
    configuration.wheelbase = vehicle_settings['wheelbase']

    configuration.curvilinear_coordinate_system = create_curvilinear_coordinate_system(
        configuration.reference_path)
    configuration.initial_state = compute_initial_state(planning_problem,
                                                        configuration)

    configuration.collision_checks_in_curvilinear_cosy = vehicle_settings['collision_checks_in_curvilinear_cosy']

    # todo: collision checker?
    # configuration.collision_checker_world = create_cartesian_collision_checker(
    #     scenario,
    #     configuration.lanelet_network,
    #     configuration.radius,
    #     True,
    #     vehicle_settings['reduce_distance_to_road_boundary'])
    return configuration


def find_reference_path_and_lanelets_leading_to_goal(
        route_planner: RoutePlanner, planning_problem: PlanningProblem, settings: Dict):
    def interval_extract(list_ids):
        list_ids = sorted(set(list_ids))
        range_start = previous_number = list_ids[0]
        merged_intervals = list()
        for number in list_ids[1:]:
            if number == previous_number + 1:
                previous_number = number
            else:
                merged_intervals.append([range_start, previous_number])
                range_start = previous_number = number
        merged_intervals.append([range_start, previous_number])
        return merged_intervals

    assert (planning_problem.planning_problem_id in settings), \
        'Cannot find settings for planning problem {}'.format(planning_problem.planning_problem_id)

    vehicle_settings = settings[planning_problem.planning_problem_id]

    source_lanelet = route_planner.lanelet_network.lanelets_in_proximity(
        planning_problem.initial_state.position, 100)

    if len(source_lanelet) < 1:
        raise ValueError('Expected exactly one source lanelet. Found no source lanelet.')
    else:
        source_lanelet = source_lanelet[0]

    start_lanelet = source_lanelet
    # find the start lanelet
    if source_lanelet.predecessor:
        start_lanelet = route_planner.lanelet_network.find_lanelet_by_id(source_lanelet.predecessor[0])
    candidate_holder = route_planner.plan_routes()
    route = candidate_holder.retrieve_first_route()
    reference_path = route.reference_path

    # visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)
    # list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()

    lanelets_leading_to_goal = route.list_ids_lanelets
    # extend the reference path:
    first_lanelet = route_planner.lanelet_network.find_lanelet_by_id(lanelets_leading_to_goal[0])
    while first_lanelet.predecessor:
        first_lanelet = route_planner.lanelet_network.find_lanelet_by_id(first_lanelet.predecessor[0])
        reference_path = np.concatenate((first_lanelet.center_vertices, reference_path))
    last_lanelet = route_planner.lanelet_network.find_lanelet_by_id(lanelets_leading_to_goal[-1])
    goal_position = planning_problem.goal.state_list[0].position.center

    while last_lanelet.successor:
        last_lanelet = route_planner.lanelet_network.find_lanelet_by_id(last_lanelet.successor[0])
        reference_path = np.concatenate((reference_path, last_lanelet.center_vertices))
        if reference_path[-1][0] > goal_position[0]:
            break
    # if 'use_complete_lanelet_network' in vehicle_settings and vehicle_settings['use_complete_lanelet_network']:
    #     lanelets_leading_to_goal = set(route_planner.lanelet_network._lanelets.keys())

    if 'overwrite_reference_path' in vehicle_settings and vehicle_settings['overwrite_reference_path'] is not None:
        reference_path = route_planner.lanelet_network.find_lanelet_by_id(
            vehicle_settings['overwrite_reference_path'][0]).center_vertices
        for element in range(1, len(vehicle_settings['overwrite_reference_path'])):
            next_lanelet = route_planner.lanelet_network.find_lanelet_by_id(
                vehicle_settings['overwrite_reference_path'][element])
            if np.isclose(reference_path[-1],
                          next_lanelet.center_vertices[0]).all():
                idx = 1
            else:
                idx = 0
            reference_path = np.concatenate((reference_path,
                                             next_lanelet.center_vertices[idx:]))
    # plot_reference_path(reference_path)
    max_curvature = vehicle_settings['max_curvature_reference_path'] + 0.2
    if vehicle_settings['resampling_reference_path']:
        while max_curvature > vehicle_settings['max_curvature_reference_path']:
            reference_path = np.array(chaikins_corner_cutting(reference_path))
            reference_path = resample_polyline(reference_path, vehicle_settings['resampling_reference_path'])
            abs_curvature = abs(compute_curvature_from_polyline(reference_path))
            max_curvature = max(abs_curvature)

        if 'resampling_reference_path_depending_on_curvature' in vehicle_settings:
#                and vehicle_settings['resampling_reference_path_depending_on_curvature']:
            # resample path with higher value where curvature is small
            resampled_path = list()
            intervals = list()
            abs_curvature[0:5] = 0.2
            merged_intervals_ids = interval_extract([i for i, v in enumerate(abs_curvature) if v < 0.01])
            for i in range(0, len(merged_intervals_ids) - 1):
                if i == 0 and merged_intervals_ids[i][0] != 0:
                    intervals.append([0, merged_intervals_ids[i][0]])
                if merged_intervals_ids[i][0] != merged_intervals_ids[i][1]:
                    intervals.append(merged_intervals_ids[i])
                intervals.append([merged_intervals_ids[i][1], merged_intervals_ids[i + 1][0]])

            if len(merged_intervals_ids) == 1:
                if merged_intervals_ids[0][0] != 0:
                    intervals.append([0, merged_intervals_ids[0][0]])
                if merged_intervals_ids[0][0] != merged_intervals_ids[0][1]:
                    intervals.append(merged_intervals_ids[0])

            if intervals and intervals[-1][1] != len(reference_path):
                intervals.append([intervals[-1][1], len(reference_path)])

            resampled_path = None
            for i in intervals:
                if i in merged_intervals_ids:
                    step = 3.
                else:
                    step = vehicle_settings['resampling_reference_path']
                if resampled_path is None:
                    resampled_path = resample_polyline(reference_path[i[0]:i[1]], step)
                else:
                    resampled_path = np.concatenate(
                        (resampled_path, resample_polyline(reference_path[i[0]:i[1]], step)))
        else:
            resampled_path = reference_path
    else:
        resampled_path = reference_path
    return resampled_path, lanelets_leading_to_goal, start_lanelet.lanelet_id

def create_lanelet_network(lanelet_network: LaneletNetwork, lanelets_leading_to_goal: List[int]) -> LaneletNetwork:
    new_lanelet_network = LaneletNetwork()

    for lanelet_id in lanelets_leading_to_goal:
        lanelet_orig = lanelet_network.find_lanelet_by_id(lanelet_id)

        predecessor = list(set(lanelet_orig.predecessor).intersection(lanelets_leading_to_goal))
        successor = list(set(lanelet_orig.successor).intersection(lanelets_leading_to_goal))

        lanelet = Lanelet(lanelet_orig.left_vertices, lanelet_orig.center_vertices, lanelet_orig.right_vertices,
                          lanelet_orig.lanelet_id, predecessor, successor)

        if {lanelet_orig.adj_left}.intersection(lanelets_leading_to_goal):
            lanelet.adj_left = lanelet_orig.adj_left
            lanelet.adj_left_same_direction = lanelet_orig.adj_left_same_direction
        if {lanelet_orig.adj_right}.intersection(lanelets_leading_to_goal):
            lanelet.adj_right = lanelet_orig.adj_right
            lanelet.adj_right_same_direction = lanelet_orig.adj_right_same_direction
        new_lanelet_network.add_lanelet(lanelet)
    return new_lanelet_network



def compute_approximating_circle_radius(ego_length, ego_width) -> float:
    """
    From Julia Kabalar
    Computes parameters of the circle approximation of the ego_vehicle

    :param ego_length: Length of ego vehicle
    :param ego_width: Width of ego vehicle
    :return: radius of circle approximation, circle center point distance
    """
    assert ego_length >= 0 and ego_width >= 0, 'Invalid vehicle dimensions = {}'.format([ego_length, ego_width])

    if np.isclose(ego_length, 0.0) and np.isclose(ego_width, 0.0):
        return 0.0, 0.0

    # Divide rectangle into 3 smaller rectangles
    square_length = ego_length / 3

    # Calculate minimum radius
    diagonal_square = np.sqrt((square_length / 2) ** 2 + (ego_width / 2) ** 2)

    # Round up value
    if diagonal_square > round(diagonal_square, 1):
        approx_radius = round(diagonal_square, 1) + 0.1
    else:
        approx_radius = round(diagonal_square, 1)

    return approx_radius, round(square_length * 2, 1)


def create_curvilinear_coordinate_system(reference_path: np.ndarray) -> commonroad_dc.pycrccosy.CurvilinearCoordinateSystem:
    cosy = pycrccosy.CurvilinearCoordinateSystem(reference_path)
    # curvature_of_reference_path = compute_curvature_from_polyline(reference_path)
    # print(len(curvature_of_reference_path), len(reference_path))
    # cosy.set_curvature(curvature_of_reference_path)
    return cosy

def compute_initial_state(
        planning_problem: PlanningProblem,
        configuration: RepairingConfigurationVehicle) -> Tuple:
    """
    This function computes the initial state of the ego vehicle for the reachable set computation given a
    planning problem according to CommonRoad. It is assumed that d*kappa_ref << 1 holds, where d is the distance of
    the ego vehicle to the reference path and kappa_ref is the curvature of the reference path,
    for the transformation of the ego vehicle's velocity to the curvilinear coordinate system.

    :param planning_problem: CommonRoad planning problem
    :param configuration: parameters of the ego vehicle
    :return: initial state of the ego vehicle in curvilinear coordinates
    """
    # if configuration.reference_point == pycrreach.ReferencePoint.REAR:
    pos = configuration.curvilinear_coordinate_system.convert_to_curvilinear_coords(
        planning_problem.initial_state.position[0]
        - configuration.wheelbase / 2 * np.cos(planning_problem.initial_state.orientation),
        planning_problem.initial_state.position[1]
        - configuration.wheelbase / 2 * np.sin(planning_problem.initial_state.orientation))
    # elif configuration.reference_point == pycrreach.ReferencePoint.CENTER:
    #     pos = configuration.curvilinear_coordinate_system.convert_to_curvilinear_coords(
    #         planning_problem.initial_state.position[0],
    #         planning_problem.initial_state.position[1])
    # else:
    #     raise ValueError("<compute_initial_state>: unknown reference point: {}".format(
    #         configuration.reference_point))

    reference_path = configuration.reference_path
    ref_orientation = compute_orientation_from_polyline(reference_path)
    ref_pathlength = compute_pathlength_from_polyline(reference_path)
    orientation_interpolated = np.interp(pos[0], ref_pathlength, ref_orientation)

    v_x = planning_problem.initial_state.velocity * np.cos(
        planning_problem.initial_state.orientation - orientation_interpolated)
    v_y = planning_problem.initial_state.velocity * np.sin(
        planning_problem.initial_state.orientation - orientation_interpolated)
    return ((pos[0], v_x),
            (pos[1], v_y))

def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    # todo: complete the function
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(ini_final_state.orientation-0.2, ini_final_state.orientation+0.2)
    goal_velocity = Interval(0, ini_final_state.velocity+5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list)+5)
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region

def create_cartesian_collision_checker(scenario: Scenario,
                                       lanelet_network_vehicle: LaneletNetwork,
                                       vehicle_radius: float,
                                       consider_traffic=False,
                                       reduce_distance_to_road_boundary=0) -> pycrcc.CollisionChecker:
    scenario_cc: Scenario = Scenario(scenario.dt, scenario.benchmark_id)
    scenario_cc.add_objects(lanelet_network_vehicle)
    if consider_traffic:
        scenario_cc.add_objects(scenario.obstacles)
    scenario_cc.add_objects(create_road_boundary(scenario_cc))

    cc = pycrcc.CollisionChecker()
    for obstacle in scenario_cc.obstacles:
        if obstacle.obstacle_type == ObstacleType.ROAD_BOUNDARY:
            cc.add_collision_object(
                create_collision_object(
                    obstacle, params={'minkowski_sum_circle': True,
                                      'minkowski_sum_circle_radius': vehicle_radius - reduce_distance_to_road_boundary,
                                      'resolution': 5}))
        else:
            cc.add_collision_object(
                create_collision_object(obstacle, params={'minkowski_sum_circle': True,
                                                          'minkowski_sum_circle_radius': vehicle_radius,
                                                          'resolution': 5}))
    return cc
