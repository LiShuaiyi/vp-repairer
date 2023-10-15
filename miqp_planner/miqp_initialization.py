from typing import Dict, Tuple, List, Union, Any
import numpy as np

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route

# commonroad-io
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import KSState, State
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.common.util import make_valid_orientation

# commonroad-curvilinear-coordinatesystem
from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import (chaikins_corner_cutting, compute_curvature_from_polyline, resample_polyline,
                                         compute_pathlength_from_polyline, compute_orientation_from_polyline)

from commonroad_qp_planner.trajectory import TrajPoint
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle, ReferencePoint
from commonroad_qp_planner.qp_lat_planner import QPLatState
from commonroad_qp_planner.qp_long_planner import QPLongState
from commonroad_qp_planner.utils import validate_orientation

from crmonitor.common.world import RoadNetwork, DynamicObstacleVehicle

def set_up_test(settings: Dict,
           scenario: Scenario,
           planning_problem: PlanningProblem,
           road_network: RoadNetwork,
           vehicle: DynamicObstacleVehicle,
        ):
    # instantiate a route planner with the scenario and the planning problem
    route_planner = RoutePlanner(scenario,
                                 planning_problem,
                                 backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    vehicle_configuration = create_optimization_configuration_vehicle_test(
        scenario,
        route_planner,
        planning_problem,
        settings['vehicle_settings'],
        road_network,
        vehicle
    )
    return vehicle_configuration


def create_optimization_configuration_vehicle_test(
        scenario: Scenario,
        route_planner: RoutePlanner,
        planning_problem: PlanningProblem,
        settings: Dict,
        road_network: RoadNetwork,
        vehicle: DynamicObstacleVehicle,
):
    assert (planning_problem.planning_problem_id in settings), \
        'Cannot find settings for planning problem {}'.format(planning_problem.planning_problem_id)

    vehicle_settings = settings[planning_problem.planning_problem_id]
    configuration = PlanningConfigurationVehicle()

    reference_path = vehicle.ref_path_lane
    lanelets_leading_to_goal = vehicle.lanelets_dir

    configuration.lanelet_network = create_lanelet_network(scenario.lanelet_network, lanelets_leading_to_goal)
    configuration.reference_path = reference_path.new_vertice

    if 'reference_point' in vehicle_settings:
        configuration.reference_point = set_reference_point(vehicle_settings['reference_point'])

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

    configuration.length = vehicle_settings['length']
    configuration.width = vehicle_settings['width']
    configuration.wheelbase = vehicle_settings['wheelbase']
    configuration.react_time = vehicle_settings['react_time']
    configuration.radius, _ = compute_approximating_circle_radius(configuration.length,
                                                                  configuration.width)
    configuration.curvilinear_coordinate_system = reference_path.clcs
    return configuration


def create_lanelet_network(lanelet_network: LaneletNetwork, lanelets_leading_to_goal: List[int]) -> LaneletNetwork:
    """
    Create a new lanelet network based on the current structure and given reference lanelets.
    """
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


def set_reference_point(reference_point: str) -> ReferencePoint:
    if reference_point == 'rear':
        return ReferencePoint.REAR
    elif reference_point == 'center':
        return ReferencePoint.CENTER
    else:
        raise ValueError("<set_reference_point>: reference point of the ego vehicle is unknown: {}".format(
            reference_point))


def compute_approximating_circle_radius(ego_length, ego_width) -> Tuple[Union[float, Any], Any]:
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

def create_curvilinear_coordinate_system(
        reference_path: np.ndarray) -> CurvilinearCoordinateSystem:
    cosy = CurvilinearCoordinateSystem(reference_path)
    return cosy
