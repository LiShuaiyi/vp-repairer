import functools
import math
import os
import shutil
from typing import Union, Dict, Tuple, List, Set
import ruamel.yaml
from commonroad.geometry.shape import Rectangle, Shape
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import StopLine, Lanelet
from commonroad.scenario.obstacle import Obstacle, ObstacleRole, ObstacleType

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.scenario.trajectory import State
from commonroad_route_planner.route_planner import RoutePlanner
from vehiclemodels.vehicle_parameters import VehicleParameters
import numpy as np
from commonroad_dc import pycrcc
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

from shapely.geometry import LineString

from minimum_violation_velocity_planner.common.node import Node


def writeRulebookFile(rulebook: List, rulebook_file_location):
    with open(rulebook_file_location, 'w+') as filehandle:
        for item in rulebook:
            filehandle.write('%s\n' % item)


def readRulebookFile(rulebook_file_location: str) -> List:
    rulebook = []
    with open(rulebook_file_location, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            rulebook.append(currentPlace)
    return rulebook


def determineTimeHorizon(scenario: Scenario, max_time_horizon, dt, dt_precision) -> float:
    max_scenario_duration = getScenarioDuration(scenario) * scenario.dt

    if max_scenario_duration == 0.0 or max_scenario_duration > max_time_horizon:
        return roundToGrid(max_time_horizon, dt_precision, dt)
    else:
        return roundToGrid(max_scenario_duration, dt_precision, dt)


def getScenarioDuration(scenario: Scenario) -> int:
    duration = 0
    for obs in scenario.dynamic_obstacles:
        if duration < obs.prediction.trajectory.state_list[-1].time_step:
            duration = obs.prediction.trajectory.state_list[-1].time_step
    return duration


def loadParameterFile(param_files_location: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file
    :param file_name: name of the yaml file
    """
    try:
        with open(param_files_location, 'r') as stream:
            try:
                config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
                return config
            except ruamel.yaml.YAMLError as exc:
                print(exc)
                return None
    except IOError:
        raise ValueError("There is no parameter file in this directory!")


def createOutputFolder(scenario_id: str):
    output_dir = os.path.abspath(os.getcwd() + '/output/' + scenario_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir


def create_reference_path(planning_problem, scenario) -> Union[None, Tuple[CurvilinearCoordinateSystem, List[int]]]:
    # Instantiate a route planner with the scenario and the planning problem
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX)
    route = route_planner.plan_routes().retrieve_best_route_by_orientation()

    if route:
        ref_path = route.reference_path

        # we apply chaikins corner cutting 10 times to smoothen the reference path
        for i in range(0, 10):
            ref_path = chaikins_corner_cutting(ref_path)
        ref_path = resample_polyline(ref_path, 2.0)

        # create curvilinear cosy
        ccosy = CurvilinearCoordinateSystem(ref_path, 25.0, 0.1)
        ccosy.compute_and_set_curvature()
        return ccosy, route.list_ids_lanelets
    else:
        return None

@functools.lru_cache(maxsize=None)
def getTimeStepFromTimeValue(time_value: float, scenario: Scenario) -> int:
    return int(round(time_value / scenario.dt, 0))


@functools.lru_cache(maxsize=None)
def getCurvatureAtStation(ccosy: CurvilinearCoordinateSystem, s: float) -> float:
    idx = next(idx for idx, station in enumerate(ccosy.segments_longitudinal_coordinates()) if station >= s)

    s_first = ccosy.segments_longitudinal_coordinates()[idx - 1]
    s_second = ccosy.segments_longitudinal_coordinates()[idx]

    kappa_first = ccosy.get_curvature()[idx - 1]
    kappa_second = ccosy.get_curvature()[idx]

    return np.interp(s, [s_first, s_second], [kappa_first, kappa_second])


@functools.lru_cache(maxsize=None)
def getOrientationAtStation(ccosy: CurvilinearCoordinateSystem, s: float) -> float:
    ego_tangent_vector = ccosy.tangent(s)
    # range [-pi, pi]
    orientation = math.atan2(ego_tangent_vector[1], ego_tangent_vector[0])
    # range [0, 2 pi]
    return orientation % (2 * math.pi)


@functools.lru_cache(maxsize=None)
def getIntervalOverlap(interval_a: tuple, interval_b: tuple) -> float:
    return max(0.0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))


@functools.lru_cache(maxsize=None)
def createEgoCollisionObject(station: float, ccosy: CurvilinearCoordinateSystem, ego_params: VehicleParameters,
                             extension_left=0.0, extension_right=0.0, extension_top=0.0, extension_bottom=0.0,
                             extension_minkovski=None) -> pycrcc.Shape:
    ego_shape = createEgoShape(station, ccosy, ego_params, extension_left, extension_right, extension_top,
                               extension_bottom)

    if extension_minkovski is None:
        return create_collision_object(ego_shape)
    else:
        return create_collision_object(ego_shape, params={'minkowski_sum_circle': True,
                                                          'minkowski_sum_circle_radius': extension_minkovski,
                                                          'resolution': 4})


@functools.lru_cache(maxsize=None)
def createEgoShape(station: float, ccosy: CurvilinearCoordinateSystem, ego_params: VehicleParameters,
                   extension_left=0.0, extension_right=0.0, extension_top=0.0, extension_bottom=0.0) -> Shape:
    ego_pos = ccosy.convert_to_cartesian_coords(station, 0.0)
    ego_orientation = getOrientationAtStation(ccosy, station)

    ego_shape = Rectangle(ego_params.l + extension_left + extension_right,
                          ego_params.w + extension_bottom + extension_top,
                          ego_pos,
                          ego_orientation)
    return ego_shape


@functools.lru_cache(maxsize=None)
def createObstacleCollisionObject(obstacle: Obstacle, obstacle_state: State) -> Union[pycrcc.Shape, None]:
    return create_collision_object(createObstacleShape(obstacle, obstacle_state))


@functools.lru_cache(maxsize=None)
def createObstacleShape(obstacle: Obstacle, obstacle_state: State) -> Shape:
    return obstacle.obstacle_shape.rotate_translate_local(obstacle_state.position, obstacle_state.orientation)


@functools.lru_cache(maxsize=None)
def getObstacleState(obstacle: Obstacle, time_step=Union[int, None], extrapolate=False, scenario_dt=0.1) -> Union[
    State, None]:
    obst_state = None

    # Obstacle is dynamic
    if obstacle.obstacle_role == ObstacleRole.DYNAMIC:

        initial_time_step = obstacle.initial_state.time_step
        final_time_step = obstacle.prediction.trajectory.final_state.time_step

        # Time step is before initial state
        if time_step < initial_time_step and extrapolate:
            obst_state = obstacle.initial_state  # copy state
            obst_state.time_step = time_step
            ds = scenario_dt * (initial_time_step - time_step) * obst_state.velocity
            obst_state.position[0] -= math.cos(obst_state.orientation) * ds
            obst_state.position[1] -= math.sin(obst_state.orientation) * ds

        # Time step is initial state
        elif time_step == obstacle.initial_state.time_step:
            obst_state = obstacle.initial_state

        # Time step is between initial and final time step
        elif initial_time_step < time_step <= final_time_step:
            obst_state = obstacle.prediction.trajectory.state_at_time_step(time_step)

        elif time_step > final_time_step and extrapolate:
            obst_state = obstacle.prediction.trajectory.final_state  # copy state
            obst_state.time_step = time_step
            ds = scenario_dt * (time_step - final_time_step) * obst_state.velocity
            obst_state.position[0] += math.cos(obst_state.orientation) * ds
            obst_state.position[1] += math.sin(obst_state.orientation) * ds

    # Obstacle is static
    else:
        obst_state = obstacle.initial_state

    return obst_state


@functools.lru_cache(maxsize=None)
def deltaAngles(angle1: float, angle2: float) -> float:
    return math.pi - abs(abs(angle1 - angle2) - math.pi)


@functools.lru_cache(maxsize=None)
def getProjectedObstacleVelocity(station: float, ccosy: CurvilinearCoordinateSystem, obstacle_state: State) -> float:
    orientation_ego = getOrientationAtStation(ccosy, station)
    orientation_obst = obstacle_state.orientation
    delta_orientation = deltaAngles(orientation_ego, orientation_obst)

    return math.cos(delta_orientation) * obstacle_state.velocity


@functools.lru_cache(maxsize=None)
def determineStationOfStopLine(stop_line: StopLine, ccosy: CurvilinearCoordinateSystem) -> Union[None, float]:
    s_of_stop_line = None

    p_intersection = lineIntersection(ccosy.reference_path(), [stop_line.start, stop_line.end])
    if p_intersection is not None:
        s_of_stop_line, _ = ccosy.convert_to_curvilinear_coords(p_intersection[0], p_intersection[1])

    return s_of_stop_line


@functools.lru_cache(maxsize=None)
def determineStationRangeOfLanelet(lanelet: Lanelet, ccosy: CurvilinearCoordinateSystem) -> Union[None, Tuple]:
    # Only when both intersection points exist, we return the tuple

    # Left Boundary
    p_intersection = lineIntersection(ccosy.reference_path(), lanelet.left_vertices)
    if p_intersection is not None:
        s_left, _ = ccosy.convert_to_curvilinear_coords(p_intersection[0], p_intersection[1])
    else:
        return None

    # Right Boundary
    p_intersection = lineIntersection(ccosy.reference_path(), lanelet.right_vertices)
    if p_intersection is not None:
        s_right, _ = ccosy.convert_to_curvilinear_coords(p_intersection[0], p_intersection[1])
    else:
        return None

    return tuple(sorted([s_left, s_right]))


def lineIntersection(points_line1, points_line2) -> Union[None, Tuple]:
    line1 = LineString(points_line1)
    line2 = LineString(points_line2)

    int_pt = line1.intersection(line2)

    if int_pt:
        return int_pt.x, int_pt.y
    else:
        return None


def getRelevantObstacles(obstacles: List[Obstacle], relevant_obstacle_types: Set[ObstacleType]) -> List[Obstacle]:
    relevant_obstacles = []
    for obstacle in obstacles:
        if obstacle.obstacle_type in relevant_obstacle_types:
            relevant_obstacles.append(obstacle)
    return relevant_obstacles


def getTimeStepMap(scenario: Scenario, dt, time_horizon, rounding_precision) -> Dict:
    n_t = int(time_horizon[1] / dt) + 1

    time_step_map = {}
    for time_step in range(n_t + 1):
        time_step_map[round(time_step * dt, rounding_precision)] = int(round(time_step * dt / scenario.dt, 0))
    return time_step_map


def getEgoLaneStationMap(scenario: Scenario, planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem, ds, station_horizon, rounding_precision) -> Dict:
    initial_s = round(ccosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])[0], rounding_precision)

    ego_lane_station_map = {}
    station = initial_s
    i = 0
    while station <= station_horizon[1]:
        i += 1
        ego_pos = ccosy.convert_to_cartesian_coords(station, 0.0)
        ego_lane_station_map[station] = scenario.lanelet_network.find_lanelet_by_position([ego_pos])[0]
        station = round(initial_s + i * ds, rounding_precision)
    return ego_lane_station_map

@functools.lru_cache(maxsize=None)
def getVelocityLimit(traffic_sign_interpreter: TrafficSigInterpreter, lanes: Tuple[int]):
    return traffic_sign_interpreter.speed_limit(frozenset(lanes))


@functools.lru_cache(maxsize=None)
def roundToGrid(x: float, prec=5, grid_base=1.0, offset=0.0) -> float:
    return round(grid_base * round((x-offset)/grid_base) + offset, prec)


def getEgoCollisionBoxes(planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                                 ego_params: VehicleParameters, ds, station_horizon, rounding_precision,
                                 extension_left=0.0, extension_right=0.0, extension_top=0.0, extension_bottom=0.0) -> Dict:
    initial_s = round(ccosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])[0], rounding_precision)

    ego_collision_object_map = {}
    station = initial_s
    i = 0
    while station <= station_horizon[1]:
        i += 1
        ego_collision_object_map[station] = createEgoCollisionObject(station, ccosy, ego_params, extension_left,
                                                                     extension_right, extension_top, extension_bottom)
        station = round(initial_s + i * ds, rounding_precision)

    return ego_collision_object_map


def getEgoShapes(planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                                 ego_params: VehicleParameters, ds, station_horizon, rounding_precision,
                                 extension_left=0.0, extension_right=0.0, extension_top=0.0, extension_bottom=0.0) -> Dict:
    initial_s = round(ccosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])[0], rounding_precision)

    ego_shape_map = {}
    station = initial_s
    i = 0
    while station <= station_horizon[1]:
        i += 1
        ego_shape_map[station] = createEgoShape( station, ccosy, ego_params, extension_left, extension_right, extension_top, extension_bottom)
        station = round(initial_s + i * ds, rounding_precision)

    return ego_shape_map


def getStandardObstacleCollisionBoxes(scenario: Scenario, time_step_map: Dict, dt: float, time_horizon: List, rounding_precision: int) -> Dict:
    n_t = int(time_horizon[1] / dt) + 1

    obstacle_collisionObject_map = {}
    # This is gird time_step size
    for time_step in range(n_t+1):
        # This is scenario time_step size
        sc_time_step = time_step_map.get(round(time_step * dt, rounding_precision))

        for obstacle in scenario.obstacles:
            obstacle_state = getObstacleState(obstacle, sc_time_step)
            if obstacle_state is not None:
                obst_bb = createObstacleCollisionObject(obstacle, obstacle_state)
                obstacle_collisionObject_map[(sc_time_step, obstacle.obstacle_id)] = obst_bb
    return obstacle_collisionObject_map


def getdObstacleRefPathProjection(scenario: Scenario, time_step_map: Dict, dt: float, time_horizon: List, rounding_precision: int, ccosy: CurvilinearCoordinateSystem) -> Dict:
    n_t = int(time_horizon[1] / dt) + 1

    obstacle_ref_path_projections = {}
    # This is gird time_step size
    for time_step in range(n_t+1):
        # This is scenario time_step size
        sc_time_step = time_step_map.get(round(time_step * dt, rounding_precision))

        for obstacle in scenario.obstacles:
            obstacle_state = getObstacleState(obstacle, sc_time_step)
            if obstacle_state is not None:
                try:
                    s_d = ccosy.convert_to_curvilinear_coords(obstacle_state.position[0], obstacle_state.position[1])
                    obstacle_ref_path_projections[(sc_time_step, obstacle.obstacle_id)] = s_d
                except ValueError:
                    obstacle_ref_path_projections[(sc_time_step, obstacle.obstacle_id)] = None
            else:
                obstacle_ref_path_projections[(sc_time_step, obstacle.obstacle_id)] = None
    return obstacle_ref_path_projections

@functools.lru_cache(maxsize=None)
def globalHeuristic(v0, time, time_horizon, dt, action, interpolation_function) -> float:
    area = 0.0
    if v0 > 0.0 and time < time_horizon:
        t = time
        te = 0.0
        while v0 > 0.0 and t < time_horizon:
            v1 = max(0.0, v0 + action * dt)
            area += interpolation_function(v0, v1, dt)
            v0 = v1
            t += dt
            te += dt
    return area


#@functools.lru_cache(maxsize=None)
def isLexicographicallyBetter(l1: tuple, l2: tuple):
    for x, y in zip(l1, l2):
        if math.isclose(x, y, abs_tol=1e-9):
            continue
        elif x < y:
            return True
        else:
            return False

# Print iterations progress
# Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


@functools.lru_cache(maxsize=None)
def violationIntegrationLinearInterpolation(v1: float, v2: float, dt:float) -> float:
    return (v1 + v2) / 2.0 * dt


@functools.lru_cache(maxsize=None)
def violationIntegrationMaxValue(v1: float, v2: float, dt: float) -> float:
    return max(v1, v2) * dt


@functools.lru_cache(maxsize=None)
def violationIntegrationCurrentValue(v1: float, v2: float, dt: float) -> float:
    return v2 * dt
