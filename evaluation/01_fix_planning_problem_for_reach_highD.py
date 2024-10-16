import numpy as np
import pandas as pd
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.geometry.shape import Rectangle

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import ObstacleType
# import necessary classes from different modules
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from matplotlib import pyplot as plt


def update_goal_state(initial_state: InitialState,
                      initial_trajectory: Trajectory,
                      lanelet_networks: LaneletNetwork) -> GoalRegion:
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    depth = 1
    initial_lanelet_id = lanelet_networks.find_lanelet_by_position([initial_state.position])[0][0]
    lanelet = lanelet_networks.find_lanelet_by_id(initial_lanelet_id)
    i = 0
    while i < depth and len(lanelet.successor) != 0:
        if len(lanelet.successor) > 1:
            lanelet = lanelet_networks.find_lanelet_by_id(lanelet.successor[0])
        else:
            lanelet = lanelet_networks.find_lanelet_by_id(lanelet.successor[0])
        i += 1
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(2, 2, np.array([lanelet.center_vertices[-1][0], lanelet.center_vertices[-1][1]])),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region

# Load the CSV file, skipping the first line
csv_file_path = 'highD_evaluation_rg1_3_filtered.csv'
file_path = "/home/liny/Documents/commonroad/highD-repair/"
target_path = "/home/liny/Documents/commonroad/highd_scenarios_2024_repaired/"

# Skip the first line and load the rest of the CSV
df = pd.read_csv(csv_file_path, skiprows=1, header=None)

author = 'Yuanfei Lin'
affiliation = 'Technical University of Munich, Germany'
source = 'highD'
tags = {Tag.CRITICAL, Tag.INTERSTATE}

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    scenario_id = row[0]
    scenario, planning_problem_set = CommonRoadFileReader(
                        file_path + scenario_id + ".xml"
                    ).open(lanelet_assignment=True)
    planning_problem = list(
        planning_problem_set.planning_problem_dict.values()
    )[0]
    ego_id = int(list(row)[1])
    print(scenario_id, ego_id, row[2])
    ego_initial = scenario.obstacle_by_id(ego_id)
    if ego_initial.obstacle_type != ObstacleType.CAR:
        continue
    # remove obstacles
    for obs in scenario.dynamic_obstacles:
        if obs.obstacle_type != ObstacleType.CAR:
            scenario.remove_obstacle(obs)

    planning_problem.initial_state = ego_initial.initial_state
    planning_problem.goal = update_goal_state(ego_initial.initial_state,
                                              ego_initial.prediction.trajectory,
                                              scenario.lanelet_network)
    #
    # rnd = MPRenderer()
    # planning_problem.draw(rnd)
    # scenario.draw(rnd)
    # rnd.render()
    # plt.show()

    route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()
    visualize_route(route, scenario, planning_problem)
    if route:
        file_name = target_path + str(scenario_id).replace('T-1', 'T-' + str(ego_id)) + '.xml'
        scenario.scenario_id = str(scenario_id).replace('T-1', 'T-' + str(ego_id))
        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
        fw.write_to_file(file_name, OverwriteExistingFile.ALWAYS)
    else:
        print('FAIL! No route found for scenario: ', scenario_id)