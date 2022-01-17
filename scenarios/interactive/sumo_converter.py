"""
Based on the initial trajectory of the ego vehicle and the initial states of all surrounding vehicles,
we use SUMO as the prediction model to update the occupancy of other vehicles.
https://commonroad.in.tum.de/sumo-interface
"""
import os
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.scenario.obstacle import DynamicObstacle

from simulation.simulations import simulate_without_ego, simulate_with_solution, simulate_with_planner

file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/"


def simulate_with_ego_sumo(interactive_scenario_path,
                           output_folder_path,
                           ego_vehicle: DynamicObstacle,
                           create_video=True):
    """
        Simulates an interactive scenario with a plugged in ego's trajectory
    """
    pass

def main():
    simulation_with_solution = True
    scenario_id = "DEU_LocationFUpper-60_38_T-1"
    path_scenario = os.path.join(file_path, scenario_id)
    scenario, planning_problem_set = CommonRoadFileReader(file_path + scenario_id + ".xml"). \
        open(lanelet_assignment=True)

    # path to store output GIFs
    path_videos = "../outputs/videos/"

    # path to store simulated scenarios
    path_scenarios_simulated = "/home/yuanfei/commonroad/highD-dataset/highD-cr-sumo/"

    vehicle_type = VehicleType.BMW_320i
    vehicle_model = VehicleModel.KS
    scenario_with_planner, pps, ego_vehicles = simulate_with_planner(interactive_scenario_path=path_scenario,
                                                                     output_folder_path=path_videos,
                                                                     create_video=True)