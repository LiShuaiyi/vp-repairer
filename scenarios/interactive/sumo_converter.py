"""
Based on the initial trajectory of the ego vehicle and the initial states of all surrounding vehicles,
we use SUMO as the prediction model to update the occupancy of other vehicles.
https://commonroad.in.tum.de/sumo-interface
"""

import matplotlib as mpl

mpl.use('TkAgg')
from simulation.simulations import load_sumo_configuration, create_video_for_simulation, \
    SimulationOption, check_trajectories
from simulation.utility import save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
from commonroad.scenario.obstacle import DynamicObstacle
import copy
import os
import pickle
from enum import unique, Enum
from math import sin, cos
from typing import Tuple, Dict, Optional, Union

import numpy as np
from sumocr.sumo_config.default import DefaultConfig
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.visualization.video import create_video
from sumocr.sumo_docker.interface.docker_interface import SumoInterface

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.solution import Solution
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt

file_path_sumo = "/home/yuanfei/commonroad/highD-dataset/highD-cr-sumo/"
file_path_init = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/"


def simulate_with_ego_trajectory(interactive_scenario_path: str,
                                 output_folder_path: str = None,
                                 create_video: bool = False,
                                 use_sumo_manager: bool = False,
                                 create_ego_obstacle: bool = False,
                                 ego_vehicle: DynamicObstacle = None) -> Tuple[
    Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles from the planned trajectories as the ego vehicles
    :return: Tuple of the simulated scenario, planning problem set, and list of ego vehicles
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    # remove ego vehicle and update the planning problem
    scenario.remove_obstacle(scenario.obstacle_by_id(ego_vehicle.obstacle_id))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_vehicle.initial_state.yaw_rate = 0
    ego_vehicle.initial_state.slip_angle = 0
    planning_problem.initial_state = ego_vehicle.initial_state
    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario
    # todo: check the time horizon
    scenario_with_planner, ego_vehicles = simulate_scenario(SimulationOption.MOTION_PLANNER, conf,
                                                            scenario_wrapper,
                                                            interactive_scenario_path,
                                                            ego_vehicle,
                                                            num_of_steps=ego_vehicle.prediction.final_time_step,
                                                            # conf.simulation_steps,
                                                            planning_problem_set=planning_problem_set,
                                                            use_sumo_manager=use_sumo_manager)
    scenario_with_planner.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_planner, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.MOTION_PLANNER.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            obstacle_ego.obstacle_id = ego_vehicle.obstacle_id
            scenario_with_planner.add_objects(obstacle_ego)

    return scenario_with_planner, planning_problem_set, ego_vehicles


def simulate_scenario(mode: SimulationOption,
                      conf: DefaultConfig,
                      scenario_wrapper: ScenarioWrapper,
                      scenario_path: str,
                      ego_vehicle_cr: DynamicObstacle,
                      num_of_steps: int = None,
                      planning_problem_set: PlanningProblemSet = None,
                      solution: Solution = None,
                      use_sumo_manager: bool = False) -> Tuple[Scenario, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with specified mode

    :param mode: 0 = without ego, 1 = with plugged in planner, 2 = with solution trajectory
    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param scenario_path: path to the interactive scenario folder
    :param num_of_steps: number of steps to simulate
    :param planning_problem_set: planning problem set of the scenario
    :param solution: solution to the planning problem
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """

    if num_of_steps is None:
        num_of_steps = conf.simulation_steps

    sumo_interface = None
    if use_sumo_manager:
        sumo_interface = SumoInterface(use_docker=True)
        sumo_sim = sumo_interface.start_simulator()

        sumo_sim.send_sumo_scenario(conf.scenario_name,
                                    scenario_path)
    else:
        sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, planning_problem_set=planning_problem_set)

    if mode is SimulationOption.WITHOUT_EGO:
        # simulation without ego vehicle
        for step in range(num_of_steps):
            # set to dummy simulation
            sumo_sim.dummy_ego_simulation = True
            sumo_sim.simulate_step()

    elif mode is SimulationOption.MOTION_PLANNER:
        # simulation with plugged in planner

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles
            for step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles

                # retrieve the CommonRoad scenario at the current time step, e.g. as an input for a prediction module
                current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
                for idx, ego_vehicle in enumerate(ego_vehicles.values()):
                    # retrieve the current state of the ego vehicle
                    state_current_ego = ego_vehicle.current_state

                    # ====== plug in your motion planner here
                    # example motion planner which decelerates to full stop
                    next_state = copy.deepcopy(ego_vehicle_cr.state_at_time(sumo_sim.current_time_step + 1))
                    # ====== end of motion planner

                    # update the ego vehicle with new trajectory with only 1 state for the current step
                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using sumo_docker
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        run_simulation()

    elif mode is SimulationOption.SOLUTION:
        # simulation with given solution trajectory

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles

            for time_step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles
                for idx_ego, ego_vehicle in enumerate(ego_vehicles.values()):
                    # update the ego vehicles with solution trajectories
                    trajectory_solution = solution.planning_problem_solutions[idx_ego].trajectory
                    next_state = copy.deepcopy(trajectory_solution.state_list[time_step])

                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using SUMO Manager
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        check_trajectories(solution, planning_problem_set, conf)
        run_simulation()

    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()
    if use_sumo_manager:
        sumo_interface.stop_simulator()

    ego_vechicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                         ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    return simulated_scenario, ego_vechicles


def main():
    name_scenario_init = "DEU_LocationELower-24_18_T-1"
    name_scenario_intr = name_scenario_init.replace('_T-', '_I-')
    path_scenario_init = os.path.join(file_path_init, name_scenario_init)
    path_scenario_intr = os.path.join(file_path_sumo, name_scenario_intr)

    simulation_with_planner = True

    path_solutions = "/home/yuanfei/commonroad/commonroad-interactive-scenarios/outputs/solutions/"
    # solution = CommonRoadSolutionReader.open(os.path.join(path_solutions, name_solution + ".xml"))

    # path to store output GIFs
    path_videos = "/home/yuanfei/commonroad/commonroad-interactive-scenarios/outputs/videos/"

    # path to store simulated scenarios
    path_scenarios_simulated = "/home/yuanfei/commonroad/commonroad-interactive-scenarios/outputs/simulated_scenarios/"

    vehicle_type = VehicleType.BMW_320i
    vehicle_model = VehicleModel.PM
    cost_function = CostFunction.JB1

    ego_id = 8
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario_init + '.xml').open(lanelet_assignment=True)
    ego_vehicle = scenario.obstacle_by_id(ego_id)
    # simulation with plugged-in motion planner
    scenario_with_planner, pps, ego_vehicles = simulate_with_ego_trajectory(
        interactive_scenario_path=path_scenario_intr,
        output_folder_path=path_videos,
        create_ego_obstacle=True,
        ego_vehicle=ego_vehicle,
        create_video=True)
    for time_step in range(ego_vehicle.prediction.final_time_step):
        rnd = MPRenderer(figsize=(40, 10))
        scenario_with_planner.lanelet_network.draw(rnd)
        for obs in scenario_with_planner.dynamic_obstacles:
            obs.draw(rnd, draw_params=ParamServer(
                {"time_begin": time_step,
                 "occupancy": {
                     "draw_occupancies": 1,
                     "shape": {"rectangle": {
                         "facecolor": "black",
                         "edgecolor": "black"}
                     }},
                 "trajectory": {
                     "draw_trajectory": False},
                 "dynamic_obstacle":
                     {"vehicle_shape": {
                         "occupancy": {
                             "shape": {"rectangle": {
                                 "facecolor": "black",
                                 "edgecolor": "black"}
                             }}}, 'show_label': False}}))
        for obs in scenario.dynamic_obstacles:
            obs.draw(rnd, draw_params=ParamServer(
                {"time_begin": time_step,
                 "occupancy": {
                     "draw_occupancies": 1,
                     "shape": {"rectangle": {
                         "facecolor": "blue",
                         "edgecolor": "blue"}
                     }},
                 "trajectory": {
                     "draw_trajectory": False},
                 "dynamic_obstacle":
                     {"vehicle_shape": {
                         "occupancy": {
                             "shape": {"rectangle": {
                                 "facecolor": "blue",
                                 "edgecolor": "blue"}
                             }}}, 'show_label': True}}))

        rnd.render()
        plt.title(str(time_step))
        plt.show()

    if scenario_with_planner:
        # write simulated scenario to file
        fw = CommonRoadFileWriter(scenario_with_planner, pps)
        fw.write_to_file(f"{path_scenarios_simulated}{name_scenario_intr}.xml", OverwriteExistingFile.ALWAYS)

        # saves trajectory to solution file
        # save_solution(scenario_with_planner, pps, ego_vehicles,
        #               vehicle_type,
        #               vehicle_model,
        #               cost_function,
        #               path_solutions, overwrite=True)


if __name__ == '__main__':
    main()
