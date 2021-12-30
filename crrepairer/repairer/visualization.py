# standard imports
from typing import List

# third party
import matplotlib.pyplot as plt
import numpy as np

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer

from commonroad_repair.crrepairer.t_solver.utils import calculate_safe_distance

def visualize_profile(target_vehicle:DynamicObstacle,
                      ego_initial: DynamicObstacle,
                      ego_repaired: DynamicObstacle):
    time_list = []
    target_pos_list = []
    ego_ini_pos_list = []
    ego_rep_pos_list = []
    safe_dis_ini_list = []
    safe_dis_rep_list = []
    ego_ini_vel_list = []
    ego_rep_vel_list = []
    for time_step in range(ego_initial.prediction.final_time_step + 1):
        time_list.append(time_step)
        target_pos_list.append(target_vehicle.state_at_time(time_step).position)
        ego_ini_pos_list.append(ego_initial.state_at_time(time_step).position)
        ego_rep_pos_list.append(ego_repaired.state_at_time(time_step).position)
        safe_dis_ini_list.append(target_pos_list[time_step][0]-
                                 target_vehicle.obstacle_shape.length/2-
                                 calculate_safe_distance(ego_initial.state_at_time(time_step).velocity,
                                                         target_vehicle.state_at_time(time_step).velocity,
                                                         -10.5,
                                                         -10.,
                                                         0.4) - ego_initial.obstacle_shape.length/2)
        safe_dis_rep_list.append(target_pos_list[time_step][0] - target_vehicle.obstacle_shape.length/2-
                                 calculate_safe_distance(ego_repaired.state_at_time(time_step).velocity,
                                                         target_vehicle.state_at_time(time_step).velocity,
                                                         -10.5,
                                                         -10.,
                                                         0.4) - ego_repaired.obstacle_shape.length/2)
        ego_ini_vel_list.append(ego_initial.state_at_time(time_step).velocity)
        ego_rep_vel_list.append(ego_repaired.state_at_time(time_step).velocity)
    plt.plot(time_list, np.array(ego_ini_pos_list)[:, 0], color='blue', linewidth=0.8,)
    plt.plot(time_list, np.array(target_pos_list)[:, 0], color='black', linewidth=0.8,)
    plt.plot(time_list, np.array(ego_rep_pos_list)[:, 0], color='green', linewidth=0.8,)
    plt.plot(time_list, safe_dis_ini_list, color='red', linewidth=1.,)
    plt.plot(time_list, safe_dis_rep_list, color='yellow', linewidth=1.,)
    plt.show()
    plt.plot(time_list, ego_ini_vel_list, color='blue', linewidth=1.,)
    plt.plot(time_list, ego_rep_vel_list, color='green', linewidth=1.,)
    plt.show()

def visualize_repairing_result(scenario: Scenario,
                               ego_initial: DynamicObstacle,
                               ego_repaired: DynamicObstacle,
                               timestep: int,
                               save_path: str = None,
                               plot_limits = None):
    """
    Function to visualize complete planning result from the reactive planner for a given time step
    :param scenario: CommonRoad scenario object
    :param: planning_problem: CommonRoad PlanningProblem object
    :param ego: Ego vehicle as CommonRoad DynamicObstacle object
    :param pos: positions of planned trajectory [(nx2) np.ndarray]
    :param timestep: current time step of scenario to plot
    :param traj_set: List of sampled trajectories (optional)
    :param ref_path: Reference path for planner as polyline [(nx2) np.ndarray] (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    ego_position = ego_repaired.state_at_time(timestep).position
    # plot_limits = [ego_position[0]-5, ego_position[0]+70, 7.5, 17.5]
    # create renderer object
    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    # visualize scenario
    scenario.draw(
        rnd,
        draw_params=ParamServer({"time_begin": timestep, "trajectory": {
            "draw_trajectory": True}, "occupancy": {
            "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': False}})
    )
    # visualize planning problem
    # planning_problem.draw(rnd, draw_params={"initial_state": {"state": {"draw_arrow": False}}})
    # visualize ego vehicle

    ego_initial.draw(rnd,
                     draw_params=ParamServer(
                         {"time_begin": timestep, "trajectory": {
                             "draw_trajectory": False},
                          "occupancy": {
                              "draw_occupancies": 1,
                              "shape": {"rectangle": {
                                  "facecolor": "#0065bd",
                                  "edgecolor": "#0065bd"}
                              }},
                          "dynamic_obstacle":
                              {"vehicle_shape": {
                                  "occupancy": {
                                      "shape": {"rectangle": {
                                          "facecolor": "#0065bd",
                                          "edgecolor": "#0065bd"}
                                      }}}}}))
    ego_repaired.draw(rnd,
                      draw_params=ParamServer(
                          {"time_begin": timestep, "trajectory": {
                              "draw_trajectory": False},
                           "occupancy": {
                               "draw_occupancies": 1,
                               "shape": {"rectangle": {
                                   "facecolor": "#a2ad00",
                                   "edgecolor": "#a2ad00"}
                               }},
                           "dynamic_obstacle":
                               {"vehicle_shape": {
                                   "occupancy": {
                                       "shape": {"rectangle": {
                                           "facecolor": "#a2ad00",
                                           "edgecolor": "#a2ad00"}
                                       }}}}}))

    # render scenario and ego vehicle
    rnd.render()
    #
    pos_x_repaired = []
    pos_y_repaired = []
    for state in ego_repaired.prediction.trajectory.state_list:
        pos_x_repaired.append(state.position[0])
        pos_y_repaired.append(state.position[1])

    # visualize optimal trajectory
    rnd.ax.plot(pos_x_repaired[timestep:], pos_y_repaired[timestep:], color='#a2ad00', marker='.', markersize=7.5, zorder=22, linewidth=1.5,
                label='repaired trajectory')
    pos_x_initial = [ego_initial.initial_state.position[0]]
    pos_y_initial = [ego_initial.initial_state.position[1]]
    for state in ego_initial.prediction.trajectory.state_list:
        pos_x_initial.append(state.position[0])
        pos_y_initial.append(state.position[1])
    rnd.ax.plot(pos_x_initial[timestep:], pos_y_initial[timestep:], color='#0065bd', marker='x', markersize=7.5, zorder=21, linewidth=1.5,
                label='initial trajectory')
    plt.xticks([])
    plt.yticks([])
    # show plot
    plt.show(block=True)

    # save as .png file
    if save_path is not None:
        plt.savefig(f"{save_path}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300,
                    bbox_inches='tight')
