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
import commonroad_dc.pycrcc as pycrcc

def visualize_profile(target_vehicle: DynamicObstacle,
                      follow_vehicle: DynamicObstacle,
                      ego_initial: DynamicObstacle,
                      ego_repaired: DynamicObstacle):
    plt.figure(figsize=(20, 8))
    time_list = []
    target_pos_list = []
    ego_ini_pos_list = []
    ego_rep_pos_list = []
    safe_dis_ini_list = []
    safe_dis_rep_list = []
    ego_ini_vel_list = []
    ego_rep_vel_list = []
    follow_pos_list = []
    for time_step in range(ego_initial.prediction.final_time_step + 1):
        time_list.append(time_step)
        target_pos_list.append(target_vehicle.state_at_time(time_step).position)
        follow_pos_list.append(follow_vehicle.state_at_time(time_step).position[0] +
                               follow_vehicle.obstacle_shape.length/2)
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
    plt.plot(time_list, np.array(ego_ini_pos_list)[:, 0], color='#0065bd', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    # plt.plot(time_list, np.array(target_pos_list)[:, 0], color='black', linewidth=0.8,)
    plt.plot(time_list[13:], np.array(ego_rep_pos_list)[13:, 0], color='#a2ad00', marker='.',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list, safe_dis_ini_list, color='red', linewidth=1.,)
    plt.plot(time_list, follow_pos_list,color='red', linewidth=1.,)
    plt.plot(time_list, safe_dis_rep_list, color='yellow', linewidth=1.,)
    plt.xlim((0, 20))
    plt.xticks(range(0, 20))
    plt.ylim((np.array(ego_ini_pos_list)[0, 0], 40))
    plt.show()
    plt.plot(time_list, ego_ini_vel_list, color='#0065bd', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list[13:], ego_rep_vel_list[13:], color='#a2ad00', marker='.',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.show()


def visualize_v_profile(
                      ego_initial: DynamicObstacle,
                      ego_repaired: DynamicObstacle,
                      tc,
                      tv):
    # plt.figure(figsize=(20, 8))
    time_list = []
    ego_ini_vel_list = []
    ego_rep_vel_list = []
    plt.axhline(y=13.88)
    for time_step in range(ego_initial.prediction.final_time_step):
        time_list.append(time_step)
        ego_ini_vel_list.append(ego_initial.state_at_time(time_step).velocity)
        ego_rep_vel_list.append(ego_repaired.state_at_time(time_step).velocity)
    plt.plot(time_list[:tv+1], ego_ini_vel_list[:tv+1], color='#0065bd', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list[tv:], ego_ini_vel_list[tv:], color='red', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list[tc:], ego_rep_vel_list[tc:], color='#a2ad00', marker='.',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.xticks(range(0, 20))
    plt.yticks(range(5, 15, 5))
    plt.show()

def visualize_a_profile(dt,
                        ego_initial: DynamicObstacle,
                        ego_repaired: DynamicObstacle,
                        time_start,
                        time_end,
                        tc,
                        tv):
    # plt.figure(figsize=(20, 8))
    time_list = []
    ego_ini_acc_list = []
    ego_rep_acc_list = []
    for time_step in range(time_start, time_end):
        time_list.append(time_step-time_start)
        if hasattr(ego_initial.state_at_time(time_step), 'acceleration'):
            ego_ini_acc_list.append(ego_initial.state_at_time(time_step).acceleration)
        else:
            ego_ini_acc_list.append((ego_initial.state_at_time(time_step+1).velocity -
                                     ego_initial.state_at_time(time_step).velocity)/dt)
        ego_rep_acc_list.append(ego_repaired.state_at_time(time_step).acceleration)
    plt.plot(time_list[:tv-time_start+1], ego_ini_acc_list[:tv-time_start+1], color='#0065bd', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list[tv-time_start:], ego_ini_acc_list[tv-time_start:], color='red', marker='x',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.plot(time_list[tc-time_start:], ego_rep_acc_list[tc-time_start:], color='#a2ad00', marker='.',
             markersize=7.5, zorder=21, linewidth=1.5)
    plt.xticks(range(0, 20))

    plt.show()

def visualize_repairing_result(scenario: Scenario,
                               ego_repaired: DynamicObstacle,
                               timestep: int,
                               save_path: str = None,
                               plot_limits = None,
                               end_time=None,
                               tc=None,
                               target_veh=None,
                               ego_initial=None):
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
    # plot_limits = [ego_position[0]-5, ego_position[0]+70, 7.5, 17.5]
    # create renderer object
    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    # visualize scenario
    scenario.draw(
        rnd,
        draw_params=ParamServer({"time_begin": timestep, "time_end": end_time, "trajectory": {
            "draw_trajectory": True},
                                 "lanelet": {"fill_lanelet": False},
                                 "occupancy": {
            "draw_occupancies": 0},
                                 "dynamic_obstacle":
                               {
                                   "vehicle_shape": {
                                   "occupancy": {
                                       "draw_occupancies": 0,
                                       "shape": {"rectangle": {
                                           "facecolor": "black",
                                           "edgecolor": "black"}
                                       }}}}})
    )
    # visualize planning problem
    # planning_problem.draw(rnd, draw_params={"initial_state": {"state": {"draw_arrow": False}}})
    if target_veh:
        ego_state_list = [ego_repaired.initial_state] + ego_repaired.prediction.trajectory.state_list
        safe_distance = calculate_safe_distance(ego_state_list[timestep].velocity,
                                                target_veh.state_at_time(timestep).velocity,
                                                -10.5, -10.0, 0.4)
        box_center = target_veh.state_at_time(timestep).position - [target_veh.obstacle_shape.length / 2, 0] - \
                     [safe_distance / 2, 0]
        # -preceding_vehicle.obstacle_shape.width/2
        # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
        obb = pycrcc.RectOBB(safe_distance / 2, 3.5 / 2, 0.0, box_center[0], box_center[1])
        obb.draw(rnd, draw_params={"opacity": 0.2,
                                   "facecolor": "red",
                                   'edgecolor': "red"})
    if timestep>=tc:
        ego_repaired.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": timestep, "time_end": end_time,"trajectory": {
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
    else:
        ego_repaired.draw(rnd,
                          draw_params=ParamServer(
                              {"time_begin": timestep,"time_end": end_time, "trajectory": {
                                  "draw_trajectory": False},
                               "occupancy": {
                                   "draw_occupancies": 1,
                                   "shape": {"rectangle": {
                                       "facecolor": "#0065bd",
                                       "edgecolor": "#0065bd"}
                                   }},
                               "dynamic_obstacle":
                                   {
                                       "vehicle_shape": {
                                       "occupancy": {
                                           "draw_occupancies": 0,
                                           "shape": {"rectangle": {
                                               "facecolor": "#0065bd",
                                               "edgecolor": "#0065bd"}
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
    if timestep>=tc:
        rnd.ax.plot(pos_x_repaired[timestep:end_time], pos_y_repaired[timestep:end_time], color='#a2ad00', marker='.',
                    markersize=7.5, zorder=22, linewidth=1.5,
                    label='repaired trajectory')
    else:
        rnd.ax.plot(pos_x_repaired[timestep:end_time], pos_y_repaired[timestep:end_time], color='#0065bd',
                marker='x', markersize=7.5, zorder=22, linewidth=1.5,
                label='repaired trajectory')

    plt.xticks([])
    plt.yticks([])
    # plt.title(timestep)

    # show the rule-violating region

    # show plot
    # plt.show(block=True)

    # save as .svg file
    if save_path is not None:
        if timestep<10:
            plt.savefig(f"{save_path}/{0}{timestep}.svg", format='svg', dpi=300,
                        bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/{timestep}.svg", format='svg', dpi=300,
                        bbox_inches='tight')

def visualize_initial_result(scenario: Scenario,
                             ego_initial: DynamicObstacle,
                             timestep: int,
                             target_veh=None,
                             save_path: str = None,
                             plot_limits=None,
                             end_time=None,
                             tv=None):
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
    # plot_limits = [ego_position[0]-5, ego_position[0]+70, 7.5, 17.5]
    # create renderer object
    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    # visualize scenario
    scenario.draw(
        rnd,
        draw_params=ParamServer({"time_begin": timestep, "time_end": end_time, "trajectory": {
            "draw_trajectory": True},
                                 "lanelet": {"fill_lanelet": False},
                                 "occupancy": {
                                     "draw_occupancies": 0},
                                 "dynamic_obstacle":
                                     {
                                         "vehicle_shape": {
                                             "occupancy": {
                                                 "draw_occupancies": 0,
                                                 "shape": {"rectangle": {
                                                     "facecolor": "black",
                                                     "edgecolor": "black"}
                                                 }}}}})
    )
    # visualize planning problem
    # planning_problem.draw(rnd, draw_params={"initial_state": {"state": {"draw_arrow": False}}})
    if target_veh:
        ego_state_list = [ego_initial.initial_state] + ego_initial.prediction.trajectory.state_list[:end_time]
        safe_distance = calculate_safe_distance(ego_state_list[timestep].velocity,
                                                target_veh.state_at_time(timestep).velocity,
                                                -10.5, -10.0, 0.4)
        box_center = target_veh.state_at_time(timestep).position - [target_veh.obstacle_shape.length / 2, 0] - \
                     [safe_distance / 2, 0]
        # -preceding_vehicle.obstacle_shape.width/2
        # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
        obb = pycrcc.RectOBB(safe_distance / 2, 3.5 / 2, 0.0, box_center[0], box_center[1])
        obb.draw(rnd, draw_params={"opacity": 0.2,
                                   "facecolor": "red",
                                   'edgecolor': "red"})
    if timestep<tv:
        ego_initial.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": timestep, "time_end": end_time,"trajectory": {
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
    else:
        ego_initial.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": timestep, "time_end": end_time,"trajectory": {
                                 "draw_trajectory": False},
                              "occupancy": {
                                  "draw_occupancies": 1,
                                  "shape": {"rectangle": {
                                      "facecolor": "red",
                                      "edgecolor": "red"}
                                  }},
                              "dynamic_obstacle":
                                  {"vehicle_shape": {
                                      "occupancy": {
                                          "shape": {"rectangle": {
                                              "facecolor": "red",
                                              "edgecolor": "red"}
                                          }}}}}))
    # render scenario and ego vehicle
    rnd.render()

    pos_x_initial = [ego_initial.initial_state.position[0]]
    pos_y_initial = [ego_initial.initial_state.position[1]]
    for state in ego_initial.prediction.trajectory.state_list:
        pos_x_initial.append(state.position[0])
        pos_y_initial.append(state.position[1])

    if timestep>=tv:
        rnd.ax.plot(pos_x_initial[timestep:end_time], pos_y_initial[timestep:end_time], color='red', marker='x',
                    markersize=7.5, zorder=35, linewidth=1.5,
                    label='repaired trajectory')
    else:
        rnd.ax.plot(pos_x_initial[timestep:end_time], pos_y_initial[timestep:end_time], color='#0065bd',
                marker='x', markersize=7.5, zorder=35, linewidth=1.5,
                label='repaired trajectory')
    plt.xticks([])
    plt.yticks([])
    # plt.title(timestep)

    # show the rule-violating region

    # show plot
    # plt.show(block=True)

    # save as .svg file
    if save_path is not None:
        if timestep<10:
            plt.savefig(f"{save_path}/{0}{timestep}.svg", format='svg', dpi=300,
                        bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/{timestep}.svg", format='svg', dpi=300,
                        bbox_inches='tight')