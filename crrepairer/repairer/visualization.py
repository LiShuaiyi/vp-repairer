# standard imports
from typing import List

# third party
import matplotlib.pyplot as plt

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer


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
            "draw_trajectory": False}, "occupancy": {
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
                                   "facecolor": "green",
                                   "edgecolor": "green"}
                               }},
                           "dynamic_obstacle":
                               {"vehicle_shape": {
                                   "occupancy": {
                                       "shape": {"rectangle": {
                                           "facecolor": "green",
                                           "edgecolor": "green"}
                                       }}}}}))

    # render scenario and ego vehicle
    rnd.render()
    #
    # pos_x_repaired = []
    # pos_y_repaired = []
    # for state in ego_repaired.prediction.trajectory.state_list:
    #     pos_x_repaired.append(state.position[0])
    #     pos_y_repaired.append(state.position[1])
    #
    # # visualize optimal trajectory
    # rnd.ax.plot(pos_x_repaired, pos_y_repaired, color='g', marker='.', markersize=5, zorder=22, linewidth=1.5,
    #             label='repaired trajectory')
    # pos_x_initial = [ego_initial.initial_state.position[0]]
    # pos_y_initial = [ego_initial.initial_state.position[1]]
    # for state in ego_initial.prediction.trajectory.state_list:
    #     pos_x_initial.append(state.position[0])
    #     pos_y_initial.append(state.position[1])
    # rnd.ax.plot(pos_x_initial, pos_y_initial, color='k', marker='x', markersize=5, zorder=21, linewidth=1.5,
    #             label='initial trajectory')

    # show plot
    plt.show(block=True)

    # save as .png file
    if save_path is not None:
        plt.savefig(f"{save_path}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300,
                    bbox_inches='tight')
