# standard imports
from enum import Enum
from typing import List, Union, Optional
from shapely.geometry.polygon import Polygon

# third party
import matplotlib.pyplot as plt
import numpy as np

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_qp_planner.utils import calculate_safe_distance

from crmonitor.common.world import World

from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer


class TUMColor(List, Enum):
    TUMblue = [0, 101 / 255, 189 / 255]
    TUMgreen = [162 / 255, 173 / 255, 0]
    TUMgray = [156 / 255, 157 / 255, 159 / 255]
    TUMdarkgray = [88 / 255, 88 / 255, 99 / 255]
    TUMorange = [227 / 255, 114 / 255, 34 / 255]
    TUMdarkblue = [0, 82 / 255, 147 / 255]
    TUMwhite = [1, 1, 1]
    TUMblack = [0, 0, 0]
    TUMlightgray = [217 / 255, 218 / 255, 219 / 255]


def visualize_repaired_result(
        config: RepairerConfiguration,
        ego_initial: DynamicObstacle,
        ego_repaired: DynamicObstacle,
        repairer: SMTTrajectoryRepairer,
        plot_velocity: bool = True,
        plot_acceleration: bool = True,
):
    if plot_velocity:
        visualize_v_profile(
            ego_initial,
            ego_repaired,
            config.repair.t_0,
            config.repair.t_f,
            repairer.tc,
            repairer.tv
        )

    if plot_acceleration:
        visualize_a_profile(
            config.scenario.dt,
            ego_initial,
            ego_repaired,
            config.repair.t_0,
            config.repair.t_f,
            repairer.tc,
            repairer.tv
        )
    for time_step in range(config.repair.t_0, config.repair.t_f):
        if config.debug.save_plots:
            path_fig = config.general.path_figures
        else:
            path_fig = None
        visualize_scenario(
            config.scenario,
            ego_initial,
            ego_repaired,
            time_step,  # Assuming time_end is the current time_step for visualization
            path_fig,
            config.debug.plot_limits,
            config.repair.t_f,
            repairer.tc,
            repairer.tv,
            repairer.target_vehicle,
        )

def plot_velocity_acceleration_profile(
    time_list: List[int],
    velocity_list: List[float],
    color: Union[List, str],
    label: str,
    linestyle: str = '-',
    marker: str = "."
):
    plt.plot(
        time_list,
        velocity_list,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.0,
        label=label
    )


def visualize_v_profile(
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_start: int,
    time_end: int,
    tc: int,
    tv: int
):
    plt.figure(figsize=(6, 2.4))
    time_list = [time_step - time_start for time_step in range(time_start, time_end)]
    ego_ini_vel = [ego_initial.state_at_time(t).velocity for t in range(time_start, time_end)]
    ego_rep_vel = [ego_repaired.state_at_time(t).velocity for t in range(time_start, time_end)]

    plt.axhline(y=0, linestyle="--", linewidth=1.0)
    plot_velocity_acceleration_profile(time_list, ego_ini_vel, TUMColor.TUMblue.value, "Initial")
    plot_velocity_acceleration_profile(time_list[tv - time_start:], ego_ini_vel[tv - time_start:], "red", "Violation")
    plot_velocity_acceleration_profile(time_list[tc - time_start:], ego_rep_vel[tc - time_start:], TUMColor.TUMgreen.value, "Repaired")

    plt.xticks(range(0, time_end - time_start, 10))
    plt.xlim([0, time_end - time_start])
    plt.xlabel("Time step")
    plt.ylabel("Velocity")
    plt.show()


def calculate_acceleration(obstacle: DynamicObstacle, time_step: int, dt: float) -> float:
    if hasattr(obstacle.state_at_time(time_step), "acceleration"):
        return obstacle.state_at_time(time_step).acceleration
    else:
        return (obstacle.state_at_time(time_step + 1).velocity - obstacle.state_at_time(time_step).velocity) / dt


def visualize_a_profile(
    dt: float,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_start: int,
    time_end: int,
    tc: int,
    tv: int
):
    time_list = [t - time_start for t in range(time_start, time_end)]
    ego_ini_acc = [calculate_acceleration(ego_initial, t, dt) for t in range(time_start, time_end)]
    ego_rep_acc = [calculate_acceleration(ego_repaired, t, dt) for t in range(time_start, time_end)]

    plt.figure(figsize=(6, 2.4))
    plot_velocity_acceleration_profile(time_list, ego_ini_acc, TUMColor.TUMblue.value, "Initial")
    plot_velocity_acceleration_profile(time_list[tv - time_start:], ego_ini_acc[tv - time_start:], "red", "Violation")
    plot_velocity_acceleration_profile(time_list[tc - time_start:], ego_rep_acc[tc - time_start:], TUMColor.TUMgreen.value, "Repaired")

    plt.xticks(range(0, time_end - time_start, 5))
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    plt.show()


def visualize_scenario(
    scenario: Scenario,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_step: int,
    save_path: str = None,
    plot_limits=None,
    end_time=None,
    tc=None,
    tv=None,
    target_veh=None,
):
    """
    Function to visualize the repairing result given time step
    :param scenario: CommonRoad scenario object
    :param ego_initial: initially-planned trajectory
    :param ego_repaired: repaired ego vehicle
    :param time_step: current time step
    :param save_path: Path to save plot as .png/.svg (optional)
    :param plot_limits: plot limits of the scenario
    :param end_time: ending time step
    :param tc: time-to-comply
    :param tv: time-to-violation
    :param target_veh: target vehicle for repairing
    :param world: world state
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 10))
    rnd_0 = MPRenderer(ax=ax0, plot_limits=plot_limits)
    rnd_1 = MPRenderer(ax=ax1, plot_limits=plot_limits)

    # visualize scenario
    for rnd in (rnd_0, rnd_1):
        rnd.draw_params.time_begin = time_step
        if end_time:
            rnd.draw_params.time_end = end_time
        rnd.draw_params.trajectory.draw_trajectory = False
        rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
        rnd.draw_params.occupancy.draw_occupancies = False
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = (
            False
        )
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.dynamic_obstacle.draw_shape = True
        rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
        rnd.draw_params.dynamic_obstacle.draw_signals = False
        rnd.draw_params.dynamic_obstacle.draw_icon = True
        # rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
        # rnd.draw_params.traffic_sign.draw_traffic_signs = True
        rnd.draw_params.lanelet_network.lanelet.stop_line_color = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.lanelet_network.lanelet.draw_stop_line = True
        scenario.draw(rnd)

    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = True
    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = True
    rnd_0.draw_params.dynamic_obstacle.draw_shape = True
    rnd_0.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False
    rnd_1.draw_params.dynamic_obstacle.draw_shape = True
    rnd_1.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    if time_step >= tv:
        ego_color = "red"
    else:
        ego_color = TUMColor.TUMblue.value
    ego_mark = "x"

    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd_0.draw_params.dynamic_obstacle.occupancy.draw_occupancies = True
    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMblue.value
    )
    ego_initial.draw(rnd_0)


    if target_veh:
        rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "red"
        rnd_0.draw_params.dynamic_obstacle.occupancy.shape.facecolor = "red"

        target_veh.draw(rnd_0)

    # render scenario and ego vehicle
    rnd_0.render()

    pos_x_initial = [ego_initial.initial_state.position[0]]
    pos_y_initial = [ego_initial.initial_state.position[1]]
    for state in ego_initial.prediction.trajectory.state_list:
        pos_x_initial.append(state.position[0])
        pos_y_initial.append(state.position[1])

    if time_step >= tv:
        rnd_0.ax.plot(
            pos_x_initial[
                time_step: end_time
            ],
            pos_y_initial[
                time_step: end_time
            ],
            color=ego_color,
            marker=ego_mark,
            markersize=7.5,
            zorder=35,
            linewidth=1.5,
            label="initial trajectory",
        )
    else:
        rnd_0.ax.plot(
            pos_x_initial[
                time_step: end_time
            ],
            pos_y_initial[
                time_step: end_time
            ],
            color=ego_color,
            marker=ego_mark,
            markersize=7.5,
            zorder=35,
            linewidth=1.5,
            label="initial trajectory",
        )

    if time_step >= tc:
        ego_color = TUMColor.TUMgreen.value
        ego_mark = "."
    else:
        ego_color = TUMColor.TUMblue.value
        ego_mark = "x"

    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd_1.draw_params.dynamic_obstacle.occupancy.draw_occupancies = True


    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMgreen.value
    )
    ego_repaired.draw(rnd_1)

    if target_veh:
        rnd_1.draw_params.dynamic_obstacle.occupancy.shape.facecolor = "red"
        rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "red"
        target_veh.draw(rnd_1)

    # render scenario and ego vehicle
    rnd_1.render()

    pos_x_repaired = []
    pos_y_repaired = []
    for state in ego_repaired.prediction.trajectory.state_list:
        pos_x_repaired.append(state.position[0])
        pos_y_repaired.append(state.position[1])

    # visualize optimal trajectory
    rnd_1.ax.plot(
        pos_x_repaired[
            time_step: end_time
        ],
        pos_y_repaired[
            time_step: end_time
        ],
        color=ego_color,
        marker=ego_mark,
        markersize=7.5,
        zorder=22,
        linewidth=1.5,
        label="repaired trajectory",
    )

    # if target_veh:
    #     ego_veh_state_ini = ego_initial.state_at_time(time_step)
    #     ego_veh_state_rep = ego_repaired.state_at_time(time_step)
    #     tar_veh_state = target_veh.state_at_time(time_step)
    #     tar_veh_lane = world.vehicle_by_id(target_veh.obstacle_id).get_lane(time_step)
    #     unsafe_poly_ini = compute_unsafe_polygon(
    #         ego_veh_state_ini, tar_veh_state, target_veh, tar_veh_lane
    #     )
    #     rnd_0.ax.fill(
    #         *unsafe_poly_ini.exterior.xy,
    #         zorder=30,
    #         alpha=0.2,
    #         facecolor=TUMcolor.TUMorange.value,
    #         edgecolor=None,
    #     )
    #     unsafe_poly_rep = compute_unsafe_polygon(
    #         ego_veh_state_rep, tar_veh_state, target_veh, tar_veh_lane
    #     )
    #     rnd_1.ax.fill(
    #         *unsafe_poly_rep.exterior.xy,
    #         zorder=30,
    #         alpha=0.2,
    #         facecolor=TUMcolor.TUMorange.value,
    #         edgecolor=None,
    #     )

    ax0.set_title("Initial configuration.")
    ax1.set_title("Repaired configuration.")

    # show plot
    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        if plot_limits:
            ax.set_xlim([plot_limits[0], plot_limits[1]])
            ax.set_ylim([plot_limits[2], plot_limits[3]])

    # save as .svg file
    if save_path is not None:
        if time_step < 10:
            plt.savefig(
                f"{save_path}/{str(scenario.scenario_id)}_{0}{time_step}.svg",
                format="svg",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{save_path}/{str(scenario.scenario_id)}_{time_step}.svg",
                format="svg",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        plt.show(block=True)
    # After you're done with a figure
    plt.close(fig)


def compute_unsafe_polygon(ego_veh_state, tar_veh_state, target_veh, tar_veh_lane):
    safe_distance = calculate_safe_distance(
        ego_veh_state.velocity, tar_veh_state.velocity, -10.5, -10.0, 0.4
    )
    tar_pos_rear_CART = [
        tar_veh_state.position[0] - target_veh.obstacle_shape.length / 2,
        tar_veh_state.position[1],
    ]
    tar_pos_rear_CVLN = tar_veh_lane.clcs.convert_to_curvilinear_coords(
        tar_pos_rear_CART[0], tar_pos_rear_CART[1]
    )
    safe_pos_CVLN = tar_pos_rear_CVLN - [safe_distance, 0.0]
    safe_pos_CART = tar_veh_lane.clcs.convert_to_cartesian_coords(
        safe_pos_CVLN[0], safe_pos_CVLN[1]
    )

    # left vertices
    tar_pos_rear_left_CART = tar_veh_lane.clcs_left.convert_to_cartesian_coords(
        tar_pos_rear_CVLN[0], 0.0
    )
    safe_pos_left_CART = tar_veh_lane.clcs_left.convert_to_cartesian_coords(
        safe_pos_CVLN[0], 0.0
    )
    ref_left = np.vstack(tar_veh_lane.clcs_left.reference_path())
    vertices_left = ref_left[
        (ref_left[:, 0] > safe_pos_left_CART[0])
        & (ref_left[:, 0] < tar_pos_rear_left_CART[0]),
        :,
    ]
    vertices_left = np.concatenate(
        ([safe_pos_left_CART], vertices_left, [tar_pos_rear_left_CART])
    )

    # right vertices
    tar_pos_rear_right_CART = tar_veh_lane.clcs_right.convert_to_cartesian_coords(
        tar_pos_rear_CVLN[0], 0.0
    )
    safe_pos_right_CART = tar_veh_lane.clcs_right.convert_to_cartesian_coords(
        safe_pos_CVLN[0], 0.0
    )
    ref_right = np.vstack(tar_veh_lane.clcs_right.reference_path())
    vertices_right = ref_right[
        (ref_right[:, 0] > safe_pos_right_CART[0])
        & (ref_right[:, 0] < tar_pos_rear_right_CART[0]),
        :,
    ]
    vertices_right = np.concatenate(
        ([safe_pos_right_CART], vertices_right, [tar_pos_rear_right_CART])
    )

    # the polygon vertices
    vertices_total = np.concatenate(
        (
            [safe_pos_CART],
            vertices_left,
            [tar_pos_rear_CART],
            np.flip(vertices_right, 0),
            [safe_pos_CART],
        )
    ).tolist()
    return Polygon(vertices_total)
