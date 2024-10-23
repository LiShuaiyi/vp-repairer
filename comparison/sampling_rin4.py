__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
from copy import deepcopy
import logging

import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib.lines import lineStyles
from networkx.algorithms.bipartite.basic import color

from crrepairer.utils.visualization import TUMColor
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_dc.geometry.util import (
    chaikins_corner_cutting,
    compute_orientation_from_polyline,
    compute_pathlength_from_polyline,
    resample_polyline,
)
import commonroad_dc.pycrccosy as pycrccosy
# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.general import update_goal_state
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.logger import initialize_logger

# cr monitor
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.evaluation import (
    get_evaluation_config,
    create_ego_vehicle_param,
)
from crmonitor.common.world import World, get_world_config


def extrapolate_resample_polyline(
        polyline: np.ndarray, step: float = 2.0
) -> np.ndarray:
    """
    Extrapolates polyline for resampling.
    """
    # extend start point
    p = np.poly1d(np.polyfit(polyline[:2, 0], polyline[:2, 1], 1))

    x = 2 * polyline[0, 0] - polyline[1, 0]
    a = np.array([[x, p(x)]])
    polyline = np.concatenate((a, polyline), axis=0)

    # extend end point
    # extrapolate final point
    p = np.poly1d(np.polyfit(polyline[-2:, 0], polyline[-2:, 1], 1))

    # x = 2 * polyline[-1, 0] - polyline[-2, 0]
    # this extension helps the ego vehicle can drive to the end of the lane.
    x = polyline[-1, 0] + 99 * (polyline[-1, 0] - polyline[-2, 0])
    a = np.array([[x, p(x)]])
    polyline_extend = resample_polyline(np.concatenate((polyline[-1, np.newaxis], a), axis=0), step=20.0)
    polyline_origin = resample_polyline(polyline, step=step)

    return np.concatenate((polyline_origin, polyline_extend[1:, :]), axis=0)

def smoothing_reference_path(
            reference_path: np.ndarray,
            smooth_factor=None,
            weight_coefficient=None) -> np.ndarray:
    """
    generates a smooth reference path using splprep
    """
    # generate a smooth reference path
    transposed_reference_path = reference_path.T
    # how to generate index okay
    okay = np.where(
        np.abs(np.diff(transposed_reference_path[0]))
        + np.abs(np.diff(transposed_reference_path[1]))
        > 0
    )
    xp = np.r_[transposed_reference_path[0][okay], transposed_reference_path[0][-1]]
    yp = np.r_[transposed_reference_path[1][okay], transposed_reference_path[1][-1]]

    curvature = pycrccosy.Util.compute_curvature(np.array([xp, yp]).T)
    # set weights for interpolation:
    # see details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
    weights = np.exp(-weight_coefficient * (abs(curvature) - np.min(abs(curvature))))
    # B spline interpolation
    tck, u = splprep([xp, yp], s=smooth_factor, w=weights)
    u_new = np.linspace(u.min(), u.max(), 2000)
    x_new, y_new = splev(u_new, tck, der=0)
    ref_path_smooth = np.array([x_new, y_new]).transpose()
    return ref_path_smooth



def main(
    config: ReactivePlannerConfiguration
) -> None:
    # initialize and get logger
    initialize_logger(config)
    logger = logging.getLogger("RP_LOGGER")

    # *************************************
    # Initialize Planner
    # *************************************

    # initialize rule monitor
    world_config = get_world_config()
    world_config["scenario"] = "intersection"
    world = World.create_from_scenario(config.scenario, config=world_config)
    config.planning.dt = config.scenario.dt
    if monitor_ego:= world.vehicle_by_id(config.planning.ego_id):
        monitor_ego.vehicle_param = create_ego_vehicle_param(
            get_evaluation_config().get("ego_vehicle_param"), world.dt
        )
        ego_initial = config.scenario.obstacle_by_id(
            config.planning.ego_id
        )
        config.scenario.remove_obstacle(ego_initial)
        config.planning_problem.initial_state = ego_initial.initial_state
        config.planning_problem.goal = update_goal_state(
            ego_initial.prediction.trajectory
        )
        config.vehicle.length = monitor_ego.shape.length
        config.vehicle.width = monitor_ego.shape.width

        config.planning.time_steps_computation = ego_initial.prediction.final_time_step - ego_initial.prediction.initial_time_step + 1
    else:
        raise ValueError(f"ego vehicle with id {config.planning.ego_id} not found in scenario")
    rule_evaluators = []
    for rule in config.planning.rules:
        rule_evaluators.append(RuleEvaluator.create_from_config(world,
                                                                config.planning.ego_id,
                                                                rule=rule,
                                                                use_boolean=True))

    # initialize reactive planner
    planner = ReactivePlanner(config)

    planner.rule_evaluators = rule_evaluators
    planner.world = world
    # from commonroad.visualization.mp_renderer import MPRenderer
    # rnd = MPRenderer()
    # config.scenario.draw(rnd)
    # rnd.render()

    # # run route planner and add reference path to config
    # route_planner = RoutePlanner(config.scenario.lanelet_network, config.planning_problem)
    # route = route_planner.plan_routes().retrieve_shortetest_route_with_least_lane_changes()
    # # plt.plot(route.reference_path[:, 0], route.reference_path[:, 1], color=TUMColor.TUMblue.value, zorder = 25)
    #
    #
    # reference_path = extrapolate_resample_polyline(route.reference_path)
    # reference_path = smoothing_reference_path(reference_path, smooth_factor=1.5, weight_coefficient=5)
    # reference_path = resample_polyline(reference_path, step=2)
    # plt.plot(route.reference_path[:, 0], route.reference_path[:, 1], color=TUMColor.TUMyellow.value, zorder = 25)
    # plt.plot(np.array(monitor_ego.ref_path_lane.clcs.reference_path())[:, 0],
    #          np.array(monitor_ego.ref_path_lane.clcs.reference_path())[:, 1],color=TUMColor.TUMgray.value, zorder = 25)
    # plt.show()
    # set reference path for curvilinear coordinate system
    planner.set_reference_path(reference_path=np.array(monitor_ego.ref_path_lane.clcs.reference_path()))
    # planner.set_reference_path(route.reference_path)
    # **************************
    # Run Planning
    # **************************
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)
    SAMPLING_ITERATION_IN_PLANNER = True

    count = 1
    while count < 2:
        count += 1
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity,
                                         desired_velocity=0)
            planner.set_v_sampling_parameters(-0.01, 2)

            if SAMPLING_ITERATION_IN_PLANNER:
                optimal = planner.plan()
            else:
                optimal = None
                i = 1
                while optimal is None and i <= planner.sampling_level:
                    optimal = planner.plan(i)

            if not optimal:
                break

            # record state and input
            planner.record_state_and_input(optimal[0].state_list[1])

            # reset planner state for re-planning
            planner.reset(initial_state_cart=planner.record_state_list[-1],
                          initial_state_curv=(optimal[2][1], optimal[3][1]),
                          collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

            # visualization: create ego Vehicle for planned trajectory and store sampled trajectory set
            if config.debug.show_plots or config.debug.save_plots:
                ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list,
                                                                              config.planning.ego_id)
                sampled_trajectory_bundle = None
                if config.debug.draw_traj_set:
                    sampled_trajectory_bundle = deepcopy(planner.stored_trajectories)
        else:
            # simulate scenario one step forward with planned trajectory
            sampled_trajectory_bundle = None

            # continue on optimal trajectory
            temp = current_count % config.planning.replanning_frequency

            # record state and input
            planner.record_state_and_input(optimal[0].state_list[1 + temp])

            # reset planner state for re-planning
            planner.reset(initial_state_cart=planner.record_state_list[-1],
                          initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                          collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)
        #
        # print(f"current time step: {current_count}")
        # # visualize the current time step of the simulation
        # for state in ego_vehicle.prediction.trajectory.state_list:
        #     print(state.time_step, state.velocity, state.acceleration)
        traj_cr = ego_vehicle.prediction.trajectory.state_list
        # plot velocity and acc
        plt.figure(figsize=(6, 1.5))
        plt.plot([state.velocity for state in traj_cr], linewidth=3, marker='D', linestyle="--", zorder=20,
                 markersize=4, color=TUMColor.TUMyellow.value)
        step = 3
        for i in range(0, len(sampled_trajectory_bundle), step):
            plt.plot(sampled_trajectory_bundle[i].cartesian.v, linewidth=0.2, color=TUMColor.TUMgray)
        plt.xticks(range(0, 20, 10))
        plt.xlim(0, 20)

        plt.ylim([-0.2, 2.2])
        plt.plot([state.acceleration for state in traj_cr])
        plt.legend(['velocity', 'acceleration'])
        plt.show()
        from commonroad.visualization.mp_renderer import MPRenderer

        def plot_scenario(crscenario, traj_cr, traj_set, plot_limits, time_step):
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            rnd = MPRenderer(ax=ax, plot_limits=plot_limits)

            # visualize scenario
            rnd.draw_params.time_begin = time_step
            rnd.draw_params.trajectory.draw_trajectory = False
            rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
            rnd.draw_params.occupancy.draw_occupancies = False
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = (
                False
            )
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
                TUMColor.TUMgray.value
            )
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = (
                TUMColor.TUMblack.value
            )
            rnd.draw_params.dynamic_obstacle.draw_shape = True
            rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
            rnd.draw_params.dynamic_obstacle.trajectory.line_width = 0.3
            rnd.draw_params.dynamic_obstacle.draw_signals = False
            rnd.draw_params.dynamic_obstacle.draw_icon = True
            # rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
            # rnd.draw_params.traffic_sign.draw_traffic_signs = True
            rnd.draw_params.lanelet_network.lanelet.stop_line_color = (
                TUMColor.TUMblack.value
            )
            rnd.draw_params.lanelet_network.lanelet.draw_stop_line = True
            crscenario.draw(rnd)

            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = False
            rnd.draw_params.dynamic_obstacle.draw_shape = True
            rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
            rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
                TUMColor.TUMblue.value
            )

            # render scenario and ego vehicle
            rnd.render()
            pos_x_replanned = []
            pos_y_replanned = []
            for state in traj_cr:
                pos_x_replanned.append(state.position[0])
                pos_y_replanned.append(state.position[1])

            rnd.ax.plot(
                pos_x_replanned[time_step:],
                pos_y_replanned[time_step:],
                color=TUMColor.TUMyellow.value,
                marker='D',
                linestyle="--",
                markersize=4,
                zorder=22,
                linewidth=3,
                label="replanned trajectory",
            )
            for i in range(0, len(traj_set), step):

                plt.plot(traj_set[i].cartesian.x, traj_set[i].cartesian.y,
                         color=TUMColor.TUMgray, zorder=20, linewidth=0.2, alpha=1.0)
            plt.show()

        plot_limits = [40, 69, -45, -17]

        for i in range(20 + 1):
            plot_scenario(config.scenario, traj_cr, sampled_trajectory_bundle, plot_limits, i)
        # if config.debug.show_plots or config.debug.save_plots:
        #     visualize_planner_at_timestep(scenario=config.scenario, planning_problem=config.planning_problem,
        #                                   ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
        #                                   ref_path=planner.reference_path, timestep=current_count, config=config)

    # make gif
    # make_gif(config, range(0, planner.record_state_list[-1].time_step))

    # **************************
    # Evaluate results
    # **************************
    evaluate = True
    if evaluate:
        cr_solution, feasibility_list = run_evaluation(planner.config, planner.record_state_list,
                                                       planner.record_input_list)


# *************************************
# Run planning
# *************************************
if __name__ == "__main__":
    filename = "DEU_AachenBendplatz-1_162280_T-2299.xml"

    # Build config object
    rp_config = ReactivePlannerConfiguration()
    rp_config.general.path_scenarios = "../scenarios/"
    rp_config.general.set_path_scenario(filename)
    rp_config.update()
    rp_config.debug.draw_traj_set = True
    rp_config.debug.show_plots = True
    rp_config.planning.ego_id = 10179
    rp_config.planning.rules = ["R_IN4"]
    # rp_config.sampling.longitudinal_mode = "stopping"
    main(
        config=rp_config
    )
