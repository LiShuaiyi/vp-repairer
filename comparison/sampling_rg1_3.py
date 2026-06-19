__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"

# standard imports
from copy import deepcopy
import logging

from matplotlib.lines import lineStyles
from networkx.algorithms.bipartite.basic import color

from crrepairer.utils.visualization import TUMColor
import matplotlib.pyplot as plt

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.general import update_goal_state
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.logger import initialize_logger

# cr monitor
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.evaluation import (
    get_evaluation_config,
    create_ego_vehicle_param,
)

file_path = "../scenarios/"
ego_id = 11


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
    world = World.create_from_scenario(config.scenario)
    config.planning.dt = config.scenario.dt
    if monitor_ego := world.vehicle_by_id(config.planning.ego_id):
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

    # run route planner and add reference path to config
    route_planner = RoutePlanner(config.scenario.lanelet_network, config.planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()

    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)

    # **************************
    # Run Planning
    # **************************
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)

    SAMPLING_ITERATION_IN_PLANNER = True

    i = 1
    count = 1
    while count < 2:
        count += 1
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
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

        # print(f"current time step: {current_count}")
        # # visualize the current time step of the simulation
        # for state in ego_vehicle.prediction.trajectory.state_list:
        #     print(state.time_step, state.velocity, state.acceleration)
        traj_cr = ego_vehicle.prediction.trajectory.state_list
        # plot velocity and acc
        plt.figure(figsize=(6, 1.5))
        plt.plot([state.velocity for state in traj_cr], linewidth=3, marker='D', linestyle="--", zorder=20,
                 markersize=4, color=TUMColor.TUMyellow.value)
        step = 2
        for i in range(0, len(sampled_trajectory_bundle), step):
            plt.plot(sampled_trajectory_bundle[i].cartesian.v, linewidth=0.2, color=TUMColor.TUMgray)
        plt.xticks(range(0, 20, 10))
        plt.xlim(0, 20)

        plt.ylim(20, 45)
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

        plot_limits = [158, 390, -32, -18.4]

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
    scenario_id = "DEU_LocationDLower-8_154_T-1"

    # Build config object
    rp_config = ReactivePlannerConfiguration()
    rp_config.general.path_scenarios = file_path
    rp_config.general.set_path_scenario(scenario_id + ".xml")

    rp_config.update()
    rp_config.debug.draw_traj_set = True
    rp_config.debug.show_plots = True
    rp_config.planning.ego_id = ego_id
    rp_config.planning.rules = ["R_G3", "R_G1"]
    main(
        config=rp_config
    )
