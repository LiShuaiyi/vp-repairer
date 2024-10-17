from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result, visualize_scenario_once, visualize_v_profile_tc_all
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import InitialState

import math


if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_LocationALower-13_241_T-20"
    config = RepairerConfiguration()
    config.general.path_scenarios = "/home/liny/Documents/commonroad/highd_scenarios_2024_repaired/"
    config.general.set_path_scenario(scenario_id)

    config.update()
    config.repair.rules = ["R_G1", "R_G3"]
    config.repair.ego_id = 20
    config.repair.use_mpr_derivative = True
    config.repair.use_mpr = True
    config.debug.show_plots = True
    config.repair.planner = 2
    config.repair.constraint_mode = 2

    # from commonroad.visualization.mp_renderer import MPRenderer
    # rnd = MPRenderer()
    # rnd.draw_params.dynamic_obstacle.show_label = True
    # config.scenario.draw(rnd)
    # config.planning_problem.draw(rnd)
    # rnd.render()
    # from matplotlib.pyplot import show
    # show()
    ego_initial = retrieve_ego_vehicle(config)
    print(ego_initial.obstacle_shape.length, "Length")
    for i in range(ego_initial.prediction.trajectory.final_state.time_step):
        ego_initial.state_at_time(i).acceleration = (
            ego_initial.state_at_time(i + 1).velocity
            - ego_initial.state_at_time(i).velocity
        ) / config.scenario.dt

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repairer.t_solver.tc_object.config.vehicle.cartesian.j_x_min = -5
        repairer.t_solver.tc_object.config.vehicle.cartesian.j_x_max = 5
        repairer.t_solver.tc_object.config.vehicle.cartesian.j_y_min = -5
        repairer.t_solver.tc_object.config.vehicle.cartesian.j_y_max = 5

        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )
            if config.debug.show_plots:
                # ============= Visualization =============
                visualize_v_profile_tc_all(repairer, ego_initial, ego_repaired, config.repair.t_0, config.repair.t_f,
                                           figsize=(6, 1.5), ylim=[20, 45], velocity_limit=43)
                # visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
                # visualize_scenario_once(config.scenario,
                #                         ego_initial,
                #                         ego_repaired,
                #                         repairer.tc,  # Assuming time_end is the current time_step for visualization
                #                         None,
                #                         config.debug.plot_limits,
                #                         config.repair.t_f,
                #                         repairer.tc,
                #                         repairer.tv,
                #                         repairer.target_vehicle,
                #                         traffic_rule_monitor.world,
                #                         flag_repair=True)