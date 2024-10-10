from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import math

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    # scenario_id = "DEU_AachenBendplatz-10011_1180_T-99"
    # scenario_id = "DEU_AachenFrankenburg-10294_223180_T-3199"
    # scenario_id = "DEU_AachenBendplatz-10075_71780_T-1799"
    scenario_id = "DEU_AachenBendplatz-10039_10500_T-519"
    scenario_id = "DEU_AachenBendplatz-10046_16640_T-659"
    # Build configuration object
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.intersection_type = "dataset"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10011

    # config.repair.rules = ["R_IN4"]
    # config.repair.ego_id = 10294

    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10075

    # config.repair.rules = ["R_IN4"]
    # config.repair.ego_id = 10039

    config.repair.rules = ["R_IN4"]
    config.repair.ego_id = 10046
    # ego vehicle does not have state at time step 0-3

    config.repair.N_r = 20

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
    # Retrieve the ego vehicle
    ego_initial = retrieve_ego_vehicle(config)

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )
            if config.debug.show_plots:
                # ============= Visualization =============
                visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
