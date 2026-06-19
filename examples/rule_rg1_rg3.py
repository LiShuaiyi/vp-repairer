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
    scenario_id = "DEU_LocationDLower-8_154_T-1"

    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()
    config.repair.rules = ["R_G1", "R_G3"]
    config.repair.ego_id = 11
    config.debug.show_plots = True
    config.repair.planner = 1
    config.repair.constraint_mode = 2
    config.repair.use_mpr = False

    ego_initial = retrieve_ego_vehicle(config)

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
        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )
            if config.debug.show_plots:
                # ============= Visualization =============
                # visualize_v_profile_tc_all(repairer, ego_initial, ego_repaired, config.repair.t_0, config.repair.t_f,
                #                            figsize=(6, 1.5), ylim=[20, 45], velocity_limit=43)
                visualize_repaired_result(config, ego_initial, ego_repaired, repairer)

                # config.scenario.remove_obstacle(config.scenario.obstacle_by_id(ego_initial.obstacle_id))
                # for i in range(ego_initial.prediction.trajectory.final_state.time_step + 1):

                #     visualize_scenario_once(config.scenario,
                #                             ego_initial,
                #                             ego_repaired,
                #                             i,  # Assuming time_end is the current time_step for visualization
                #                             './img/rg13',
                #                             config.debug.plot_limits,
                #                             config.repair.t_f,
                #                             repairer.tc,
                #                             repairer.tv,
                #                             None,
                #                             traffic_rule_monitor.world,
                #                             flag_repair=True,
                #                             background_file='rg13')