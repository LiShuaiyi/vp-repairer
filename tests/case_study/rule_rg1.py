from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.visualization import (
    visualize_repairing_result,
    visualize_a_profile,
    visualize_v_profile,
)
from crrepairer.utils.configuration import RepairerConfiguration

from commonroad.prediction.prediction import Trajectory
import math


if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_Gar-1_1_T-1"

    # Build configuration object
    config = RepairerConfiguration.load(f"../../config/{scenario_id}.yaml", scenario_id)
    config.update()

    ego_id = 200
    rule = ["R_G1"]
    N = 21
    ego_initial = config.scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory = Trajectory(
        1, ego_initial.prediction.trajectory.state_list[:N]
    )
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config.scenario, ego_id, rule[0])
    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(
            traffic_rule_monitor, ego_initial, config
        )
        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )

            # ============= Visualization =============
            plot_limits = [-5, 50, -4.5, 3]
            target_veh = config.scenario.obstacle_by_id(traffic_rule_monitor.other_id)
            visualize_v_profile(
                ego_initial,
                ego_repaired,
                time_start=0,
                time_end=N,
                tc=repairer.tc,
                tv=repairer.tv,
            )
            visualize_a_profile(
                config.scenario.dt,
                ego_initial,
                ego_repaired,
                time_start=0,
                time_end=N,
                tc=repairer.tc,
                tv=repairer.tv,
            )
            for time_step in range(N):
                visualize_repairing_result(
                    config.scenario,
                    ego_initial,
                    ego_repaired,
                    time_step,
                    tc=repairer.tc,
                    tv=repairer.tv,
                    plot_limits=plot_limits,
                    target_veh=target_veh,
                    world=traffic_rule_monitor.world,
                )  # , save_path=figure_path)
