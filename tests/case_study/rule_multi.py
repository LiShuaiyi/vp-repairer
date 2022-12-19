from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.visualization import (visualize_repairing_result,
                                                                   visualize_a_profile,
                                                                   visualize_v_profile)

from commonroad.common.file_reader import CommonRoadFileReader
import math

scenario_id = "DEU_Gar-1_1_T-1"
file_path = "../../scenarios/" \
            + scenario_id + ".xml"
figure_path = "./figures"

flag_visualization = True

if __name__ == '__main__':
    # ========== Scenario and Configuration =========
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 200
    rule = ["R_G1", "R_G2", "R_G3"]
    N = 21
    ego_initial = scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory.state_list = ego_initial.prediction.trajectory.state_list[:N]
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(scenario,
                                          ego_id, rule)
    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor,
                                         planning_problem,
                                         ego_initial)
        repaired_traj = repairer.repair()
        if repaired_traj is not None and flag_visualization:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                                                ego_initial.initial_state,
                                                                repaired_traj)

            # ============= Visualization =============
            plot_limits = [-5, 50, -4.5, 3]
            target_veh = scenario.obstacle_by_id(traffic_rule_monitor.other_id)
            visualize_v_profile(ego_initial, ego_repaired, time_start=0, time_end=N,
                                tc=repairer.tc, tv=repairer.tv)
            visualize_a_profile(scenario.dt, ego_initial, ego_repaired, time_start=0,
                                time_end=N, tc=repairer.tc, tv=repairer.tv)
            for time_step in range(N):
                visualize_repairing_result(scenario,
                                           ego_initial,
                                           ego_repaired,
                                           time_step,
                                           tc=repairer.tc,
                                           tv=repairer.tv,
                                           plot_limits=plot_limits,
                                           target_veh=target_veh,
                                           world=traffic_rule_monitor.world)  #, save_path=figure_path)
