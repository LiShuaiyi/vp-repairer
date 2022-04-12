from commonroad_repair.crrepairer.smt.monitor_wrapper import STLRuleMonitor
from commonroad_repair.crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from commonroad_repair.crrepairer.repairer.visualization import (visualize_repairing_result,
                                                                 visualize_a_profile,
                                                                 visualize_v_profile)

from commonroad.common.file_reader import CommonRoadFileReader

import math

scenario_id = "ZAM_Zip-1_56_T-1"
file_path = "../../../commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" + scenario_id + ".xml"
figure_path = "./figures"

flag_visualization = False

if __name__ == '__main__':
    # ========== Scenario and Configuration =========
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 2
    rule = "R_G2"

    ego_initial = scenario.obstacle_by_id(ego_id)
    initial_time_step = 50
    final_time_step = initial_time_step + 20
    # # change the time horizon
    for veh in scenario.obstacles:
        for state in veh.prediction.trajectory.state_list:
            state.time_step -= initial_time_step
        veh.initial_state = veh.prediction.trajectory.state_list[initial_time_step-1]
        veh.prediction.trajectory.state_list = veh.prediction.trajectory.state_list[initial_time_step:
                                                                                    final_time_step]
        veh.prediction.occupancy_set = veh.prediction.occupancy_set[initial_time_step:final_time_step]
        veh.prediction.final_time_step = 20

    for i in range(ego_initial.prediction.trajectory.final_state.time_step):
        ego_initial.state_at_time(i).acceleration = (ego_initial.state_at_time(i + 1).velocity -
                                                     ego_initial.state_at_time(i).velocity) / scenario.dt

    # ========== Traffic Rule Monitor =========
    rule_monitor = STLRuleMonitor(scenario,
                                  planning_problem,
                                  ego_id, rule)
    # ========== Trajectory Repairing =========
    if rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(rule_monitor,
                                         ego_initial)
        repaired_traj = repairer.repair()
        plot_limits = [-50, 10, 3.5, 11.2]
        if repaired_traj is not None and flag_visualization:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                                                ego_initial.initial_state,
                                                                repaired_traj)
            # ============= Visualization =============
            visualize_v_profile(ego_initial, ego_repaired, time_start=initial_time_step,
                                time_end=final_time_step, tc=repairer.tc, tv=repairer.tv)
            visualize_a_profile(scenario.dt, ego_initial, ego_repaired, time_start=initial_time_step,
                                time_end=final_time_step, tc=repairer.tc, tv=repairer.tv)
            for time_step in range(initial_time_step, final_time_step):
                visualize_repairing_result(scenario,
                                           ego_initial,
                                           ego_repaired,
                                           time_step,
                                           end_time=final_time_step,
                                           tc=repairer.tc,
                                           tv=repairer.tv,
                                           plot_limits=plot_limits,
                                           target_veh=None,
                                           world_state=rule_monitor.world_state)  # , save_path=figure_path)
