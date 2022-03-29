from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from commonroad_repair.crrepairer.repairer.visualization import (visualize_repairing_result,
                                                                 visualize_a_profile,
                                                                 visualize_v_profile)

from commonroad.common.file_reader import CommonRoadFileReader

scenario_id = "DEU_Gar-1_1_T-1"
file_path = "../../scenarios/" \
            + scenario_id + ".xml"
figure_path = "./figures"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 200
    rule = "R_G1"
    N = 21
    ego_initial = scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory.state_list = ego_initial.prediction.trajectory.state_list[:N]
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)

    repairer = SMTTrajectoryRepairer(rule_abstracter,
                                     ego_initial)
    repaired_traj = repairer.repair()

    ego_repaired = repairer.convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                                       ego_initial.initial_state,
                                                       repaired_traj)
    plot_limits = [-5, 50, -4.5, 3]
    # plot_limits = [-380, -150, 7.5, 17.5]
    target_veh = scenario.obstacle_by_id(repairer.rule_abstracter.other_veh_id)
    # following_veh = scenario.obstacle_by_id(203)
    # visualize_profile(target_veh, following_veh, ego_initial,   ego_vehicle)
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
                                   world_state=rule_abstracter.world_state)  #, save_path=figure_path)