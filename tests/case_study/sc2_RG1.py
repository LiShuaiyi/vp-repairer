from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver
from commonroad_repair.crrepairer.t_solver.qp_planner import QPPlannerRepair
from commonroad_repair.crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from commonroad_repair.crrepairer.t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_profile, visualize_initial_result

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt

# scenario_id = "DEU_LocationBUpper-1_22_T-1"
# scenario_id = "ZAM_Zip-1_67_T-1"
scenario_id = "DEU_Gar-1_1_T-1"
# scenario_id = "ZAM_Tutorial-1_2_T-1"
file_path = "/home/yuanfei/commonroad/commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" \
            + scenario_id + ".xml"
# file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
#             + scenario_id + ".xml"
file_path = "/home/yuanfei/commonroad/commonroad_repair/scenarios/" \
            + scenario_id + ".xml"

figure_path = "/home/yuanfei/commonroad/commonroad_repair/tests/figures/"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 200
    rule = "R_G1"

    ego_initial = scenario.obstacle_by_id(ego_id)

    ego_initial.prediction.trajectory.state_list = ego_initial.prediction.trajectory.state_list[:21]
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:21]
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)
    repairer = SMTTrajectoryRepairer(rule_abstracter,
                                     ego_initial)
    repaired_traj = repairer.repair()

    ego_vehicle = convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                              ego_initial.initial_state,
                                              repaired_traj)
    ego_initial.prediction.shape = ego_vehicle.prediction.shape
    plot_limits = [-5, 50, -4.5, 3]
    # plot_limits = [-380, -150, 7.5, 17.5]
    target_veh = scenario.obstacle_by_id(repairer.rule_abstracter.other_veh_id)
    # following_veh = scenario.obstacle_by_id(203)
    # visualize_profile(target_veh, following_veh, ego_initial, ego_vehicle)
    for time_step in range(21):
        visualize_initial_result(scenario, ego_initial,
                                 time_step,
                                 target_veh=target_veh,
                                 plot_limits=plot_limits,
                                 # save_path=figure_path,
                                 tv=int(repairer.tv))
        # visualize_repairing_result(scenario,
        #                            ego_vehicle,
        #                            time_step,
        #                            tc=repairer.tc,
        #                            plot_limits=plot_limits,
        #                            target_veh=target_veh,
        #                            ego_initial=ego_initial,
        #                            save_path=figure_path)