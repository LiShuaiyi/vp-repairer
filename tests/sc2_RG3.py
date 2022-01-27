from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair
from repairer.smt_repairer import SMTTrajectoryRepairer
from t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_v_profile,\
    visualize_initial_result

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt
import math

scenario_id = "DEU_Muc-4_2_T-1"

file_path = "../../commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" + scenario_id + ".xml"

figure_path = "/home/yuanfei/commonroad/commonroad_repair/tests/figures/"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 202
    rule = "R_G3"

    # time_step = 0
    # rnd = MPRenderer(figsize=(40, 10))
    # scenario.draw(
    #     rnd,
    #     draw_params=ParamServer({"time_begin": time_step, "trajectory": {
    #              "draw_trajectory": False}, "occupancy": {
    #         "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': True}})
    # )
    # rnd.render()
    # plt.title(str(time_step))
    # plt.show()
    ego_initial = scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory.state_list = ego_initial.prediction.trajectory.state_list[:21]
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:21]
    final_time_step = 21
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)
    repairer = SMTTrajectoryRepairer(rule_abstracter,
                                     ego_initial)
    repaired_traj = repairer.repair()
    if repaired_traj is not None:
        ego_vehicle = convert_traj_to_ego_vehicle(ego_initial.obstacle_shape,
                                                  ego_initial.initial_state,
                                                  repaired_traj)
        ego_initial.prediction.shape = ego_vehicle.prediction.shape
        # plot_limits = [-10, 100, -8, 8]
        plot_limits = [-20, 35, -5, 2]
        visualize_v_profile(ego_initial, ego_vehicle, int(repairer.tc), int(repairer.tv))
        for time_step in range(ego_vehicle.prediction.final_time_step):
            # visualize_initial_result(scenario, ego_initial,
            #                          time_step, None, plot_limits=plot_limits, end_time=final_time_step,
            #                          tv=int(repairer.tv), save_path=figure_path)
            visualize_repairing_result(scenario,
                                       ego_vehicle,
                                       time_step,
                                       tc=repairer.tc,
                                       end_time=final_time_step,
                                       plot_limits=plot_limits,
                                       ego_initial=ego_initial,
                                       save_path=figure_path)