from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair
from repairer.smt_repairer import SMTTrajectoryRepairer
from t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_profile

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt
import math

scenario_id = "DEU_LocationAUpper-26_21_T-1"
scenario_id = "DEU_LocationAUpper-47_35_T-1"
scenario_id = "DEU_LocationFUpper-60_43_T-1"
# scenario_id = "DEU_LocationAUpper-27_26_T-1"
# scenario_id = "DEU_LocationAUpper-35_38_T-1"
# scenario_id = "DEU_LocationAUpper-25_52_T-1"
# scenario_id = "DEU_LocationELower-16_11_T-1"
# scenario_id = "DEU_LocationALower-48_27_T-1"
# scenario_id = "DEU_LocationAUpper-27_26_T-1"
# scenario_id = 'ZAM_Zip-1_56_T-1'
# scenario_id = "DEU_LocationFUpper-60_43_T-1"
# scenario_id = "DEU_LocationAUpper-48_7_T-1"
# scenario_id = "DEU_LocationAUpper-35_3_T-1"
# scenario_id = "DEU_LocationAUpper-56_33_T-1"
# scenario_id = "DEU_LocationFLower-60_35_T-1"
# scenario_id = "DEU_LocationALower-54_5_T-1"
# scenario_id = "DEU_LocationBUpper-1_22_T-1"
# scenario_id = "DEU_LocationAUpper-34_9_T-1"
# # scenario_id = "DEU_LocationAUpper-44_30_T-1"
# # scenario_id = "ZAM_Zip-1_67_T-1"
# # scenario_id = "DEU_Gar-1_1_T-1"
# # scenario_id = "ZAM_Tutorial-1_2_T-1"
# scenario_id = "DEU_LocationAUpper-52_30_T-1"
# scenario_id = "DEU_LocationAUpper-47_35_T-1"
# scenario_id = "DEU_LocationAUpper-50_32_T-1"
file_path = "/home/yuanfei/commonroad/commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" \
            + scenario_id + ".xml"
file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
            + scenario_id + ".xml"
# file_path = "/home/yuanfei/commonroad/commonroad_repair/scenarios/" \
#             + scenario_id + ".xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 22
    rule = "R_G1"
    ego_initial = scenario.obstacle_by_id(ego_id)
    time_step = 0
    rnd = MPRenderer(figsize=(40, 10))
    scenario.draw(
        rnd,
        draw_params=ParamServer({"time_begin": time_step, "trajectory": {
                 "draw_trajectory": False}, "occupancy": {
            "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': True}})
    )
    rnd.render()
    plt.title(str(time_step))
    plt.show()
    # scenario.remove_obstacle(scenario.obstacle_by_id(3))
    # scenario.remove_obstacle(scenario.obstacle_by_id(4))
    # scenario.remove_obstacle(scenario.obstacle_by_id(5))


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
        plot_limits = [-380, -150, 7.5, 17.5]
        target_veh = scenario.obstacle_by_id(repairer.rule_abstracter.other_veh_id)
        # visualize_profile(target_veh, ego_initial, ego_vehicle)
    for time_step in range(ego_vehicle.prediction.final_time_step):
        # visualize_repairing_result(scenario, ego_initial,
        #                            ego_vehicle, time_step, target_veh, plot_limits=plot_limits)
        rnd = MPRenderer(figsize=(40, 10), plot_limits=plot_limits)
        scenario.draw(
            rnd,
            draw_params=ParamServer({"time_begin": time_step, "trajectory": {
                     "draw_trajectory": False}, "occupancy": {
                "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': True}})
        )
        # scenario.obstacle_by_id()
        ego_initial.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": time_step,
                              "occupancy": {
                                  "draw_occupancies": 1,
                                  "shape": {"rectangle": {
                                      "facecolor": "green",
                                      "edgecolor": "green"}
                                  }},
                              "dynamic_obstacle":
                                  {"vehicle_shape": {
                                      "occupancy": {
                                          "shape": {"rectangle": {
                                              "facecolor": "green",
                                              "edgecolor": "green"}
                                          }}}}}))
        ego_vehicle.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": time_step,
                              "occupancy": {
                                  "draw_occupancies": 1,
                                  "shape": {"rectangle": {
                                      "facecolor": "black",
                                      "edgecolor": "black"}
                                  }},
                              "trajectory": {
                                  "draw_trajectory": False},
                              "dynamic_obstacle":
                                  {"vehicle_shape": {
                                      "occupancy": {
                                          "shape": {"rectangle": {
                                              "facecolor": "black",
                                              "edgecolor": "black"}
                                          }}}, 'show_label': True}}))
        rnd.render()
        plt.title(str(time_step))
        plt.show()