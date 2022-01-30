from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair
from repairer.smt_repairer import SMTTrajectoryRepairer
from t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_v_profile

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt
import math

scenario_id = "DEU_LocationALower-50_36_T-1"

file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
            + scenario_id + ".xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 9
    rule = "R_G3"

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
    ego_initial = scenario.obstacle_by_id(ego_id)

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
        plot_limits = None #[-380, -150, 7.5, 17.5]
        # visualize_v_profile(ego_initial, ego_vehicle)
        for time_step in range(ego_vehicle.prediction.final_time_step):
            visualize_repairing_result(scenario, ego_initial,
                                       ego_vehicle, time_step, None, plot_limits=plot_limits)
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