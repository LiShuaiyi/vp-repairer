from abstraction.abstracter import RuleAbstracter
from t_solver.t_solver import TSolver
from t_solver.qp_planner import QPPlannerRepair

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
import matplotlib.pyplot as plt

scenario_id = "DEU_LocationBUpper-1_22_T-1"
# scenario_id = "ZAM_Zip-1_67_T-1"
# scenario_id = "DEU_Gar-1_1_T-1"
# scenario_id = "ZAM_Tutorial-1_2_T-1"
file_path = "/home/yuanfei/commonroad/commonroad-scenarios-master-scenarios/scenarios/hand-crafted/" \
            + scenario_id + ".xml"
# file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
#             + scenario_id + ".xml"
file_path = "/home/yuanfei/commonroad/commonroad_repair/scenarios/" \
            + scenario_id + ".xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 9
    rule = "R_G1"
    # scenario.remove_obstacle(scenario.obstacle_by_id(3))
    # scenario.remove_obstacle(scenario.obstacle_by_id(4))
    # scenario.remove_obstacle(scenario.obstacle_by_id(5))

    ego_veh = scenario.obstacle_by_id(ego_id)
    # veh_shape = ego_veh.obstacle_shape

    ego_veh.prediction.trajectory.state_list = ego_veh.prediction.trajectory.state_list[:20]
    #
    # for time_step in range(ego_veh.prediction.final_time_step):
    #     rnd = MPRenderer(figsize=(20, 10))
    #     scenario.draw(
    #         rnd,
    #         draw_params=ParamServer({"time_begin": time_step, "occupancy": {
    #             "draw_occupancies": 1}})
    #     )
    #     # scenario.obstacle_by_id()
    #     ego_veh.draw(rnd,
    #                      draw_params=ParamServer(
    #                          {"time_begin": time_step,
    #                           "occupancy": {
    #                               "draw_occupancies": 1,
    #                               "shape": {"rectangle": {
    #                                   "facecolor": "black",
    #                                   "edgecolor": "black"}
    #                               }},
    #                           "dynamic_obstacle":
    #                               {"vehicle_shape": {
    #                                   "occupancy": {
    #                                       "shape": {"rectangle": {
    #                                           "facecolor": "black",
    #                                           "edgecolor": "black"}
    #                                       }}}, 'show_label': True}}))
    #     ego_veh.prediction.trajectory.draw(rnd, draw_params={
    #         "trajectory": {"shape": {"rectangle": {"facecolor": "black"}}}})
    #     rnd.render()
    #     plt.show()
    rule_abstracter = RuleAbstracter(scenario,
                                     planning_problem,
                                     ego_id, rule)
    t_solver = TSolver(rule_abstracter.rule_monitor)
    proposition2 = next((prop for prop in list(rule_abstracter.propositions)
                         if prop.name == '(keeps_safe_distance_prec__a0_a1 >= 0)'), None)
    assign_prop = [proposition2]
    t_solver.assign_proposition(assign_prop)
    tc = t_solver.search_tc()
    tc_object = t_solver.tc_object
    print(tc_object.tv_time_step,
          tc_object.tc_time_step,
          tc_object.compliant_maneuver)
    qp_repairer = QPPlannerRepair(rule_abstracter,
                                  tc_object,
                                  assign_prop)
    repaired_trajectory = qp_repairer.plan()
    ego_vehicle = qp_repairer.convert_traj_to_ego_vehicle(repaired_trajectory)
    ego_veh.prediction.shape = ego_vehicle.prediction.shape
    # plot_limits = [-10, 100, -8, 8]
    plot_limits = [-380, -180, 0, 20]
    for time_step in range(ego_vehicle.prediction.final_time_step):
        rnd = MPRenderer(figsize=(40, 10), plot_limits=plot_limits)
        scenario.draw(
            rnd,
            draw_params=ParamServer({"time_begin": time_step, "trajectory": {
                     "draw_trajectory": False}, "occupancy": {
                "draw_occupancies": 0}, 'dynamic_obstacle': {'show_label': True}})
        )
        # scenario.obstacle_by_id()
        ego_veh.draw(rnd,
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
        # ego_vehicle.prediction.trajectory.draw(rnd, draw_params={"time_begin": time_step,
        #     "trajectory": {"shape": {"rectangle": {"facecolor": "black"}}}})
        rnd.render()
        plt.title(str(time_step))
        plt.show()