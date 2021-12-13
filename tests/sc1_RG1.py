import os
import math
import unittest
from sympy.logic.boolalg import is_cnf

from lazy_smt.abstracter import RuleAbstracter
from lazy_smt.monitor import STLRuleMonitor, MTLRuleMonitor
from lazy_smt.sat_solver import SATSolver, SATISFIABILITY
from lazy_smt.t_solver import TSolver, CutOffAction
from lazy_smt.dpll import DPLL
from repairer.qp_repairer import QPRepairer
from repairer.rule_constraints import RuleConstraints
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import PropositionNode
from stl_crmonitor.crmonitor.common.road_network import Lane

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer

import matplotlib.pyplot as plt

from z3 import sat, unsat

scenario_id = "DEU_LocationBUpper-1_22_T-1"

file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/" \
            + scenario_id + ".xml"

if __name__ == '__main__':
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(lanelet_assignment=True)
    # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 9
    rule = "R_G1"
    ego_veh = scenario.obstacle_by_id(ego_id)
    ego_veh.prediction.trajectory.state_list = ego_veh.prediction.trajectory.state_list[:40]

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
    #                                       }}}}}))
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
    qp_repairer = QPRepairer(rule_abstracter,
                             tc_object,
                             assign_prop)
    repaired_trajectory = qp_repairer.repair()
    ego_vehicle = qp_repairer.convert_traj_to_ego_vehicle(repaired_trajectory)
    ego_veh.prediction.shape =  ego_vehicle.prediction.shape
    for time_step in range(ego_vehicle.prediction.final_time_step):
        rnd = MPRenderer(figsize=(40, 10))
        scenario.draw(
            rnd,
            draw_params=ParamServer({"time_begin": time_step, "occupancy": {
                "draw_occupancies": 1}, 'dynamic_obstacle': {'show_label': True}})
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
        # 'dynamic_obstacle': {'show_label': True}}
        ego_vehicle.draw(rnd,
                         draw_params=ParamServer(
                             {"time_begin": time_step,
                              "occupancy": {
                                  "draw_occupancies": 1,
                                  "shape": {"rectangle": {
                                      "facecolor": "black",
                                      "edgecolor": "black"}
                                  }},
                              "dynamic_obstacle":
                                  {"vehicle_shape": {
                                      "occupancy": {
                                          "shape": {"rectangle": {
                                              "facecolor": "black",
                                              "edgecolor": "black"}
                                          }}}, 'show_label': True}}))
        ego_vehicle.prediction.trajectory.draw(rnd, draw_params={"time_begin": time_step,
            "trajectory": {"shape": {"rectangle": {"facecolor": "black"}}}})
        rnd.render()
        plt.show()