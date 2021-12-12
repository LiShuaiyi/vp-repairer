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