import os
import math
import unittest
from sympy.logic.boolalg import is_cnf

from abstraction.abstracter import RuleAbstracter
from sat_solver.sat_solver import SATSolver, SATISFIABILITY
from t_solver.t_solver import TSolver, CutOffAction
from sat_solver.dpll import DPLL
from t_solver.qp_planner import QPPlannerRepair
from crmonitor.predicates.rule import PropositionNode

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer

import matplotlib.pyplot as plt

from z3 import sat, unsat

class TestSMTSolver(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
        self.planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        ego_id = 1003
        obs = self.scenario.obstacle_by_id(1000)
        obs.initial_state.position[0] += 10.0
        for state in obs.prediction.trajectory.state_list:
            state.position[0] += 10.0
        obs.prediction.occupancy_set = obs.prediction._create_occupancy_set()
        obs._initial_occupancy_shape.center[0] += 10
        self.scenario.obstacle_by_id(ego_id).prediction.trajectory.state_list = self.scenario.obstacle_by_id(ego_id).prediction.trajectory.state_list[:30]
        rule = "R_G1"
        self.rule_abstracter = RuleAbstracter(self.scenario,
                                              self.planning_problem,
                                              ego_id, rule)

    def test_construction(self):
        self.assertEqual(len(self.rule_abstracter.propositions), 4)
        rule_monitor = self.rule_abstracter.rule_monitor
        for node in self.rule_abstracter.propositions:
            self.assertEqual(
                rule_monitor.prop_robust_ttv.query('alphabet == @node.alphabet')["robustness"].values[0],
                node.ttv_value)
        self.assertTrue(any([isinstance(abstraction, PropositionNode)
                             for abstraction in self.rule_abstracter.propositions]))
        exp_compliance = False
        rob_value = all([r >= 0.0 for r in rule_monitor.prop_robust_all["robustness"].values])
        self.assertEqual(
            exp_compliance, rob_value,
        )
        self.assertEqual(
            rule_monitor.other_id, 1004
        )

    def test_select_predicates(self):
        predicates = self.rule_abstracter.select_predicates()
        self.assertEqual(
            predicates[0].base_name, "keeps_safe_distance_prec"
        )

    def test_sat_solver(self):
        sat_solver = SATSolver(self.rule_abstracter.sat_encoding,
                               self.rule_abstracter.propositions,
                               self.rule_abstracter.prop_robust_ttv)
        # check whether the formula in the sat solver is CNF or not
        self.assertTrue(is_cnf(sat_solver.formula))
        sat_re = sat_solver.solve()
        self.assertEqual(
            sat_re, sat
        )
        m = sat_solver.model()
        self.assertEqual(list(m), ['d'])
        abstraction_nodes = self.rule_abstracter.propositions
        # after negating all the abstractions
        for abs_node in abstraction_nodes:
            sat_solver.update_formula(abs_node)
        sat_re = sat_solver.solve()
        self.assertEqual(
            sat_re, unsat
        )

    def test_t_solver(self):
        t_solver = TSolver(self.rule_abstracter.rule_monitor)
        proposition = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(keeps_safe_distance_prec__a0_a1 >= 0)'), None)
        t_solver.assign_proposition(proposition)
        # safe distance
        self.assertEqual(t_solver.compliant_maneuvers,
                         [CutOffAction.BRAKE,
                          CutOffAction.KICKDOWN])
        tc = t_solver.search_tc()
        self.assertEqual(tc, -math.inf)
        proposition = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(in_same_lane__a0_a1_i >= 0)'), None)
        t_solver.assign_proposition(proposition)
        tc = t_solver.search_tc()
        self.assertEqual(tc, 0.4)

    def test_satisfiability_checking(self):
        # initial assignment: a b ~c ~d
        sat_encoding = '(a | b) & c & ~d'
        sat_solver = SATSolver(sat_encoding,
                               self.rule_abstracter.propositions,
                               self.rule_abstracter.rule_monitor.prop_robust_ttv)
        self.assertEqual(sat_solver.satisfiable_subformula_list,
                         ["c"])

    def test_dpll(self):
        dpll_solver = DPLL('~a | ~b | c | d', self.rule_abstracter.prop_robust_ttv)
        self.assertEqual(dpll_solver.solve(),
                         sat)
        self.assertEqual(list(dpll_solver.model),
                         ['d'])
        dpll_solver.update_cnf('~a & a')
        self.assertEqual(dpll_solver.solve(),
                         unsat)
        self.assertEqual(dpll_solver.model,
                         set())

    def test_construct_qp_repair(self):
        t_solver = TSolver(self.rule_abstracter)
        proposition1 = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(in_same_lane__a0_a1_i >= 0)'), None)
        proposition2 = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(keeps_safe_distance_prec__a0_a1 >= 0)'), None)
        assign_prop = [proposition1]
        t_solver.assign_proposition(assign_prop, ["~b"])
        tc = t_solver.search_tc()
        tc_object = t_solver.tc_object
        qp_repairer = QPPlannerRepair(self.rule_abstracter,
                                      tc_object,
                                      assign_prop)
        self.assertIsInstance(qp_repairer, QPPlannerRepair)
        repaired_trajectory = qp_repairer.plan()
        ego_vehicle = qp_repairer.convert_traj_to_ego_vehicle(repaired_trajectory)
        for time_step in range(ego_vehicle.prediction.final_time_step):
            rnd = MPRenderer(figsize=(20, 10))
            self.scenario.draw(
                rnd,
                draw_params=ParamServer({"time_begin": time_step, "occupancy": {
                                      "draw_occupancies": 1}})
            )
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
                                              }}}}}))
            ego_vehicle.prediction.trajectory.draw(rnd, draw_params={
                "trajectory": {"shape": {"rectangle": {"facecolor": "black"}}}})
            rnd.render()
            plt.show()

    #
    # def test_rule_constraints(self):
    #     t_solver = TSolver(self.rule_abstracter.rule_monitor)
    #     proposition = next((prop for prop in list(self.rule_abstracter.propositions)
    #                         if prop.name == '(in_same_lane__a0_a1_i >= 0)'), None)
    #     t_solver.assign_proposition(proposition)
    #     tc = t_solver.search_tc()
    #     tc_object = t_solver.tc_object
    #     target_lanes_id_exp = list()
    #     for _ in range(tc_object.tc_time_step, tc_object.tv_time_step):
    #         target_lanes_id_exp.append(1)
    #     for _ in range(tc_object.tv_time_step, tc_object.N):
    #         target_lanes_id_exp.append(0)
    #     rule_constraints = RuleConstraints(tc_object,
    #                                        self.rule_abstracter,
    #                                        proposition)
    #     target_lanes = rule_constraints.set_target_lanes()
    #     target_lanes_id = list()
    #     for lane in target_lanes.values():
    #         target_lanes_id.append(lane.lane_id)
    #     self.assertEqual(target_lanes_id_exp, target_lanes_id)
    #
    #
