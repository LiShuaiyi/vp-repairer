import os
import math
import unittest
from sympy.logic.boolalg import is_cnf, is_dnf

from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.sat_solver.sat_solver import SATSolver
from commonroad_repair.crrepairer.sat_solver.dpll import DPLL
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver, CutOffAction
from commonroad_repair.crrepairer.t_solver.qp_planner import QPPlannerRepair

from commonroad.common.file_reader import CommonRoadFileReader

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
        exp_compliance = False
        rob_value = all([r >= 0.0 for r in rule_monitor.prop_robust_all["robustness"].values])
        self.assertEqual(
            exp_compliance, rob_value,
        )
        self.assertEqual(
            rule_monitor.other_id, 1004
        )

    def test_sat_solver(self):
        sat_solver = SATSolver(self.rule_abstracter)
        # check whether the formula in the sat solver is CNF or not
        self.assertTrue(is_cnf(sat_solver.formula))
        sat_re = sat_solver.solve()
        self.assertEqual(
            sat_re, sat
        )
        _, m = sat_solver.model()
        self.assertEqual(list(m), ['d'])
        abstraction_nodes = self.rule_abstracter.propositions
        # after negating all the possible solutions
        while len(m) != 0:
            sat_solver.update_formula()
            sat_re = sat_solver.solve()
            _, m = sat_solver.model()
            print(sat_solver.formula)
        self.assertEqual(
            sat_re, unsat
        )

    def test_t_solver(self):
        t_solver = TSolver(self.rule_abstracter)
        proposition = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(keeps_safe_distance_prec__a0_a1 >= 0)'), None)
        t_solver.assign_proposition([proposition], ["d"])
        # safe distance
        self.assertEqual(set(t_solver.compliant_maneuvers),
                         {CutOffAction.BRAKE, CutOffAction.KICKDOWN})
        tc = t_solver.search_tc()
        assert math.isclose(tc,
                            1.9,
                            abs_tol=1e-2)
        proposition = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(in_same_lane__a0_a1_i >= 0)'), None)
        t_solver.assign_proposition([proposition], ["~a"])
        tc = t_solver.search_tc()
        assert math.isclose(tc,
                            0.5,
                            abs_tol=1e-2)

    def test_dpll(self):
        dpll_solver = DPLL('~a | ~b | c | d', self.rule_abstracter.rule_monitor.prop_robust_ttv)
        self.assertEqual(dpll_solver.solve(),
                         sat)
        self.assertEqual(list(dpll_solver.model),
                         ['d'])
        dpll_solver.update_cnf('~a & a')
        self.assertEqual(dpll_solver.solve(),
                         unsat)
        self.assertEqual(dpll_solver.model,
                         set())

    def test_cnf_dnf_converter(self):
        original_formula = '(a and b and !c) implies d'
        sat_solver = SATSolver(self.rule_abstracter)
        cnf_formula = sat_solver.construct_cnf(original_formula)
        self.assertTrue(is_cnf(cnf_formula))
        dnf_formula = sat_solver.construct_dnf(original_formula)
        self.assertTrue(is_dnf(dnf_formula))

    def test_construct_qp_repair(self):
        t_solver = TSolver(self.rule_abstracter)
        proposition2 = next((prop for prop in list(self.rule_abstracter.propositions)
                            if prop.name == '(keeps_safe_distance_prec__a0_a1 >= 0)'), None)
        assign_prop = [proposition2]
        t_solver.assign_proposition(assign_prop, ["d"])
        t_solver.search_tc()
        tc_object = t_solver.tc_object
        qp_repairer = QPPlannerRepair(self.rule_abstracter,
                                      tc_object,
                                      assign_prop)
        self.assertIsInstance(qp_repairer, QPPlannerRepair)
