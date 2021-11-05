import os
import math
import unittest
from sympy.logic.boolalg import is_cnf
from commonroad.common.file_reader import CommonRoadFileReader

from lazy_smt.encoding import RuleEncoder
from lazy_smt.sat_solver import SATSolver, SATISFIABILITY
from crmonitor.predicates.rule import AbstractionNode


class TestSMTSolver(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        ego_id = 1003
        rule = "R_G1"
        self.ttv = 20
        self.rule_encoder = RuleEncoder(self.ttv, self.scenario, ego_id, rule)

    def test_construction(self):
        self.assertEqual(len(self.rule_encoder.prop_abs), 4)
        for node in self.rule_encoder.prop_abs:
            self.assertEqual(self.rule_encoder.abs_robust_ttv.query('abstraction == @node.name')["robustness"].values[0],
                             node.ttv_value)
        self.assertTrue(any([isinstance(abstraction, AbstractionNode)
                             for abstraction in self.rule_encoder.prop_abs]))
        exp_compliance = False
        rob_value = all([r >= 0.0 for r in self.rule_encoder.abs_robustness["robustness"].values])
        self.assertEqual(
            exp_compliance, rob_value,
        )

    def test_select_predicates(self):
        predicates = self.rule_encoder.select_predicates()
        self.assertEqual(
            predicates[0].base_name, "keeps_safe_distance_prec"
        )

    def test_sat_solver(self):
        sat_encoding = self.rule_encoder.sat_encoding
        sat_solver = SATSolver(sat_encoding)
        # check whether the formula in the sat solver is CNF or not
        self.assertTrue(is_cnf(sat_solver.formula))
        sat = sat_solver.solve()
        self.assertEqual(
            sat, SATISFIABILITY.SAT
        )
        abstraction_nodes = self.rule_encoder.prop_abs
        # after negating all the abstractions
        for abs_node in abstraction_nodes:
            sat_solver.update_formula(abs_node)
        sat = sat_solver.solve()
        self.assertEqual(
            sat, SATISFIABILITY.UNSAT
        )