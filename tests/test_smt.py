import os
import math
import unittest
from sympy.logic.boolalg import is_cnf
from commonroad.common.file_reader import CommonRoadFileReader

from lazy_smt.encoding import RuleEncoder
from lazy_smt.sat_solver import SATSolver
from crmonitor.predicates.rule import AbstractionNode


class TestSMTSolver(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_construction(self):
        ego_id = 1003
        rule = "R_G1"
        rule_encoder = RuleEncoder(self.scenario, ego_id, rule)
        self.assertEqual(len(rule_encoder.prop_abs), 4)
        self.assertTrue(any([isinstance(abstraction, AbstractionNode) for abstraction in rule_encoder.prop_abs]))
        exp_compliance = False
        rob_value = all([r >= 0.0 for r in rule_encoder.abs_robustness["robustness"].values])
        self.assertEqual(
            exp_compliance, rob_value, f"Test failed for ego_id={ego_id}"
        )
        sat_encoding = rule_encoder.sat_encoding
        sat_solver = SATSolver(sat_encoding)
        # check whether the formula in the sat solver is CNF or not
        self.assertTrue(is_cnf(sat_solver.formula))

    def test_select_predicates(self):
        ego_id = 1003
        rule = "R_G1"
        rule_encoder = RuleEncoder(self.scenario, ego_id, rule)
        predicates = rule_encoder.select_predicates(ttv=math.inf)
        self.assertEqual(
            predicates[0].base_name, "keeps_safe_distance_prec"
        )
