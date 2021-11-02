import os
import unittest
import math
from copy import deepcopy
from typing import List
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState

from cut_off.ttcc import TTCC
from cut_off.simulation import SimulationLong, SimulationLateral, CutOffAction
from cut_off.utils import check_velocity_feasibility, visualize_state_list

from encoding.rule_encoding import RuleEncoder

from crmonitor.predicates.rule import AbstractionNode


class TestMonitor(unittest.TestCase):
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
        self.assertEqual(type(rule_encoder.prop_abs.pop()), AbstractionNode)