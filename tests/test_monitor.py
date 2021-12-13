import os
import unittest
import math
from copy import deepcopy
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world_state import WorldState
from lazy_smt.monitor import STLRuleMonitor, MTLRuleMonitor


class TestMonitor(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_mtl_monitor(self):
        ego_id = 1003
        mtl_monitor = MTLRuleMonitor(self.scenario, ego_id, ["B_SRG1"])
        mtl_monitor.evaluate_initially()

    def test_robustness_monitor(self):
        pass

    def test_boolean_monitor(self):
        ego_id = 1003
        world_state = WorldState.create_from_scenario(self.scenario, ego_id)
        rule_monitor = STLRuleMonitor(world_state, ["R_G1"])
