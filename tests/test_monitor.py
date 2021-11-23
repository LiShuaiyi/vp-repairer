import os
import unittest
import math
from copy import deepcopy
from commonroad.common.file_reader import CommonRoadFileReader

from lazy_smt.monitor import STLRuleMonitor, MTLRuleMonitor


class TestMonitor(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_mtl_monitor(self):
        mtl_monitor = MTLRuleMonitor(self.scenario, ["B_SRG1"])
        mtl_monitor.evaluate_initially()