import os
import unittest
import math
from copy import deepcopy
from commonroad.common.file_reader import CommonRoadFileReader

from cut_off.ttr import TTR
from cut_off.simulation import SimulationLong, SimulationLateral, CutOffAction
from cut_off.utils import check_velocity_feasibility, visualize_state_list
from decimal import *


class TestTTR(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "ZAM_Urban-3_3_Repair.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_steering(self):
        visualize_state_list(self.scenario.obstacles[2].prediction.trajectory.state_list, self.scenario,
                             self.scenario.obstacles[1].obstacle_shape)

    def test_ttc(self):
        ego_id = 8
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        ttr_object = TTR(self.scenario, ego_vehicle)
        ttc = ttr_object.ttc
        self.assertEqual(round(ttc, 1), 2.4)

    def test_ttr(self):
        ego_id = 8
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        ttr_object = TTR(self.scenario, ego_vehicle)
        maneuver_set = [CutOffAction.BRAKE, CutOffAction.KICKDOWN]
        ttr = ttr_object.generate(maneuver_set)
        self.assertEqual(round(ttr, 1), 2.1)