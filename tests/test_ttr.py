import os
import unittest
import math
from copy import deepcopy
from commonroad.common.file_reader import CommonRoadFileReader

from cut_off.ttr import TTR
from cut_off.simulation import SimulationLong, SimulationLateral, CutOffAction

from crmonitor.common.world_state import WorldState


class TestTTR(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "ZAM_Urban-3_3_Repair.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        ego_id = 8
        self.world_state = WorldState.create_from_scenario(self.scenario, ego_id)

    def test_ttc(self):
        ttr_object = TTR(self.world_state)
        ttc = ttr_object.ttc
        assert math.isclose(ttc, 2.4, abs_tol=1e-2)

    def test_ttr_1(self):
        ttr_object = TTR(self.world_state)
        maneuver_set = [CutOffAction.STEERLEFT,
                        CutOffAction.BRAKE,
                        CutOffAction.KICKDOWN,
                        CutOffAction.STEERRIGHT]
        ttr = ttr_object.generate(maneuver_set)
        assert math.isclose(ttr, 2.3, abs_tol=1e-2)

    def test_ttr_2(self):
        ttr_object = TTR(self.world_state)
        maneuver_set = [CutOffAction.STEERRIGHT]  # impossible maneuver
        ttr = ttr_object.generate(maneuver_set)
        assert math.isclose(ttr, -math.inf, abs_tol=1e-2)