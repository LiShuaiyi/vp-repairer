import os
import unittest
import math
import matplotlib.pyplot as plt


from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState

from cut_off.ttcc import TTCC
from cut_off.simulation import SimulationLong, SimulationLateral, CutOffAction
from cut_off.utils import check_velocity_feasibility, transfer_state_list_to_trajectory

class TestTTCC(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_ttv(self):
        ego_id = 1003
        world_state_1 = WorldState.create_from_scenario(self.scenario,
                                                        ego_id)
        ttcc_object_1 = TTCC(world_state_1, ["R_G1"])
        ttv = ttcc_object_1.calc_ttv()
        self.assertEqual(ttv, 2.0)
        ego_id = 1002
        world_state_2 = WorldState.create_from_scenario(self.scenario,
                                                        ego_id)
        ttcc_object_2 = TTCC(world_state_2, ["R_G1"])
        ttv = ttcc_object_2.calc_ttv()
        self.assertEqual(ttv, -math.inf)

    def test_simulation_long(self):
        ego_id = 1003
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        SL1 = SimulationLong(CutOffAction.KICKDOWN, ego_vehicle, 0)
        simulated_state1 = SL1.simulate_state_list()
        self.assertEqual(simulated_state1[-1].time_step, 50)
        self.assertEqual(check_velocity_feasibility(simulated_state1[-1], SL1.parameters), True)
        SL2 = SimulationLong(CutOffAction.STEADYSPEED, ego_vehicle, 0)
        simulated_state2 = SL2.simulate_state_list()
        self.assertEqual(simulated_state2[-1].time_step, 50)
        self.assertEqual(simulated_state2[-1].velocity, simulated_state2[0].velocity)
        SL3 = SimulationLong(CutOffAction.KICKDOWN, ego_vehicle, 0)
        simulated_state3 = SL3.simulate_state_list()
        self.assertEqual(simulated_state3[-1].time_step, 50)
        self.assertEqual(check_velocity_feasibility(simulated_state3[-1], SL3.parameters), True)

    def test_simulate_lateral(self):
        ego_id = 1003
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        world_state_1 = WorldState.create_from_scenario(self.scenario,
                                                        ego_id)
        SL1 = SimulationLateral(CutOffAction.LANECHANGELEFT, ego_vehicle, 0, world_state_1)
        simulated_state_list1 = SL1.simulate_state_list()
        rnd = MPRenderer()
        self.scenario.draw(rnd)
        trajectory1 = transfer_state_list_to_trajectory(self.scenario, simulated_state_list1,
                                                        SL1.parameters)
        trajectory1.draw(rnd, draw_params={'time_begin': 0,
                                           'trajectory': {'draw_trajectory': True}})
        rnd.render()
        plt.show()