import os
import unittest
import math
from copy import deepcopy
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState

from cut_off.ttcc import TTCC
from cut_off.simulation import SimulationLong, SimulationLateral, CutOffAction
from cut_off.utils import check_velocity_feasibility, visualize_state_list

from lazy_smt.monitor import RuleMonitor


class TestTTCC(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        self.scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

    def test_ttv(self):
        ego_id = 1003
        world_state = WorldState.create_from_scenario(self.scenario, ego_id)
        rule_monitor = RuleMonitor(world_state, ["R_G1"])
        ttcc_object = TTCC(rule_monitor)
        assert math.isclose(ttcc_object.ttv, 2.0, abs_tol=1e-2)

    def test_simulation_long(self):
        ego_id = 1003
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        SL1 = SimulationLong(CutOffAction.BRAKE, ego_vehicle, 0)
        simulated_state1 = SL1.simulate_state_list()
        self.assertEqual(
            simulated_state1[-1].time_step,
            50)
        self.assertEqual(check_velocity_feasibility(
            simulated_state1[-1],
            SL1.parameters),
            True)
        # visualize the scenario and the trajectory
        # self.scenario.lanelet_network
        # visualize_state_list(simulated_state1, self.scenario, SL1.vehicle_dynamics.shape)
        SL2 = SimulationLong(CutOffAction.STEADYSPEED, ego_vehicle, 0)
        simulated_state2 = SL2.simulate_state_list()
        self.assertEqual(
            simulated_state2[-1].time_step,
            50)
        self.assertEqual(
            simulated_state2[-1].velocity,
            simulated_state2[0].velocity)
        SL3 = SimulationLong(CutOffAction.KICKDOWN, ego_vehicle, 0)
        simulated_state3 = SL3.simulate_state_list()
        self.assertEqual(
            simulated_state3[-1].time_step,
            50)
        self.assertEqual(
            check_velocity_feasibility(
                simulated_state3[-1],
                SL3.parameters),
            True)
        # visualize the scenario and the trajectory
        # visualize_state_list(simulated_state3, self.scenario, SL3.vehicle_dynamics.shape)

    def test_simulate_lateral(self):
        ego_id = 1003
        ego_vehicle = self.scenario.obstacle_by_id(ego_id)
        world_state: WorldState = WorldState.create_from_scenario(self.scenario, ego_id)
        SL1 = SimulationLateral(
            CutOffAction.LANECHANGELEFT,
            ego_vehicle,
            0,
            world_state)
        simulated_state_list1 = SL1.simulate_state_list()
        final_lanelet = self.scenario.lanelet_network.find_lanelet_by_position(
            [simulated_state_list1[-1].position])[0]
        final_lane = world_state.road_network.find_lane_by_lanelet(final_lanelet[0])
        self.assertEqual(
            world_state.ego_vehicle.lane.adj_left.lane_id,
            final_lane.lane_id)
        # visualize the scenario and the trajectory
        # visualize_state_list(simulated_state_list1, self.scenario, SL1.parameters)
        SL2 = SimulationLateral(CutOffAction.LANECHANGERIGHT, ego_vehicle, 0, world_state)
        simulated_state_list2 = SL2.simulate_state_list()
        final_lanelet = self.scenario.lanelet_network.find_lanelet_by_position(
            [simulated_state_list2[-1].position])[0]
        final_lane = world_state.road_network.find_lane_by_lanelet(final_lanelet[0])
        self.assertEqual(
            world_state.ego_vehicle.lane.adj_right.lane_id,
            final_lane.lane_id)

        # # visualize the scenario and the trajectory
        # visualize_state_list(simulated_state_list2, self.scenario, SL2.parameters)

    def test_ttcc_1(self):
        ego_id = 1003
        world_state = WorldState.create_from_scenario(self.scenario, ego_id)
        rule_monitor = RuleMonitor(world_state, ["R_G1"])
        ttcc_object = TTCC(rule_monitor)
        # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1007))
        # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
        ttcc = ttcc_object.generate(CutOffAction.LANECHANGELEFT)
        self.assertEqual(
            ttcc,
            -math.inf)

    def test_ttcc_2(self):
        ego_id = 1003
        # self.scenario.remove_obstacle(self.scenario.obstacle_by_id(1006))
        world_state = WorldState.create_from_scenario(self.scenario, ego_id)
        rule_monitor = RuleMonitor(world_state, ["R_G1"])
        ttcc_object = TTCC(rule_monitor)
        ttcc = ttcc_object.generate(CutOffAction.LANECHANGERIGHT)
        assert math.isclose(ttcc, 0.5, abs_tol=1e-2)

    def test_ttcc_3(self):
        ego_id = 1003
        world_state = WorldState.create_from_scenario(self.scenario, ego_id)
        rule_monitor = RuleMonitor(world_state, ["R_G1"])
        ttcc_object = TTCC(rule_monitor)
        ttcc = ttcc_object.generate(CutOffAction.BRAKE)
        self.assertEqual(
            round(ttcc, 1),
            -math.inf)
