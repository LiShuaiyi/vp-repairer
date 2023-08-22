"""
Unit tests of the module time-to-comply computation
"""

import os
import unittest
import math

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory
from crmonitor.common.world import World

from crrepairer.cut_off.tc import TC
from crrepairer.cut_off.utils import update_ego_vehicle
from crrepairer.smt.monitor_wrapper import STLRuleMonitor

from commonroad_crime.utility.simulation import Maneuver, SimulationLong
from commonroad_crime.data_structure.configuration import CriMeConfiguration


class TestTC(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "DEU_test-1_1_T-1.xml")
        self.scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        self.ego_id = 1003
        self._ego_obs = self.scenario.obstacle_by_id(self.ego_id)
        self.rule_monitor = STLRuleMonitor(self.scenario,
                                           self.ego_id,
                                           ["R_G1"][0])

    def test_tv(self):
        tc_object = TC(self._ego_obs, self.rule_monitor)
        assert math.isclose(tc_object.tv,
                            2.0,
                            abs_tol=1e-2)

    def test_tc_1(self):
        tc_object = TC(self._ego_obs, self.rule_monitor)
        tc = tc_object.generate([Maneuver.STEERLEFT])
        self.assertEqual(
            tc,
            -math.inf)

    def test_tc_2(self):
        tc_object = TC(self._ego_obs, self.rule_monitor)
        tc = tc_object.generate([Maneuver.STEERRIGHT])
        self.assertEqual(
            round(tc, 1),
            .4)

    def test_tc_3(self):
        tc_object = TC(self._ego_obs, self.rule_monitor)
        tc = tc_object.generate([Maneuver.BRAKE])
        self.assertEqual(
            round(tc, 1),
            1.9)

    def test_tc_total(self):
        tc_object = TC(self._ego_obs, self.rule_monitor)
        tc = tc_object.generate([Maneuver.STEERLEFT,
                                 Maneuver.STEERRIGHT,
                                 Maneuver.KICKDOWN,
                                 Maneuver.BRAKE])
        self.assertEqual(
            round(tc, 1),
            1.9)
        self.assertEqual(
            tc_object.compliant_maneuver, Maneuver.BRAKE
        )

    def test_update_world_state(self):
        # simulate a new trajectory of the ego vehicle
        ego_vehicle = self.scenario.obstacle_by_id(self.ego_id)
        world_state = self.rule_monitor.world
        config = CriMeConfiguration()
        config.scenario = self.scenario
        sim_long = SimulationLong(Maneuver.BRAKE, ego_vehicle, config)
        new_state_list = sim_long.simulate_state_list(0)[1:]
        # 1. directly update the ego vehicle
        update_ego_vehicle(world_state.road_network,
                           world_state.vehicle_by_id(ego_vehicle.obstacle_id),
                           new_state_list,
                           0,
                           world_state.dt)
        # 2. recreate the world state
        ego_vehicle.prediction.trajectory = Trajectory(1, new_state_list)
        world_state_updated = World.create_from_scenario(self.scenario) #, self.ego_id)
        ego_former = world_state.vehicle_by_id(ego_vehicle.obstacle_id)
        ego_updated = world_state_updated.vehicle_by_id(ego_vehicle.obstacle_id)
        # comparison
        # ---> length of the state list
        self.assertEqual(
            len(ego_former.states_cr),
            len(ego_updated.states_cr),
        )
        #self.assertEqual(
        #    len(ego_former.states_lon),
        #    len(ego_updated.states_lon),
        #)
        #self.assertEqual(
        #    len(ego_former.states_lat),
        #    len(ego_updated.states_lat),
        #)
        # ---> check the final state
        # whether its the same
        self.assertTrue(ego_former.state_list_cr[-1] ==
                        ego_updated.state_list_cr[-1])
        self.assertTrue(ego_former.get_lat_state(ego_vehicle.prediction.final_time_step).d ==
                        ego_updated.get_lat_state(ego_vehicle.prediction.final_time_step).d)
        self.assertTrue(ego_former.get_lon_state(ego_vehicle.prediction.final_time_step).s ==
                        ego_updated.get_lon_state(ego_vehicle.prediction.final_time_step).s)
        # ---> check the lane
        self.assertEqual(ego_former.get_lane(0).contained_lanelets,
                         ego_updated.get_lane(0).contained_lanelets)