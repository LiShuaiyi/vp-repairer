import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from optimizer.qp_planner import QPPlanner, QPLongReference, QPLongState
from optimizer.configuration import PlanningConfigurationVehicle
from optimizer.constraints import LatConstraints, LonConstraints

from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline


class TestOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.scenario_root_path = os.path.join(root_path, "scenarios")
        scenario_file = os.path.join(self.scenario_root_path, "ZAM_Tutorial-1_2_T-1.xml")
        self.scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        self.planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    def test_optimizer(self):
        time_horizon = 4.0
        # initial state of vehicle for the optimization problem (longitudinal position, velocity, acceleration, jerk)
        initial_state = self.planning_problem.initial_state
        x_0 = np.array([initial_state.position[0],
                        initial_state.velocity,
                        0.0,
                        0.0]).reshape([4, ])

        ##################################
        # Configuration and reference path
        ##################################
        vehicle_configuration = PlanningConfigurationVehicle()
        initial_lanelet = self.scenario.lanelet_network.lanelets_in_proximity(initial_state.position, 100)[0]
        reference_path = initial_lanelet.center_vertices
        while initial_lanelet.successor:
            initial_lanelet = initial_lanelet.successor
            reference_path = np.concatenate((reference_path, initial_lanelet.center_vertices))
        reference_path = np.array(chaikins_corner_cutting(reference_path))
        vehicle_configuration.reference_path = resample_polyline(reference_path, 2.5)
        qp_planner = QPPlanner(self.scenario,
                               self.planning_problem,
                               time_horizon,
                               vehicle_configuration,
                               verbose=True)

        ############################
        # long. and lat. constraints
        ############################
        # create constraints for longitudinal minimum and maximum position
        s_min = []  # minimum position constraint
        s_max = []  # maximum position constraint
        # extract obstacle from scenario
        dyn_obstacles = self.scenario.dynamic_obstacles
        # go through obstacle list and distinguish between following and leading vehicle
        for o in dyn_obstacles:
            if o.initial_state.position[0] < x_0[0]:
                print('Following vehicle id={}'.format(o.obstacle_id))
                prediction = o.prediction.trajectory.state_list
                for p in prediction:
                    s_min.append(p.position[0] + o.obstacle_shape.length / 2. + 2.5)
            else:
                print('Leading vehicle id={}'.format(o.obstacle_id))
                prediction = o.prediction.trajectory.state_list
                for p in prediction:
                    s_max.append(p.position[0] - o.obstacle_shape.length / 2. - 2.5)
        s_min = np.array(s_min)
        s_max = np.array(s_max)
        c_long = LonConstraints.construct_constraints(s_min, s_max, s_min, s_max)
        d_min_single = np.full_like(s_min, -2.5)
        d_max_single = np.full_like(s_max, 2.5)
        d_min = np.array((d_min_single, d_min_single, d_min_single)).transpose()
        d_max = np.array((d_max_single, d_max_single, d_max_single)).transpose()
        c_lat = LatConstraints.construct_constraints(d_min, d_max, d_min, d_max)

        ######################################
        # reference for the long. optimization
        ######################################
        v_ref = 30  #m/s
        x_ref = list()
        for i in range(len(s_min)):
            x_ref.append(QPLongState(0, v_ref, 0., 0., 0.))
        reference = QPLongReference(x_ref)

        #####################################
        # trajectory generation and conversion
        #####################################
        trajectory = qp_planner.plan_trajectories(c_long, c_lat, reference)
        trajectory_cartesian = qp_planner.transform_trajectory_to_cartesian_coordinates(trajectory)
        ego_vehicle = trajectory_cartesian.convert_to_cr_ego_vehicle(
                vehicle_configuration.width, vehicle_configuration.length,
                vehicle_configuration.wheelbase, vehicle_configuration.vehicle_id)
        self.scenario.add_objects(ego_vehicle)
        rnd = MPRenderer()
        self.scenario.draw(rnd)
        rnd.render()
        plt.show()