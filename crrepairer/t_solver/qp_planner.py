import numpy as np
from commonroad_dc.geometry.util import (compute_pathlength_from_polyline, compute_orientation_from_polyline)

from commonroad_qp_planner.qp_planner import QPPlanner, QPLongState, QPLongReference
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import set_up, convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode

from commonroad_repair.crrepairer.cut_off.tc import TC
from commonroad_repair.crrepairer.t_solver.rule_constraints import RuleConstraints
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter

from commonroad.scenario.trajectory import Trajectory, State
from commonroad.scenario.scenario import DynamicObstacle, TrajectoryPrediction, ObstacleType
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle

from typing import List
import yaml
import os


class QPPlannerRepair(QPPlanner):
    def __init__(self,
                 rule_abstracter: RuleAbstracter,
                 tc_object: TC,
                 sel_proposition: List[PropositionNode]):
        self._scenario = rule_abstracter.world_state.scenario
        self._ego_vehicle = tc_object.ego_vehicle
        # remove the existing ego vehicle from the scenario to avoid the conflict
        self._planning_problem = rule_abstracter.world_state.planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory
        self._cut_off_time_step = tc_object.tc_time_step
        self._N = tc_object.N
        if self._cut_off_time_step == 0:
            self._cut_off_state = self._ego_vehicle.initial_state
        else:
            self._cut_off_state = self._initial_trajectory.state_at_time_step(self._cut_off_time_step)
        self._settings = self.config_settings()
        self._reformulate_planning_problem()
        self._time_horizon = round((self._N - self._cut_off_time_step) * self._scenario.dt, 1)
        self._planning_problem.initial_state = self._cut_off_state
        self._vehicle_configuration: PlanningConfigurationVehicle = set_up(self._settings,
                                                                           self._scenario,
                                                                           self._planning_problem)
        # self._planning_problem.initial_state.time_step = 0 # todo: check the time steps
        super().__init__(self._scenario,
                         self._planning_problem,
                         self._time_horizon,
                         self._vehicle_configuration)
        self._rule_constraints = RuleConstraints(tc_object,
                                                 rule_abstracter,
                                                 sel_proposition,
                                                 self._vehicle_configuration,
                                                 self._initial_trajectory)

    def _reformulate_planning_problem(self,):
        if not hasattr(self._planning_problem, "initial_state"):
            raise ValueError("<QPPlannerRepair>: the initial state needs to be specified")
        self._planning_problem.initial_state = self._ego_vehicle.initial_state
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)

    def plan(self):
        long_constr = self._rule_constraints.longitudinal_constraints()
        reference_lon = self._formulate_reference()
        traj_lon, status = self.longitudinal_trajectory_planning(long_constr, reference_lon,
                                                                 safe_dis_modes=self._rule_constraints.
                                                                 safe_distance_modes)
        if status is not 'optimal':
            return None
            # raise ValueError('<QPPlannerRepair/_longitudinal_trajectory_planning>: failed')
        print('\t\t Lateral optimization')
        lat_constr = self._rule_constraints.lateral_constraints(traj_lon)
        trajectory, status = self.lateral_trajectory_planning(traj_lon, lat_constr)
        # convert trajectory to cartesian space
        if status is not 'optimal':
            return None
            # raise ValueError('<QPPlannerRepair/_lateral_trajectory_planning>: failed')
        cr_trajectory = self.transform_merge_trajectory(trajectory)
        return cr_trajectory

    def convert_traj_to_ego_vehicle(self,
                                    cr_trajectory: Trajectory,
                                    vehicle_id: int = 0) -> DynamicObstacle:
        """
        Converts trajectory object to CommonRoad obstacle with specified width and length
        :param width: The width of the ego vehicle
        :param length: The length of the ego vehicle
        :param vehicle_id: ID of ego vehicle
        :return: The CommonRoad DynamicObstacle object containing the current trajectory
        """
        # get trajectory
        shape = Rectangle(self._vehicle_configuration.length,
                          self._vehicle_configuration.width)
        pred = TrajectoryPrediction(cr_trajectory, shape)

        # create new object
        ego = DynamicObstacle(obstacle_id=vehicle_id,
                              obstacle_type=ObstacleType.CAR,
                              prediction=pred,
                              obstacle_shape=shape,
                              initial_state=self._ego_vehicle.initial_state)
        return ego

    def transform_merge_trajectory(self, trajectory: QPTrajectory):
        cartesian_traj_points = list()
        for state in trajectory.states:
            cart_pos = self.vehicle_configuration.curvilinear_coordinate_system.convert_to_cartesian_coords(
                state.position[0], state.position[1])
            # convert the orientation here
            ref_orientation = compute_orientation_from_polyline(self.vehicle_configuration.reference_path)
            ref_pathlength = compute_pathlength_from_polyline(self.vehicle_configuration.reference_path)
            orientation_interpolated = 0 # np.interp(state.position[0], ref_pathlength, ref_orientation)
            cartesian_traj_points.append(TrajPoint(
               t=state.t, x=cart_pos[0], y=cart_pos[1], theta=state.orientation + orientation_interpolated, v=state.v, a=state.a,
               kappa=state.kappa, kappa_dot=state.kappa_dot, j=state.j, lane=state.lane))

        traj = QPTrajectory(cartesian_traj_points, TrajectoryType.CARTESIAN)

        traj._u_lon = trajectory.u_lon
        traj._u_lat = trajectory.u_lat
        cr_traj_repaired = traj.convert_to_cr_trajectory(self._vehicle_configuration.wheelbase)
        if self._cut_off_time_step == 0:
            remaining_states = []
        else:
            remaining_states = [] + \
                               self._initial_trajectory.state_list[:self._cut_off_time_step-1]
        for state in cr_traj_repaired.state_list:
            state.time_step += self._cut_off_time_step + 1
        cr_traj_repaired.state_list = remaining_states + cr_traj_repaired.state_list
        return cr_traj_repaired

    def _formulate_reference(self):
        x_ref = list()
        for state in self._initial_trajectory.state_list[self._cut_off_time_step:]:
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            x_ref.append(QPLongState(pos[0], state.velocity, 0., 0., 0.))
        return QPLongReference(x_ref)

    def config_settings(self):
        config_file = 'config_highd.yaml'
        # config_file = 'config_' + str(self._scenario.scenario_id) + '.yaml'
        config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                   "../../config"))

        with open(os.path.join(config_dir, config_file), 'r') as stream:
            try:
                settings = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
        return settings


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2)
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region
