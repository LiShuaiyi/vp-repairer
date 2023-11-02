import numpy as np

from miqp_planner.miqp_planner_base import MIQPPlanner
from miqp_planner.miqp_long_planner import MIQPLongState, MIQPLongReference
from miqp_planner.miqp_constraints import LongitudinalConstraint, LateralConstraint

from commonroad_qp_planner.initialization import convert_pos_curvilinear
from miqp_planner.miqp_initialization import set_up_miqp
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory

from crrepairer.smt.monitor_wrapper import PropositionNode

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.scenario import (
    DynamicObstacle,
    TrajectoryPrediction,
    ObstacleType,
)
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle

from typing import List
import yaml
import os
import time


class MIQPPlannerRepair(MIQPPlanner):
    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        tc_object: TC,
        sel_proposition: List[PropositionNode],
        proposition_full: List[PropositionNode],
        planning_problem: PlanningProblem,
    ):
        # initialize the scenario and planning problem
        self._scenario = rule_monitor.world.scenario
        self._ego_vehicle = tc_object.ego_vehicle
        self._planning_problem = planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory

        self.road_network = rule_monitor.world.road_network
        self.ego_vehicle_roadnetwork = rule_monitor.world.vehicle_by_id(
            rule_monitor.vehicle_id
        )
        self._start_time_step = tc_object.ego_vehicle.initial_state.time_step

        # set the cut-off state as the initial state
        self._cut_off_time_step = tc_object.tc_time_step
        self._N = tc_object.N
        if self._cut_off_time_step == self._start_time_step:
            self._cut_off_state = self._ego_vehicle.initial_state
        else:
            self._cut_off_state = self._initial_trajectory.state_at_time_step(
                self._cut_off_time_step
            )
        self._time_horizon = round(
            (self._N - self._cut_off_time_step) * self._scenario.dt,
            tc_object.round_tolerance,
        )
        self._planning_problem.initial_state = InitialState(
            position=self._cut_off_state.position,
            velocity=self._cut_off_state.velocity,
            orientation=self._cut_off_state.orientation,
            time_step=self._cut_off_state.time_step,
            acceleration=self._cut_off_state.acceleration,
            # not needed but mandatory field
            yaw_rate=0,
            slip_angle=0,
        )
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)
        # load and set up the configuration
        self._settings = self.config_settings()
        self._vehicle_configuration = set_up_miqp(
            self._settings,
            self._scenario,
            self._planning_problem,
            self.ego_vehicle_roadnetwork,
        )

        # update the vehicle shape
        self._vehicle_configuration.width = self._ego_vehicle.obstacle_shape.width
        self._vehicle_configuration.length = self._ego_vehicle.obstacle_shape.length

        # initialize the MIQP planner
        super().__init__(
            self._scenario,
            self._planning_problem,
            self._time_horizon,
            self._vehicle_configuration,
        )

        # construct constraints
        self._long_constraints = LongitudinalConstraint(
            tc_object,
            rule_monitor,
            sel_proposition,
            proposition_full,
            self._vehicle_configuration,
            self._initial_trajectory,
            self._start_time_step,
        )

    def long_constraints(self):
        return self._long_constraints

    def total_time_steps(self):
        return self._N - self._cut_off_time_step

    def plan(self):
        """
        Plans a trajectory starting from the cut-off state.
            First: constructs the constraints and the reference path
            Then: generates the trajectory in both longitudinal and lateral directions
        """
        print("* \t\t MIQP Longitudinal optimization")
        reference_lon = self.construct_s_reference()
        traj_lon = self.longitudinal_trajectory_planning(
            reference_lon, self._long_constraints, slack=True
        )
        if traj_lon is None:
            return None
        print("* \t\t MIQP Lateral optimization")
        # TODO: fix inputs
        lateral_constraints = LateralConstraint(
            self._long_constraints._tc_obj,
            self._long_constraints._rule_monitor,
            self._long_constraints._veh_config,
            self._long_constraints.target_lanes,
            traj_lon,
            self._long_constraints.sel_prop_full,
        )
        lateral_constraints.create_d_constraints(traj_lon)
        trajectory = self.lateral_trajectory_planning(
            traj_lon, lateral_constraints, d_reference=None
        )
        cr_trajectory = self.transform_merge_trajectory(trajectory)
        return cr_trajectory

    def construct_s_reference(self):
        """
        Constructs the longitudinal reference from the initially-planned trajectory.
        """
        x_ref = list()
        for state in self._initial_trajectory.states_in_time_interval(
            self._cut_off_time_step, self._ego_vehicle.prediction.final_time_step
        ):
            if state is None:
                state = self._ego_vehicle.initial_state
            # TODO: create new instead of using QP planner
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            # TODO: get correct velocity. In state there are two variables related to velocity: velocity and velocity_y
            x_ref.append(MIQPLongState(pos[0], state.velocity, 0.0, 0.0, 0.0))
        return MIQPLongReference(x_ref)

    # TODO: create new trajectory construction for MIQP
    def transform_merge_trajectory(self, trajectory: QPTrajectory):
        """
        Transforms and merges the trajectory (before and after repairing)
        """
        cartesian_traj_points = list()
        for state in trajectory.states:
            cart_pos = self.vehicle_configuration.curvilinear_coordinate_system.convert_to_cartesian_coords(
                state.position[0], state.position[1]
            )
            orientation_interpolated = np.interp(
                state.position[0],
                self.vehicle_configuration.curvilinear_coordinate_system.ref_pos,
                self.vehicle_configuration.curvilinear_coordinate_system.ref_theta,
            )

            v = state.v / np.cos(state.orientation - orientation_interpolated)
            cartesian_traj_points.append(
                TrajPoint(
                    t=state.t,
                    x=cart_pos[0],
                    y=cart_pos[1],
                    theta=state.orientation,
                    v=v,
                    a=state.a,
                    kappa=state.kappa,
                    kappa_dot=state.kappa_dot,
                    j=state.j,
                    lane=state.lane,
                )
            )

        traj = QPTrajectory(cartesian_traj_points, TrajectoryType.CARTESIAN)

        traj._u_lon = trajectory.u_lon
        traj._u_lat = trajectory.u_lat
        cr_traj_repaired = traj.convert_to_cr_trajectory(
            self._vehicle_configuration.wheelbase
        )
        # TODO: fix time step
        if self._cut_off_time_step == 1:
            remaining_states = [self._ego_vehicle.initial_state]
        elif self._cut_off_time_step == self._start_time_step:
            remaining_states = []
        else:
            remaining_states = [
                self._ego_vehicle.initial_state
            ] + self._initial_trajectory.states_in_time_interval(
                self._start_time_step + 1, self._cut_off_time_step - 1
            )
        for state in cr_traj_repaired.state_list:
            state.time_step += self._cut_off_time_step
        state_list = [
            CustomState(
                time_step=state.time_step,
                position=state.position,
                velocity=state.velocity,
                orientation=state.orientation,
                acceleration=state.acceleration,
            )
            for state in remaining_states + cr_traj_repaired.state_list
        ]
        cr_traj_repaired = Trajectory(self._start_time_step, state_list)
        return cr_traj_repaired

    def config_settings(self):
        """
        Configuration settings.
        """
        config_file = "config_" + str(self._scenario.scenario_id) + ".yaml"
        config_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../../config")
        )
        if not os.path.exists(os.path.join(config_dir, config_file)):
            config_file = "config_default.yaml"
        with open(os.path.join(config_dir, config_file), "r") as stream:
            try:
                settings = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
        if config_file == "config_default.yaml":
            settings["vehicle_settings"][
                self._planning_problem.planning_problem_id
            ] = settings["vehicle_settings"].pop(1)
        return settings


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region
