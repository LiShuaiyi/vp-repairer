import numpy as np
import time

from miqp_planner.miqp_initialization import set_up_miqp
from miqp_planner.miqp_planner_base import MIQPPlanner
from miqp_planner.miqp_lat_planner import MIQPLatPlanner
from miqp_planner.miqp_long_planner import MIQPLongState, MIQPLongReference, MIQPLongPlanner
from miqp_planner.miqp_constraints import (
    LongitudinalConstraint,
    LateralConstraint,
    RuleConstraint,
)

from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import compute_initial_state

from crrepairer.smt.monitor_wrapper import PropositionNode
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.cut_off.tc import TC
from crrepairer.utils.configuration import RepairerConfiguration, IntersectionType

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState

from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle

from typing import List, Optional
import yaml
import os


class MIQPPlannerRepair(MIQPPlanner):
    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        tc_object: TC,
        config: RepairerConfiguration,
    ):
        self.rule_monitor = rule_monitor
        self.tc_object = tc_object

        # initialize from the TC object
        self._ego_vehicle = tc_object.ego_vehicle
        self._initial_trajectory: Optional[Trajectory] = self._ego_vehicle.prediction.trajectory
        self._start_time_step = tc_object.ego_vehicle.initial_state.time_step
        self._round_tolerance = tc_object.round_tolerance

        # initialize from the configuration
        self.config: Optional[RepairerConfiguration] = None
        self.reset(config)
        self._settings = self.config_settings(config)

        # update the configuration
        self._monitor_ego_vehicle = rule_monitor.world.vehicle_by_id(
            rule_monitor.vehicle_id
        )

        # update and initialize the vehicle configuration
        # >>> side note: the initial state is not updated to keep the reference path possibly long
        self.config.planning_problem.goal = update_goal_state(self._initial_trajectory)
        self._vehicle_configuration = set_up_miqp(
            self._settings,
            self.config.scenario,
            self.config.planning_problem,
            self._monitor_ego_vehicle,
        )
        if rule_monitor.scenario_type == "intersection":
            self._vehicle_configuration.CLCS = (
                rule_monitor.world.vehicle_by_id(
                    self._ego_vehicle.obstacle_id
                ).ref_path_lane.clcs
            )
        else:
            self._vehicle_configuration.CLCS = (
                rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id)
                .get_lane(0)
                .clcs
            )

        # empty objects
        self._N: Optional[int] = None
        self._cut_off_time_step: Optional[float, int] = None
        self._cut_off_state: Optional[CustomState, InitialState] = None
        self._time_horizon: Optional[float] = None
        self._constraints: Optional[RuleConstraint] = None

        # initialize the MIQP planner
        super().__init__(config)

        # initialize the Longitudinal Planner
        self.long_planner = MIQPLongPlanner(
            config=self.config,
        )

        # initialize the Lateral Planner
        self.lat_planner = MIQPLatPlanner(
            config=self.config,
        )

    def construct_constraints(self,
                              sel_proposition: List[PropositionNode],
                              proposition_full: List[PropositionNode],):
        if self._vehicle_configuration is not None:
            self._constraints = RuleConstraint(
                self.tc_object,
                self.rule_monitor,
                sel_proposition,
                proposition_full,
                self._vehicle_configuration,
                self._initial_trajectory,
                self._start_time_step,
            )
        else:
            assert self.config is not None, "<Repairer.construct_constraints(). No Configuration object initialized>"

    def reset(self, config: RepairerConfiguration = None,
              initial_trajectory: Optional[Trajectory] = None,
              tc_object: TC = None,
              rule_monitor: STLRuleMonitor = None):
        """
        Initializes/resets configuration of the repairer for re-planning purposes
        """

        # set updated config
        if config is not None:
            self.config = config
        else:
            assert self.config is not None, "<Repairer.reset(). No Configuration object provided>"

        if rule_monitor is not None:
            self.rule_monitor = rule_monitor

        if tc_object is not None:
            self.tc_object = tc_object
            self._N = tc_object.N

            self._cut_off_time_step = tc_object.tc_time_step
            # initialize the cut-off state
            if self._cut_off_time_step == self._start_time_step:
                self._cut_off_state = self._ego_vehicle.initial_state
            else:
                self._cut_off_state = self._initial_trajectory.state_at_time_step(
                    self._cut_off_time_step
                )

            self._time_horizon = self.config.miqp_planner.horizon = round(
                (self._N - self._cut_off_time_step) * self.config.scenario.dt,
                self._round_tolerance,
            )

            self.config.planning_problem.initial_state = InitialState(
                position=self._cut_off_state.position,
                velocity=self._cut_off_state.velocity,
                orientation=self._cut_off_state.orientation,
                time_step=self._cut_off_state.time_step,
                acceleration=self._cut_off_state.acceleration,
                # not needed but mandatory field
                yaw_rate=0,
                slip_angle=0,
            )

            # use the coordinate system from the world
            if self.config.repair.scenario_type == "interstate":
                self._vehicle_configuration.curvilinear_coordinate_system = (
                    self._monitor_ego_vehicle
                    .get_lane(0)
                    .clcs
                )
            # update the config from the qp planner
            self.config.vehicle.qp_veh_config = self._vehicle_configuration
            # update the vehicle shape
            self._vehicle_configuration.width = self.time_invariant_constraints.width = (
                self._ego_vehicle.obstacle_shape.width)
            self._vehicle_configuration.length = self.time_invariant_constraints.length = (
                self._ego_vehicle.obstacle_shape.length)

            # update the initial state accordingly
            self.initial_state = compute_initial_state(
                self.config.planning_problem.initial_state,
                self.config.vehicle.qp_veh_config
            )

            # reset the N and horizon
            self.lat_planner.reset(nr_steps=self._N - self._cut_off_time_step, horizon=self._time_horizon)
            self.long_planner.reset(nr_steps=self._N - self._cut_off_time_step,
                                    horizon=self._time_horizon, initial_state=self.initial_state)

        if initial_trajectory is not None:
            self._initial_trajectory = initial_trajectory
            # update the goal state for better planning/updating the high-level route

    @property
    def total_time_steps(self):
        return self._N - self._cut_off_time_step

    def plan(self):
        """
        Plans a trajectory starting from the cut-off state.
            First: constructs the constraints and the reference path
            Then: generates the trajectory in both longitudinal and lateral directions
        """
        print("* \t\t MIQP Longitudinal optimization")
        start_time_lon = time.time()
        reference_lon = self.construct_s_reference()
        self._constraints.construct_longitudinal_constraints(self._cut_off_time_step)
        traj_lon = self.longitudinal_trajectory_planning(
            reference_lon,
            self._constraints.longitudinal_constraints,
            self._constraints.safe_distance_modes,
            self._constraints.target_vehicle,
        )
        print(
            "* \t\t -- run time {} s --".format(round(time.time() - start_time_lon, 3))
        )
        if traj_lon is None:
            return None
        print("* \t\t MIQP Lateral optimization")
        # TODO: fix inputs
        start_time_lat = time.time()

        self._constraints.create_d_constraints(traj_lon)
        # select_proposition = self._constraints.sel_prop_full
        trajectory = self.lateral_trajectory_planning(
            traj_lon, self._constraints.lateral_constraints
        )
        print(
            "* \t\t -- run time {} s --".format(round(time.time() - start_time_lat, 3))
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
            cart_pos = self.vehicle_configuration.qp_veh_config.CLCS.convert_to_cartesian_coords(
                state.position[0], state.position[1]
            )
            orientation_interpolated = np.interp(
                state.position[0],
                self.vehicle_configuration.qp_veh_config.CLCS.ref_pos,
                self.vehicle_configuration.qp_veh_config.CLCS.ref_theta,
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
            self._vehicle_configuration.wheelbase,
            self._vehicle_configuration.wb_ra
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

    def config_settings(self, config: RepairerConfiguration):
        """
        Configuration settings.
        """
        # different vehicle setting for inD scenarios
        if (
            config.repair.scenario_type == "intersection"
            and config.repair.intersection_type == IntersectionType.DATASET
        ):
            config_file = "config_intersection.yaml"
        else:
            config_file = "config_" + str(self.config.scenario.scenario_id) + ".yaml"
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
        if (
            config_file == "config_default.yaml"
            or config_file == "config_intersection.yaml"
        ):
            settings["vehicle_settings"][
                self.config.planning_problem.planning_problem_id
            ] = settings["vehicle_settings"].pop(1)
        settings["scenario_type"] = config.repair.scenario_type
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
