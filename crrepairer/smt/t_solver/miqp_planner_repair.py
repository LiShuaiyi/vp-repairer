from miqp_planner.miqp_planner_base import MIQPPlanner
from miqp_planner.miqp_long_planner import MIQPLongState, MIQPLongReference
from miqp_planner.miqp_constraints import LongitudinalConstraint, LateralConstraint

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import set_up, convert_pos_curvilinear

from crrepairer.smt.monitor_wrapper import PropositionNode

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.rule_constraints import RuleConstraints
from crrepairer.smt.monitor_wrapper import STLRuleMonitor

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.scenario import DynamicObstacle, TrajectoryPrediction, ObstacleType
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle

from typing import List
import yaml
import os
import time

class MIQPPlannerRepair(MIQPPlanner):
    def __init__(self,
                 rule_monitor: STLRuleMonitor,
                 tc_object: TC,
                 sel_proposition: List[PropositionNode],
                 proposition_full: List[PropositionNode],
                 planning_problem: PlanningProblem):
        # initialize the scenario and planning problem
        self._scenario = rule_monitor.world.scenario
        self._ego_vehicle = tc_object.ego_vehicle
        self._planning_problem = planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory

        # set the cut-off state as the initial state
        self._cut_off_time_step = tc_object.tc_time_step
        self._N = tc_object.N
        if self._cut_off_time_step == 0:
            self._cut_off_state = self._ego_vehicle.initial_state
        else:
            self._cut_off_state = self._initial_trajectory.state_at_time_step(self._cut_off_time_step)
        self._time_horizon = round((self._N - self._cut_off_time_step) * self._scenario.dt, 1)
        self._planning_problem.initial_state = InitialState(
            position=self._cut_off_state.position,
            velocity=self._cut_off_state.velocity,
            orientation=self._cut_off_state.orientation,
            time_step=self._cut_off_state.time_step,
            acceleration=self._cut_off_state.acceleration,
            # not needed but mandatory field
            yaw_rate=0,
            slip_angle=0
        )
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)
        # load and set up the configuration
        self._settings = self.config_settings() #TODO: need to add self setting for miqp
        self._vehicle_configuration: PlanningConfigurationVehicle = set_up(self._settings,
                                                                           self._scenario,
                                                                           self._planning_problem)

        # update the vehicle shape
        self._vehicle_configuration.width = self._ego_vehicle.obstacle_shape.width
        self._vehicle_configuration.length = self._ego_vehicle.obstacle_shape.length

        # initialize the QP planner
        super().__init__(self._scenario,
                         self._planning_problem,
                         self._time_horizon,
                         self._vehicle_configuration)

        # construct constraints
        self._long_constraints = LongitudinalConstraint(tc_object,
                                                        rule_monitor,
                                                        sel_proposition,
                                                        proposition_full,
                                                        self._vehicle_configuration,
                                                        self._initial_trajectory)

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
        print('* \t\t Longitudinal optimization')
        reference_lon = self.construct_s_reference()
        traj_lon = self.longitudinal_trajectory_planning(reference_lon, self._long_constraints)
        # TODO: fix inputs
        lateral_constraints = LateralConstraint(self._long_constraints._tc_obj,
                                                self._long_constraints._rule_monitor,
                                                self._long_constraints._veh_config,
                                                self._long_constraints.target_lanes,
                                                traj_lon,
                                                self._long_constraints.sel_prop_full)
        lateral_constraints.create_d_constraints(traj_lon)
        self.lateral_trajectory_planning(traj_lon, lateral_constraints, d_reference=None)


    def construct_s_reference(self):
        """
        Constructs the longitudinal reference from the initially-planned trajectory.
        """
        x_ref = list()
        for state in self._initial_trajectory.states_in_time_interval(self._cut_off_time_step,
                                                                      self._ego_vehicle.prediction.final_time_step):
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            # TODO: get correct velocity. In state there are two variables related to velocity: velocity and velocity_y
            x_ref.append(MIQPLongState(pos[0], state.velocity, 0., 0., 0.))
        return MIQPLongReference(x_ref)

    def construct_d_reference(self):
        pass

    def convert_traj_to_ego_vehicle(self):
        pass

    def transform_merge_trajectory(self):
        pass

    def config_settings(self):
        """
        Configuration settings.
        """
        config_file = 'config_' + str(self._scenario.scenario_id) + '.yaml'
        config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                   "../../../config"))
        if not os.path.exists(os.path.join(config_dir, config_file)):
            config_file = 'config_default.yaml'
        with open(os.path.join(config_dir, config_file), 'r') as stream:
            try:
                settings = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
        if config_file == 'config_default.yaml':
            # for HighD scnarios
            settings["vehicle_settings"][self._planning_problem.planning_problem_id] = \
                settings["vehicle_settings"].pop(1)
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
    goal_state = CustomState(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region
