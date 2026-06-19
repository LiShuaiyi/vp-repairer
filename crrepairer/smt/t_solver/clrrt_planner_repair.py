import numpy as np
import time

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad_route_planner.route_planner import RoutePlanner

from informed_cl_rrt.rule_informed_cl_rrt import RuleInformedCLRRT
from informed_cl_rrt.search_space import SearchSpace
from miqp_planner.miqp_initialization import set_up_miqp
from miqp_planner.miqp_planner_base import MIQPPlanner
from miqp_planner.miqp_lat_planner import MIQPLatPlanner
from miqp_planner.miqp_long_planner import MIQPLongState, MIQPLongReference, MIQPLongPlanner
from miqp_planner.miqp_constraints_manual import (
    LongitudinalConstraint,
    LateralConstraint,
    RuleConstraint as RuleConstraintMIQPManual
)
from miqp_planner.miqp_constraints_reach import RuleConstraintMIQPReach

from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import compute_initial_state

from crrepairer.smt.monitor_wrapper import PropositionNode
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.cut_off.tc import TC
from crrepairer.utils.configuration import RepairerConfiguration, IntersectionType
from crrepairer.smt.t_solver.qp_planner_repair import update_goal_state_extension, update_goal_state

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState

from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle

from typing import List, Optional
import yaml
import os

from reference_path.reference_generator import ReferenceGenerator


class CLRRTPlannerRepair:
    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        tc_object: TC,
        config: RepairerConfiguration,
    ):
        self.rrt_time = None
        self.rule_monitor = rule_monitor
        self.tc_object = tc_object

        self.config: Optional[RepairerConfiguration] = None
        self.reset(config)

        # initialize from the TC object
        self._ego_vehicle = tc_object.ego_vehicle
        self._initial_trajectory: Optional[Trajectory] = self._ego_vehicle.prediction.trajectory
        self._start_time_step = tc_object.ego_vehicle.initial_state.time_step

        # empty objects
        self._N: Optional[int] = None
        self._cut_off_time_step: Optional[float, int] = None
        self._cut_off_state: Optional[CustomState, InitialState] = None

    def reset(self, config: RepairerConfiguration = None,
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

            self.config.planning_problem.initial_state = InitialState(
                position=self._cut_off_state.position,
                velocity=self._cut_off_state.velocity,
                orientation=self._cut_off_state.orientation,
                time_step=self._cut_off_state.time_step,
                acceleration=getattr(self._cut_off_state, "acceleration", 0.0),
                # not needed but mandatory field
                yaw_rate=0,
                slip_angle=0,
            )
            self.config.planning_problem.goal = update_goal_state(
                self._cut_off_state,
                self._initial_trajectory,
                self.config.scenario.lanelet_network
            )

    def plan(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the config file
        config_path = os.path.join(script_dir, "config_clrrt.yaml")
        settings = load_settings(config_path)
        self.config.scenario.remove_obstacle(
            self.config.scenario.obstacle_by_id(self._ego_vehicle.obstacle_id)
        )
        route_planner = RoutePlanner(self.config.scenario.lanelet_network, self.config.planning_problem)
        candidate_holder = route_planner.plan_routes()
        route = candidate_holder.retrieve_first_route()
        route_id = route.lanelet_ids[0]
        lane = self.config.scenario.lanelet_network.find_lanelet_by_id(route_id)

        self._monitor_ego_vehicle = self.rule_monitor.world.vehicle_by_id(
            self._ego_vehicle.obstacle_id
        )
        reference_path = self._monitor_ego_vehicle.ref_path_lane

        # rg = ReferenceGenerator(
        #     reference_path=reference_path.center_vertices,
        #     scenario=self.config.scenario,
        #     planning_problem=self.config.planning_problem,
        #     first_violation_state=self._ego_vehicle.state_at_time(
        #         self.tc_object.tv_time_step
        #     ),
        #     settings=settings,
        # )
        # reference_path = rg.choose_optimal_trajectory(obstacle_position=None)
        search_space = SearchSpace(
            self.config.scenario,
            self.config.planning_problem,
            SearchSpace.SampleMode.RP_GMM,
            gaussian_deviation=0.5,
            reference_path=reference_path,
        )

        rrt = RuleInformedCLRRT(
            self.config.scenario, self.config.planning_problem, search_space, reference_path=reference_path,
            world=self.rule_monitor.world,
            rule_evaluators=self.rule_monitor.rule_eval, ego_id=self._ego_vehicle.obstacle_id
        )
        time_start = time.time()
        final_path, goal_cost, samples_taken, sample_time, valid_sample = (
            rrt.expand_tree()
        )

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
        for state in final_path:
            state.time_step += self._cut_off_time_step
        state_list = [
            CustomState(
                time_step=state.time_step,
                position=state.position,
                velocity=state.velocity,
                orientation=state.orientation,
                acceleration=getattr(state, "acceleration", 0.0),
            )
            for state in remaining_states + final_path
        ]
        cr_traj_repaired = Trajectory(self._start_time_step, state_list)
        self.rrt_time = time.time() - time_start
        return cr_traj_repaired

def update_goal_state(initial_state: InitialState,
                      initial_trajectory: Trajectory,
                      lanelet_networks: LaneletNetwork) -> GoalRegion:
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    depth = 1
    initial_lanelet_id = lanelet_networks.find_lanelet_by_position([initial_state.position])[0][0]
    lanelet = lanelet_networks.find_lanelet_by_id(initial_lanelet_id)
    i = 0
    while i < depth and len(lanelet.successor) != 0:
        if len(lanelet.successor) > 1:
            lanelet = lanelet_networks.find_lanelet_by_id(lanelet.successor[0])
        else:
            lanelet = lanelet_networks.find_lanelet_by_id(lanelet.successor[0])
        i += 1
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_position = [ini_final_state.position[0] + 20 * np.cos(ini_final_state.orientation),
                     ini_final_state.position[1] + 20 * np.sin(ini_final_state.orientation)]
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list))
    goal_state = CustomState(
        position=Rectangle(20, 5, np.asarray(goal_position)),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region

def load_settings(config_name):
    """
    Loads the settings of given scenario
    :param config_name: the name of configuration file
    :return: settings
    """
    with open(config_name, "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return settings
