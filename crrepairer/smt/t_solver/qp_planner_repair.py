
from commonroad_qp_planner.qp_planner import QPPlanner, QPLongState, QPLongReference
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import set_up, convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType

from stl_crmonitor.crmonitor.predicates.rule import PropositionNode

from commonroad_repairer.crrepairer.cut_off.tc import TC
from commonroad_repairer.crrepairer.smt.t_solver.rule_constraints import RuleConstraints
from commonroad_repairer.crrepairer.smt.monitor_wrapper import STLRuleMonitor

from commonroad.scenario.trajectory import Trajectory, State
from commonroad.scenario.scenario import DynamicObstacle, TrajectoryPrediction, ObstacleType
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle

from typing import List
import yaml
import os
import time


class QPPlannerRepair(QPPlanner):
    """
    QP-planner for trajectory repairing starting from the cut-off state.
    """
    def __init__(self,
                 rule_monitor: STLRuleMonitor,
                 tc_object: TC,
                 sel_proposition: List[PropositionNode],
                 verbose=False):
        # initialize the scenario and planning problem
        self._scenario = rule_monitor.world_state.scenario
        self._ego_vehicle = tc_object.ego_vehicle
        self._planning_problem = rule_monitor.world_state.planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory

        # set the cut-off state as the initial state
        self._cut_off_time_step = tc_object.tc_time_step
        self._N = tc_object.N
        if self._cut_off_time_step == 0:
            self._cut_off_state = self._ego_vehicle.initial_state
        else:
            self._cut_off_state = self._initial_trajectory.state_at_time_step(self._cut_off_time_step)
        self._reformulate_planning_problem()
        self._time_horizon = round((self._N - self._cut_off_time_step) * self._scenario.dt, 1)
        self._planning_problem.initial_state = self._cut_off_state

        # load and set up the configuration
        self._settings = self.config_settings()
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
                         self._vehicle_configuration,
                         qp_long_parameters=self._settings["qp_planner"]["longitudinal_parameters"],
                         qp_lat_parameters=self._settings["qp_planner"]["lateral_parameters"],
                         verbose=verbose)

        # construct the rule constraints based on the traffic rules and proposition to be repaired
        self._rule_constraints = RuleConstraints(tc_object,
                                                 rule_monitor,
                                                 sel_proposition,
                                                 self._vehicle_configuration,
                                                 self._initial_trajectory)

    @property
    def rule_constraints(self):
        return self._rule_constraints

    @property
    def total_time_steps(self):
        return self._N - self._cut_off_time_step

    def _reformulate_planning_problem(self, ):
        """
        Reformulates the planning problem: initial state and goal
        """
        if not hasattr(self._planning_problem, "initial_state"):
            raise ValueError("<QPPlannerRepair>: the initial state needs to be specified")
        self._planning_problem.initial_state = self._ego_vehicle.initial_state
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)

    def plan(self):
        """
        Plans a trajectory starting from the cut-off state.
            First: constructs the constraints and the reference path
            Then: generates the trajectory in both longitudinal and lateral directions
        """
        print('* \t<QPPlanner>: process starts')
        print('* \t\t Longitudinal optimization')
        long_constr = self._rule_constraints.longitudinal_constraints()
        reference_lon = self.construct_s_reference()
        start_time_lon = time.time()
        traj_lon, status = self.longitudinal_trajectory_planning(long_constr, reference_lon,
                                                                 safe_dis_modes=self._rule_constraints.
                                                                 safe_distance_modes)
        print('* \t\t -- run time {} s --'.format(round(time.time()-start_time_lon, 3)))
        if status is not 'optimal':
            return None
            # raise ValueError('<QPPlannerRepair/_longitudinal_trajectory_planning>: failed')
        print('* \t\t Lateral optimization')
        lat_constr = self._rule_constraints.lateral_constraints(traj_lon)
        lat_constr.select_proposition = long_constr.select_proposition
        start_time_lat = time.time()
        trajectory, status = self.lateral_trajectory_planning(traj_lon,
                                                              lat_constr,
                                                              None)
        print('* \t\t -- run time {} s --'.format(round(time.time()-start_time_lat, 3)))
        # convert trajectory to cartesian space
        if status is not 'optimal':
            return None
            # raise ValueError('<QPPlannerRepair/_lateral_trajectory_planning>: failed')
        cr_trajectory = self.transform_merge_trajectory(trajectory)
        return cr_trajectory

    def construct_s_reference(self):
        """
        Constructs the longitudinal reference from the initially-planned trajectory.
        """
        x_ref = list()
        for state in self._initial_trajectory.states_in_time_interval(self._cut_off_time_step,
                                                                      self._ego_vehicle.prediction.final_time_step):
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            x_ref.append(QPLongState(pos[0], state.velocity, 0., 0., 0.))
        return QPLongReference(x_ref)

    def construct_d_reference(self):
        """
        Constructs the lateral reference from the initially-planned trajectory.
        """
        d_ref = list()
        for state in self._initial_trajectory.states_in_time_interval(self._cut_off_time_step,
                                                                      self._ego_vehicle.prediction.final_time_step):
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            d_ref.append(pos[1])
        return d_ref

    def convert_traj_to_ego_vehicle(self,
                                    cr_trajectory: Trajectory,
                                    vehicle_id: int = 0) -> DynamicObstacle:
        """
        Converts trajectory object to CommonRoad obstacle with specified width and length
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
        """
        Transforms and merges the trajectory (before and after repairing)
        """
        cartesian_traj_points = list()
        for state in trajectory.states:
            cart_pos = self.vehicle_configuration.curvilinear_coordinate_system.convert_to_cartesian_coords(
                state.position[0], state.position[1])
            cartesian_traj_points.append(TrajPoint(
                t=state.t, x=cart_pos[0], y=cart_pos[1], theta=state.orientation, v=state.v,
                a=state.a,
                kappa=state.kappa, kappa_dot=state.kappa_dot, j=state.j, lane=state.lane))

        traj = QPTrajectory(cartesian_traj_points, TrajectoryType.CARTESIAN)

        traj._u_lon = trajectory.u_lon
        traj._u_lat = trajectory.u_lat
        cr_traj_repaired = traj.convert_to_cr_trajectory(self._vehicle_configuration.wheelbase)
        if self._cut_off_time_step == 0:
            remaining_states = [self._ego_vehicle.initial_state]
        else:
            remaining_states = [self._ego_vehicle.initial_state] + \
                               self._initial_trajectory.states_in_time_interval(1, self._cut_off_time_step-1)
        for state in cr_traj_repaired.state_list:
            state.time_step += self._cut_off_time_step
        cr_traj_repaired.state_list = remaining_states + cr_traj_repaired.state_list
        return cr_traj_repaired

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
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region
