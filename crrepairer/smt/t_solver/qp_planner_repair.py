from commonroad_qp_planner.qp_planner import (
    QPPlanner,
    QPLongState,
    QPLongDesired,
    LonConstraints,
)
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import (
    set_up,
    convert_pos_curvilinear,
    create_optimization_configuration_vehicle,
)
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType
from commonroad_qp_planner.utils import plot_result, plot_position_constraints

from crrepairer.smt.monitor_wrapper import PropositionNode

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.rule_constraints_manual import RuleConstraintsManual
from crrepairer.smt.t_solver.rule_constraints_reach import RuleConstraintsReach
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


class QPPlannerRepair(QPPlanner):
    """
    QP-planner for trajectory repairing starting from the cut-off state.
    """

    def __init__(
            self,
            rule_monitor: STLRuleMonitor,
            tc_object: TC,
            sel_proposition: List[PropositionNode],
            proposition_full: List[PropositionNode],
            planning_problem: PlanningProblem,
            verbose=False,
    ):
        # initialize the scenario and planning problem
        self._scenario = rule_monitor.world.scenario
        self._ego_vehicle = tc_object.ego_vehicle
        self._start_time_step = tc_object.ego_vehicle.initial_state.time_step
        self._planning_problem = planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory

        # set the cut-off state as the initial state
        self._cut_off_time_step = tc_object.tc_time_step
        self._N = tc_object.N
        if self._cut_off_time_step == 0:
            self._cut_off_state = self._ego_vehicle.initial_state
        else:
            self._cut_off_state = self._initial_trajectory.state_at_time_step(
                self._cut_off_time_step
            )
        self._time_horizon = round(
            (self._N - self._cut_off_time_step) * self._scenario.dt, tc_object.round_tolerance
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
        # self._vehicle_configuration: PlanningConfigurationVehicle = set_up(self._settings,
        #                                                                    self._scenario,
        #                                                                    self._planning_problem)
        self._vehicle_configuration: PlanningConfigurationVehicle = (
            create_optimization_configuration_vehicle(
                self._scenario, self._planning_problem, self._settings
            )
        )

        # use the coordinate system from the world
        # TODO: separate intersection and interstate
        if rule_monitor.scenario_type == "intersection":
            self._vehicle_configuration.curvilinear_coordinate_system = (
                rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id)
                .ref_path_lane
                .clcs
            )
        else:
            self._vehicle_configuration.curvilinear_coordinate_system = (
                rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id)
                .get_lane(0)
                .clcs
            )

        # update the vehicle shape
        self._vehicle_configuration.width = self._ego_vehicle.obstacle_shape.width
        self._vehicle_configuration.length = self._ego_vehicle.obstacle_shape.length
        # self._rule_constraints = RuleConstraintsManual(tc_object,
        #                                                rule_monitor,
        #                                                sel_proposition,
        #                                                proposition_full,
        #                                                self._vehicle_configuration,
        #                                                self._initial_trajectory)

        # construct the rule constraints based on the traffic rules and proposition to be repaired
        self._rule_constraints = RuleConstraintsReach(
            tc_object,
            rule_monitor,
            sel_proposition,
            proposition_full,
            self._vehicle_configuration,
            self._initial_trajectory,
            self._planning_problem,
        )

        # initialize the QP planner
        super().__init__(
            vehicle_configuration=self._vehicle_configuration,
            num_planning_steps=self._N - self._cut_off_time_step,
            qp_long_parameters=self._settings["qp_planner"]["longitudinal_parameters"],
            qp_lat_parameters=self._settings["qp_planner"]["lateral_parameters"],
            verbose=verbose,
            safe_dis_modes=None,
        )

    @property
    def rule_constraints(self):
        return self._rule_constraints

    @property
    def total_time_steps(self):
        return self._N - self._cut_off_time_step

    def plan(self):
        """
        Plans a trajectory starting from the cut-off state.
            First: constructs the constraints and the reference path
            Then: generates the trajectory in both longitudinal and lateral directions
        """
        print("* \t<QPPlanner>: process starts")
        print("* \t\t Longitudinal optimization")
        long_constr = self._rule_constraints.longitudinal_constraints(
            self._vehicle_configuration
        )
        reference_lon = self.construct_s_reference(long_constr)
        self.reset(self._scenario)
        start_time_lon = time.time()
        self.step(
            self._planning_problem.initial_state,
            self._planning_problem.initial_state.velocity,
        )

        traj_lon, status = self.longitudinal_trajectory_planning(
            long_constr, reference_lon
        )
        print(
            "* \t\t -- run time {} s --".format(round(time.time() - start_time_lon, 3))
        )
        if status is not "optimal":
            return None
            # raise ValueError('<QPPlannerRepair/_longitudinal_trajectory_planning>: failed')
        print("* \t\t Lateral optimization")
        lat_constr = self._rule_constraints.lateral_constraints(
            traj_lon, self._vehicle_configuration
        )
        lat_constr.select_proposition = long_constr.select_proposition
        start_time_lat = time.time()
        trajectory, status = self.lateral_trajectory_planning(
            traj_lon, lat_constr, None
        )
        print(
            "* \t\t -- run time {} s --".format(round(time.time() - start_time_lat, 3))
        )
        # convert trajectory to cartesian space
        if status is not "optimal":
            return None
            # raise ValueError('<QPPlannerRepair/_lateral_trajectory_planning>: failed')
        cr_trajectory = self.transform_merge_trajectory(trajectory)

        # plot_position_constraints(trajectory, (long_constr.s_hard_min, long_constr.s_hard_max), (lat_constr.d_hard_min, lat_constr.d_hard_max))
        return cr_trajectory

    def construct_s_reference(self, lon_constr: LonConstraints):
        """
        Constructs the longitudinal reference from the initially-planned trajectory.
        """
        x_ref = list()
        # based on the initially planned trajectory
        # for state in self._initial_trajectory.states_in_time_interval(
        #         self._cut_off_time_step, self._ego_vehicle.prediction.final_time_step
        # ):
        #     pos = convert_pos_curvilinear(state, self._vehicle_configuration)
        #     x_ref.append(QPLongState(pos[0], state.velocity, 0.0, 0.0, 0.0))
        # use the constraint as the reference
        for i in range(len(lon_constr.s_hard_min)):
            pos = (lon_constr.s_hard_max[i] + lon_constr.s_hard_min[i])/2
            vel = (lon_constr.v_max[i] + lon_constr.v_min[i])/2
            x_ref.append(QPLongState(pos, vel, 0.0, 0.0, 0.0))
        return QPLongDesired(x_ref)

    def construct_d_reference(self):
        """
        Constructs the lateral reference from the initially-planned trajectory.
        """
        d_ref = list()
        for state in self._initial_trajectory.states_in_time_interval(
                self._cut_off_time_step, self._ego_vehicle.prediction.final_time_step
        ):
            pos = convert_pos_curvilinear(state, self._vehicle_configuration)
            d_ref.append(pos[1])
        return d_ref

    def convert_traj_to_ego_vehicle(
            self, cr_trajectory: Trajectory, vehicle_id: int = 0
    ) -> DynamicObstacle:
        """
        Converts trajectory object to CommonRoad obstacle with specified width and length
        :param vehicle_id: ID of ego vehicle
        :return: The CommonRoad DynamicObstacle object containing the current trajectory
        """
        # get trajectory
        shape = Rectangle(
            self._vehicle_configuration.length, self._vehicle_configuration.width
        )
        pred = TrajectoryPrediction(cr_trajectory, shape)

        # create new object
        ego = DynamicObstacle(
            obstacle_id=vehicle_id,
            obstacle_type=ObstacleType.CAR,
            prediction=pred,
            obstacle_shape=shape,
            initial_state=self._ego_vehicle.initial_state,
        )
        return ego

    def transform_merge_trajectory(self, trajectory_CLCS: QPTrajectory):
        """
        Transforms and merges the trajectory (before and after repairing)
        """
        trajectory = self.transform_trajectory_to_cartesian_coordinates(trajectory_CLCS)
        cr_traj_repaired = trajectory.convert_to_cr_ego_vehicle(
            self._vehicle_configuration.width,
            self._vehicle_configuration.length,
            self._vehicle_configuration.wheelbase,
            self._vehicle_configuration.wb_ra,
            vehicle_id=self._ego_vehicle.obstacle_id,
        )

        if self._cut_off_time_step == 0:
            remaining_states = [self._ego_vehicle.initial_state]
        else:
            remaining_states = [
                                   self._ego_vehicle.initial_state
                               ] + self._initial_trajectory.states_in_time_interval(
                self._start_time_step + 1, self._cut_off_time_step - 1
            )
        for state in cr_traj_repaired.prediction.trajectory.state_list:
            state.time_step += self._cut_off_time_step
        state_list = [
            CustomState(
                time_step=state.time_step,
                position=state.position,
                velocity=state.velocity,
                orientation=state.orientation,
                acceleration=state.acceleration,
            )
            for state in remaining_states
            + cr_traj_repaired.prediction.trajectory.state_list
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
            # for HighD scnarios
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
