import copy
import numpy as np
from typing import Optional

import commonroad_qp_planner.qp_long_planner as qp_long_planner_module
import commonroad_qp_planner.qp_lat_planner as qp_lat_planner_module
from commonroad_qp_planner.qp_planner import (
    QPPlanner,
    QPLongState,
    LonConstraints,
    QPLongDesired,
    QPLatReference,
    QPLatDesired
)
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import (
    set_up,
    convert_pos_curvilinear,
    compute_initial_state,
    create_optimization_configuration_vehicle,
)
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory
from commonroad_qp_planner.trajectory import TrajPoint, TrajectoryType

from crrepairer.smt.monitor_wrapper import PropositionNode

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.rule_constraints import RuleConstraintsManual
from crrepairer.smt.t_solver.rule_constraints_reach import RuleConstraintsReach
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.shape import shape_dimensions

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.scenario import (
    DynamicObstacle,
    TrajectoryPrediction,
    ObstacleType,
    Scenario
)
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad_route_planner.route_planner import RoutePlanner

from typing import List
import yaml
import os
import time
import sys
from types import MethodType

qp_long_planner_module.GUROBI = qp_long_planner_module.OSQP
qp_lat_planner_module.GUROBI = qp_lat_planner_module.OSQP


def _tv_constraints_with_hard_rule_bounds(planner, c_long, c_ti):
    """Apply collision bounds with slack and rule bounds without slack.

    The upstream planner selects either ``s_soft_*`` or ``s_hard_*``.  Manual
    intersection repair needs both: collision estimates can initially be
    infeasible, while stop-line/conflict-entry bounds must never be crossed.
    """
    constraints = []
    for k in range(planner.N):
        constraints.append(
            planner._x[:, k + 1]
            == planner._A @ planner._x[:, k] + planner._B @ planner._u[:, k]
        )
        if c_long.s_soft_max[k] != np.inf:
            constraints.append(
                planner._x[0, k + 1] - planner._u[:, planner.N + 1]
                <= c_long.s_soft_max[k]
            )
        if c_long.s_hard_max[k] != np.inf:
            constraints.append(
                planner._x[0, k + 1] <= c_long.s_hard_max[k]
            )
        if planner._safe_distance_modes is not None and planner._safe_distance_modes[k]:
            pred_state = c_long.preceding_vehicle.get_lon_state(
                k + c_long.tc_time_step
            )
            for velocity in planner._velocity_samples:
                safe_distance_0 = qp_long_planner_module.calculate_safe_distance(
                    velocity, pred_state.v, -10.5, -10, c_ti.react_time
                )
                safe_distance_derivative = (
                    qp_long_planner_module.derivative_safe_distance(
                        velocity, -10.0, c_ti.react_time
                    )
                )
                safe_distance = safe_distance_0 + safe_distance_derivative * (
                    planner._x[1, k + 1] - velocity
                )
                constraints.append(
                    planner._x[0, k + 1]
                    <= c_long.preceding_vehicle.rear_s(k + c_long.tc_time_step)
                    - c_ti.length / 2
                    - c_ti.wheelbase / 2
                    - safe_distance
                    - 1.0e-2
                )
        if c_long.s_soft_min[k] != -np.inf:
            constraints.append(
                planner._x[0, k + 1] + planner._u[:, planner.N]
                >= c_long.s_soft_min[k]
            )
        if c_long.s_hard_min[k] != -np.inf:
            constraints.append(
                planner._x[0, k + 1] >= c_long.s_hard_min[k]
            )
        if c_long.v_max[k] != np.inf:
            constraints.append(planner._x[1, k + 1] <= c_long.v_max[k])
        if c_long.v_min[k] != -np.inf:
            constraints.append(planner._x[1, k + 1] >= c_long.v_min[k])
        if c_long.a_max[k] != np.inf:
            constraints.append(planner._x[2, k + 1] <= c_long.a_max[k])
        if c_long.a_min[k] != -np.inf:
            constraints.append(planner._x[2, k + 1] >= c_long.a_min[k])
    return constraints


class QPPlannerRepair(QPPlanner):
    """
    QP-planner for trajectory repairing starting from the cut-off state.
    """

    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        tc_object: TC,
        config: RepairerConfiguration,
        verbose=False,
    ):
        # initialize the scenario and planning problem
        self._scenario = rule_monitor.world.scenario
        self._tc_object = tc_object
        self._ego_vehicle = tc_object.ego_vehicle
        self._start_time_step = tc_object.ego_vehicle.initial_state.time_step
        self._planning_problem = config.planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory
        self._rule_monitor = rule_monitor

        # set the cut-off state as the initial state
        self._cut_off_time_step: Optional[float, int] = None
        self._N: Optional[int] = None
        self._cut_off_state: Optional[CustomState, InitialState] = None
        self._time_horizon: Optional[float] = None
        self.sel_proposition = None
        self.full_proposition = None

        # load and set up the configuration
        # >>> side note: the initial state is not updated to keep the reference path possibly long
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)
        self._settings = self.config_settings()
        try:
            self._qp_configuration: PlanningConfigurationVehicle = set_up(
                self._settings, self._scenario, self._planning_problem
            )
        except ValueError as err:
            if "could not find a single route" not in str(err):
                raise
            ego_vehicle_world = rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id)
            ref_lane = self._select_fallback_reference_lane(ego_vehicle_world)
            if ref_lane is None:
                raise RuntimeError(
                    "route planner failed and no fallback reference lane could be "
                    "derived from the monitor world"
                ) from err
            print(
                "* \t<QPPlannerRepair>: route planner failed; "
                "reusing fallback ego reference path from monitor world."
            )
            self._qp_configuration = create_optimization_configuration_vehicle(
                self._scenario,
                self._planning_problem,
                self._settings["vehicle_settings"],
                route_planner=RoutePlanner(self._scenario.lanelet_network, self._planning_problem),
                reference_path=np.array(ref_lane.clcs.reference_path()),
                lanelets_leading_to_goal=ref_lane.contained_lanelets,
                cosy=ref_lane.clcs,
            )
        # The route planner can return an intersection reference path opposite
        # to the monitor vehicle's travel direction.  Then a non-negative CR
        # speed becomes negative in QP coordinates, while rule bounds are
        # extracted from the monitor's forward CLCS.  Reuse the monitor lane
        # only for that diagnosed mismatch; keep the normal Lin2022 route in
        # every aligned case.
        try:
            projected_initial = compute_initial_state(
                self._ego_vehicle.initial_state, self._qp_configuration
            )
        except (ValueError, AssertionError) as err:
            message = str(err)
            if not (
                "outside of projection domain" in message
                or "Provided velocity not valid" in message
            ):
                raise
            projected_initial = None
        if projected_initial is None or (
            float(self._ego_vehicle.initial_state.velocity) >= 0.0
            and float(projected_initial.v) < -1.0e-6
        ):
            ego_vehicle_world = rule_monitor.world.vehicle_by_id(
                self._ego_vehicle.obstacle_id
            )
            ref_lane = self._select_fallback_reference_lane(ego_vehicle_world)
            if ref_lane is not None:
                self._qp_configuration = create_optimization_configuration_vehicle(
                    self._scenario,
                    self._planning_problem,
                    self._settings["vehicle_settings"],
                    route_planner=RoutePlanner(
                        self._scenario.lanelet_network, self._planning_problem
                    ),
                    reference_path=np.array(ref_lane.clcs.reference_path()),
                    lanelets_leading_to_goal=ref_lane.contained_lanelets,
                    cosy=ref_lane.clcs,
                )
        self.config = config

        self.reach_set_time = 0
        self.opti_plan_time = 0

        # use the coordinate system from the world
        # if rule_monitor.scenario_type == "intersection":
        #     ref_lane = rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id).ref_path_lane
        # else:
        #     # fixme: first or the last
        #     obs = rule_monitor.world.vehicle_by_id(self._ego_vehicle.obstacle_id)
        #     ref_lane = obs.get_lane(obs.end_time)

        # self._qp_configuration.CLCS = ref_lane.clcs
        # import matplotlib.pyplot as plt
        # from commonroad.visualization.mp_renderer import MPRenderer
        # rnd = MPRenderer()
        # rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
        # self._scenario.lanelet_network.draw(rnd)
        # rnd.render()
        # import numpy as np
        # plt.plot(np.array(self._qp_configuration.CLCS.reference_path())[:, 0],
        #          np.array(self._qp_configuration.CLCS.reference_path())[:, 1])
        # plt.show()

        # update the vehicle shape
        ego_length, ego_width = shape_dimensions(self._ego_vehicle.obstacle_shape)
        self._qp_configuration.width = ego_width
        self._qp_configuration.length = ego_length

        # construct the rule constraints based on the traffic rules and proposition to be repaired
        if config.repair.constraint_mode == 1:
            pass
        #     print(  "Using Manual Rule Constraints for QP Planner Repairer"  )
        #     # self._rule_constraints = RuleConstraintsManual(
        #     #     self._qp_configuration,
        #     #     rule_monitor,
        #     #     self._initial_trajectory,
        #     # )
        #     print(f"tc_object.tc_time_step: {tc_object.tc_time_step}")
        #     self._rule_constraints = RuleConstraintsManual(
        #         tc_object=self._tc_object,
        #         rule_monitor=rule_monitor,
        #         veh_config=self._qp_configuration,
        #         proposition_full=self.full_proposition,
        #         sel_proposition_full=self.sel_proposition,
        #         initial_trajectory=self._initial_trajectory,
        #         start_time_step=self._tc_object.ego_vehicle.initial_state.time_step
        #     )
        elif config.repair.constraint_mode == 2:
            self._rule_constraints = RuleConstraintsReach(
                tc_object,
                rule_monitor,
                self._qp_configuration,
                self._initial_trajectory,
                self.config
            )
            self._qp_configuration.CLCS = self._rule_constraints.reach_config.planning.CLCS
            self._qp_configuration.reference_path = self._rule_constraints.reach_config.planning.reference_path

        else:
            raise AssertionError(f"the constraint mode {config.repair.constraint_mode} is not supported")
        super().__init__(
            self._qp_configuration,
            qp_long_parameters=self._settings["qp_planner"]["longitudinal_parameters"],
            qp_lat_parameters=self._settings["qp_planner"]["lateral_parameters"],
            verbose=verbose,
        )
        if any(
            rule in config.repair.rules
            for rule in ("R_IN1", "R_IN3", "R_IN3_hand_draft", "R_IN4", "R_IN5")
        ):
            self.lon_planner.tv_constraints = MethodType(
                _tv_constraints_with_hard_rule_bounds, self.lon_planner
            )

    @staticmethod
    def _select_fallback_reference_lane(ego_vehicle_world):
        ref_lane = getattr(ego_vehicle_world, "ref_path_lane", None)
        if ref_lane is not None and getattr(ref_lane, "clcs", None) is not None:
            return ref_lane

        candidate_times = []
        start_time = getattr(ego_vehicle_world, "start_time", None)
        end_time = getattr(ego_vehicle_world, "end_time", None)
        if start_time is not None:
            candidate_times.append(start_time)
        if end_time is not None and end_time not in candidate_times:
            candidate_times.append(end_time)
        if start_time is not None and end_time is not None:
            candidate_times.extend(
                time_step
                for time_step in range(start_time, end_time + 1)
                if time_step not in candidate_times
            )

        for time_step in candidate_times:
            try:
                candidate_lanes = list(ego_vehicle_world.lanes_at_state(time_step))
            except Exception:
                candidate_lanes = []

            best_lane = None
            best_lon_v = -np.inf
            for lane in candidate_lanes:
                if lane is None or getattr(lane, "clcs", None) is None:
                    continue
                try:
                    lon_state = ego_vehicle_world.get_lon_state(time_step, lane)
                    lon_v = getattr(lon_state, "v", -np.inf) if lon_state is not None else -np.inf
                except Exception:
                    lon_v = -np.inf
                if lon_v > best_lon_v:
                    best_lon_v = lon_v
                    best_lane = lane

            if best_lane is not None and best_lon_v >= 0.0:
                return best_lane
            if best_lane is not None and getattr(best_lane, "clcs", None) is not None:
                return best_lane

        return None
    
    def construct_constraints(self,
                              sel_proposition: List[PropositionNode],
                              proposition_full: List[PropositionNode]):
        if self.config.repair.constraint_mode == 1:
            print(  "Using Manual Rule Constraints for QP Planner Repairer"  )
            # self._rule_constraints = RuleConstraintsManual(
            #     self._qp_configuration,
            #     rule_monitor,
            #     self._initial_trajectory,
            # )
            self._rule_constraints = RuleConstraintsManual(
                tc_object=self._tc_object,
                rule_monitor=self._rule_monitor,
                veh_config=self._qp_configuration,
                proposition_full=proposition_full,
                sel_proposition_full=sel_proposition,
                initial_trajectory=self._initial_trajectory,
                start_time_step=self._tc_object.ego_vehicle.initial_state.time_step
            )
        else:
            pass

    def reset(self,
              tc_object: TC = None,
              sel_proposition: List[PropositionNode] = None,
              full_proposition: List[PropositionNode] = None,
              scenario: Scenario = None,
              ):
        """
        Initializes/resets configuration of the repairer for re-planning purposes
        """

        # The selected branch is needed while constructing the initial QP state.
        # Assign it before handling tc_object; the old order left reset-time
        # normalization dependent on the previous iteration's branch.
        if sel_proposition is not None:
            self.sel_proposition = sel_proposition
        if full_proposition is not None:
            self.full_proposition = full_proposition

        if scenario is not None:
            self.scenario = scenario
            if not hasattr(scenario, 'dt'):
                self.dt = 0.1  # default time step
            else:
                self.dt = scenario.dt

            self.lon_planner.reset(self.dt)
            self.lat_planner.reset(self.dt)

        if tc_object is not None:
            self._tc_object = tc_object
            self._cut_off_time_step = tc_object.tc_time_step
            self._N = tc_object.N
            if self._cut_off_time_step == self._ego_vehicle.initial_state.time_step:
                self._cut_off_state = self._ego_vehicle.initial_state
            else:
                self._cut_off_state = self._initial_trajectory.state_at_time_step(
                    self._cut_off_time_step
                )
            self._time_horizon = round(
                (self._N - self._cut_off_time_step) * self._scenario.dt,
                tc_object.round_tolerance,
            )
            # !! update the initial state as the cut off state
            self._planning_problem.initial_state = InitialState(
                position=self._cut_off_state.position,
                velocity=self._normalized_initial_velocity(self.sel_proposition),
                orientation=self._cut_off_state.orientation,
                time_step=self._cut_off_state.time_step,
                acceleration=self._normalized_initial_acceleration(self.sel_proposition),
                # not needed but mandatory field
                yaw_rate=0,
                slip_angle=0,
            )
            self.lon_planner.set_time_step(self._N - self._cut_off_time_step)
            self.lat_planner.set_time_step(self._N - self._cut_off_time_step)

    def _selected_positive_standstill(self, selected):
        return any(
            not proposition.alphabet.startswith("~")
            and proposition.name.count("not(") % 2 == 0
            and "in_standstill__0" in proposition.name
            for proposition in selected or ()
        )

    def _selected_negative_ego_conflict(self, selected):
        return any(
            proposition.alphabet.startswith("~")
            and "in_intersection_conflict_area__0_1" in proposition.name
            for proposition in selected or ()
        )

    def _normalized_initial_velocity(self, selected):
        velocity = float(self._cut_off_state.velocity)
        if self._selected_positive_standstill(selected) and abs(velocity) <= 0.05:
            return 0.0
        return velocity

    def _normalized_initial_acceleration(self, selected):
        if self._selected_positive_standstill(selected) and abs(
            float(self._cut_off_state.velocity)
        ) <= 0.05:
            return 0.0
        return getattr(self._cut_off_state, "acceleration", 0.0)


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
        start_time_lon_constr = time.time()

        self._rule_constraints.reset(self._start_time_step,
                                     self._tc_object,
                                     self._rule_monitor,
                                     self.sel_proposition,
                                     self.full_proposition,
                                     )
        
        try:
            long_constr = self._rule_constraints.longitudinal_constraints(
                self._qp_configuration
            )
        except Exception as e:
            self.reach_set_time = time.time() - start_time_lon_constr
            print(f"Error in constructing longitudinal constraints: {e}")
            return None
        reference_lon = self.construct_s_reference(long_constr)
        long_constr_construction_time = time.time() - start_time_lon_constr
        print(
            "* \t\t -- longi constraint construction {} s --".format(round(long_constr_construction_time, 3))
        )

        start_time_lon = time.time()
        # Rule-constraint construction can refresh shared planning state.  Apply
        # the narrow standstill normalization at the final hand-off to QP too.
        qp_initial_state = copy.deepcopy(self._planning_problem.initial_state)
        self.step(
            qp_initial_state,
            qp_initial_state.velocity,
        )
        # CommonRoad stores scalar speed as a non-negative magnitude.  A
        # locally reversed/intersection CLCS can make its cosine projection
        # negative, although this QP formulation and all extracted bounds use
        # forward speed v >= 0 and increasing progress.  Normalize that
        # representation mismatch at the final hand-off; acceleration keeps
        # its physical sign (negative still means braking).
        if (
            hasattr(self, "initial_state")
            and
            float(qp_initial_state.velocity) >= 0.0
            and float(self.initial_state.v) < 0.0
        ):
            self.initial_state._v = abs(float(self.initial_state.v))
        # Standstill is longitudinal.  Normalize after Cartesian-to-CLCS
        # projection, since a vehicle with nonzero scalar speed can still have
        # near-zero longitudinal speed on a noisy/misaligned first frame.
        if (
            self._selected_positive_standstill(self.sel_proposition)
            and abs(float(self.initial_state.v)) <= 0.05
        ):
            # TrajPoint's public v/a setters are intentionally no-ops in the
            # upstream planner, so update its backing fields.
            self.initial_state._v = 0.0
            self.initial_state._a = 0.0
            self.set_desired_velocity(0.0)
        elif (
            self._selected_negative_ego_conflict(self.sel_proposition)
            and self._tc_object.tc_time_step == self._start_time_step
        ):
            # Match VP's repair-prefix semantics: at TC=0 the first optimized
            # sample is reconstructed, rather than treated as an immutable
            # predecessor with its noisy/original v and a.  This permits the
            # emergency hold used by VP when the ego is immediately before a
            # conflict entrance; all following samples still obey QP dynamics.
            self.initial_state._v = 0.0
            self.initial_state._a = 0.0
            self.set_desired_velocity(0.0)

        traj_lon, status = self.longitudinal_trajectory_planning(
            long_constr, reference_lon
        )
        long_optimization_time = time.time() - start_time_lon
        self.reach_set_time = long_constr_construction_time
        self.opti_plan_time = long_optimization_time
        print(
            "* \t\t -- longi optimization time {} s --".format(round(long_optimization_time, 3))
        )
        if status != "optimal":
            return None
            # raise ValueError('<QPPlannerRepair/_longitudinal_trajectory_planning>: failed')
        print("* \t\t Lateral optimization")
        start_time_lat_constr = time.time()

        lat_constr = self._rule_constraints.lateral_constraints(
            traj_lon, self.vehicle_configuration
        )
        reference_lat = self.construct_d_reference(lat_constr)

        lat_constr.select_proposition = long_constr.select_proposition
        lat_constr_construction_time = time.time() - start_time_lat_constr
        self.reach_set_time = (
            long_constr_construction_time + lat_constr_construction_time
        )
        print(
            "* \t\t -- lateral constraint construction {} s --".format(round(lat_constr_construction_time, 3))
        )
        start_time_lat = time.time()
        
        trajectory, status = self.lateral_trajectory_planning(
            traj_lon, lat_constr, d_ref=reference_lat
        )
        lat_optimization_time = time.time() - start_time_lat
        self.opti_plan_time = long_optimization_time + lat_optimization_time
        print(
            "* \t\t -- lateral optimization time {} s --".format(round(lat_optimization_time, 3))
        )
        
        self.reach_set_time = long_constr_construction_time + lat_constr_construction_time
        self.opti_plan_time = long_optimization_time + lat_optimization_time

        # convert trajectory to cartesian space
        if status != "optimal":
            if (
                self._selected_negative_ego_conflict(self.sel_proposition)
                and self._is_standstill_longitudinal_trajectory(traj_lon)
            ):
                # The lateral QP is speed-parameterized and can become
                # singular for a stationary longitudinal solution.  Only in
                # that narrow case retain the fixed-path longitudinal result.
                fixed_path_trajectory = self._fixed_path_trajectory(traj_lon)
                return self.transform_merge_trajectory(fixed_path_trajectory)
            # print(f"[DEBUG] QP Planner Repair failed with status: {status}")
            return None
            # raise ValueError('<QPPlannerRepair/_lateral_trajectory_planning>: failed')
        
        # TEST: do not transform to cartesian coordinates
        cr_trajectory = self.transform_merge_trajectory(trajectory)
        # cr_trajectory = trajectory

        # plot_position_constraints(trajectory, (long_constr.s_hard_min, long_constr.s_hard_max), (lat_constr.d_hard_min, lat_constr.d_hard_max))
        return cr_trajectory

    @staticmethod
    def _is_standstill_longitudinal_trajectory(trajectory, tolerance=0.05):
        return bool(trajectory.states) and all(
            abs(float(state.v)) <= tolerance for state in trajectory.states
        )

    def _fixed_path_trajectory(self, longitudinal_trajectory):
        """Attach the current lateral offset to a longitudinal QP result."""
        lateral_offset = float(self.initial_state.position[1])
        orientation = float(self.initial_state.orientation)
        curvature = float(self.initial_state.kappa)
        points = []
        for longitudinal_state in longitudinal_trajectory.states:
            point = copy.deepcopy(longitudinal_state)
            point._position[1] = lateral_offset
            point._orientation = orientation
            point._kappa = curvature
            point._kappa_dot = 0.0
            points.append(point)
        trajectory = QPTrajectory(points, TrajectoryType.FRENET)
        trajectory._u_lon = longitudinal_trajectory.u_lon
        trajectory._u_lat = np.zeros(max(0, len(points) - 1))
        return trajectory

    def construct_s_reference(self, lon_constr: LonConstraints):
        """
        Constructs the longitudinal reference from the initially-planned trajectory.
        """
        x_ref = list()
        for ts in range(0, lon_constr.N):
            if lon_constr.s_hard_min[ts] != -np.inf:
                x_ref.append(
                    QPLongState(
                        lon_constr.s_hard_min[ts], lon_constr.v_min[ts], 0.0, 0.0, 0.0
                    )
                )
            else:
                x_ref.append(
                    QPLongState(
                        None, lon_constr.v_min[ts], 0.0, 0.0, 0.0
                    )
                )
        return QPLongDesired(x_ref)

    def construct_d_reference(self, lat_constraints):
        """
        Constructs the lateral reference from the initially-planned trajectory.
        """
        d_ref = list()
        initial_pos = convert_pos_curvilinear(
            self._ego_vehicle.initial_state, self._qp_configuration
        )[1]
        last_pos = initial_pos

        for abs_time_step in range(self._cut_off_time_step, self._N + 1):
            state = self._initial_trajectory.state_at_time_step(abs_time_step)
            rel_time_step = abs_time_step - self._cut_off_time_step - 1

            if 0 <= rel_time_step < len(lat_constraints.d_hard_min):
                if np.any(lat_constraints.d_hard_min[rel_time_step]) == -np.inf and \
                    np.any(lat_constraints.d_hard_max[rel_time_step]) == np.inf:
                    if state is not None:
                        pos = convert_pos_curvilinear(state, self._qp_configuration)[1]
                    else:
                        pos = last_pos
                elif lat_constraints.d_hard_min[rel_time_step][0] != np.inf and \
                    lat_constraints.d_hard_min[rel_time_step][0] != -np.inf:
                    pos = (
                        lat_constraints.d_hard_min[rel_time_step][0]
                        + lat_constraints.d_hard_max[rel_time_step][0]
                    ) / 2
                else:
                    pos = last_pos
            else:
                if state is not None:
                    pos = convert_pos_curvilinear(state, self._qp_configuration)[1]
                else:
                    pos = last_pos

            d_ref.append(pos)
            last_pos = pos
        return QPLatReference(d_ref)

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
            self._qp_configuration.length, self._qp_configuration.width
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
            self._qp_configuration.width,
            self._qp_configuration.length,
            self._qp_configuration.wheelbase,
            self._qp_configuration.wb_ra,
            vehicle_id=self._ego_vehicle.obstacle_id,
        )

        # ``convert_to_cr_ego_vehicle`` already returns the QP initial state at
        # relative time step 0.  At TC=0, prepending the CommonRoad initial
        # state therefore created two states carrying ``time_step == 0``.  A
        # CommonRoad Trajectory is index based, so the duplicate shifted every
        # subsequent state by one and also made monitor traces one sample
        # longer than traces of the other vehicles.  TC validation reports
        # that structural mismatch as ``updated_tv = -inf``.
        if self._cut_off_time_step == 0:
            remaining_states = []
        else:
            remaining_states = [self._ego_vehicle.initial_state]
            prefix_begin = self._start_time_step + 1
            prefix_end = self._cut_off_time_step - 1
            if prefix_end >= prefix_begin:
                remaining_states += self._initial_trajectory.states_in_time_interval(
                    prefix_begin, prefix_end
                )
        remaining_states = [state for state in remaining_states if state is not None]
        for state in cr_traj_repaired.prediction.trajectory.state_list:
            state.time_step += self._cut_off_time_step
        repaired_states = cr_traj_repaired.prediction.trajectory.state_list
        # The input datasets (and the strict monitor) use forward-interval
        # acceleration: a[t] belongs to the transition t -> t+1.  The QP
        # exports its acceleration state at the interval end.  Shift that
        # field back by one sample so reconstructed trajectory values use the
        # exact same index as the constraint arrays.  Keep the final value;
        # it is already constrained as x[N].
        for state, next_state in zip(repaired_states, repaired_states[1:]):
            state.acceleration = next_state.acceleration
        state_list = [
            CustomState(
                time_step=state.time_step,
                position=state.position,
                velocity=state.velocity,
                orientation=state.orientation,
                acceleration=getattr(state, "acceleration", 0.0),
            )
            for state in remaining_states
            + repaired_states
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


def update_goal_state_extension(initial_trajectory: Trajectory, lanelet_network: LaneletNetwork):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    ini_final_lanelet = lanelet_network.find_lanelet_by_position([ini_final_state.position])[0]
    for lanelet_id in lanelet_network.find_lanelet_by_id(ini_final_lanelet[0]).successor:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        # if lanelet
    successor_lanelet = lanelet_network.find_lanelet_by_id(
        lanelet_network.find_lanelet_by_id(ini_final_lanelet[0]).successor[0]
    )
    ref_point = successor_lanelet.center_vertices[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(1, 1, ref_point),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region
