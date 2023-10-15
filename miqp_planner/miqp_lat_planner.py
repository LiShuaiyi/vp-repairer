from typing import Dict, List
import numpy as np
import sys, os
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.common.util import Interval

from commonroad_dc.geometry.util import (
    compute_curvature_from_polyline,
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
)

from miqp_planner.gurobi_optimizer import GurobiSolver
from miqp_planner.miqp_constraints import LongitudinalConstraint, LateralConstraint

from commonroad_qp_planner.configuration import (
    PlanningConfigurationVehicle,
    ReferencePoint,
)
from commonroad_qp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType


class MIQPLatState(object):
    """
    Class representing a state <d,theta,kappa,kappa_dot> within the QPLatPlanner
    """

    def __init__(
        self,
        d: float,
        theta: float,
        kappa: float,
        kappa_dot: float,
        t=0.0,
        s=None,
        v=None,
        a=None,
        j=None,
        u_lon=None,
    ):
        self.d = d
        self.theta = theta
        self.kappa = kappa
        self.kappa_dot = kappa_dot
        self.t = t

        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.u_lon = u_lon


class MIQPLatRefState(object):
    """
    Class representing a state <s,v,theta,kappa> of QPLatPlanner reference (longitudinal profile and
    curvature/orientation of reference)
    """

    def __init__(
        self, s: float, v: float, a: float, j: float, theta: float, kappa: float
    ):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.theta = theta
        self.kappa = kappa


class MIQPLatReference(object):
    """
    Class representing a QPLatReference made up of a list of QPLatRefStates
    """

    def __init__(self, reference: list()):
        self.reference = reference

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference: list()):
        # check if reference is list of reference states
        assert isinstance(reference, list) and (
            isinstance(s, MIQPLatRefState) for s in reference
        )
        self._reference = reference

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1

    @classmethod
    def construct_from_lon_traj_and_reference(
        cls,
        lon_traj: Trajectory,
        reference: np.ndarray,
        vehicle_configuration: PlanningConfigurationVehicle,  # TODO: fix time invariate constraint
        ti=None,
    ) -> "MIQPLatReference":
        assert (
            isinstance(lon_traj, Trajectory)
            and lon_traj.coord_type is TrajectoryType.CARTESIAN
        ), "<QPLatReference>: Provided longitudinal trajectory is invalid or not in Frenet. traj = {}".format(
            lon_traj
        )
        assert np.isclose(
            np.sum(lon_traj.get_positions()[:, 1]), 0.0
        ), "<QPLatReference>: Provided longitudinal trajectory containts lateral information != 0. d = {}".format(
            lon_traj.get_positions()[:, 1]
        )
        assert (
            isinstance(reference, np.ndarray)
            and reference.ndim == 2
            and len(reference) > 1
            and len(reference[0, :]) == 2
        ), "<QPLatReference>: Provided reference is not valid. reference = {}".format(
            reference
        )

        # compute orientation, curvature and pathlength of reference
        ref_orientation = compute_orientation_from_polyline(reference)
        ref_curvature = compute_curvature_from_polyline(reference)
        ref_pathlength = compute_pathlength_from_polyline(reference)

        # get s coordinates of longitudinal motion for interpolation of theta and kappa of reference
        s = lon_traj.get_positions()[:, 0]
        v = lon_traj.get_velocities()
        a = lon_traj.get_accelerations()
        j = lon_traj.get_jerk()
        # check if numerical errors have happened in lon trajectory
        for i in range(0, len(a)):
            if np.greater(a[i], vehicle_configuration.a_max_x):
                a[i] = vehicle_configuration.a_max_x
            if np.greater(vehicle_configuration.a_min_x, a[i]):
                a[i] = vehicle_configuration.a_min_x

        assert np.greater_equal(
            np.max(ref_pathlength), np.max(s)
        ), "<QPLatReference>: Provided reference is not long enough for interpolation! ref = {}, traj = {}".format(
            np.max(ref_pathlength), np.max(s)
        )
        CLCS = vehicle_configuration.curvilinear_coordinate_system

        # interpolate curvature at s positions of trajectory
        # curvature_interpolated = np.interp(s, ref_pathlength, ref_curvature)
        curvature_interpolated = np.interp(s, CLCS.ref_pos, CLCS.ref_curv)

        # interpolate orientation at s positions of trajectory
        # orientation_interpolated = np.interp(s, ref_pathlength, ref_orientation)
        orientation_interpolated = np.interp(s, CLCS.ref_pos, CLCS.ref_theta)
        assert (
            len(curvature_interpolated) == len(orientation_interpolated) == len(s)
        ), "<QPLatReference>: interpolation failed!"

        # create QPLat reference
        states = list()
        # TODO: state starts from 1 or 0?
        # for i in range(1, len(s)):
        for i in range(0, len(s)):
            states.append(
                MIQPLatRefState(
                    s[i],
                    v[i],
                    a[i],
                    j[i],
                    orientation_interpolated[i],
                    curvature_interpolated[i],
                )
            )

        return MIQPLatReference(states)


class MIQPLatPlanner:
    def __init__(
        self,
        horizon: float,
        N: int,
        dT: float,
        length: float,
        lateral_constraints: LateralConstraint,
        x_init_lat: MIQPLatState,
        x_ref_lat: MIQPLatReference,
        vehicle_configuration: PlanningConfigurationVehicle,
        miqp_lat_params,
    ):
        self.time_horizon = horizon
        self.N = N
        self.dt = dT

        # number of x <d, theta, kappa, kappa dot>
        self._n = 4
        # number of u <kappa dot dot>
        self._m = 1

        # wheelbase length
        self._length = length

        self._lateral_constraints = lateral_constraints
        self._x_init_lat = x_init_lat
        self._x_ref_lat = x_ref_lat

        self.vehicle_configuration = vehicle_configuration
        self.lat_params = miqp_lat_params  # TODO: cost weight

        self._init_time_invariant_constraints()
        self._init_dynamic_constraints()

        self.solver = GurobiSolver()

        self.weight = [0.05, 15.1, 40.0, 20.0, 1.0]

        self.d_reference = np.zeros(self.N + 1)

    def plan(self):
        self._init_state_var()
        self._init_control_var()
        self.solver.add_lat_dynamic_cons(
            self._lateral_constraints.dynamic_matrix_list,
            self._lateral_constraints.init_state,
            self._lateral_constraints.theta_r,
        )
        # self.solver.add_lat_dis_cons(self._lateral_constraints.lat_dis_cons_matrix,
        #                              self._lateral_constraints.theta_r,
        #                              self._lateral_constraints.d_min,
        #                              self._lateral_constraints.d_max)
        # self.solver.add_kappa_limit(self._lateral_constraints.kappa_lim)
        self.solver.costfunc_lat(
            self._x_ref_lat, self.weight, d_reference=self.d_reference
        )
        self.solver.solve()
        self.var_x = self.solver.get_var_x()
        self.control_u = self.solver.get_control_u()
        trajectory = self.create_output_trajectory()
        return trajectory

    def _init_state_var(self):
        x_shape = self._lateral_constraints.var_lat_x_lb.shape
        x = np.empty(x_shape, dtype=object)
        self.solver.add_lat_state_var(
            x,
            x_shape,
            self._lateral_constraints.var_lat_x_lb,
            self._lateral_constraints.var_lat_x_ub,
        )

    def _init_control_var(self):
        u_shape = self._lateral_constraints.var_lat_u_lb.shape
        u = np.empty(u_shape, dtype=object)
        self.solver.add_lat_control_var(
            u,
            u_shape,
            self._lateral_constraints.var_lat_u_lb,
            self._lateral_constraints.var_lat_u_ub,
        )

    def _init_time_invariant_constraints(self):
        # TODO: need to add in parameters
        self.kappa_dot_dot_min = -100
        self.kappa_dot_dot_max = 100
        self.kappa_dot_min = -0.4
        self.kappa_dot_max = 0.4
        self.kappa_min = -0.5
        self.kappa_max = 0.5
        # lower and upper bound for control u
        self._lateral_constraints.var_lat_u_lb = self.kappa_dot_dot_min * np.ones(
            self.N
        )
        self._lateral_constraints.var_lat_u_ub = self.kappa_dot_dot_max * np.ones(
            self.N
        )
        # lower and upper bound for states x
        self._lateral_constraints.var_lat_x_lb = -1000 * np.ones((self._n, self.N + 1))
        self._lateral_constraints.var_lat_x_ub = 1000 * np.ones((self._n, self.N + 1))
        # TODO: why from t = 2
        # lower and upper bound for kappa
        self._lateral_constraints.var_lat_x_lb[2, 2:] = self.kappa_min
        self._lateral_constraints.var_lat_x_ub[2, 2:] = self.kappa_max
        # lower and upper bound for kappa_dot
        self._lateral_constraints.var_lat_x_lb[3, 2:] = self.kappa_dot_min
        self._lateral_constraints.var_lat_x_ub[3, 2:] = self.kappa_dot_max

    def _init_dynamic_constraints(self):
        # TODO: what is theta_r
        self.theta_r = list()
        for i in range(self.N):
            self.theta_r.append(self._x_ref_lat.reference[i].theta)
        self._lateral_constraints.theta_r = self.theta_r

        kappa_lim = list()

        for i in range(self.N):
            v = self._x_ref_lat.reference[i].v
            a = self._x_ref_lat.reference[i].a
            theta = self._x_ref_lat.reference[i].theta

            # x = Ax+Bu+Dz
            A = np.array(
                [
                    [
                        1,
                        self.dt * v,
                        (self.dt**2) * 0.5 * (v**2),
                        (self.dt**3) / 6 * (v**2),
                    ],
                    [0, 1, self.dt * v, (self.dt**2) * 0.5 * v],
                    [0, 0, 1, self.dt],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array(
                [
                    [(self.dt**4) / 24 * (v**2)],
                    [(self.dt**3) / 6 * v],
                    [(self.dt**2) * 0.5],
                    [self.dt],
                ]
            )
            # disturbances on input
            D = np.array([-self.dt * v, 0, 0, 0]).reshape([-1, 1])
            # TODO: do we need S C E in qp_lat_planner?
            #  (maybe Using the three-circle-approximation to enforce positional constraints) (d1, d2, d3)
            self._lateral_constraints.dynamic_matrix_list.append(
                {"A": A, "B": B, "D": D}
            )

            if i == 0:
                # initial condition
                ini_kappa = (self.theta_r[1] - self.theta_r[0]) / (
                    self._x_ref_lat.reference[1].s - self._x_ref_lat.reference[0].s
                )
                # TODO: initial state t = 0s
                self._lateral_constraints.init_state = np.array(
                    [self._x_init_lat.d, self._x_init_lat.theta, self._x_init_lat.kappa, self._x_init_lat.kappa_dot]
                ).transpose()

            # selection matrix for output
            S = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
            C = np.array(
                [
                    [1, 0, 0, 0],
                    [1, 0.5 * self._length, 0, 0],
                    [1, self._length, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            # disturbances on output
            E = np.transpose(np.array([0, -0.5 * self._length, -self._length, 0, 0]))
            self._lateral_constraints.lat_dis_cons_matrix.append(
                {"S": S, "C": C, "E": E}
            )

            kappa_lim_k = min(
                np.sqrt(self.vehicle_configuration.a_max**2 - a**2)
                / (np.max([v, 0.5]) ** 2),
                self.kappa_max,
            )
            kappa_lim.append(kappa_lim_k)
        self._lateral_constraints.kappa_lim = np.array(kappa_lim)

    def create_output_trajectory(self):
        long_traj_states = self._lateral_constraints.long_traj.states
        traj = list()
        # add initial state
        traj.append(
            TrajPoint(
                self._x_init_lat.t,
                self._x_init_lat.s,
                self._x_init_lat.d,
                self._x_init_lat.theta,
                self._x_init_lat.v,
                self._x_init_lat.a,
                j=self._x_init_lat.j,
            )
        )
        for k in range(self.N):
            traj.append(
                TrajPoint(
                    t=self._x_init_lat.t + self.dt * (k + 1),
                    x=long_traj_states[k + 1].position[0],
                    y=self.var_x[0, k + 1],
                    theta=self.var_x[1, k + 1],
                    v=long_traj_states[k + 1].v,
                    a=long_traj_states[k + 1].a,
                    kappa=self.var_x[2, k + 1],
                    j=long_traj_states[k + 1].j,
                    kappa_dot=self.var_x[3, k + 1],
                    lane=-1,
                )
            )
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = self._lateral_constraints.long_traj.u_lon
        traj._u_lat = self.control_u
        return traj
