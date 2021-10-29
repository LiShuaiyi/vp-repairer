import sys
import numpy as npy
import matplotlib.pyplot as plt
from cvxpy import *
from collections import defaultdict
from typing import Union, Tuple

from optimizer.abstract import TrajectoryPlanner
from optimizer.constraints import TIConstraints, TVConstraints, LonConstraints
from optimizer.trajectory import Trajectory, TrajPoint, TrajectoryType

import commonroad.common.validity as val


# Weights
class QPLongPARAMS:
    W_S = 0 # weight for state deviation
    W_V = 4  # weight for velocity deviation
    W_A = 1  # weight for acceleration deviation
    W_J = 2  # weight for jerk deviation
    W_U = 0.1  # weight for input
    W_S_Q = 10.0  # weight for quadratic slack costs SOFT POS CONSTRAINTS
    W_S_L = 50.0  # weight for linear slack costs SOFT POS CONSTRAINTS
    W_S_A_L = 10  # weight for lower acceleration slack
    W_S_A_U = 2  # weight for upper acceleration slack
    L_ENLARGE = 0.0  # added to the length of every vehicle to increase distance


class QPLongState(object):
    def __init__(self, s: float, v: float, a: float, j: float, t=0.):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.t = t

    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, s: float):
        assert s is not None, "<QPLongState> s is not valid! s = {}".format(s)
        self._s = s

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, v: float):
        assert val.is_valid_velocity(v), "<QPLongState> v is not valid! v = {}".format(v)
        self._v = v

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, a: float):
        assert val.is_valid_acceleration(a), "<QPLongState> a is not valid! a = {}".format(a)
        self._a = a

    @property
    def j(self) -> float:
        return self._j

    @j.setter
    def j(self, j):
        assert j is not None and isinstance(j, val.ValidTypes.NUMBERS), "<QPLongState> j is not valid! j = {}".format(j)
        self._j = j

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, t: float):
        assert isinstance(t, val.ValidTypes.NUMBERS) and t >= 0., '<QPLongState>: t is not valid. t = {}'.format(t)
        self._t = t

    def to_array(self) -> list:
        return npy.array([self.s, self.v, self.a, self.j])

    def __str__(self):
        # to represents the class objects as a string
        return '<QPLongState> (s={}, v={}, a={}, j={}, t={})'.format(self.s, self.v, self.a, self.j, self.t)


class QPLongReference(object):
    def __init__(self, state):
        self.reference = state

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, state):
        # check if state is single state or list of states
        assert isinstance(state, QPLongState) or (
                isinstance(state, list) and (isinstance(s, QPLongState) for s in state))
        self._reference = state

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1


class QPLongPlanner(TrajectoryPlanner):
    def __init__(self,
                 horizon: float,
                 N: int,
                 dT: float,
                 slack=False,
                 qp_long_params: QPLongPARAMS = QPLongPARAMS()):
        super().__init__(horizon, N, dT)

        # turn off verbose mode as default
        self.verbose = False

        # set parameters
        self._n = 4
        self._m = 1
        # ---------------------------------------- slack variable ----------------------------------------------------#
        self._slack = slack
        self._slack_soft_pos = False
        # number of slack variables if slack has been set to True <s_l_soft, s_u_soft>
        self._n_s = 2 if self._slack_soft_pos else 0
        # plus additional 2 soft for acceleration bath tubs
        self._slack_acc = False
        self._n_a = 3 if self._slack_acc else 0
        # NEW SLACK APPROACH -> only slacking s_u_hard otherwise vehicle crashes on purpose with following vehicles
        # In total N slacks now to check collision at each time step
        self._n_c = 1 if self._slack else 0  # self.N # currently only one slack support
        # ------------------------------------------------------------------------------------------------------------#
        # define variables and matrices
        self._x = Variable(shape=(self._n,
                                  self.N + 1))
        self._u = Variable(shape=(self._m,
                                  self.N + self._n_s + self._n_a + self._n_c))

        # store parameters
        self._qp_long_params = qp_long_params

        # state transition matrices
        self._A = npy.array(
            [[1, dT, (dT ** 2.) / 2., (dT ** 3.) / 6.], [0, 1., dT, (dT ** 2.) / 2.], [0, 0, 1., dT], [0, 0, 0, 1]])
        self._B = npy.array([[(dT ** 4.) / 24.], [(dT ** 3.) / 6.], [(dT ** 2.) / 2.], [dT]])
        # weight matrices states and inputs
        self._Q = npy.eye(self._n) * npy.transpose(npy.array(
            [self._qp_long_params.W_S, self._qp_long_params.W_V, self._qp_long_params.W_A, self._qp_long_params.W_J]))
        self._R = self._qp_long_params.W_U
        # weight matrices for slack variables
        self._H_Q = npy.identity(self._n_s) * self._qp_long_params.W_S_Q
        self._H_L = npy.repeat(1, self._n_s) * self._qp_long_params.W_S_L


    @property
    def slack(self):
        return self._slack

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        assert isinstance(verbose, bool), "<QPLatPlanner>: verbose flag must be of type bool!"
        self._verbose = verbose

    def plan(self,
             x_initial: QPLongState,
             x_ref: QPLongReference,
             c_ti: TIConstraints,
             c_tv: TVConstraints) \
            -> Tuple[Union[Trajectory, None], int]:
        traj, status, cost = self._cvxpy_plan(x_initial, x_ref, c_ti, c_tv)
        # check result
        if not 'optimal' == status:
            print('\t\t\t Status longitudinal trajectory planner: {}'.format(status))
        return traj, status

    def _cvxpy_plan(self,
                    x_initial: QPLongState,
                    x_ref: QPLongReference,
                    c_ti: TIConstraints,
                    c_tv: TVConstraints) \
            -> Tuple[Union[Trajectory, None], str, float]:
        """
            Plans a longitudinal trajectory for a given initial state and reference with respect to
            time-variant and invariant constraints
            :param x_initial: The initial state of the vehicle
            :param x_ref: The reference state or list of states (goals)
            :param c_ti: The time-invariant constraints
            :param c_tv: The time-variant longitudinal constraints
            :return: A longitudinal trajectory where the lateral component is set to zero
        """
        # Prepare constraints
        if isinstance(c_tv, TVConstraints):
            c_tv = c_tv.lon
        if isinstance(c_tv, LonConstraints):
            c_tv = c_tv

        # initialize cost and constraints
        cost = 0
        constr = []

        # create all states of the problem along the horizon N
        for k in range(self.N):
            #########################################
            # Define cost function including reference
            #########################################
            if x_ref:
                cost += quad_form(self._x[:, k + 1] - npy.transpose(
                                [x_ref.reference[k].s, x_ref.reference[k].v,
                                 x_ref.reference[k].a, x_ref.reference[k].j]), self._Q) +\
                        square(self._u[:, k]) * self._R
            else:
                cost += quad_form(self._x[:, k + 1], self._Q) +\
                        square(self._u[:, k]) * self._R
            ##################################
            # Specify time-variant constraints
            ##################################
            # state transition based on kinematic model
            constr += [self._x[:, k + 1] == self._A @ self._x[:, k] + self._B @ self._u[:, k]]
            # position constraints
            if c_tv.s_hard_min[k] != npy.inf:
                constr += [
                    self._x[0, k + 1] >= c_tv.s_hard_min[k] + self._qp_long_params.L_ENLARGE]
            if c_tv.s_hard_max[k] != npy.inf:
                constr += [self._x[0, k + 1] <= c_tv.s_hard_max[k] - self._qp_long_params.L_ENLARGE]
            # consider soft position constraints including slack
            if self._slack_soft_pos:
                if c_tv.s_soft_min[k] != npy.inf:
                    constr += [self._x[0, k + 1] + self._u[:, self.N] >= c_tv.s_soft_min[k]]  # position constraints
                if c_tv.s_soft_max[k] != npy.inf:
                    constr += [self._x[0, k + 1] - self._u[:, self.N + 1] <= c_tv.s_soft_max[k]]

        ###################################
        # Set up time-invariant constraints
        ###################################
        constr += [self._x[1, :] >= c_ti.v_min, self._x[1, :] <= c_ti.v_max]  # velocity
        constr += [self._x[2, :] >= c_ti.a_x_min, self._x[2, :] <= c_ti.a_x_max]  # acceleration
        constr += [self._x[3, :] >= c_ti.j_x_min, self._x[3, :] <= c_ti.j_x_max]  # jerk
        constr += [self._x[:, 0] == x_initial.to_array()]  # initial state constraint

        ############################
        # Solve optimization problem
        ############################
        prob = Problem(Minimize(cost), constr)
        print("Problem is convex:", prob.is_dcp())
        prob.solve(verbose=self.verbose)
        if self.verbose and not prob.status == 'infeasible':
            print('Created optimization with |x|={} and |u|={}'.format(self._x.size, self._u.size))
            if self._slack_soft_pos:
                print("Soft Pos Slack variables = {}".format(self._u[self.N:(self.N + self._n_s)].value.A.flatten()))
            if self._slack_acc:
                print("Soft Acc Slack variables = {}".format(
                    self._u[(self.N + self._n_s):(self.N + self._n_s + 2)].value.A.flatten()))
            if self.slack:
                if self._n_c > 1:
                    print("Hard Pos Slack variables = {}".format(self._u[(self.N + self._n_s):].value.A.flatten()))
                else:
                    print("Hard Pos Slack variable = {}".format(self._u[(self.N + self._n_s + self._n_a)].value))
            print("Costs = {}".format(cost))

        ##########################
        # Create output trajectory
        ##########################
        traj = None
        if not prob.status == 'infeasible':
            traj = list()
            # add initial state
            traj.append(TrajPoint(x_initial.t, x_initial.s, 0, 0,
                                  x_initial.v, x_initial.a, j=x_initial.j))
            for k in range(self.N):
                traj.append(TrajPoint(x_initial.t + self.dT * (k + 1), self._x[0, k + 1].value, 0, 0,
                                      self._x[1, k + 1].value if self._x[1, k + 1].value >= 0. else 0.,
                                      self._x[2, k + 1].value, j=self._x[3, k + 1].value))
            traj = Trajectory(traj, TrajectoryType.FRENET)
            traj._u_lon = npy.transpose(self._u.value.flatten())[:self.N]
        return traj, prob.status, prob.value
