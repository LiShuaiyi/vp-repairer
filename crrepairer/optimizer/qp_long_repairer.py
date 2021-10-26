import sys
import numpy as npy
import matplotlib.pyplot as plt
from cvxpy import *
from collections import defaultdict
from typing import Union, Tuple

from optimization.abstract_repairer import TrajectoryRepairer
from optimization.constraints import TIConstraints, TVConstraints, LonConstraints
from optimization.trajectory import Trajectory, TrajPoint, TrajectoryType

import commonroad.common.validity as val


# QPLongRepairer Weights
class QPLongPARAMS:
    W_S = 4.0  # weight for state deviation
    W_V = 4  # weight for velocity deviation
    W_A = 1  # weight for acceleration deviation
    W_J = 2  # weight for jerk deviation
    W_U = 0.1  # weight for input
    W_S_Q = 10.0  # weight for quadratic slack costs SOFT POS CONSTRAINTS
    W_S_L = 50.0  # weight for linear slack costs SOFT POS CONSTRAINTS
    W_C_Q = 5  # weight for quadratic slack costs for collision mitigation
    W_C_L = 50  # weight for linear slack costs for collision mitigation
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


class QPLongRepairer(TrajectoryRepairer):
    def __init__(self,
                 tstcc: int,
                 horizon: float,
                 N: int,
                 dT: float,
                 slack=True,
                 qp_long_params: QPLongPARAMS = QPLongPARAMS()):
        super().__init__(tstcc, horizon, N, dT)
        self.verbose = True
        self._slack = slack

        self.tstcc = tstcc
        self._n = 4
        self._m = 1

        self._slack_soft_pos = False
        # number of slack variables if slack has been set to True <s_l_soft, s_u_soft>
        self._n_s = 2 if self._slack_soft_pos else 0
        # plus additional 2 soft for acceleration bath tubs
        self._slack_acc = False
        self._n_a = 3 if self._slack_acc else 0
        # NEW SLACK APPROACH -> only slacking s_u_hard otherwise vehicle crashes on purpose with following vehicles
        # In total N slacks now to check collision at each time step
        self._n_c = 1 if self._slack else 0  # self.N # currently only one slack support

        # define variables and matrices
        self._x = Variable((self._n, self.N + 1))
        self._u = Variable((self._m, self.N + self._n_s + self._n_a + self._n_c))

        # state transition matrices
        self._A = npy.array(
            [[1, dT, (dT ** 2.) / 2., (dT ** 3.) / 6.], [0, 1., dT, (dT ** 2.) / 2.], [0, 0, 1., dT], [0, 0, 0, 1]])
        self._B = npy.array([[(dT ** 4.) / 24.], [(dT ** 3.) / 6.], [(dT ** 2.) / 2.], [dT]])

        # store parameters
        self._qp_long_params = qp_long_params

        # weight matrices states and inputs
        self._Q = npy.eye(self._n) * npy.transpose(npy.array(
            [self._qp_long_params.W_S, self._qp_long_params.W_V, self._qp_long_params.W_A, self._qp_long_params.W_J]))
        self._R = self._qp_long_params.W_U

        # weight matrices for slack variables
        self._H_Q = npy.identity(self._n_s) * self._qp_long_params.W_S_Q
        self._H_L = npy.repeat(1, self._n_s) * self._qp_long_params.W_S_L
        # weight matrices for slack variables for collision mitigation
        self._C_Q = self._qp_long_params.W_C_Q
        self._C_L = self._qp_long_params.W_C_L

        # self._codegen_solver_dict = self._set_codegen_solver_dict()

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

    def plot_state_vector(self, c: TIConstraints, cv: TVConstraints):
        x = self._x
        N = x.size[1] - 1
        if isinstance(cv, TVConstraints):
            lon = cv.lon
        else:
            lon = cv

        plt.ion()
        plt.figure()

        # Plot (x_t)_1.
        plt.subplot(4, 1, 1)
        x1 = x[0, :].value.A.flatten()
        plt.plot(npy.array(range(N + 1)), x1)
        plt.plot(npy.array(range(N)) + 1, lon.s_soft_min, 'C1')
        plt.plot(npy.array(range(N)) + 1, lon.s_soft_max, 'C1')
        plt.plot(npy.array(range(N)) + 1, lon.s_hard_min, 'C2')
        plt.plot(npy.array(range(N)) + 1, lon.s_hard_max, 'C2')
        plt.ylabel(r"$s$", fontsize=16)
        plt.xticks([])

        # Plot (x_t)_2.    plt.ion()
        plt.subplot(4, 1, 2)
        x2 = x[1, :].value.A.flatten()
        plt.plot(npy.array(range(N + 1)), x2)
        plt.yticks(npy.linspace(c.v_min, c.v_max, 3))
        plt.ylim([c.v_min, c.v_max + 2])
        plt.ylabel(r"$v$", fontsize=16)
        plt.xticks([])

        # Plot (x_t)_3.
        plt.subplot(4, 1, 3)
        x2 = x[2, :].value.A.flatten()
        plt.plot(npy.array(range(N + 1)), x2)
        plt.yticks(npy.linspace(c.a_x_min, c.a_x_max, 3))
        plt.ylim([c.a_x_min, c.a_x_max + 2])
        plt.ylabel(r"$a$", fontsize=16)
        plt.xticks([])

        # Plot (x_t)_4.
        plt.subplot(4, 1, 4)
        x2 = x[3, :].value.A.flatten()

        plt.plot(npy.array(range(N + 1)), x2)
        plt.ylabel(r"$j$", fontsize=16)
        plt.xticks(range(0, N + 1))
        plt.xlabel(r"$k$", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)

    def repair(self, x_initial: QPLongState, x_ref: QPLongReference, ti: TIConstraints, tv: TVConstraints) \
            -> Tuple[Union[Trajectory, None], int]:
        # if self.N in self._codegen_solver_dict:
        #     traj, status, cost = self._codegen_plan(x_initial, x_ref, ti, tv)
        # else:
        traj, status, cost = self._cvxpy_plan(x_initial, x_ref, ti, tv)

        # check result
        if not 'optimal' == status:
            print('\t\t\t Status longitudinal trajectory planner: {}'.format(status))
        return traj, status

    # def _codegen_plan(self, x_initial: QPLongState, x_ref: QPLongReference, ti: TIConstraints, tv: TVConstraints) \
    #         -> Tuple[str, float]:
    #
    #     if isinstance(tv, TVConstraints):
    #         c = tv.lon
    #     if isinstance(tv, LonConstraints):
    #         c = tv
    #
    #     initial_state = npy.array([[x_initial.s], [x_initial.v], [x_initial.a], [x_initial.j]])
    #
    #     x_ref_array = npy.empty([self._n, self.N])
    #     for i, s in enumerate(x_ref.reference):
    #         x_ref_array[:, i] = s.to_array()
    #
    #     vars_dict, stats_dict = self._codegen_solver_dict[self.N](
    #         W_S=self._qp_long_params.W_S,
    #         W_V=self._qp_long_params.W_V,
    #         W_A=self._qp_long_params.W_A,
    #         W_J=self._qp_long_params.W_J,
    #         W_U=self._qp_long_params.W_U,
    #         W_S_A_L=self._qp_long_params.W_S_A_L,
    #         W_S_A_U=self._qp_long_params.W_S_A_U,
    #         _A=self._A,
    #         _B=self._B,
    #         x_initial=initial_state,
    #         s_hard_min=c.s_hard_min,
    #         s_hard_max=c.s_hard_max,
    #         v_min=ti.v_min,
    #         v_max=ti.v_max,
    #         a_x_min=ti.a_x_min,
    #         a_x_max=ti.a_x_max,
    #         j_x_min=ti.j_x_min,
    #         j_x_max=ti.j_x_max,
    #         x_ref=x_ref_array)
    #     self._x = vars_dict['_x']
    #     self._u = vars_dict['_u']
    #
    #     #######
    #     # Create output trajectory
    #     #######
    #     traj = None
    #     if stats_dict['status'] == 'optimal':
    #         traj = list()
    #         # add initial state
    #         traj.append(TrajPoint(x_initial.t, x_initial.s, 0, 0,
    #                               x_initial.v, x_initial.a, j=x_initial.j))
    #         for k in range(self.N):
    #             traj.append(TrajPoint(x_initial.t + self.dT * (k + 1), self._x[0, k + 1], 0, 0,
    #                                   self._x[1, k + 1] if self._x[1, k + 1] >= 0. else 0.,
    #                                   self._x[2, k + 1], j=self._x[3, k + 1]))
    #
    #         traj = Trajectory(traj, TrajectoryType.FRENET)
    #         traj._u_lon = self._u[0, : self.N]
    #
    #     return traj, stats_dict['status'], stats_dict['objective']

    def _cvxpy_plan(self, x_initial: QPLongState, x_ref: QPLongReference, ti: TIConstraints, tv: TVConstraints) \
            -> Tuple[Union[Trajectory, None], str, float]:
        """
        Plans a longitudinal trajectory for a given initial state and reference with respect to
         time-variant and invariant constraints
        :param x_initial: The initial state of the vehicle
        :param x_ref: The reference state or list of states (goals)
        :param ti: The time-invariant constraints
        :param tv: The time-variant longitudinal constraints
        :return: A longitudinal trajectory where the lateral component is set to zero
        """

        # Prepare constraints
        if isinstance(tv, TVConstraints):
            c = tv.lon

        if isinstance(tv, LonConstraints):
            c = tv

        # NEW TTC
        # s_predicted = [x_initial.s + k * self.dT * x_initial.v for k in range(1, self.N + 1)]
        # TTC = npy.argmin(npy.abs(c.s_hard_max - s_predicted))

        # check if reference is single state or list
        ref_len = x_ref.length()
        if ref_len > 1:
            assert ref_len == self.N

        states = []
        cost = 0
        constr = []
        # create all states of the problem along the horizon N
        for k in range(self.N):

            #####
            # Define cost function including reference
            #####

            # in case the reference is a single state
            # if ref_len == 1:
            #     cost = quad_form(
            #         self._x[:, k + 1] - npy.transpose(
            #             [x_ref.reference.s, x_ref.reference.v, x_ref.reference.a, x_ref.reference.j]),
            #         self._Q) + self._u[:, k] * self._u[:, k] * self._R
            # # in case the reference is a list of states
            # else:

            cost += quad_form(
                        self._x[:, k + 1] - npy.transpose(
                            [x_ref.reference[k].s, x_ref.reference[k].v, x_ref.reference[k].a, x_ref.reference[k].j]),
                        self._Q) + square(self._u[:, k]) * self._R


            #######
            # Add collision slack costs
            #######
            # if self.slack:
            #     if self._n_c > 1:
            #         cost += quad_form(self._u[(self.N + self._n_s + self._n_a) + k], self._qp_long_params.W_C_Q) + \
            #                 self._u[(self.N + self._n_s + self._n_a) + k] * self._qp_long_params.W_C_L
            #     else:
            #         if k == 0:
            #             cost += quad_form(self._u[(self.N + self._n_s + self._n_a)], self._qp_long_params.W_C_Q) + \
            #                     self._u[(
            #                             self.N + self._n_s + self._n_a)] * self._qp_long_params.W_C_L
            #
            #     # linear quadratic velocity costs from collision on
            #     if k >= TTC:
            #         cost += quad_form(self._x[1, k + 1], self._qp_long_params.W_C_Q) + self._x[
            #             1, k + 1] * self._qp_long_params.W_C_L

            # OTHER COSTS OF SLACK VARIABLES
            # if k == 1:
            #     if self._slack_soft_pos:
            #         # soft slack 1
            #         cost += quad_form(self._u[self.N], self._qp_long_params.W_S_Q) + self._qp_long_params.W_S_L * \
            #                 self._u[self.N]
            #
            #     if self._slack_acc:
            #         # lower slack acceleration
            #         cost += quad_form(self._u[self.N + self._n_s],
            #                           self._qp_long_params.W_S_A_L) + self._qp_long_params.W_S_A_L * \
            #                 self._u[self.N + self._n_s]
            #         # upper slack acceleration
            #         cost += quad_form(self._u[self.N + self._n_s + 1],
            #                           self._qp_long_params.W_S_A_U) + self._qp_long_params.W_S_A_U * \
            #                 self._u[self.N + self._n_s + 1]

            ########
            # Specify time-variant constraints
            ########

            # state transition based on kinematic model
            constr += [self._x[:, k + 1] == self._A @ self._x[:, k] + self._B @ self._u[:, k]]
            # constr += [self._x[0, k + 1] >= self._x[0, k]]
            # consider hard position constraints depending on collision slack on or off
            # if self.slack:
            #     # slack k allows solver to enlarge constraint for step k and beyond
            #     if c.s_hard_max[k] != npy.inf:
            #         if self._n_c > 1:
            #             constr += [self._x[0, k + 1] - self._u[:, self.N + self._n_s + self._n_a + k] <= c.s_hard_max[
            #                 k] - self._qp_long_params.L_ENLARGE]
            #         else:
            #             constr += [self._x[0, k + 1] - self._u[:, self.N + self._n_s + self._n_a] <= c.s_hard_max[
            #                 k] - self._qp_long_params.L_ENLARGE]
            #
            # else:
            if c.s_hard_min[k] != npy.inf:
                constr += [
                    self._x[0, k + 1] >= c.s_hard_min[k] + self._qp_long_params.L_ENLARGE]  # position constraints
            if c.s_hard_max[k] != npy.inf:
                constr += [self._x[0, k + 1] <= c.s_hard_max[k] - self._qp_long_params.L_ENLARGE]

            # # consider soft position constraints including slack
            # if False and self._slack_soft_pos:
            #     if c.s_soft_min[k] != npy.inf:
            #         constr += [self._x[0, k + 1] + self._u[:, self.N] >= c.s_soft_min[k]]  # position constraints
            #     if c.s_soft_max[k] != npy.inf:
            #         constr += [self._x[0, k + 1] - self._u[:, self.N + 1] <= c.s_soft_max[k]]

            # consider soft position constraints including slack
            # if False and self._slack_soft_pos:
            #     if c.s_soft_min[k] != npy.inf:
            #         constr += [self._x[0, k + 1] + self._u[:, self.N] >= c.s_soft_min[k]]  # position constraints
            #     if c.s_soft_max[k] != npy.inf:
            #         constr += [self._x[0, k + 1] - self._u[:, self.N + 1] <= c.s_soft_max[k]]

            # states.append(Problem(Minimize(cost), constr))  # problem

        # sums problem objectives and concatenates constraints
        # prob = sum(states)

        #######
        # Set up time-invariant constraints
        #######
        constr += [self._x[1, :] >= ti.v_min, self._x[1, :] <= ti.v_max]  # velocity
        constr += [self._x[2, :] >= ti.a_x_min, self._x[2, :] <= ti.a_x_max]  # acceleration
        # c += [self._x[2, :] >= ti.a_x_min / 4. - self._u[:, self.N + self._n_s]] # slack linear
        # c += [self._x[2, :] <= ti.a_x_max / 4. + self._u[:, self.N + self._n_s + 1]]
        # constr += [self._x[3, :] >= ti.j_x_min, self._x[3, :] <= ti.j_x_max]  # jerk
        constr += [self._x[:, 0] == x_initial.to_array()]  # initial state constraint
        # c += [self._u[:, self.N:].T >= npy.repeat(0., self._n_s + self._n_a + self._n_c)]  # slack variables >= 0
        # if self.zero_v_constr:
        #     # c += [self._x[1, self.N] == 0 + self._u[:,self.N]]  # velocity must be zero in the end!
        #     c += [self._x[1, self.N] >= 0]
        #     c += [self._x[1, self.N] <= 0.1]

        prob = Problem(Minimize(cost), constr)

        #######
        # Solve optimization problem
        #######
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

        # #######
        # # Create output trajectory
        # #######
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
