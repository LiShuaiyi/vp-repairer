from typing import Dict, List
import numpy as np
import sys, os
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.common.util import Interval
from miqp_planner.gurobi_optimizer import GurobiSolver
from miqp_planner.miqp_constraints import LongitudinalConstraint

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle, ReferencePoint
from commonroad_qp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType

import matplotlib.pyplot as plt

class MIQPLongState(object):
    def __init__(self, s: float, v: float, a: float, j: float, t=0.):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.t = t


class MIQPLongReference(object):
    def __init__(self, state):
        self.reference = state

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, state):
        # check if state is single state or list of states
        assert isinstance(state, MIQPLongState) or (
                isinstance(state, list) and (isinstance(s, MIQPLongState) for s in state))
        self._reference = state

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1


class MIQPLongPlanner:
    def __init__(self,
                 horizon: float,
                 N: int,
                 dT: float,
                 qp_long_params,
                 scenario: Scenario,
                 vehicle_configuration: PlanningConfigurationVehicle,
                 initial_state: TrajPoint,
                 long_constraints: LongitudinalConstraint):
        self.time_horizon = horizon
        self.N = N
        self.dt = dT
        self.scenario = scenario
        self.vehicle_configuration = vehicle_configuration
        self.initial_state = initial_state
        self.s0 = MIQPLongState(self.initial_state.position[0], self.initial_state.v, self.initial_state.a, self.initial_state.j)
        self.long_constraints = long_constraints
        self.weight = [0.1, 0.4, 1, 2, 0.1]

        # number of x
        self._n = 4
        # number of u
        self._m = 1

        self._init_time_invariant_constraints()
        self._init_dynamic_constraints()

        self.solver = GurobiSolver()

    def plan(self, reference_path):
        self._init_state_var()
        self._init_control_var()
        self.solver.add_long_dynamic_cons(self.long_constraints.dynamic_matrix_list, self.long_constraints.init_state)
        self.long_constraints.add_rule_constraints()
        self.long_constraints.add_collision_free_constraints()
        self.solver.add_collision_free_cons(self.long_constraints.collision_free_constraints)
        self.solver.add_rule_cons(self.long_constraints.rule_constraints)
        self.solver.costfunc_long(reference_path, self.weight)
        self.solver.solve()

        self.var_x = self.solver.get_var_x()
        self.all_delta = self.solver.get_delta()
        self.control_u = self.solver.get_control_u()
        trajectory = self.create_output_trajectory()

        # # plot result
        # fig, ax = plt.subplots()
        # t = np.linspace(0, 36, 37)
        # ax.plot(t, self.var_x[0, :], label='s')
        # ax.plot(self.long_constraints.collision_free_constraints['index_ub'],
        #         self.long_constraints.collision_free_constraints['collision_free_ub'], label='collision free')
        # index = list()
        # s_limit_front = list()
        # s_limit_behind = list()
        # for cons_name in self.long_constraints.rule_constraints:
        #     constraint = self.long_constraints.rule_constraints[cons_name]
        #     for i in range(len(constraint['s_limit_front'])):
        #         if constraint['s_limit_front'][i] != np.inf:
        #             index.append(i)
        #             s_limit_front.append(constraint['s_limit_front'][i])
        #             s_limit_behind.append(constraint['s_limit_behind'][i])
        # ax.plot(index, s_limit_front, label='s_limit_front')
        # ax.plot(index, s_limit_behind, label='s_limit_behind')
        # ax.set_xlabel('time step')
        # ax.set_ylabel('s')
        # ax.legend()
        #
        # plt.show()
        #
        # fig1, ax1 = plt.subplots()
        # ax1.plot(t, self.var_x[1, :])
        # ax1.set_xlabel('time step')
        # ax1.set_ylabel('velocity')
        # ax1.legend()
        # plt.show()
        #
        # fig2, ax2 = plt.subplots()
        # ax2.plot(t, self.var_x[2, :])
        # ax2.set_xlabel('time step')
        # ax2.set_ylabel('acceleration')
        # ax2.legend()
        # plt.show()

        return trajectory

    def _init_time_invariant_constraints(self):
        # lower and upper bound for control u
        self.long_constraints.var_long_u_lb = -5000 * np.ones(self.N)
        self.long_constraints.var_long_u_ub = 5000 * np.ones(self.N)
        # lower and upper bound for states x
        self.long_constraints.var_long_x_lb = -1000 * np.ones((self._n, self.N + 1))
        self.long_constraints.var_long_x_ub = 1000 * np.ones((self._n, self.N + 1))
        # lower and upper bound for velocity
        self.long_constraints.var_long_x_lb[1, :] = self.vehicle_configuration.min_speed_x
        self.long_constraints.var_long_x_ub[1, :] = self.vehicle_configuration.max_speed_x
        # lower and upper bound for acceleration
        self.long_constraints.var_long_x_lb[2, :] = self.vehicle_configuration.a_min_x
        self.long_constraints.var_long_x_ub[2, :] = self.vehicle_configuration.a_max_x
        # lower and upper bound for jerk
        self.long_constraints.var_long_x_lb[3, :] = self.vehicle_configuration.j_min_x
        self.long_constraints.var_long_x_ub[3, :] = self.vehicle_configuration.j_max_x

    def _init_dynamic_constraints(self):
        dT = self.dt
        A = np.array(
            [[1, dT, (dT ** 2.) / 2., (dT ** 3.) / 6.], [0, 1., dT, (dT ** 2.) / 2.], [0, 0, 1., dT], [0, 0, 0, 1]])
        B = np.array([[(dT ** 4.) / 24.], [(dT ** 3.) / 6.], [(dT ** 2.) / 2.], [dT]])
        D = np.array([0, 0, 0, 0]).reshape([-1, 1])
        self.long_constraints.dynamic_matrix_list = [{'A': A, 'B': B, 'D': D}] * self.N
        self.long_constraints.init_state = np.array([self.s0.s, self.s0.v, self.s0.a, self.s0.j]).transpose()

    def _init_state_var(self):
        x_shape = self.long_constraints.var_long_x_lb.shape
        x = np.empty(x_shape, dtype=object)
        self.solver.add_long_state_var(x, x_shape, self.long_constraints.var_long_x_lb,
                                         self.long_constraints.var_long_x_ub)

    def _init_control_var(self):
        u_shape = self.long_constraints.var_long_u_lb.shape
        u = np.empty(u_shape, dtype=object)
        self.solver.add_long_control_var(u, u_shape, self.long_constraints.var_long_u_lb, self.long_constraints.var_long_u_ub)

    def create_output_trajectory(self):
        traj = list()
        # add initial state
        traj.append(TrajPoint(self.initial_state.t, self.initial_state.position[0], 0, 0,
                              self.initial_state.v, self.initial_state.a, j=self.initial_state.j))
        for k in range(self.N):
            traj.append(TrajPoint(self.initial_state.t + self.dt * (k + 1), self.var_x[0, k + 1], 0, 0,
                                  self.var_x[1, k + 1] if self.var_x[1, k + 1] >= 0. else 0.,
                                  self.var_x[2, k + 1], j=self.var_x[3, k + 1]))
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = self.control_u
        return traj
