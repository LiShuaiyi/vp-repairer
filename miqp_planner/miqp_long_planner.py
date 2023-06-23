from typing import Dict, List
import numpy as np
import sys, os
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import utils
from configuration import ScenarioParams
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.common.util import Interval
import gurobi_optimizer
from miqp_constraints import LongitudinalConstraint

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle, ReferencePoint
from commonroad_qp_planner.trajectory import TrajPoint

class MIQPLongState(object):
    def __init__(self, s: float, v: float, a: float, j: float, t=0.):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.t = t

class MIQPLongPlanner:
    def __init__(self,
                 horizon: float,
                 N: int,
                 dT: float,
                 qp_long_params,
                 scenario: Scenario,
                 vehicle_configuration: PlanningConfigurationVehicle,
                 initial_state: TrajPoint):
        self.time_horizon = horizon
        self.N = N
        self.dt = dT
        self.scenario = scenario
        self.vehicle_configuration = vehicle_configuration
        self.initial_state = initial_state
        self.s0 = MIQPLongState(self.initial_state.position[0], self.initial_state.v, self.initial_state.a, self.initial_state.j)
        self.long_constraints = LongitudinalConstraint()

        # number of x
        self._n = 4
        # number of u
        self._m = 1

        self._init_time_invariant_constraints()
        self._init_dynamic_constraints()

        self.solver = gurobi_optimizer.GurobiSolver()

    def plan(self):
        self._init_state_var()
        self._init_control_var()
        self.solver.add_long_dynamic_cons(self.long_constraints.dynamic_matrix_list, self.long_constraints.init_state)

    def _init_time_invariant_constraints(self):
        # lower and upper bound for control u
        self.long_constraints.var_long_u_lb = -5000 * np.ones((self._m, self.N))
        self.long_constraints.var_long_u_ub = 5000 * np.ones((self._m, self.N))
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
        self.solver.add_long_control_var(x, x_shape, self.long_constraints.var_long_x_lb,
                                         self.long_constraints.var_long_x_ub)

    def _init_control_var(self):
        u_shape = self.long_constraints.var_long_u_lb.shape
        u = np.empty(u_shape, dtype=object)
        self.solver.add_long_control_var(u, u_shape, self.long_constraints.var_long_u_lb, self.long_constraints.var_long_u_ub)
