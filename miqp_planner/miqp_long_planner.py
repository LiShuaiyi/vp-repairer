from typing import Dict, List
import numpy as np
from commonroad.scenario.scenario import Scenario
from miqp_planner.gurobi_optimizer import GurobiSolver
from miqp_planner.miqp_constraints import LongitudinalConstraint

from commonroad_qp_planner.configuration import (
    PlanningConfigurationVehicle,
    ReferencePoint,
)
from commonroad_qp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType


class MIQPLongState(object):
    def __init__(self, s: float, v: float, a: float, j: float, t=0.0):
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
            isinstance(state, list) and (isinstance(s, MIQPLongState) for s in state)
        )
        self._reference = state

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1


class MIQPLongPlanner:
    def __init__(
        self,
        horizon: float,
        N: int,
        dT: float,
        scenario: Scenario,
        vehicle_configuration: PlanningConfigurationVehicle,
        initial_state: TrajPoint,
        long_constraints: LongitudinalConstraint,
        slack=False,
    ):
        self.time_horizon = horizon
        self.N = N
        self.dt = dT
        self.scenario = scenario
        self.vehicle_configuration = vehicle_configuration
        self.initial_state = initial_state
        self.s0 = MIQPLongState(
            self.initial_state.position[0],
            self.initial_state.v,
            self.initial_state.a,
            self.initial_state.j,
        )
        self.long_constraints = long_constraints
        # TODO: add parameter in config file
        self.weight = [0.1, 0.2, 0.5, 1, 0.1, 1000000]

        # number of x
        self._n = 4
        # number of u
        self._m = 1

        # slack variable
        self._slack_pos = slack
        self._n_s = 2 if self._slack_pos else 0

        self._init_time_invariant_constraints()
        self._init_dynamic_constraints()

        self.solver = GurobiSolver()

    def plan(self, reference_path):
        # initial state and control variables in solver
        self._init_state_var()
        self._init_control_var()
        if self._slack_pos:
            self._init_slack_var()
        # add longitudinal dynamic constraints in solver
        self.solver.add_long_dynamic_cons(
            self.long_constraints.dynamic_matrix_list, self.long_constraints.init_state
        )
        # create rule constraints
        self.long_constraints.add_rule_constraints()
        # create collision free constraints
        self.long_constraints.add_collision_free_constraints()
        # add collision free constraints in solver
        self.solver.add_collision_free_cons(
            self.long_constraints.collision_free_constraints
        )
        # add rule constraits in solver
        self.solver.add_rule_cons(self.long_constraints.rule_constraints)
        # cost function
        self.solver.costfunc_long(reference_path, self.weight)
        self.solver.solve()

        try:
            # extract solution
            self.var_x = self.solver.get_var_x()
            self.all_delta = self.solver.get_delta()
            self.control_u = self.solver.get_control_u()
            self.slack_var = self.solver.get_slack_var()
            print("slack variable: ", self.slack_var)
            trajectory = self.create_output_trajectory()
        except:
            return None

        return trajectory

    def _init_time_invariant_constraints(self):
        # lower and upper bound for control u
        self.long_constraints.var_long_u_lb = -5000 * np.ones(self.N)
        self.long_constraints.var_long_u_ub = 5000 * np.ones(self.N)
        # lower and upper bound for slack variable
        if self._slack_pos:
            self.long_constraints.var_slack_lb = np.zeros(self._n_s)
            self.long_constraints.var_slack_ub = 5000 * np.ones(self._n_s)
        # lower and upper bound for states x
        self.long_constraints.var_long_x_lb = -1000 * np.ones((self._n, self.N + 1))
        self.long_constraints.var_long_x_ub = 1000 * np.ones((self._n, self.N + 1))
        # lower and upper bound for velocity
        self.long_constraints.var_long_x_lb[
            1, :
        ] = self.vehicle_configuration.min_speed_x
        self.long_constraints.var_long_x_ub[
            1, :
        ] = self.vehicle_configuration.max_speed_x
        # lower and upper bound for acceleration
        self.long_constraints.var_long_x_lb[2, :] = self.vehicle_configuration.a_min_x
        self.long_constraints.var_long_x_ub[2, :] = self.vehicle_configuration.a_max_x
        # lower and upper bound for jerk
        self.long_constraints.var_long_x_lb[3, :] = self.vehicle_configuration.j_min_x
        self.long_constraints.var_long_x_ub[3, :] = self.vehicle_configuration.j_max_x

    def _init_dynamic_constraints(self):
        dT = self.dt
        A = np.array(
            [
                [1, dT, (dT**2.0) / 2.0, (dT**3.0) / 6.0],
                [0, 1.0, dT, (dT**2.0) / 2.0],
                [0, 0, 1.0, dT],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [[(dT**4.0) / 24.0], [(dT**3.0) / 6.0], [(dT**2.0) / 2.0], [dT]]
        )
        D = np.array([0, 0, 0, 0]).reshape([-1, 1])
        self.long_constraints.dynamic_matrix_list = [{"A": A, "B": B, "D": D}] * self.N
        self.long_constraints.init_state = np.array(
            [self.s0.s, self.s0.v, self.s0.a, self.s0.j]
        ).transpose()

    def _init_state_var(self):
        x_shape = self.long_constraints.var_long_x_lb.shape
        x = np.empty(x_shape, dtype=object)
        self.solver.add_long_state_var(
            x,
            x_shape,
            self.long_constraints.var_long_x_lb,
            self.long_constraints.var_long_x_ub,
        )

    def _init_control_var(self):
        u_shape = self.long_constraints.var_long_u_lb.shape
        u = np.empty(u_shape, dtype=object)
        self.solver.add_long_control_var(
            u,
            u_shape,
            self.long_constraints.var_long_u_lb,
            self.long_constraints.var_long_u_ub,
        )

    def _init_slack_var(self):
        slack_shape = self.long_constraints.var_slack_ub.shape
        slack = np.empty(slack_shape, dtype=object)
        self.solver.add_slack_var(
            slack,
            slack_shape,
            self.long_constraints.var_slack_lb,
            self.long_constraints.var_slack_ub,
        )

    def create_output_trajectory(self):
        traj = list()
        # add initial state
        traj.append(
            TrajPoint(
                self.initial_state.t,
                self.initial_state.position[0],
                0,
                0,
                self.initial_state.v,
                self.initial_state.a,
                j=self.initial_state.j,
            )
        )
        for k in range(self.N):
            traj.append(
                TrajPoint(
                    self.initial_state.t + self.dt * (k + 1),
                    self.var_x[0, k + 1],
                    0,
                    0,
                    self.var_x[1, k + 1] if self.var_x[1, k + 1] >= 0.0 else 0.0,
                    self.var_x[2, k + 1],
                    j=self.var_x[3, k + 1],
                )
            )
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = self.control_u
        return traj
