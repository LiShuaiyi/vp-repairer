import numpy as np

from miqp_planner.gurobi_optimizer import GurobiSolver
from miqp_planner.miqp_constraints import LongitudinalConstraint

from crrepairer.utils.configuration import RepairerConfiguration

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
            config: RepairerConfiguration,
            initial_state: TrajPoint,
    ):
        # basic configuration
        self.time_horizon = config.miqp_planner.horizon
        self.N = config.miqp_planner.N_p
        self.dt = config.scenario.dt
        self.scenario = config.scenario
        self.vehicle_configuration = config.vehicle

        # construct the initial state
        self.initial_state = initial_state
        self.s0 = MIQPLongState(
            self.initial_state.position[0],
            self.initial_state.v,
            self.initial_state.a,
            self.initial_state.j,
        )

        self.weight = config.miqp_planner.weight_long

        # number of x
        self._n = 4
        # number of u
        self._m = 1

        # slack variable
        self._slack_pos = config.miqp_planner.slack_long
        self._n_s = 2 if self._slack_pos else 0

        # Dynamic matrix and initial state
        A = np.array(
            [
                [1, self.dt, (self.dt ** 2.0) / 2.0, (self.dt ** 3.0) / 6.0],
                [0, 1.0, self.dt, (self.dt ** 2.0) / 2.0],
                [0, 0, 1.0, self.dt],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [[(self.dt ** 4.0) / 24.0], [(self.dt ** 3.0) / 6.0], [(self.dt ** 2.0) / 2.0], [self.dt]]
        )
        D = np.array([0, 0, 0, 0]).reshape([-1, 1])

        self.dynamic_matrix_list = [{"A": A, "B": B, "D": D}] * self.N
        self.init_state = np.array(
            [self.s0.s, self.s0.v, self.s0.a, self.s0.j]
        ).transpose()

        # initialize solver
        self.solver = GurobiSolver()


    def plan(self, long_ref: MIQPLongReference, long_constraints: LongitudinalConstraint):
        # initialize constraints
        self._init_time_invariant_constraints(long_constraints)

        # initial state and control variables in solver
        self._init_state_var(long_constraints)
        self._init_control_var(long_constraints)

        # add longitudinal dynamic constraints
        self.solver.add_long_dynamic_cons(
            self.dynamic_matrix_list, self.init_state
        )

        if self._slack_pos:
            self._init_slack_var(long_constraints)
        # add longitudinal dynamic constraints in solver

        # fixme: this should be outside the plan function? how is the rule constraints applied in the end?
        # create rule constraints
        long_constraints.add_rule_constraints()
        # create collision free constraints
        long_constraints.add_collision_free_constraints()
        # add collision free constraints in solver
        self.solver.add_collision_free_cons(
            long_constraints.collision_free_constraints
        )
        # add rule constraints in solver
        self.solver.add_rule_cons(long_constraints.rule_constraints)
        # cost function
        self.solver.costfunc_long(long_ref, self.weight)
        self.solver.solve()

        try:
            # extract solution
            all_delta = self.solver.get_delta()  # fixme: is this used? if not, maybe remove
            print("slack variable: ", self.solver.get_slack_var())
            trajectory = self.create_output_trajectory()
        except:
            return None  # fixme: better handling needed

        return trajectory

    def _init_time_invariant_constraints(self, long_constraints: LongitudinalConstraint):
        # lower and upper bound for control u
        # todo: can this be added to the constraints directly instead of via the long_constraints?
        long_constraints.var_long_u_lb = -5000 * np.ones(self.N)
        long_constraints.var_long_u_ub = 5000 * np.ones(self.N)
        # lower and upper bound for slack variable
        if self._slack_pos:
            long_constraints.var_slack_lb = np.zeros(self._n_s)
            long_constraints.var_slack_ub = 5000 * np.ones(self._n_s)
        # lower and upper bound for states x
        long_constraints.var_long_x_lb = -1000 * np.ones((self._n, self.N + 1))
        long_constraints.var_long_x_ub = 1000 * np.ones((self._n, self.N + 1))
        # lower and upper bound for velocity
        long_constraints.var_long_x_lb[1, :] = self.vehicle_configuration.qp_veh_config.min_speed_x
        long_constraints.var_long_x_ub[1, :] = self.vehicle_configuration.qp_veh_config.max_speed_x
        # lower and upper bound for acceleration
        long_constraints.var_long_x_lb[2, :] = self.vehicle_configuration.qp_veh_config.a_min_x
        long_constraints.var_long_x_ub[2, :] = self.vehicle_configuration.qp_veh_config.a_max_x
        # lower and upper bound for jerk
        long_constraints.var_long_x_lb[3, :] = self.vehicle_configuration.qp_veh_config.j_min_x
        long_constraints.var_long_x_ub[3, :] = self.vehicle_configuration.qp_veh_config.j_max_x

    def _init_state_var(self, long_constraints: LongitudinalConstraint):
        """Initializes the state variables"""
        x_shape = long_constraints.var_long_x_lb.shape
        x = np.empty(x_shape, dtype=object)
        self.solver.add_long_state_var(
            x,
            x_shape,
            long_constraints.var_long_x_lb,
            long_constraints.var_long_x_ub,
        )

    def _init_control_var(self, long_constraints: LongitudinalConstraint):
        """Initializes the input variables"""
        u_shape = long_constraints.var_long_u_lb.shape
        u = np.empty(u_shape, dtype=object)
        self.solver.add_long_control_var(
            u,
            u_shape,
            long_constraints.var_long_u_lb,
            long_constraints.var_long_u_ub,
        )

    def _init_slack_var(self, long_constraints: LongitudinalConstraint):
        """Initializes slack variables"""
        slack_shape = long_constraints.var_slack_ub.shape
        slack = np.empty(slack_shape, dtype=object)
        self.solver.add_slack_var(
            slack,
            slack_shape,
            long_constraints.var_slack_lb,
            long_constraints.var_slack_ub,
        )

    def create_output_trajectory(self):
        """creates the output trajectory"""
        # extract the solution
        var_x = self.solver.get_var_x()
        var_u = self.solver.get_control_u()

        # generate the trajectory
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
                    var_x[0, k + 1],
                    0,
                    0,
                    var_x[1, k + 1] if var_x[1, k + 1] >= 0.0 else 0.0,
                    var_x[2, k + 1],
                    j=var_x[3, k + 1],
                )
            )
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = var_u
        return traj
