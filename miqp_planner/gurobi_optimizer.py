from gurobipy import Model as Gmodel
from gurobipy import QuadExpr, LinExpr, GRB

import numpy as np

from miqp_constraints import LongitudinalConstraint


class GurobiSolver:
    def __init__(self):
        self.u_shape = None
        self.u = None
        self.x_shape = None
        self.x = None
        self.constraint_obj = None
        self.model = Gmodel()

    def add_long_state_var(self, x, x_shape, x_lb, x_ub):
        self.x = x
        self.x_shape = x_shape
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                self.x[i, j] = self.add_var("continuous", "x_long_{}_{}_".format(i, j), x_lb[i, j], x_ub[i, j])

    def add_long_control_var(self, u, u_shape, u_lb, u_ub):
        self.u = u
        self.u_shape = u_shape
        for i in range(self.u_shape[0]):
            self.u[i] = self.add_var("continuous", "u_long_{}".format(i), u_lb[i], u_ub[i])

    def add_var(self, typeofvar, name, lb=0, ub=0):
        if typeofvar == "continuous":
            return self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
        if typeofvar == "binary":
            return self.model.addVar(vtype=GRB.BINARY, name=name)

    def add_long_dynamic_cons(self, dynamic_matrix_list, init_state):
        z = np.zeros((self.x_shape[1], 1))
        self.add_dynamic_cons(dynamic_matrix_list, init_state, z)

    def add_dynamic_cons(self, dynamic_matrix_list, init_state, theta_r):
        for i in range(len(dynamic_matrix_list)):
            dynamic_matrix_dict = dynamic_matrix_list[i]
            if i == 0:
                self.add_matrix_eq_cons(np.eye(dynamic_matrix_dict["A"].shape[0]),
                                        np.zeros_like(dynamic_matrix_dict["B"]),
                                        np.zeros_like(dynamic_matrix_dict["D"]),
                                        self.x[:, i],
                                        init_state,
                                        0, -1, 0)
                self.add_matrix_eq_cons(dynamic_matrix_dict["A"],
                                        dynamic_matrix_dict["B"],
                                        dynamic_matrix_dict["D"],
                                        self.x[:, i + 1],
                                        self.x[:, i],
                                        self.u[i], i, theta_r[i])
            else:
                self.add_matrix_eq_cons(dynamic_matrix_dict["A"],
                                        dynamic_matrix_dict["B"],
                                        dynamic_matrix_dict["D"],
                                        self.x[:, i + 1],
                                        self.x[:, i],
                                        self.u[i], i, theta_r[i])

    def add_matrix_eq_cons(self, A, B, D, x_left, x_right, u, t, z):
        """
        Add an equality constraint in form:
        x = Ax+Bu+Dz
        """
        x_right = x_right.reshape([-1, 1])
        x_prop = A.dot(x_right) + B.dot(u) +D.dot(z)
        # Constraints for enforcing consistency of motion
        for i_x in range(x_right.size):
            self.model.addConstr(x_left[i_x] == x_prop[i_x, 0], "state_trans{}_at_time{}".format(i_x, t))



