import math

from gurobipy import Model as Gmodel
from gurobipy import QuadExpr, LinExpr, GRB

import numpy as np

from miqp_planner.miqp_constraints import LongitudinalConstraint


class GurobiSolver:
    def __init__(self):
        self.u_shape = None
        self.u = None
        self.x_shape = None
        self.x = None
        self.slack = None
        self.constraint_obj = None
        self.delta = {}
        self.model = Gmodel()

    def add_long_state_var(self, x, x_shape, x_lb, x_ub):
        self.x = x
        self.x_shape = x_shape
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                self.x[i, j] = self.add_var(
                    "continuous", "x_long_{}_{}".format(i, j), x_lb[i, j], x_ub[i, j]
                )

    def add_long_control_var(self, u, u_shape, u_lb, u_ub):
        self.u = u
        self.u_shape = u_shape
        for i in range(self.u_shape[0]):
            self.u[i] = self.add_var(
                "continuous", "u_long_{}".format(i), u_lb[i], u_ub[i]
            )

    def add_slack_var(self, slack, slack_shape, slack_lb, slack_ub):
        self.slack = slack
        self.slack_shape = slack_shape
        # for i in range(self.slack_shape[0]):
        #     for j in range(self.slack_shape[1]):
        #         self.slack[i, j] = self.add_var(
        #             "continuous", "slack_{}_{}".format(i, j), slack_lb[i, j], slack_ub[i, j]
        #         )
        for i in range(self.slack_shape[0]):
            self.slack[i] = self.add_var(
                "continuous", "u_long_{}".format(i), slack_lb[i], slack_ub[i]
            )

    def add_lat_state_var(self, x, x_shape, x_lb, x_ub):
        self.x = x
        self.x_shape = x_shape
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                self.x[i, j] = self.add_var(
                    "continuous", "x_lat_{}_{}".format(i, j), x_lb[i, j], x_ub[i, j]
                )

    def add_lat_control_var(self, u, u_shape, u_lb, u_ub):
        self.u = u
        self.u_shape = u_shape
        for i in range(self.u_shape[0]):
            self.u[i] = self.add_var(
                "continuous", "u_lat_{}".format(i), u_lb[i], u_ub[i]
            )

    def add_var(self, typeofvar, name, lb=0, ub=0):
        if typeofvar == "continuous":
            return self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
        if typeofvar == "binary":
            return self.model.addVar(vtype=GRB.BINARY, name=name)

    def add_long_dynamic_cons(self, dynamic_matrix_list, init_state):
        z = np.zeros(self.x_shape[1])
        self.add_dynamic_cons(dynamic_matrix_list, init_state, z)

    def add_lat_dynamic_cons(self, dynamic_matrix_list, init_state, theta_r):
        z = theta_r
        self.add_dynamic_cons(dynamic_matrix_list, init_state, z)

    def add_dynamic_cons(self, dynamic_matrix_list, init_state, theta_r):
        for i in range(len(dynamic_matrix_list)):
            dynamic_matrix_dict = dynamic_matrix_list[i]
            if i == 0:
                # TODO: initial states constraints t = 0 (this should be a time invariant cons)
                self.add_matrix_eq_cons(
                    np.eye(dynamic_matrix_dict["A"].shape[0]),
                    np.zeros_like(dynamic_matrix_dict["B"]),
                    np.zeros_like(dynamic_matrix_dict["D"]),
                    self.x[:, i],
                    init_state,
                    0,
                    -1,
                    0,
                )
                self.add_matrix_eq_cons(
                    dynamic_matrix_dict["A"],
                    dynamic_matrix_dict["B"],
                    dynamic_matrix_dict["D"],
                    self.x[:, i + 1],
                    self.x[:, i],
                    self.u[i],
                    i,
                    theta_r[i],
                )
            else:
                self.add_matrix_eq_cons(
                    dynamic_matrix_dict["A"],
                    dynamic_matrix_dict["B"],
                    dynamic_matrix_dict["D"],
                    self.x[:, i + 1],
                    self.x[:, i],
                    self.u[i],
                    i,
                    theta_r[i],
                )

    def add_rule_cons(self, rule_constraints):
        for rule_constraint_name in rule_constraints:
            rule_constraint = rule_constraints[rule_constraint_name]
            if rule_constraint["decision_variable"]:
                self.create_binary_variable_in_cons(
                    rule_constraint["num_decision_variables"],
                    rule_constraint["constraint_name"],
                )
                self.add_binary_variables_cons(rule_constraint)
                # TODO: fix constraint name
                self.add_binary_rule_constraint(rule_constraint, big_M=3000)
            else:
                self.add_rule_constraint(rule_constraint)

    def add_matrix_eq_cons(self, A, B, D, x_left, x_right, u, t, z):
        """
        Add an equality constraint in form:
        x = Ax+Bu+Dz
        """
        x_right = x_right.reshape([-1, 1])
        x_prop = A.dot(x_right) + B.dot(u) + D.dot(z)
        # Constraints for enforcing consistency of motion
        for i_x in range(x_right.size):
            self.model.addConstr(
                x_left[i_x] == x_prop[i_x, 0], "state_trans{}_at_time{}".format(i_x, t)
            )

    def create_binary_variable_in_cons(self, num_variables, constraint_name):
        for i in range(num_variables):
            delta_tmp = list()
            for k in range(self.x_shape[1]):
                delta_tmp.append(
                    self.add_var(
                        typeofvar="binary",
                        name="delta_{}_{}_{}".format(constraint_name, i + 1, k),
                    )
                )
            self.delta["{}_{}".format(constraint_name, i + 1)] = np.array(delta_tmp)

    def add_binary_variables_cons(self, rule_constraint):
        for i in range(self.x_shape[1] - 1):
            # if i == 0:
            #     params_dict = {"vars": [[1, self.delta['conflict_area_1'][i]]]}
            #     self.add_eq_cons(params_dict, "binary_variable_conflict_area_constraint_init")
            params_dict = {
                "vars": [
                    [1, self.delta["{}_1".format(rule_constraint["constraint_name"])][i]],
                    [-1, self.delta["{}_1".format(rule_constraint["constraint_name"])][i + 1]],
                ]
            }
            self.add_eq_cons(
                params_dict, "binary_variable_conflict_area_constraint_{}".format(i)
            )

    def add_binary_rule_constraint(self, rule_constraint, big_M):
        if rule_constraint["constraint_name"] in ["conflict_area", "intersection"]:
            for i in rule_constraint["time_step"]:
                time_step = rule_constraint["time_step"].index(i)
                if rule_constraint["s_limit_front"][time_step] != math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [1, self.x[0, time_step]],
                        [
                            -big_M,
                            self.delta["{}_1".format(rule_constraint["constraint_name"])][
                                time_step
                            ],
                        ],
                    ]
                    params_dict["constants"] = [
                        -rule_constraint["s_limit_front"][time_step]
                    ]
                    if self.slack is not None:
                        # params_dict["vars"].append([-1, self.slack[1, time_step]])
                        params_dict["vars"].append([-1, self.slack[1]])
                    self.add_ineq_cons(
                        params_dict,
                        "{}_front_t{}".format(
                            rule_constraint["constraint_name"], time_step
                        ),
                    )
                if rule_constraint["s_limit_behind"][time_step] != -math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [-1, self.x[0, time_step]],
                        [
                            big_M,
                            self.delta["{}_1".format(rule_constraint["constraint_name"])][
                                time_step
                            ],
                        ],
                    ]
                    params_dict["constants"] = [
                        rule_constraint["s_limit_behind"][time_step],
                        -big_M,
                    ]
                    if self.slack is not None:
                        pass
                        # params_dict["vars"].append([-1, self.slack[0, time_step]])
                        # params_dict["vars"].append([-1, self.slack[0]])
                    self.add_ineq_cons(
                        params_dict,
                        "{}_behind_t{}".format(
                            rule_constraint["constraint_name"], time_step
                        ),
                    )
        else:
            print("warning: no constraints added")

    def add_rule_constraint(self, rule_constraint):
        if rule_constraint["constraint_name"] in ["stop_line"]:
            for i in rule_constraint["time_step"]:
                time_step = rule_constraint["time_step"].index(i)
                if rule_constraint["s_limit_front"][time_step] != math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [1, self.x[0, time_step]],
                    ]
                    params_dict["constants"] = [-rule_constraint["s_limit_front"][time_step]]
                    if self.slack is not None:
                        # params_dict["vars"].append([-1, self.slack[1, time_step]])
                        params_dict["vars"].append([-1, self.slack[1]])
                    self.add_ineq_cons(
                        params_dict,
                        "{}_front_t{}".format(rule_constraint["constraint_name"], time_step),
                    )
                if rule_constraint["s_limit_behind"][time_step] != -math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [-1, self.x[0, time_step]],
                    ]
                    params_dict["constants"] = [
                        rule_constraint["s_limit_behind"][time_step],
                    ]
                    if self.slack is not None:
                        pass
                        # params_dict["vars"].append([-1, self.slack[0, time_step]])
                        # params_dict["vars"].append([-1, self.slack[0]])
                    self.add_ineq_cons(
                        params_dict,
                        "{}_behind_t{}".format(rule_constraint["constraint_name"], time_step),
                    )
        else:
            print("warning: no constraints added")

    def add_collision_free_cons(self, collision_free_constraint):
        for index in range(len(collision_free_constraint["index_lb"])):
            params_dict = {}
            params_dict["vars"] = [
                [-1, self.x[0, collision_free_constraint["index_lb"][index]]]
            ]
            params_dict["constants"] = [
                collision_free_constraint["collision_free_lb"][index]
            ]
            self.add_ineq_cons(
                params_dict,
                "collision_free_cons_lb_at_time_{}".format(
                    collision_free_constraint["index_lb"][index]
                ),
            )

        for index in range(len(collision_free_constraint["index_ub"])):
            params_dict = {}
            params_dict["vars"] = [
                [1, self.x[0, collision_free_constraint["index_ub"][index]]]
            ]
            params_dict["constants"] = [
                -collision_free_constraint["collision_free_ub"][index]
            ]
            self.add_ineq_cons(
                params_dict,
                "collision_free_cons_ub_at_time_{}".format(
                    collision_free_constraint["index_ub"][index]
                ),
            )

    def add_lat_dis_cons(self, lat_dis_cons_matrix, theta, d_min, d_max):
        # TODO: need to be tested
        for i in range(self.x_shape[1] - 1):
            if list(d_min[i]) != [-np.inf, -np.inf, -np.inf]:
                S = lat_dis_cons_matrix[i]["S"]
                C = lat_dis_cons_matrix[i]["C"]
                E = lat_dis_cons_matrix[i]["E"]
                distance = S.dot(C.dot(self.x[:, i + 1]) + E.dot(theta[i]))
                for k in range(distance.size):
                    params_dict = {}
                    params_dict["vars"] = [[-1, distance[k]]]
                    params_dict["constants"] = [d_min[i, k]]
                    self.add_ineq_cons(
                        params_dict, "lat_dist_cons_min_{}_{}".format(i + 1, k + 1)
                    )

            if list(d_max[i]) != [np.inf, np.inf, np.inf]:
                S = lat_dis_cons_matrix[i]["S"]
                C = lat_dis_cons_matrix[i]["C"]
                E = lat_dis_cons_matrix[i]["E"]
                distance = S.dot(C.dot(self.x[:, i + 1]) + E.dot(theta[i]))
                for k in range(distance.size):
                    params_dict = {}
                    params_dict["vars"] = [[1, distance[k]]]
                    params_dict["constants"] = [-d_max[i, k]]
                    self.add_ineq_cons(
                        params_dict, "lat_dist_cons_max_{}_{}".format(i + 1, k + 1)
                    )

    def add_kappa_limit(self, kappa_lim):
        for i in range(self.x_shape[1] - 1):
            params_dict = {}
            params_dict["vars"] = [[-1, self.x[2, i + 1]]]
            params_dict["constants"] = [-kappa_lim[i]]
            self.add_ineq_cons(params_dict, "lat_kappa_lim_min_{}".format(i + 1))

            params_dict = {}
            params_dict["vars"] = [[1, self.x[2, i + 1]]]
            params_dict["constants"] = [-kappa_lim[i]]
            self.add_ineq_cons(params_dict, "lat_kappa_lim_max_{}".format(i + 1))

    def add_ineq_cons(self, params_dict: dict, name):
        equation = LinExpr()
        for params in params_dict["vars"]:
            equation.add(params[1], params[0])
        if "constants" in params_dict.keys():
            for constant in params_dict["constants"]:
                equation.add(constant)
        self.model.addConstr(equation <= 0, name=name)

    def add_eq_cons(self, params_dict: dict, name):
        equation = LinExpr()
        for params in params_dict["vars"]:
            equation.add(params[1], params[0])
        if "constants" in params_dict.keys():
            for constant in params_dict["constants"]:
                equation.add(constant)
        self.model.addConstr(equation == 0, name=name)

    def costfunc_long(self, x_ref, weight):
        # TODO: cost function definition
        long_costs = QuadExpr()
        weight_s = weight[0]
        weight_v = weight[1]
        weight_a = weight[2]
        weight_j = weight[3]
        weight_u = weight[4]
        weight_slack = weight[5]
        for i in range(self.x_shape[1]):
            diff_ref = LinExpr()
            diff_ref.add(self.x[0, i])
            diff_ref.addConstant(-x_ref.reference[i].s)
            long_costs.add(diff_ref * diff_ref, weight_s)

            diff_ref.clear()
            diff_ref.add(self.x[1, i])
            diff_ref.addConstant(-x_ref.reference[i].v)
            long_costs.add(diff_ref * diff_ref, weight_v)

            diff_ref.clear()
            diff_ref.add(self.x[2, i])
            diff_ref.addConstant(-x_ref.reference[i].a)
            long_costs.add(diff_ref * diff_ref, weight_a)

            diff_ref.clear()
            diff_ref.add(self.x[3, i])
            diff_ref.addConstant(-x_ref.reference[i].j)
            long_costs.add(diff_ref * diff_ref, weight_j)

        for u in self.u:
            long_costs.add(u * u, weight_u)

        if self.slack is not None:
            # for i in range(self.slack_shape[1]):
            #     long_costs.add(self.slack[0, i] * self.slack[0, i], weight_slack)
            #     long_costs.add(self.slack[1, i] * self.slack[1, i], weight_slack)
            for slack in self.slack:
                long_costs.add(slack, weight_slack)

        self.model.setObjective(long_costs, GRB.MINIMIZE)

    def costfunc_lat(self, x_ref, weight, d_reference):
        # TODO: cost function definition
        # TODO: check the length of reference
        lat_costs = QuadExpr()
        weight_d = weight[0]
        weight_theta = weight[1]
        weight_kappa = weight[2]
        weight_kappa_dot = weight[3]
        weight_u = weight[4]
        for i in range(1, self.x_shape[1]):
            diff_ref = LinExpr()
            diff_ref.add(self.x[0, i])
            diff_ref.addConstant(-d_reference[i])
            lat_costs.add(diff_ref * diff_ref, weight_d)

            diff_ref.clear()
            diff_ref.add(self.x[1, i])
            diff_ref.addConstant(-round(x_ref.reference[i].theta, 2))
            lat_costs.add(diff_ref * diff_ref, weight_theta)

            diff_ref.clear()
            diff_ref.add(self.x[2, i])
            diff_ref.addConstant(-round(x_ref.reference[i].kappa, 2))
            lat_costs.add(diff_ref * diff_ref, weight_kappa)

            diff_ref.clear()
            diff_ref.add(self.x[3, i])
            diff_ref.addConstant(0)
            lat_costs.add(diff_ref * diff_ref, weight_kappa_dot)

        for u in self.u:
            lat_costs.add(u * u, weight_u)

        self.model.setObjective(lat_costs, GRB.MINIMIZE)

    def solve(self):
        self.model.update()
        self.model.optimize()

    def get_var_x(self):
        """
        Get state variables from solution
        """
        x_value = np.empty(self.x_shape)
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                x_value[i, j] = self.get_var(self.x[i, j])
        return x_value

    def get_delta(self):
        """
        Get deta_1 and delta_2 from solution
        """
        all_delta = list()
        for delta_name in self.delta:
            delta = self.delta[delta_name]
            delta_value = np.empty(delta.shape)
            for i in range(len(delta)):
                delta_value[i] = self.get_var(self.delta[delta_name][i])
            all_delta.append(delta_value)
        return all_delta

    def get_control_u(self):
        u_value = np.empty(self.u_shape)
        for i in range(self.u_shape[0]):
            u_value[i] = self.get_var(self.u[i])
        return u_value

    def get_slack_var(self):
        if self.slack is not None:
            slack_value = np.empty(self.slack_shape)
            # for i in range(self.slack_shape[0]):
            #     for j in range(self.slack_shape[1]):
            #         slack_value[i, j] = self.get_var(self.slack[i, j])
            # return slack_value
            for i in range(self.slack_shape[0]):
                slack_value[i] = self.get_var(self.slack[i])
            return slack_value
        else:
            return None

    def get_var(self, var):
        return var.x
