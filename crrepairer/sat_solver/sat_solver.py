import sympy as sp
from enum import Enum

from sympy.logic.inference import satisfiable
from commonroad_repair.crrepairer.sat_solver.dpll import DPLL

from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter


class SATSolver:
    def __init__(self,
                 rule_abstracter: RuleAbstracter):
        self._formula = self.construct_cnf(rule_abstracter.sat_encoding)
        self._prop_nodes = rule_abstracter.propositions
        self._prop_robust_all = rule_abstracter.prop_robust_all
        self._init_assign = list()
        self._dpll_solver = DPLL(self._formula, self._prop_robust_all, rule_abstracter.rule_monitor.tv_time_step)
        self._dpll_model = None

    @property
    def formula(self):
        return self._formula

    @property
    def initial_assignment(self):
        return self._init_assign

    @staticmethod
    def stl2sympy(input_formula: str):
        return input_formula.replace("and", "&").replace("or", "|").replace("!", "~").replace("implies", ">>")

    @staticmethod
    def construct_cnf(stl_formula):
        """
        Construct Conjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
        """
        sp_formula = SATSolver.stl2sympy(stl_formula)
        cnf_formula = str(sp.to_cnf(sp_formula))
        return cnf_formula

    @staticmethod
    def construct_dnf(stl_formula):
        """
        Construct Disjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
        """
        sp_formula = SATSolver.stl2sympy(stl_formula)
        dnf_formula = str(sp.to_dnf(sp_formula))
        return dnf_formula

    def check_prior_satisfiability(self, abs_robust_tv):
        self._init_assign = self.obtain_initial_assignment(abs_robust_tv)
        satisfiable_list = list()
        for i in range(len(self._init_assign)):
            updated_formula = self._formula + '& ~' + self._init_assign[i]
            for j in range(len(self._init_assign)):
                if j is not i:
                    updated_formula += '&' + self._init_assign[j]
            if satisfiable(eval(updated_formula)):
                satisfiable_list.append(self._init_assign[i][-1])
        return satisfiable_list

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        # sat_result = satisfiable(eval(self._formula))
        self._dpll_solver.update_cnf(self._formula)
        sat_result = self._dpll_solver.solve()
        return sat_result

    def model(self) -> (list, str):
        """
        return a satisfiable proposition - based on robustness
        """
        # select the unvisited predicates within the least robust proposition at time step TTV.
        # prop_rob_min = self._prop_robust_ttv[self._prop_robust_ttv.robustness.abs()
        #                                      == self._prop_robust_ttv.robustness.abs().min()].alphabet.values
        self._dpll_model = self._dpll_solver.model
        prop_list = list()
        for m in list(self._dpll_model):
            sel_prop_node = next((prop_node for prop_node in self._prop_nodes
                                  if prop_node.alphabet == m[-1]), None)
            prop_list.append(sel_prop_node)
        print("<SATSolver>: model is {}".format(self._dpll_model))
        return prop_list, self._dpll_model

    def update_formula(self):
        """
        Based on the syntax for sympy, the SAT formula is updated by negating the unsatisfiable abstraction:
        phi_SAT = phi_SAT and (not abs)
        """
        if self._formula[0] is not '(':
            self._formula = '(' + self.formula + ')'
        # generate counterexample
        counter_ex = '~' + list(self._dpll_model)[0]
        if len(list(self._dpll_model)) > 1:
            counter_ex = '(' + counter_ex
            for atom in list(self._dpll_model)[1:]:
                counter_ex += ' | ~' + atom
            counter_ex += ')'
        self._formula += ' & ' + counter_ex
        print("<SATSolver>: the formula is updated to {}".format(self._formula))

    @staticmethod
    def obtain_initial_assignment(robustness_tv):
        ini_assign = list()
        for _, row in robustness_tv.iterrows():
            if row['robustness'] > 0:
                ini_assign.append(row['alphabet'])
            else:
                ini_assign.append('~' + row['alphabet'])
        return ini_assign
