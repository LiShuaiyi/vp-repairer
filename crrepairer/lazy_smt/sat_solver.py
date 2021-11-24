import sympy as sp
from enum import Enum
from sympy.logic.inference import satisfiable

from crmonitor.predicates.rule import AbstractionNode
from sympy.abc import *


class SATISFIABILITY(Enum):
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"


class SATSolver:
    def __init__(self, sat_encoding, abs_robust_ttv):
        self._formula = self.construct_cnf(sat_encoding)
        self._sat_list = self.check_prior_satisfiability(abs_robust_ttv)

    @property
    def formula(self):
        return self._formula

    @property
    def satisfiable_subformula_list(self):
        return self._sat_list

    @staticmethod
    def construct_cnf(stl_formula):
        """
        Construct Conjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
        """

        def stl2sympy(input_formula: str):
            return input_formula.replace("and", "&").replace("or", "|").replace("!", "~").replace("implies", ">>")

        sp_formula = stl2sympy(stl_formula)
        cnf_formula = str(sp.to_cnf(sp_formula))
        return cnf_formula

    def check_prior_satisfiability(self, abs_robust_tv):
        def obtain_initial_assignment(robustness_tv):
            ini_assign = list()
            for _, row in robustness_tv.iterrows():
                if row['robustness'] > 0:
                    ini_assign.append(row['alphabet'])
                else:
                    ini_assign.append('~' + row['alphabet'])
            return ini_assign
        initial_assignment = obtain_initial_assignment(abs_robust_tv)
        satisfiable_list = list()
        for i in range(len(initial_assignment)):
            updated_formula = self._formula + '& ~' + initial_assignment[i]
            for j in range(len(initial_assignment)):
                if j is not i:
                    updated_formula += '&' + initial_assignment[j]
            if satisfiable(eval(updated_formula)):
                satisfiable_list.append(initial_assignment[i][-1])
        return satisfiable_list

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        sat_result = satisfiable(eval(self._formula))
        if sat_result is False:
            return SATISFIABILITY.UNSAT
        else:
            return SATISFIABILITY.SAT

    def update_formula(self, abstraction: AbstractionNode):
        """
        Based on the syntax for sympy, the SAT formula is updated by negating the unsatisfiable abstraction:
        phi_SAT = phi_SAT and (not abs)
        """
        if self._formula[0] is not '(':
            self._formula = '(' + self.formula + ')'
        # generate counterexample
        sign = self._formula[self.formula.index(abstraction.alphabet)-1]
        if sign != '~':
            sign = ''
        self._formula += ' & ~' + sign + abstraction.alphabet
