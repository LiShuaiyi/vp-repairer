import sympy as sp
from enum import Enum
from sympy.logic.inference import satisfiable

from crmonitor.predicates.rule import AbstractionNode


class SATISFIABILITY(Enum):
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"


class SATSolver:
    def __init__(self, sat_encoding):
        self._formula = self.construct_cnf(sat_encoding)

    @property
    def formula(self):
        return self._formula

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

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        sat_result = satisfiable(self._formula)
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
        self._formula += ' & ~' + abstraction.alphabet

