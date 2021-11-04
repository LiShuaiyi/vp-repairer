import sympy as sp


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


