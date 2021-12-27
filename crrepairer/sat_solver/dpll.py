from sympy.logic.boolalg import is_cnf
from copy import deepcopy
from z3 import sat, unsat


class DPLL:
    def __init__(self, sympy_cnf: str,
                 prop_robust_ttv=None):
        """
        Based on the pseudocode in Wikipedia page:
        https://en.wikipedia.org/wiki/DPLL_algorithm
        """
        assert is_cnf(sympy_cnf), "<DPLL>: the given formula {} is not CNF or" \
                                  " not in the sympy CNF standard".format(sympy_cnf)
        self._cnf = self._assign_cnf(sympy_cnf)
        self._literals = self.get_literal(self._cnf, prop_robust_ttv)


    @property
    def literals(self):
        return self._literals

    @property
    def cnf(self):
        return self._cnf

    @staticmethod
    def get_literal(cnf, prop_robust_ttv):
        def robustness_degree(alp):
            return abs(prop_robust_ttv[prop_robust_ttv['alphabet'] == alp[-1]].robustness.values[0])
        literals = []
        for sub in cnf:
            split_cnf = sub.split()
            for lit in split_cnf:
                if lit[-1] not in literals and '~' + lit[-1] not in literals:
                    literals.append(lit)
        # use robustness as heuristics to rank the literals
        if prop_robust_ttv is not None:
            return sorted(literals, key=robustness_degree)
        else:
            return literals

    @staticmethod
    def _assign_cnf(sympy_cnf):
        return sympy_cnf.replace('(', '').replace(')', '').replace('|', '').split(' & ')

    def update_cnf(self, cnf):
        self._cnf = self._assign_cnf(cnf)

    def solve(self):
        return self._solve(deepcopy(self._cnf))

    def _solve(self, cnf):
        cnf = list(set(cnf))
        units = [i for i in cnf if len(i) < 3]
        if len(units):
            cnf = self.unit_propagation(cnf, units)
        if len(cnf) == 0:
            # if \phi is a consistent set of literals
            return sat
        if sum(len(clause) == 0 for clause in cnf):
            # if \phi contains an empty clause
            return unsat
        literals = self.get_literal(''.join(cnf), None)
        lit = self.choose_literal(literals)
        print('<DPLL>: literal ({}) is selected'.format(lit))
        if self._solve(deepcopy(cnf) + [lit]):
            return sat
        elif self._solve(deepcopy(cnf) + ['~'+lit]):
            return sat
        else:
            return unsat

    def choose_literal(self, literals):
        return literals[0]

    def unit_propagation(self, cnf, units):
        print(units, cnf)
        for unit in units:
            if '~' in unit:
                i = 0
                while True:
                    if unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    elif unit[-1] in cnf[i]:
                        cnf[i] = cnf[i].replace(unit[-1], '').strip()
                    i += 1
                    if i >= len(cnf):
                        break
            else:
                i = 0
                while True:
                    if '~'+unit in cnf[i]:
                        cnf[i] = cnf[i].replace('~'+unit, '').strip()
                        if '  ' in cnf[i]:
                            cnf[i] = cnf[i].replace('  ', ' ')
                    elif unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    i += 1
                    if i >= len(cnf):
                        break
        return cnf

if __name__ == '__main__':
    dpll_solver = DPLL('(a | b) & ~a')
    # dpll_solver.update_cnf('a & ~a')
    print(dpll_solver.solve())