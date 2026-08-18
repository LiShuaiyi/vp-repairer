import math
import functools

from sympy.logic.boolalg import is_cnf
from copy import deepcopy
from z3 import sat, unsat

from crrepairer.smt.sat_solver.dpll import DPLL


class DomainDPLL:
    def __init__(
        self,
        sympy_cnf: str,
        prop_nodes=None,
        tv_time_step=0,
        domains=None,
        hard_domain_vars=None,
        repair_literals=None,
    ):
        """
        Based on the pseudocode in Wikipedia page:
        https://en.wikipedia.org/wiki/DPLL_algorithm

        domains: dict[str, set[int]]
          e.g. {"a": {0,1}, "b": {1}, "c": {0}}
        """
        assert is_cnf(sympy_cnf), (
            "<DPLL>: the given formula {} is not CNF or"
            " not in the sympy CNF standard".format(sympy_cnf)
        )
        self._prop_nodes = prop_nodes
        self._tv_time_step = tv_time_step
        self._domains = domains or {}
        self._hard_domain_vars = set(hard_domain_vars or ())
        self._repair_literals = list(dict.fromkeys(repair_literals or ()))
        self._base_cnf = self._assign_cnf(sympy_cnf)
        self._cnf = list()
        self._literals = list()
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []
        self._model = set()
        self._rebuild_cnf()

    @property
    def model(self):
        return set.union(self._assign_true, self._assign_false)

    @property
    def literals(self):
        return self._literals

    @property
    def cnf(self):
        return self._cnf

    @property
    def domains(self):
        return self._domains

    @property
    def hard_domain_vars(self):
        return self._hard_domain_vars

    @property
    def repair_literals(self):
        return self._repair_literals

    @staticmethod
    def get_literal(cnf, prop_nodes, tv_time_step: int):
        def robustness_degree(alp):
            rob_min_tv_h = 0
            node = next((x for x in prop_nodes if x.alphabet[-1] == alp[-1]), None)
            for predicate in node.children:
                if predicate.agent_placeholders == (1, 0):
                    rob_min_tv_h += 1
            rob_min_tv_h += abs(node.ttv_h_min)
            return rob_min_tv_h

        literals = []
        for sub in cnf:
            split_cnf = sub.split()
            for lit in split_cnf:
                if lit[-1] not in literals and "~" + lit[-1] not in literals:
                    literals.append(lit)

        if prop_nodes is not None and tv_time_step is not math.inf:
            return sorted(literals, key=robustness_degree)
        else:
            return literals

    @staticmethod
    def _assign_cnf(sympy_cnf):
        return (
            sympy_cnf.replace("(", "")
            .replace("~~", "")
            .replace(")", "")
            .replace("|", "")
            .split(" & ")
        )

    @staticmethod
    def _apply_domains(cnf, domains):
        """
        Turn variable domains into unit clauses and append to CNF.
        - {1} => x
        - {0} => ~x
        - {0,1} => no constraint
        """
        if not domains:
            return cnf

        extra_units = []
        for var, allowed in domains.items():
            if not isinstance(var, str) or len(var) != 1:
                raise ValueError(f"<DPLL>: domain key must be single-character var, got: {var}")
            allowed = set(allowed)
            if allowed == {1}:
                extra_units.append(var)
            elif allowed == {0}:
                extra_units.append("~" + var)
            elif allowed == {0, 1}:
                pass
            else:
                raise ValueError(f"<DPLL>: invalid domain for {var}: {allowed} (must be {{0}}, {{1}}, or {{0,1}})")
        return cnf + extra_units

    def _reset_search_state(self):
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []

    def _rebuild_cnf(self):
        self._cnf = self._apply_domains(deepcopy(self._base_cnf), self._domains)
        if self._repair_literals:
            self._cnf.append(" ".join(self._repair_literals))
        self._literals = self.get_literal(
            self._cnf, self._prop_nodes, self._tv_time_step
        )
        self._reset_search_state()

    def set_domains(self, domains=None):
        self._domains = dict(domains) if domains is not None else {}
        self._rebuild_cnf()

    def set_search_guidance(
        self,
        domains=None,
        hard_domain_vars=None,
        repair_literals=None,
    ):
        self._domains = dict(domains) if domains is not None else {}
        self._hard_domain_vars = set(hard_domain_vars or ())
        self._repair_literals = list(dict.fromkeys(repair_literals or ()))
        self._rebuild_cnf()

    def relax_domains_for_model(self, model=None):
        """
        Remove domain restrictions for variables in a failed SAT model.

        Domains are pruning hints. If a repair attempt rejects a model, keeping
        a unit domain for variables in that model may block the updated CNF from
        exploring alternative repair choices.
        """
        model = self.model if model is None else model
        if not self._domains or not model:
            return []

        relaxed = []
        for literal in model:
            var = literal[-1]
            if var in self._domains and var not in self._hard_domain_vars:
                self._domains.pop(var)
                relaxed.append(var)

        if relaxed:
            self._rebuild_cnf()
        return sorted(set(relaxed))

    def update_cnf(self, cnf, domains=None):
        """
        cnf: sympy CNF string
        domains: optional new domains
        """
        self._base_cnf = self._assign_cnf(cnf)
        if domains is not None:
            self._domains = dict(domains)
        self._rebuild_cnf()

    def solve(self):
        self._active_cnf_states = set()
        try:
            return self._solve(deepcopy(self._cnf))
        finally:
            self._active_cnf_states.clear()

    def back_tracking(self):
        for i in self._new_true:
            self._assign_true.remove(i)
        for i in self._new_false:
            self._assign_false.remove(i)

    def _solve(self, cnf):
        # A repeated CNF on the current recursion path means that the selected
        # literal did not simplify the formula.  Recursing again can never
        # discover a new assignment and previously ended in RecursionError for
        # larger IN5 formulas after counterexample clauses were appended.
        state = tuple(sorted(cnf))
        if state in self._active_cnf_states:
            return unsat
        self._active_cnf_states.add(state)
        try:
            return self._solve_active(cnf)
        finally:
            self._active_cnf_states.remove(state)

    def _solve_active(self, cnf):
        cnf = [clause.replace("~~", "") for clause in cnf]

        # An empty clause is a contradiction and must be handled before unit
        # propagation (the empty string otherwise behaves like a substring of
        # every clause in the string-based representation).
        if any(len(clause) == 0 for clause in cnf):
            self.back_tracking()
            return unsat

        units = [i for i in cnf if len(i) < 3]
        units = list(set(units))
        self._new_true = []
        self._new_false = []
        self._assign_true = set(self._assign_true)
        self._assign_false = set(self._assign_false)

        if len(units):
            cnf = self.unit_propagation(cnf, units)

        if len(cnf) == 0:
            return sat

        if any(len(clause) == 0 for clause in cnf):
            self.back_tracking()
            return unsat

        literals = self.get_literal(cnf, self._prop_nodes, self._tv_time_step)
        lit = self.choose_literal(literals)

        if self._solve(deepcopy(cnf) + [lit]) == sat:
            return sat
        opposite_lit = lit[1:] if lit.startswith("~") else "~" + lit
        if self._solve(deepcopy(cnf) + [opposite_lit]) == sat:
            return sat
        else:
            self._assign_true = set()
            self._assign_false = set()
            return unsat

    def choose_literal(self, literals):
        literal_vars = {literal[-1] for literal in literals}
        for preferred in self._repair_literals:
            if preferred[-1] in literal_vars:
                return preferred
        return literals[0]

    def unit_propagation(self, cnf, units):
        for unit in units:
            if "~" in unit:
                self._assign_false.add(unit)
                self._new_false.append(unit)
                i = 0
                while True:
                    if unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    elif unit[-1] in cnf[i]:
                        cnf[i] = cnf[i].replace(unit[-1], "").strip()
                        if "  " in cnf[i]:
                            cnf[i] = cnf[i].replace("  ", " ")
                    i += 1
                    if i >= len(cnf):
                        break
            else:
                self._assign_true.add(unit)
                self._new_true.append(unit)
                i = 0
                while True:
                    if "~" + unit in cnf[i]:
                        cnf[i] = cnf[i].replace("~" + unit, "").strip()
                        if "  " in cnf[i]:
                            cnf[i] = cnf[i].replace("  ", " ")
                    elif unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    i += 1
                    if i >= len(cnf):
                        break
        return cnf


if __name__ == "__main__":
    # Example 1: a & b, with a in {0,1}, b in {1}
    dpll_solver = DomainDPLL("a & b")
    dpll_solver.set_domains({"a": {0, 1}, "b": {1}})
    print(dpll_solver.solve())
    print(dpll_solver.model)

    # Example 2: a & b, with b in {0} => UNSAT
    dpll_solver2 = DomainDPLL("a & b")
    dpll_solver2.set_domains({"a": {0, 1}, "b": {0}})
    print(dpll_solver2.solve())
    print(dpll_solver2.model)
