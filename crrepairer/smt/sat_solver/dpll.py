import math
import functools

from sympy.logic.boolalg import is_cnf
from copy import deepcopy
from z3 import sat, unsat


class DPLL:
    def __init__(self, sympy_cnf: str, prop_nodes=None, tv_time_step=0):
        """
        Based on the pseudocode in Wikipedia page:
        https://en.wikipedia.org/wiki/DPLL_algorithm
        """
        assert is_cnf(sympy_cnf), (
            "<DPLL>: the given formula {} is not CNF or"
            " not in the sympy CNF standard".format(sympy_cnf)
        )
        self._cnf = self._assign_cnf(sympy_cnf)
        self._prop_nodes = prop_nodes
        self._tv_time_step = tv_time_step
        self._literals = self.get_literal(self._cnf, prop_nodes, tv_time_step)
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []
        self._model = set()

    @property
    def model(self):
        return set.union(self._assign_true, self._assign_false)

    @property
    def literals(self):
        return self._literals

    @property
    def cnf(self):
        return self._cnf

    @staticmethod
    def get_literal(cnf, prop_nodes, tv_time_step: int):
        def robustness_degree(alp):
            node = next(
                (x for x in prop_nodes if x.alphabet[-1] == alp[-1]),
                None,
            )
            if node is None:
                return math.inf
            value = getattr(node, "ttv_h_min", math.inf)
            try:
                distance = abs(float(value))
            except (TypeError, ValueError):
                distance = math.inf
            if not math.isfinite(distance):
                distance = math.inf
            target_penalty = sum(
                getattr(predicate, "agent_placeholders", None) == (1, 0)
                for predicate in node.children
            )
            return target_penalty + distance

        literals = []
        for sub in cnf:
            split_cnf = sub.split()
            for lit in split_cnf:
                if lit[-1] not in literals and "~" + lit[-1] not in literals:
                    test = lit[-1]
                    literals.append(lit)
        # use robustness as heuristics to rank the literals
        if prop_nodes is not None and tv_time_step is not math.inf:
            return sorted(literals, key=robustness_degree)
        else:
            return literals

    @staticmethod
    def get_domain_guided_literal(cnf, prop_nodes, tv_time_step: int):
        """Return the VP-aware ordering used only by DomainDPLL.

        Plain DPLL deliberately keeps the original robustness-only order.
        DomainDPLL may additionally prioritize constraint-backed predicates
        and the polarity that changes the monitored value.
        """
        def node_for(alphabet):
            return next(
                (
                    node
                    for node in prop_nodes or ()
                    if node.alphabet[-1] == alphabet[-1]
                ),
                None,
            )

        def predicate_name(predicate):
            name = getattr(predicate, "base_name", None)
            if name is None:
                name = getattr(
                    getattr(predicate, "evaluator", None),
                    "predicate_name",
                    "",
                )
            return str(getattr(name, "value", name)).lower()

        def repairability_rank(node):
            if node is None:
                return 2
            node_name = str(getattr(node, "name", ""))
            if node_name.startswith(("previous", "prev", "pre")):
                return 2
            names = [predicate_name(predicate) for predicate in node.children]
            if any("stop_line_in_front" in name for name in names) and not any(
                token in node_name for token in ("historically", "once")
            ):
                return 0
            if any(
                "stop_line_in_front" in name or "in_standstill" in name
                for name in names
            ):
                return 1
            if any(
                "in_intersection_conflict_area" in predicate_name(predicate)
                and getattr(predicate, "agent_placeholders", None) == (0, 1)
                for predicate in node.children
            ):
                return 1
            if any(
                token in name
                for name in names
                for token in (
                    "speed_limit",
                    "abrupt",
                    "safe_distance",
                    "in_front_of",
                    "precedes",
                    "in_same_lane",
                )
            ):
                return 2
            priority_tokens = (
                "same_priority",
                "target_has_priority",
                "has_priority",
            )
            if any(token in node_name.lower() for token in priority_tokens) or any(
                token in name
                for name in names
                for token in priority_tokens
            ):
                return 4
            return 3

        def guided_rank(literal):
            node = node_for(literal)
            if node is None:
                return (2, math.inf, literal[-1])
            try:
                distance = abs(float(getattr(node, "ttv_h_min", math.inf)))
            except (TypeError, ValueError):
                distance = math.inf
            if not math.isfinite(distance):
                distance = math.inf
            target_penalty = sum(
                getattr(predicate, "agent_placeholders", None) == (1, 0)
                for predicate in node.children
            )
            return (
                repairability_rank(node),
                target_penalty,
                distance,
                literal[-1],
            )

        def desired_repair_literal(literal):
            node = node_for(literal)
            if node is None:
                return literal
            names = [predicate_name(predicate) for predicate in node.children]
            node_name = str(getattr(node, "name", ""))
            if node_name.startswith(("previous", "prev", "pre")):
                return literal
            if any(
                "stop_line_in_front" in name or "in_standstill" in name
                for name in names
            ):
                return ("~" if node_name.count("not(") % 2 else "") + literal[-1]
            if any(
                "in_intersection_conflict_area" in predicate_name(predicate)
                and getattr(predicate, "agent_placeholders", None) == (0, 1)
                for predicate in node.children
            ):
                return "~" + literal[-1]
            try:
                value = float(getattr(node, "ttv_value", 0.0))
            except (TypeError, ValueError):
                return literal
            if not math.isfinite(value) or value == 0.0:
                return literal
            return literal[-1] if value < 0.0 else "~" + literal[-1]

        literals = DPLL.get_literal(cnf, None, tv_time_step)
        if prop_nodes is None or tv_time_step is math.inf:
            return literals
        return [
            desired_repair_literal(literal)
            for literal in sorted(literals, key=guided_rank)
        ]

    @staticmethod
    def _assign_cnf(sympy_cnf):
        return (
            sympy_cnf.replace("(", "")
            .replace("~~", "")
            .replace(")", "")
            .replace("|", "")
            .split(" & ")
        )

    def update_cnf(self, cnf):
        self._cnf = self._assign_cnf(cnf)
        self._literals = self.get_literal(
            self._cnf, self._prop_nodes, self._tv_time_step
        )
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []

    def solve(self):
        # Plain DPLL is the unguided comparison baseline. Keep the recursive
        # enumeration used by the original implementation (da120f4); all
        # domain-aware search remains isolated in DomainDPLL.
        return self._solve(deepcopy(self._cnf))

    def back_tracking(self):
        for i in self._new_true:
            self._assign_true.remove(i)
        for i in self._new_false:
            self._assign_false.remove(i)

    def _solve(self, cnf):
        cnf = [clause.replace("~~", "") for clause in cnf]
        units = [i for i in cnf if len(i) < 3]
        units = list(set(units))
        self._new_true = []
        self._new_false = []
        self._assign_true = set(self._assign_true)
        self._assign_false = set(self._assign_false)
        if len(units):
            # cnf = [clause.replace("~~", "") for clause in cnf]
            cnf = self.unit_propagation(cnf, units)
        if len(cnf) == 0:
            # if \phi is a consistent set of literals
            return sat
        if sum(len(clause) == 0 for clause in cnf):
            # if \phi contains an empty clause
            self.back_tracking()
            return unsat
        literals = self.get_literal(cnf, self._prop_nodes, self._tv_time_step)
        lit = self.choose_literal(literals)
        # print('<DPLL>: literal ({}) is selected'.format(lit))

        if self._solve(deepcopy(cnf) + [lit]) == sat:
            return sat
        elif self._solve(deepcopy(cnf) + ["~" + lit]) == sat:
            return sat
        else:
            self._assign_true = set()
            self._assign_false = set()
            return unsat

    def choose_literal(self, literals):
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
    dpll_solver = DPLL("a & ~a")
    print(dpll_solver.solve())
    print(dpll_solver.model)
