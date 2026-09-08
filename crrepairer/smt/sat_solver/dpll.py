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
        def node_for(alp):
            return next(
                (x for x in prop_nodes or () if x.alphabet[-1] == alp[-1]), None
            )

        def predicate_name(predicate):
            name = getattr(predicate, "base_name", None)
            if name is None:
                name = getattr(getattr(predicate, "evaluator", None), "predicate_name", "")
            return str(getattr(name, "value", name)).lower()

        def repairability_rank(node):
            """Prefer the small set of predicates p1c1 can encode exactly."""
            if node is None:
                return 2
            # Past truth values cannot be changed by replanning the suffix.
            if str(getattr(node, "name", "")).startswith(("previous", "prev", "pre")):
                return 2
            names = [predicate_name(predicate) for predicate in node.children]
            node_name = str(getattr(node, "name", "")).lower()
            # For R_IN1, keeping the stop line in front falsifies the crossing
            # antecedent and needs no three-second stop.  Try that direct,
            # cheaper repair before constructing the historical stop window.
            if any("stop_line_in_front" in name for name in names) and not any(
                token in node_name for token in ("historically", "once")
            ):
                return 0
            if any("stop_line_in_front" in name or "in_standstill" in name for name in names):
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
                    "speed_limit", "abrupt", "safe_distance", "in_front_of",
                    "precedes", "in_same_lane",
                )
            ):
                return 2
            # Priority is an environment/right-of-way fact, not a maneuver
            # that p1c1 can realize directly.  Keep it searchable (no hard
            # domain restriction), but try it only after other predicates so
            # it cannot cheaply satisfy the Boolean formula before the
            # conflict/stop-line action has been considered.
            priority_tokens = (
                "same_priority",
                "target_has_priority",
                "has_priority",
            )
            if any(token in node_name for token in priority_tokens) or any(
                token in name for name in names for token in priority_tokens
            ):
                return 4
            return 3

        def robustness_degree(alp):
            node = node_for(alp)
            if node is None:
                return (2, math.inf, alp[-1])
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
            return (repairability_rank(node), target_penalty, distance, alp[-1])

        def desired_repair_literal(alp):
            """Try the polarity which flips the proposition at TV first."""
            node = node_for(alp)
            if node is None:
                return alp
            names = [predicate_name(predicate) for predicate in node.children]
            node_name = str(getattr(node, "name", ""))
            if node_name.startswith(("previous", "prev", "pre")):
                return alp
            if any(
                "stop_line_in_front" in name or "in_standstill" in name
                for name in names
            ):
                # A proposition wrapper can contain an explicit Boolean not.
                return ("~" if node_name.count("not(") % 2 else "") + alp[-1]
            if any(
                "in_intersection_conflict_area" in predicate_name(predicate)
                and getattr(predicate, "agent_placeholders", None) == (0, 1)
                for predicate in node.children
            ):
                return "~" + alp[-1]
            try:
                value = float(getattr(node, "ttv_value", 0.0))
            except (TypeError, ValueError):
                return alp
            if not math.isfinite(value) or value == 0.0:
                return alp
            return alp[-1] if value < 0.0 else "~" + alp[-1]

        literals = []
        for sub in cnf:
            split_cnf = sub.split()
            for lit in split_cnf:
                if lit[-1] not in literals and "~" + lit[-1] not in literals:
                    test = lit[-1]
                    literals.append(lit)
        # use robustness as heuristics to rank the literals
        if prop_nodes is not None and tv_time_step is not math.inf:
            ordered = sorted(literals, key=robustness_degree)
            return [desired_repair_literal(literal) for literal in ordered]
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
        # Keep the assignment local to each recursive branch. The legacy
        # implementation shared ``_assign_*`` across recursion levels while
        # ``_new_*`` only described the latest level, so a failed branch could
        # leak literals and yield a model violating a newly added blocking
        # clause.
        clauses = [tuple(clause.split()) for clause in self._cnf]
        model = self._solve_with_assignment(clauses, {})
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []
        if model is None:
            return unsat
        self._assign_true = {var for var, value in model.items() if value}
        self._assign_false = {"~" + var for var, value in model.items() if not value}
        return sat

    @staticmethod
    def _literal_value(literal, assignment):
        var = literal[-1]
        if var not in assignment:
            return None
        return assignment[var] != literal.startswith("~")

    def _solve_with_assignment(self, clauses, assignment):
        residual = []
        for clause in clauses:
            values = [self._literal_value(literal, assignment) for literal in clause]
            if any(value is True for value in values):
                continue
            remaining = tuple(
                literal for literal, value in zip(clause, values) if value is None
            )
            if not remaining:
                return None
            residual.append(remaining)

        if not residual:
            return assignment

        unit = next((clause[0] for clause in residual if len(clause) == 1), None)
        if unit is not None:
            extended = dict(assignment)
            extended[unit[-1]] = not unit.startswith("~")
            return self._solve_with_assignment(residual, extended)

        residual_strings = [" ".join(clause) for clause in residual]
        selected = self.choose_literal(
            self.get_literal(residual_strings, self._prop_nodes, self._tv_time_step)
        )
        opposite = selected[1:] if selected.startswith("~") else "~" + selected
        for literal in (selected, opposite):
            extended = dict(assignment)
            extended[literal[-1]] = not literal.startswith("~")
            model = self._solve_with_assignment(residual, extended)
            if model is not None:
                return model
        return None

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
