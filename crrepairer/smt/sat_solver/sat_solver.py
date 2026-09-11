import os

import sympy as sp

from crrepairer.smt.sat_solver.dpll import DPLL
from crrepairer.smt.sat_solver.dpll_domain import DomainDPLL
from crrepairer.smt.monitor_wrapper import PropositionNode, STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.smt import construct_cnf, stl2sympy

class SATSolver:
    def __init__(self, rule_monitor: STLRuleMonitor, config: RepairerConfiguration):
        # nnf is constructed with the monitor
        # self._formula = construct_nnf(rule_monitor.sat_formula)
        self._formula = construct_cnf(rule_monitor.sat_formula)
        self._source_formula = rule_monitor.sat_formula
        print("* \t<SATSolver>: the formula in CNF is {}".format(self._formula))
        self._prop_nodes = rule_monitor.proposition_nodes
        self._prop_robust_all = rule_monitor.rob_abstraction
        self._init_assign = list()

        self._config = config
        self._solver_mode = getattr(config.repair, "sat_solver_mode", "dpll")
        self._domain_dict = {}
        self._hard_domain_vars = set()
        self._repair_literals = []
        self._dpll_solver = self._build_solver(rule_monitor.tv_time_step)
        self._solver_formula = self._formula
        self._dpll_model = None
        self._expanded_decomposition_groups = set()

    def _build_solver(self, tv_time_step):
        if self._solver_mode == "domain_dpll":
            return DomainDPLL(
                self._formula,
                self._prop_nodes,
                tv_time_step,
            )
        return DPLL(self._formula, self._prop_nodes, tv_time_step)

    @property
    def formula(self):
        return self._formula

    @property
    def initial_assignment(self):
        return self._init_assign

    @property
    def solver_mode(self):
        return self._solver_mode

    @property
    def domain_dict(self):
        return self._domain_dict

    def set_domain_dict(
        self,
        domain_dict,
        hard_domain_vars=None,
        repair_literals=None,
    ):
        self._domain_dict = dict(domain_dict) if domain_dict is not None else {}
        self._hard_domain_vars = set(hard_domain_vars or ())
        self._repair_literals = list(dict.fromkeys(repair_literals or ()))

    def add_once_time_selectors(self, once_time_plans):
        """Encode finite existential time alternatives with local Tseitin clauses.

        The original compound proposition ``p`` remains in the rule CNF.  For
        each discrete end time ``w``, a fresh ``z_w`` represents the conjunction of its
        atomic VP obligations, and ``p`` represents the disjunction of all
        time alternatives::

            z_w <-> (q_w_1 & ... & q_w_n)
            p   <-> (z_1 | ... | z_k)

        The corresponding definitional clauses are already CNF, so this path
        never asks SymPy to distribute the expanded temporal expression.
        Pairwise exclusion retains the repairer's one-common-time search
        policy.
        """
        used = {
            str(node.alphabet).lstrip("~")
            for node in self._prop_nodes
        }
        # The in-tree DPLL representation uses single-character atoms.  A
        # plain-DPLL baseline expands every discrete time alternative, which can exceed
        # the 26 Latin lowercase symbols; use additional Unicode identifier
        # characters rather than silently applying estimate-based pruning.
        symbol_pool = (
            "abcdefghijklmnopqrstuvwxyz"
            "αβγδεζηθικλμνξοπρστυφχψω"
            "абвгдежзийклмнопрстуфхцчшщъыьэюя"
        )
        available = [character for character in symbol_pool if character not in used]
        synthetic_nodes = []
        selector_debug = {}
        linking_clauses = []
        once_parent_nodes = []

        for group, plan in once_time_plans.items():
            parent = plan.get("parent")
            members = tuple(plan.get("children", ()))
            candidates = tuple(plan.get("candidates", ()))
            if parent is None or not members:
                continue
            if len(available) < len(candidates):
                raise RuntimeError(
                    "SAT once-time expansion exhausted the single-character "
                    "DPLL symbol pool."
                )

            selectors = []
            selector_nodes = []
            for window in candidates:
                if len(available) < 1 + len(members):
                    raise RuntimeError(
                        "SAT once-time expansion exhausted the current "
                        "single-character DPLL symbol pool."
                    )
                variable = available.pop(0)
                selector = PropositionNode(
                    name=f"vp_once_time[{window[1]}]({group})",
                    alphabet=variable,
                    source_rule=members[0].source_rule,
                    children=[],
                    # A positive selector is a repair choice; negative is the
                    # monitored/default state for this synthetic proposition.
                    ttv_value=-1.0,
                    ttv_h_min=-1.0,
                )
                selector.vp_once_group = group
                selector.vp_once_interval = tuple(window)
                selector.vp_once_time_auxiliary = True
                synthetic_nodes.append(selector)
                selectors.append(variable)
                selector_nodes.append(selector)

                obligation_variables = []
                obligations = []
                for member in members:
                    obligation_variable = available.pop(0)
                    obligation = PropositionNode(
                        name=(
                            f"vp_once_obligation[{window[1]}]("
                            f"{member.name})"
                        ),
                        alphabet=obligation_variable,
                        source_rule=member.source_rule,
                        children=list(getattr(member, "children", ()) or ()),
                        ttv_value=member.ttv_value,
                        ttv_h_min=member.ttv_h_min,
                    )
                    obligation.vp_once_group = group
                    obligation.vp_once_interval = tuple(window)
                    obligation.vp_once_obligation = True
                    obligation.vp_once_selector = variable
                    obligation.vp_once_source = member
                    synthetic_nodes.append(obligation)
                    obligations.append(obligation)
                    obligation_variables.append(obligation_variable)
                    linking_clauses.append(
                        sp.Or(
                            sp.Not(sp.Symbol(variable)),
                            sp.Symbol(obligation_variable),
                        )
                    )
                selector.vp_once_obligations = tuple(obligations)
                # Complete the Tseitin equivalence
                # z_w <-> (q_w_1 & ... & q_w_n).  The clauses above encode
                # z_w -> q_w_i; this one encodes the reverse implication.
                linking_clauses.append(
                    sp.Or(
                        sp.Symbol(variable),
                        *(
                            sp.Not(sp.Symbol(obligation_variable))
                            for obligation_variable in obligation_variables
                        ),
                    )
                )
            parent_variable = str(parent.alphabet).lstrip("~")
            parent_symbol = sp.Symbol(parent_variable)
            # p -> OR(z_w).  With no dynamically possible end time this reduces
            # to ~p, which is the exact finite-trace result.
            linking_clauses.append(
                sp.Or(
                    sp.Not(parent_symbol),
                    *(sp.Symbol(variable) for variable in selectors),
                )
            )
            # Each selected end time implies the original existential proposition.
            for variable in selectors:
                linking_clauses.append(
                    sp.Or(parent_symbol, sp.Not(sp.Symbol(variable)))
                )
            parent.vp_once_parent_auxiliary = True
            once_parent_nodes.append(parent)
            selector_debug[group] = tuple(
                (
                    selector.alphabet,
                    tuple(selector.vp_once_interval),
                    tuple(
                        obligation.alphabet
                        for obligation in selector.vp_once_obligations
                    ),
                )
                for selector in selector_nodes
            )

            # Selecting several end times is logically unnecessary and would
            # ask VP to enforce several distinct historical windows at once.
            for left_index, left in enumerate(selectors):
                for right in selectors[left_index + 1 :]:
                    linking_clauses.append(
                        sp.Or(sp.Not(sp.Symbol(left)), sp.Not(sp.Symbol(right)))
                    )

        if not selector_debug:
            return {}
        # ``self._formula`` is already CNF.  Every definitional item above is
        # one CNF clause, hence their conjunction is CNF without ``to_cnf``.
        base_expression = sp.sympify(stl2sympy(self._formula))
        expanded_expression = sp.And(base_expression, *linking_clauses)
        if not sp.logic.boolalg.is_cnf(expanded_expression):
            raise RuntimeError("Local temporal Tseitin encoding is not CNF.")
        parent_variables = {
            str(parent.alphabet).lstrip("~") for parent in once_parent_nodes
        }
        self._prop_nodes = list(self._prop_nodes) + synthetic_nodes
        self._dpll_solver._prop_nodes = self._prop_nodes
        for variable in parent_variables:
            self._domain_dict.pop(variable, None)
            self._hard_domain_vars.discard(variable)
        self._repair_literals = [
            literal
            for literal in self._repair_literals
            if literal.lstrip("~") not in parent_variables
        ]
        self._formula = str(expanded_expression)
        self._once_time_selectors = selector_debug
        self._expanded_decomposition_groups.update(selector_debug)
        # In exact once-time mode each positive selector denotes an immediately
        # executable VP candidate.  Prefer these candidates to arbitrary
        # completions of the surrounding Boolean formula.
        self._repair_literals = list(
            dict.fromkeys(
                list(self._repair_literals)
                + [
                    selector
                    for selectors in selector_debug.values()
                    for selector, _window, _obligations in selectors
                ]
            )
        )
        return selector_debug

    def sort_domain_repair_literals(self):
        """Order DomainDPLL guidance using executable polarity first."""
        if self._solver_mode != "domain_dpll":
            return
        # Reuse the plain DPLL's established repairability/robustness ordering
        # as a stable priority over DomainDPLL's admissible repair literals.
        # This changes search order only, never CNF or domains.  Match by
        # variable because domain guidance may request the opposite polarity.
        plain_order = self._dpll_solver.get_literal(
            self._dpll_solver._assign_cnf(self._formula),
            self._prop_nodes,
            self._dpll_solver._tv_time_step,
        )
        rank = {
            literal.lstrip("~"): index
            for index, literal in enumerate(plain_order)
        }
        original_order = {
            literal: index
            for index, literal in enumerate(self._repair_literals)
        }
        node_by_variable = {
            str(prop.alphabet).lstrip("~"): prop
            for prop in self._prop_nodes
        }

        def already_has_requested_polarity(literal):
            variable = literal.lstrip("~")
            node = node_by_variable.get(variable)
            if node is None:
                return 1
            requested = not literal.startswith("~")
            current = float(node.ttv_value) >= 0.0
            return int(requested == current)

        self._repair_literals.sort(
            key=lambda literal: (
                already_has_requested_polarity(literal),
                rank.get(literal.lstrip("~"), len(rank)),
                original_order[literal],
            )
        )

    def add_hard_false_units(self, variables):
        """Conjoin certified-false auxiliary variables as root CNF units."""
        variables = tuple(sorted(set(variables)))
        if not variables:
            return
        # ``self._formula`` was already checked as CNF when the local
        # Tseitin clauses were installed.  Conjoining unit clauses preserves
        # CNF by construction, so reparsing and serializing the whole formula
        # through SymPy here only adds avoidable preprocessing time.
        units = " & ".join(f"~{variable}" for variable in variables)
        self._formula = f"({self._formula}) & {units}"

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        if self._solver_mode == "domain_dpll":
            if (
                self._dpll_solver.domains != self._domain_dict
                or self._dpll_solver.hard_domain_vars != self._hard_domain_vars
                or self._dpll_solver.repair_literals != self._repair_literals
            ):
                self._dpll_solver.set_search_guidance(
                    self._domain_dict,
                    self._hard_domain_vars,
                    self._repair_literals,
                )
            if self._solver_formula != self._formula:
                self._dpll_solver.update_cnf(self._formula)
                self._solver_formula = self._formula
        else:
            if self._solver_formula != self._formula:
                self._dpll_solver.update_cnf(self._formula)
                self._solver_formula = self._formula
        sat_result = self._dpll_solver.solve()
        return sat_result

    def model(self) -> (list, str):
        """
        return a satisfiable proposition - based on robustness
        """
        self._dpll_model = list(self._dpll_solver.model)
        # DomainDPLL intentionally returns a partial model and can omit unit-
        # propagated literals.  For an active once-time choice those implied
        # literals are precisely the theory obligations which VP must enforce,
        # so materialize them explicitly.  Inactive time-local obligations are
        # removed; q <-> z in the expanded CNF fixes them to false anyway.
        active_once_selectors = {
            literal
            for literal in self._dpll_model
            if not literal.startswith("~")
            and any(
                getattr(prop, "vp_once_time_auxiliary", False)
                and prop.alphabet[-1] == literal[-1]
                for prop in self._prop_nodes
            )
        }
        self._dpll_model = [
            literal
            for literal in self._dpll_model
            if not any(
                getattr(prop, "vp_once_obligation", False)
                and prop.alphabet[-1] == literal[-1]
                for prop in self._prop_nodes
            )
        ]
        assigned_once_obligations = set()
        for prop in self._prop_nodes:
            if (
                getattr(prop, "vp_once_obligation", False)
                and getattr(prop, "vp_once_selector", None)
                in active_once_selectors
            ):
                variable = prop.alphabet[-1]
                if variable not in assigned_once_obligations:
                    self._dpll_model.append(variable)
                    assigned_once_obligations.add(variable)
        # A DPLL model is intentionally partial: once one literal satisfies a
        # clause, unrelated variables may be absent.  For a proposition which
        # VP split into independently executable children, however, blocking
        # that partial model would also block a stronger candidate that keeps
        # the same literals and activates the remaining child constraints.
        # Complete only decomposition children with their monitored truth
        # value.  This makes failed-candidate blocking exact in the new
        # abstraction dimensions and leaves every ordinary rule unchanged.
        assigned_variables = {literal[-1] for literal in self._dpll_model}
        for prop_node in self._prop_nodes:
            if (
                getattr(prop_node, "vp_decomposition_group", None) is None
                or prop_node.vp_decomposition_group
                in self._expanded_decomposition_groups
                or prop_node.alphabet[-1] in assigned_variables
            ):
                continue
            variable = prop_node.alphabet[-1]
            domain = getattr(self._dpll_solver, "domains", {}).get(variable)
            if domain is not None and len(domain) == 1:
                desired_positive = bool(next(iter(domain)))
            else:
                desired_positive = float(prop_node.ttv_value) > 0.0
            self._dpll_model.append(
                variable if desired_positive else f"~{variable}"
            )
            assigned_variables.add(variable)
        prop_list = list()
        for m in list(self._dpll_model):
            sel_prop_node = next(
                (
                    prop_node
                    for prop_node in self._prop_nodes
                    if prop_node.alphabet[-1] == m[-1]
                ),
                None,
            )
            if sel_prop_node:
                sel_prop_node.alphabet = m  # Assign m to the alphabet attribute
            prop_list.append(sel_prop_node)
        print("* \t<SATSolver>: model is {}".format(self._dpll_model))
        return prop_list, self._dpll_model

    def update_formula(self, blocking_literals=None):
        """
        Based on the syntax for sympy, the SAT formula is updated by negating the unsatisfiable abstraction:
        phi_SAT = phi_SAT and (not abs)
        """
        if self._formula[0] != "(":
            self._formula = "(" + self.formula + ")"
        # Block exactly the failed SAT model.  DomainDPLL may then relax only
        # the domain variables which occur in this model, leaving all other
        # domain restrictions active for the next search.
        model = list(
            self._dpll_model if blocking_literals is None else blocking_literals
        )
        if not model:
            return
        def negate_literal(literal):
            return literal[1:] if literal.startswith("~") else "~" + literal

        counter_ex = negate_literal(model[0])
        if len(model) > 1:
            counter_ex = "(" + counter_ex
            for atom in model[1:]:
                counter_ex += " | " + negate_literal(atom)
            counter_ex += ")"
        self._formula += " & " + counter_ex
        print("* \t<SATSolver>: the formula is updated to {}".format(self._formula))
