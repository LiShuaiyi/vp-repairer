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

    def add_temporal_witness_selectors(self, witness_plans):
        """Encode finite existential witness alternatives in the SAT CNF.

        A selector is a Boolean-only witness choice.  Each member of the
        selected temporal conjunction gets its own theory atom, so no atom
        handed to VP contains a hidden Boolean conjunction.  For witness ``w``
        the encoding is ``z_w -> (q_w_1 & ... & q_w_n)``; the original parent
        conjunction is replaced by ``z_1 | ... | z_k``.  Pairwise exclusion
        keeps one common witness active at a time.
        """
        used = {
            str(node.alphabet).lstrip("~")
            for node in self._prop_nodes
        }
        # The in-tree DPLL representation uses single-character atoms.  A
        # plain-DPLL baseline expands every temporal witness, which can exceed
        # the 26 Latin lowercase symbols; use additional Unicode identifier
        # characters rather than silently applying estimate-based pruning.
        symbol_pool = (
            "abcdefghijklmnopqrstuvwxyz"
            "αβγδεζηθικλμνξοπρστυφχψω"
            "абвгдежзийклмнопрстуфхцчшщъыьэюя"
        )
        available = [character for character in symbol_pool if character not in used]
        synthetic_nodes = []
        replacements = {}
        selector_debug = {}
        linking_clauses = []

        for group, plan in witness_plans.items():
            members = tuple(plan.get("children", ()))
            member_vars = tuple(
                str(member.alphabet).lstrip("~") for member in members
            )
            candidates = tuple(plan.get("candidates", ()))
            if not members:
                continue
            if len(available) < len(candidates):
                raise RuntimeError(
                    "SAT witness expansion exhausted the single-character "
                    "DPLL symbol pool."
                )

            selectors = []
            selector_nodes = []
            for window in candidates:
                if len(available) < 1 + len(members):
                    raise RuntimeError(
                        "SAT witness expansion exhausted the current "
                        "single-character DPLL symbol pool."
                    )
                variable = available.pop(0)
                selector = PropositionNode(
                    name=f"vp_witness[{window[1]}]({group})",
                    alphabet=variable,
                    source_rule=members[0].source_rule,
                    children=[],
                    # A positive selector is a repair choice; negative is the
                    # monitored/default state for this synthetic proposition.
                    ttv_value=-1.0,
                    ttv_h_min=-1.0,
                )
                selector.vp_witness_group = group
                selector.vp_witness_window = tuple(window)
                selector.vp_witness_auxiliary = True
                synthetic_nodes.append(selector)
                selectors.append(variable)
                selector_nodes.append(selector)

                obligation_variables = []
                obligations = []
                for member in members:
                    obligation_variable = available.pop(0)
                    obligation = PropositionNode(
                        name=(
                            f"vp_witness_leaf[{window[1]}]("
                            f"{member.name})"
                        ),
                        alphabet=obligation_variable,
                        source_rule=member.source_rule,
                        children=list(getattr(member, "children", ()) or ()),
                        ttv_value=member.ttv_value,
                        ttv_h_min=member.ttv_h_min,
                    )
                    obligation.vp_witness_group = group
                    obligation.vp_witness_window = tuple(window)
                    obligation.vp_witness_obligation = True
                    obligation.vp_witness_selector = variable
                    obligation.vp_witness_source = member
                    synthetic_nodes.append(obligation)
                    obligations.append(obligation)
                    obligation_variables.append(obligation_variable)
                    linking_clauses.append(
                        sp.Or(
                            sp.Not(sp.Symbol(variable)),
                            sp.Symbol(obligation_variable),
                        )
                    )
                    # The obligation is a witness-local theory atom.  Making
                    # the gate bidirectional gives it a canonical false value
                    # whenever its witness is inactive, instead of leaving an
                    # irrelevant Boolean degree of freedom in failed models.
                    linking_clauses.append(
                        sp.Or(
                            sp.Not(sp.Symbol(obligation_variable)),
                            sp.Symbol(variable),
                        )
                    )
                selector.vp_witness_obligations = tuple(obligations)
            parent_conjunction = sp.And(
                *(sp.Symbol(variable) for variable in member_vars)
            )
            witness_disjunction = (
                sp.Or(*(sp.Symbol(variable) for variable in selectors))
                if selectors
                else sp.false
            )
            replacements[parent_conjunction] = witness_disjunction
            selector_debug[group] = tuple(
                (
                    selector.alphabet,
                    tuple(selector.vp_witness_window),
                    tuple(
                        obligation.alphabet
                        for obligation in selector.vp_witness_obligations
                    ),
                )
                for selector in selector_nodes
            )

            # Selecting several witnesses is logically unnecessary and would
            # ask VP to enforce several distinct historical windows at once.
            for left_index, left in enumerate(selectors):
                for right in selectors[left_index + 1 :]:
                    linking_clauses.append(
                        sp.Or(sp.Not(sp.Symbol(left)), sp.Not(sp.Symbol(right)))
                    )

        if not replacements:
            return {}
        source_expression = (
            sp.sympify(stl2sympy(self._source_formula))
            if isinstance(self._source_formula, str)
            else sp.sympify(self._source_formula)
        )
        expanded_expression = source_expression.xreplace(replacements)
        if expanded_expression == source_expression:
            missing = ", ".join(str(item) for item in replacements)
            raise RuntimeError(
                "Could not locate decomposed temporal conjunction(s) in the "
                f"SAT source formula: {missing}."
            )
        if linking_clauses:
            expanded_expression = sp.And(
                expanded_expression, *linking_clauses
            )
        self._prop_nodes = list(self._prop_nodes) + synthetic_nodes
        self._dpll_solver._prop_nodes = self._prop_nodes
        self._formula = str(sp.to_cnf(expanded_expression))
        self._temporal_witness_selectors = selector_debug
        self._expanded_decomposition_groups.update(selector_debug)
        # In exact witness mode each positive selector denotes an immediately
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
        # propagated literals.  For an active temporal witness those implied
        # literals are precisely the theory obligations which VP must enforce,
        # so materialize them explicitly.  Inactive witness obligations are
        # removed; q <-> z in the expanded CNF fixes them to false anyway.
        active_witness_selectors = {
            literal
            for literal in self._dpll_model
            if not literal.startswith("~")
            and any(
                getattr(prop, "vp_witness_auxiliary", False)
                and prop.alphabet[-1] == literal[-1]
                for prop in self._prop_nodes
            )
        }
        self._dpll_model = [
            literal
            for literal in self._dpll_model
            if not any(
                getattr(prop, "vp_witness_obligation", False)
                and prop.alphabet[-1] == literal[-1]
                for prop in self._prop_nodes
            )
        ]
        assigned_witness_obligations = set()
        for prop in self._prop_nodes:
            if (
                getattr(prop, "vp_witness_obligation", False)
                and getattr(prop, "vp_witness_selector", None)
                in active_witness_selectors
            ):
                variable = prop.alphabet[-1]
                if variable not in assigned_witness_obligations:
                    self._dpll_model.append(variable)
                    assigned_witness_obligations.add(variable)
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
