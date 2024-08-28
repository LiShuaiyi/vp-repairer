import sympy as sp

from crrepairer.smt.sat_solver.dpll import DPLL
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration


class SATSolver:
    def __init__(self, rule_monitor: STLRuleMonitor, config: RepairerConfiguration):
        self._formula = self.construct_nnf(rule_monitor.sat_formula)
        self._formula = self.construct_cnf(self._formula)
        self._prop_nodes = rule_monitor.proposition_nodes
        self._prop_robust_all = rule_monitor.rob_abstraction
        self._init_assign = list()
        self._dpll_solver = DPLL(
            self._formula, self._prop_nodes, rule_monitor.tv_time_step
        )
        self._dpll_model = None

        self._config = config

    @property
    def formula(self):
        return self._formula

    @property
    def initial_assignment(self):
        return self._init_assign

    @staticmethod
    def stl2sympy(input_formula: str):
        return (
            input_formula.replace("and", "&")
            .replace("or", "|")
            .replace("!", "~")
            .replace("implies", ">>")
        )

    @staticmethod
    def construct_cnf(stl_formula):
        """
        Construct Conjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
        """
        if isinstance(stl_formula, str):
            sp_formula = SATSolver.stl2sympy(stl_formula)
        else:
            sp_formula = stl_formula
        cnf_formula = str(sp.to_cnf(sp_formula))
        return cnf_formula

    @staticmethod
    def construct_dnf(stl_formula):
        """
        Construct Disjunctive Normal Form (DNF) using sympy - first needs to convert the formula to sp's interface.
        """
        if isinstance(stl_formula, str):
            sp_formula = SATSolver.stl2sympy(stl_formula)
        else:
            sp_formula = stl_formula
        dnf_formula = str(sp.to_dnf(sp_formula))
        return dnf_formula

    @staticmethod
    def construct_nnf(stl_formula):
        """
        Construct Negation Normal Form (NNF) using sympy - first needs to convert the formula to sp's interface.
        """
        sp_formula = SATSolver.stl2sympy(stl_formula)
        nnf_formula = sp.to_nnf(sp.simplify(sp_formula))
        return nnf_formula

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        self._dpll_solver.update_cnf(self._formula)
        sat_result = self._dpll_solver.solve()
        return sat_result

    def model(self) -> (list, str):
        """
        return a satisfiable proposition - based on robustness
        """
        self._dpll_model = self._dpll_solver.model
        prop_list = list()
        for m in list(self._dpll_model):
            sel_prop_node = next(
                (
                    prop_node
                    for prop_node in self._prop_nodes
                    if prop_node.alphabet == m[-1]
                ),
                None,
            )
            if sel_prop_node:
                sel_prop_node.alphabet = m  # Assign m to the alphabet attribute
            prop_list.append(sel_prop_node)
        print("* \t<SATSolver>: model is {}".format(self._dpll_model))
        return prop_list, self._dpll_model

    def update_formula(self):
        """
        Based on the syntax for sympy, the SAT formula is updated by negating the unsatisfiable abstraction:
        phi_SAT = phi_SAT and (not abs)
        """
        if self._formula[0] is not "(":
            self._formula = "(" + self.formula + ")"
        # generate counterexample
        counter_ex = "~" + list(self._dpll_model)[0]
        if len(list(self._dpll_model)) > 1:
            counter_ex = "(" + counter_ex
            for atom in list(self._dpll_model)[1:]:
                counter_ex += " | ~" + atom
            counter_ex += ")"
        self._formula += " & " + counter_ex
        print("* \t<SATSolver>: the formula is updated to {}".format(self._formula))
