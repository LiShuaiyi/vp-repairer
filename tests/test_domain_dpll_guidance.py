from z3 import sat, unsat

from crrepairer.smt.sat_solver.dpll_domain import DomainDPLL


def test_repair_action_clause_forces_repair_literal_into_model():
    solver = DomainDPLL(
        "a | b",
        domains={"a": {0}},
        hard_domain_vars={"a"},
        repair_literals=["b"],
    )

    assert solver.solve() == sat
    assert "~a" in solver.model
    assert "b" in solver.model


def test_hard_domain_is_not_relaxed_after_failed_model():
    solver = DomainDPLL(
        "a | b",
        domains={"a": {0}},
        hard_domain_vars={"a"},
        repair_literals=["b"],
    )
    assert solver.solve() == sat

    assert solver.relax_domains_for_model(solver.model) == []
    assert solver.domains == {"a": {0}}


def test_conflicting_hard_domain_and_repair_action_is_unsat():
    solver = DomainDPLL(
        "a",
        domains={"a": {0}},
        hard_domain_vars={"a"},
        repair_literals=["a"],
    )

    assert solver.solve() == unsat
