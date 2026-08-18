from types import SimpleNamespace

from z3 import sat, unsat

from crrepairer.repairer.vp.domain import VPPredicateEstimation
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


def test_rg_cut_in_is_hard_fixed_to_current_truth_value():
    estimator = object.__new__(VPPredicateEstimation)
    estimator._hard_domain_vars = set()
    cut_in = SimpleNamespace(name="once[0,30](cut_in__1_0)", alphabet="c", ttv_value=-0.4)
    safe_distance = SimpleNamespace(
        name="keeps_safe_distance_prec__0_1",
        alphabet="d",
        ttv_value=-1.0,
    )
    domains = {"c": {0, 1}, "d": {0, 1}}

    fixed_count = estimator._fix_uncontrollable_rg_predicates(
        domains,
        [cut_in, safe_distance],
    )

    assert fixed_count == 1
    assert domains == {"c": {0}, "d": {0, 1}}
    assert estimator._hard_domain_vars == {"c"}
