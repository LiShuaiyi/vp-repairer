import math

import numpy as np
import pytest
from stlpy.STL import LinearPredicate
from stlpy.systems.linear import LinearSystem

from stlccp_comparison.decomposition import decompose_formula, reversed_robustness
from stlccp_comparison.solver import STLCCPSolver, smooth_min_and_gradient


def predicate(coefficient, bound=0.0):
    return LinearPredicate(np.asarray([[coefficient]], dtype=float), bound)


def test_conjunction_is_fully_epigraphic():
    formula = predicate(1.0, 1.0) & predicate(-1.0, -2.0)
    result = decompose_formula(formula, 1)
    assert result.num_auxiliary == 1
    assert len(result.convex_bounds) == 2
    assert len(result.disjunctive_bounds) == 0


def test_disjunction_keeps_one_ccp_constraint_and_weights_full_subtree():
    formula = (predicate(1.0) & predicate(-1.0)) | predicate(1.0, 2.0)
    result = decompose_formula(formula, 1)
    assert result.num_auxiliary == 2  # root and the conjunctive branch
    assert len(result.convex_bounds) == 2
    assert len(result.disjunctive_bounds) == 1
    assert result.disjunctive_bounds[0].leaf_count == 3


def test_temporal_offsets_and_exact_reversed_robustness():
    formula = predicate(1.0, 1.0).eventually(0, 1)
    y = np.asarray([[0.0, 3.0]])
    assert reversed_robustness(formula, y) == pytest.approx(-2.0)
    result = decompose_formula(formula, 2)
    times = sorted(child.time for child in result.disjunctive_bounds[0].children)
    assert times == [0, 1]


def test_mellowmin_is_sound_over_approximation_of_minimum():
    values = np.asarray([-2.0, 1.0, 4.0])
    lse, lse_gradient = smooth_min_and_gradient(values, "lse", 10.0)
    mellow, mellow_gradient = smooth_min_and_gradient(values, "mellowmin", 10.0)
    assert lse <= values.min()
    assert mellow >= values.min()
    assert mellow - values.min() <= math.log(len(values)) / 10.0 + 1e-12
    assert lse_gradient.sum() == pytest.approx(1.0)
    assert mellow_gradient.sum() == pytest.approx(1.0)


def test_smooth_min_tangent_is_global_upper_bound():
    center = np.asarray([-1.2, 0.3, 2.0])
    sample = np.asarray([0.2, -0.5, 1.1])
    value, gradient = smooth_min_and_gradient(center, "mellowmin", 4.0)
    sample_value, _ = smooth_min_and_gradient(sample, "mellowmin", 4.0)
    tangent = value + gradient @ (sample - center)
    assert tangent >= sample_value - 1e-12


def test_exact_auxiliary_values_satisfy_decomposed_tree():
    p1, p2, p3 = predicate(1.0), predicate(-1.0), predicate(1.0, 0.5)
    formula = ((p1 | p2) & p3) | (p1 & (p2 | p3))
    y = np.asarray([[0.25]])
    decomposition = decompose_formula(formula, 1)
    auxiliary = decomposition.initialize_auxiliary(y)
    for bound in decomposition.convex_bounds:
        assert bound.child.evaluate(y, auxiliary) <= auxiliary[bound.parent] + 1e-12
    for bound in decomposition.disjunctive_bounds:
        child_values = [child.evaluate(y, auxiliary) for child in bound.children]
        assert min(child_values) <= auxiliary[bound.parent] + 1e-12


def test_small_convex_formula_solves_without_binary_variables():
    formula = predicate(1.0, 0.5)
    system = LinearSystem(
        np.eye(1), np.zeros((1, 1)), np.eye(1), np.zeros((1, 1))
    )
    decomposition = decompose_formula(formula, 1)
    solver = STLCCPSolver(
        formula,
        decomposition,
        system,
        np.asarray([1.0]),
        1,
        Q=np.zeros((1, 1)),
        R=np.eye(1),
        control_min=np.asarray([0.0]),
        control_max=np.asarray([0.0]),
        robustness_margin=0.01,
    )
    try:
        result = solver.solve(phases=("mellowmin",), max_iterations=3)
    except Exception as exc:
        if "license" in str(exc).lower():
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise
    assert result.x is not None
    assert result.robustness == pytest.approx(0.5, abs=1e-6)
    assert result.max_slack == 0.0


def test_disjunctive_formula_runs_lse_to_mellowmin_ccp():
    formula = predicate(1.0, 1.0) | predicate(-1.0, 1.0)
    system = LinearSystem(
        np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)), np.eye(1)
    )
    decomposition = decompose_formula(formula, 1)
    solver = STLCCPSolver(
        formula,
        decomposition,
        system,
        np.asarray([0.0]),
        1,
        Q=np.zeros((1, 1)),
        R=0.1 * np.eye(1),
        control_min=np.asarray([-2.0]),
        control_max=np.asarray([2.0]),
        robustness_margin=0.01,
    )
    try:
        result = solver.solve(
            phases=("lse", "mellowmin"), max_iterations=15,
            initial_u=np.asarray([[1.5]]),
        )
    except Exception as exc:
        if "license" in str(exc).lower():
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise
    assert result.x is not None
    assert result.robustness >= 0.01 - 1e-7
    assert result.max_slack <= 1e-5 + 1e-7
    assert {entry["phase"] for entry in result.phase_history} == {
        "lse", "mellowmin"
    }
