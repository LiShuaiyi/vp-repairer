import math

import numpy as np
import pytest
from stlpy.STL import LinearPredicate
from stlpy.systems.linear import LinearSystem

from micp_comparison.fewer_binary_solver import FewerBinaryGurobiSolver
from micp_comparison.rules import l1_speed_limit_formula


def predicate(index, sign=1.0, bound=0.0):
    a = np.zeros((1, 2))
    a[0, index] = sign
    return LinearPredicate(a, bound)


def static_system():
    return LinearSystem(np.eye(2), np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))


def make_solver(spec):
    try:
        return FewerBinaryGurobiSolver(
            spec, static_system(), np.ones(2), 0,
            robustness_cost=False, verbose=False,
        )
    except Exception as exc:
        if "license" in str(exc).lower():
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise


def test_conjunction_uses_no_binary_variables():
    solver = make_solver(predicate(0) & predicate(1))
    solver.model.update()
    assert solver.model.NumBinVars == 0


def test_flat_disjunction_uses_logarithmic_binary_variables():
    # Python's | operator creates a nested tree; the encoder flattens it to
    # four alternatives plus the inactive parent lambda.
    spec = predicate(0) | predicate(1) | predicate(0, -1) | predicate(1, -1)
    solver = make_solver(spec)
    solver.model.update()
    assert solver.model.NumBinVars == math.ceil(math.log2(5))


def test_disjunction_solution_is_sound():
    spec = predicate(0, 1, 2) | predicate(1, 1, 2)
    solver = make_solver(spec)
    solver.model.addConstr(solver.rho[0] >= 0.01)
    x, _, _, _ = solver.Solve()
    assert x is None  # fixed x0=[1,1] satisfies neither branch


def test_polygonal_speed_limit_is_inside_monitor_circle():
    vmax = 60.0
    formula = l1_speed_limit_formula(vmax, 2, 3, 10)
    for vs, vd in ((0, 0), (10, -3), (20, 5), (vmax, 0)):
        y = np.zeros((10, 1))
        y[2, 0], y[3, 0] = vs, vd
        assert formula.robustness(y, 0)[0] >= -1e-9
        assert math.hypot(vs, vd) <= vmax + 1e-9
