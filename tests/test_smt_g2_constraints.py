import numpy as np

from crrepairer.smt.t_solver.rule_constraints import RuleConstraintsManual


def test_abrupt_braking_constraint_respects_sat_assignment_and_strict_margin():
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)

    abrupt = constraints.ConstrAbruptBreaking(-2.0, prop_assignment=1.0)
    not_abrupt = constraints.ConstrAbruptBreaking(-2.0, prop_assignment=-1.0)

    assert abrupt[0] == -np.inf
    assert abrupt[1] < -2.0
    assert not_abrupt[0] > -2.0
    assert not_abrupt[1] == np.inf


def test_legacy_not_abrupt_wrapper_uses_strict_margin():
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)

    lower, upper = constraints.ConstrAccNotAbruptly(-2.0)

    assert lower > -2.0
    assert upper == np.inf
