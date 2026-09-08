import numpy as np

from fixed_path_micp_comparison.traffic_rule_fixed_path import (
    _and_at_each_step,
    _outside_rectangle_on_fixed_path,
)
from comparison.micp.vehicle_models_dt import VehicleModel
from fixed_path_micp_comparison.vehicle_models_fixed_path import FixedPathVehicleModel
from stlpy.STL import LinearPredicate


def test_fixed_path_dynamics_are_longitudinal_block_of_legacy_model():
    full = VehicleModel(0.2)
    fixed = FixedPathVehicleModel(0.2)
    longitudinal_indices = np.array([0, 2, 4, 6])

    np.testing.assert_allclose(
        fixed.A, full.A[np.ix_(longitudinal_indices, longitudinal_indices)]
    )
    np.testing.assert_allclose(fixed.B[:, 0], full.B[longitudinal_indices, 0])
    assert (fixed.n, fixed.m, fixed.p) == (4, 1, 5)


def test_outside_rectangle_projection_eliminates_lateral_true_case():
    assert _outside_rectangle_on_fixed_path((1.0, 2.0, 1.0, 2.0), 0.0) is None


def test_outside_rectangle_projection_keeps_longitudinal_disjunction():
    projected = _outside_rectangle_on_fixed_path((1.0, 2.0, -1.0, 1.0), 0.0)
    signal = np.zeros((5, 3))
    signal[0, :] = [0.0, 1.5, 3.0]

    assert projected.robustness(signal, 0)[0] >= 0.0
    assert projected.robustness(signal, 1)[0] < 0.0
    assert projected.robustness(signal, 2)[0] >= 0.0


def test_per_step_tree_uses_only_valid_sample_indices():
    predicate = LinearPredicate(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]), 0.0)
    tree = _and_at_each_step([predicate] * 4, 4)
    assert tree.timesteps == [0, 1, 2, 3]
