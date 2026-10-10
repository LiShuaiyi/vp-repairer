from types import SimpleNamespace

import numpy as np

from crrepairer.repairer.vp.constraints import VPConstraintExtraction
from crrepairer.repairer.vp.domain import VPPredicateEstimation
from crrepairer.repairer.vp.optimization import VPOptimization
from crrepairer.repairer.vp.predicate_piecewise import (
    FALSE_DOMAIN,
    TRUE_DOMAIN,
    UNKNOWN_DOMAIN,
)
from crrepairer.repairer.vp.semantic_predicate_regions import (
    SemanticINPredicateRegionBuilder,
    estimate_acceleration_predicate_frame,
)
from crrepairer.repairer.vp.trajectory_context import VPTrajectoryContext


class _Evaluator:
    def __init__(self, name, arity, threshold=-2.0):
        self.predicate_name = name
        self.arity = arity
        self.config = {"a_abrupt": threshold}


class _World:
    def __init__(self, accelerations):
        self._vehicles = {
            vehicle_id: SimpleNamespace(
                states_cr={
                    step: SimpleNamespace(acceleration=value)
                    for step, value in by_step.items()
                }
            )
            for vehicle_id, by_step in accelerations.items()
        }

    def vehicle_by_id(self, vehicle_id):
        return self._vehicles[vehicle_id]


def _estimate(name, ids, reachable, other=-1.0, threshold=-2.0):
    return estimate_acceleration_predicate_frame(
        _Evaluator(name, len(ids), threshold),
        1,
        ids,
        ego_id=10,
        world=_World({20: {1: other}}),
        reachable_acceleration_by_time={1: reachable},
    ).domain


def test_acceleration_rollout_obeys_jerk_and_saturates():
    lower, upper = VPConstraintExtraction._rollout_reachable_acceleration_intervals(
        0.0,
        4,
        dt=0.5,
        amin=-2.0,
        amax=1.0,
        jmin=-2.0,
        jmax=1.0,
    )
    assert np.allclose(lower, [-1.0, -2.0, -2.0, -2.0])
    assert np.allclose(upper, [0.5, 1.0, 1.0, 1.0])


def test_unary_abrupt_braking_true_false_crossing_and_zero():
    assert _estimate("brakes_abruptly", (10,), (-4.0, -3.0)) == TRUE_DOMAIN
    assert _estimate("brakes_abruptly", (10,), (-1.0, 0.0)) == FALSE_DOMAIN
    assert _estimate("brakes_abruptly", (10,), (-3.0, -1.0)) == UNKNOWN_DOMAIN
    assert _estimate("brakes_abruptly", (10,), (-2.0, -2.0)) == UNKNOWN_DOMAIN


def test_relative_abrupt_braking_handles_both_ego_argument_positions():
    # -a_ego + a_other + threshold, with a_other=-1 and threshold=-2.
    assert _estimate(
        "rel_brakes_abruptly", (10, 20), (-5.0, -4.0)
    ) == TRUE_DOMAIN
    assert _estimate(
        "rel_brakes_abruptly", (10, 20), (-2.0, -1.0)
    ) == FALSE_DOMAIN
    # -a_other + a_ego + threshold for the reversed proposition.
    assert _estimate(
        "rel_brakes_abruptly", (20, 10), (0.0, 1.0), other=-4.0
    ) == TRUE_DOMAIN
    assert _estimate(
        "rel_brakes_abruptly", (20, 10), (-4.0, -3.0), other=-1.0
    ) == FALSE_DOMAIN


def test_relative_abrupt_braking_missing_or_nonfinite_target_is_unknown():
    evaluator = _Evaluator("rel_brakes_abruptly", 2)
    missing = estimate_acceleration_predicate_frame(
        evaluator,
        1,
        (10, 20),
        ego_id=10,
        world=_World({20: {}}),
        reachable_acceleration_by_time={1: (-3.0, -2.0)},
    )
    nonfinite = _estimate(
        "rel_brakes_abruptly", (10, 20), (-3.0, -2.0), other=np.nan
    )
    assert missing.domain == UNKNOWN_DOMAIN
    assert nonfinite == UNKNOWN_DOMAIN


def test_missing_acceleration_reachability_is_unknown():
    result = estimate_acceleration_predicate_frame(
        _Evaluator("brakes_abruptly", 1),
        1,
        (10,),
        ego_id=10,
        world=_World({}),
        reachable_acceleration_by_time={},
    )
    assert result.domain == UNKNOWN_DOMAIN


def test_acceleration_predicates_are_supported_by_semantic_builder():
    assert SemanticINPredicateRegionBuilder.supports_evaluator(
        _Evaluator("brakes_abruptly", 1)
    )
    assert SemanticINPredicateRegionBuilder.supports_evaluator(
        _Evaluator("rel_brakes_abruptly", 2)
    )


def test_abrupt_braking_constraint_path_remains_present():
    extractor = object.__new__(VPConstraintExtraction)
    qp_config = SimpleNamespace(a_lon_min=-6.0)
    extractor.config = SimpleNamespace(
        vehicle=SimpleNamespace(qp_veh_config=qp_config)
    )
    predicate = SimpleNamespace(
        base_name="brakes_abruptly",
        evaluator=_Evaluator("brakes_abruptly", 1, threshold=-2.0),
    )
    extractor._constraint_not_break_abruptly(
        SimpleNamespace(children=(predicate,))
    )
    assert qp_config.a_lon_min == -1.99


class _RG2DomainEstimator(
    VPPredicateEstimation,
    VPConstraintExtraction,
    VPOptimization,
    VPTrajectoryContext,
):
    pass


def test_rg2_temporal_domain_uses_reachable_acceleration_frames():
    estimator = object.__new__(_RG2DomainEstimator)
    evaluator = _Evaluator("brakes_abruptly", 1, threshold=-4.0)
    prop = SimpleNamespace(
        name="brakes_abruptly__0",
        alphabet="a",
        children=(SimpleNamespace(evaluator=evaluator),),
        ttv_value=1.0,
    )
    relative_evaluator = _Evaluator(
        "rel_brakes_abruptly", 2, threshold=-4.0
    )
    relative_prop = SimpleNamespace(
        name="rel_brakes_abruptly__0_1",
        alphabet="b",
        children=(SimpleNamespace(evaluator=relative_evaluator),),
        ttv_value=1.0,
    )
    initial = SimpleNamespace(
        time_step=0,
        acceleration=0.0,
        position=np.zeros(2),
        velocity=1.0,
    )
    future = SimpleNamespace(
        time_step=1,
        acceleration=0.0,
        position=np.ones(2),
        velocity=1.0,
    )
    estimator._tc = 0
    estimator.config = SimpleNamespace(
        repair=SimpleNamespace(ego_id=10),
        scenario=SimpleNamespace(dt=1.0),
        vehicle=SimpleNamespace(
            qp_veh_config=SimpleNamespace(
                a_lon_min=-3.0,
                a_lon_max=2.0,
                j_lon_min=-3.0,
                j_lon_max=2.0,
            )
        ),
    )
    estimator.ego_vehicle = SimpleNamespace(
        initial_state=initial,
        prediction=SimpleNamespace(
            trajectory=SimpleNamespace(state_list=[future])
        ),
    )
    estimator.rule_monitor = SimpleNamespace(
        other_id=20,
        world=_World({20: {0: -5.0, 1: -5.0}}),
        rob_abstraction=[[[]]],
        abstraction_names=[[[]]],
    )
    estimator.sat_solver = SimpleNamespace(_prop_nodes=[prop, relative_prop])
    estimator._semantic_fixed_domain_cache = {}

    domains = estimator._build_rg2_reachable_acceleration_domains()

    assert domains == {"a": {0}, "b": {0}}
    assert estimator._predicate_frame_domain_cache["a"] == {
        0: {0},
        1: {0},
    }
    assert estimator._predicate_frame_domain_cache["b"] == {
        0: {0},
        1: {0},
    }
