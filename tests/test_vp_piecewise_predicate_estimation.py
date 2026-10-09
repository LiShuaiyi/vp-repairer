from types import SimpleNamespace

import math
import numpy as np

from crrepairer.repairer.vp.predicate_piecewise import (
    FALSE_DOMAIN,
    TRUE_DOMAIN,
    UNKNOWN_DOMAIN,
    classify_region_piecewise,
    estimate_piecewise_monotone_1d,
)
from crrepairer.repairer.vp.semantic_predicate_regions import (
    SemanticINPredicateRegionBuilder,
    SemanticIntervalSet,
)
from crrepairer.repairer.vp.domain import VPPredicateEstimation


def test_affine_monotone_interval_uses_endpoint_range():
    estimate = estimate_piecewise_monotone_1d(
        (2.0, 4.0), (), lambda value: value - 1.0, uncertainty=0.0
    )

    assert estimate.domain == TRUE_DOMAIN
    assert estimate.partition_points == (2.0, 4.0)
    assert estimate.segments[0].robustness_min == 1.0
    assert estimate.segments[0].robustness_max == 3.0


def test_tent_false_endpoints_and_true_interior_is_unknown():
    robustness = lambda value: min(value, 2.0 - value) - 0.25
    estimate = estimate_piecewise_monotone_1d(
        (0.0, 2.0), (1.0,), robustness, uncertainty=0.0
    )

    assert estimate.domain == UNKNOWN_DOMAIN
    assert estimate.partition_points == (0.0, 1.0, 2.0)


def test_tent_interval_crossing_maximum_can_be_certified_true():
    robustness = lambda value: min(value, 2.0 - value) - 0.25
    estimate = estimate_piecewise_monotone_1d(
        (0.5, 1.5), (1.0,), robustness, uncertainty=0.0
    )

    assert estimate.domain == TRUE_DOMAIN


def test_critical_point_cleanup_and_boundary_band():
    estimate = estimate_piecewise_monotone_1d(
        (0.0, 1.0),
        (0.0, 0.5, 0.5 + 1.0e-12, 1.0, math.nan, math.inf),
        lambda value: value,
        uncertainty=0.0,
    )

    assert estimate.partition_points == (0.0, 0.5, 1.0)
    assert estimate.domain == UNKNOWN_DOMAIN


def test_piecewise_boolean_region_matches_exact_union():
    region = SemanticIntervalSet.from_nominal(
        ((0.0, 1.0), (1.0, 2.0)),
        uncertainty=0.0,
        source="test",
    )

    estimate = classify_region_piecewise(region, (0.25, 1.75))

    assert estimate.domain == TRUE_DOMAIN
    assert estimate.partition_points == (0.25, 1.75)


def test_piecewise_boolean_region_preserves_uncertainty_band():
    region = SemanticIntervalSet.from_nominal(
        ((0.0, 1.0),), uncertainty=0.1, source="test"
    )

    assert classify_region_piecewise(region, (-0.05, 0.2)).domain == UNKNOWN_DOMAIN
    assert classify_region_piecewise(region, (0.2, 0.8)).domain == TRUE_DOMAIN
    assert classify_region_piecewise(region, (1.2, 1.4)).domain == FALSE_DOMAIN


def _stop_line_builder(region, *, uncertainty=0.1):
    evaluator = SimpleNamespace()
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.uncertainty = float(uncertainty)
    builder._front_extent = 0.0
    builder._region_cache = {("stop_line_in_front", id(evaluator)): region}
    builder._timing = {"region_build": 0.0}
    return builder, evaluator


def _stop_line_region(candidates, *, uncertainty=0.1, complete=True):
    intervals = [candidate["truth_overlap"] for candidate in candidates]
    region = SemanticIntervalSet.from_nominal(
        intervals,
        uncertainty=uncertainty,
        source="stop_line",
        complete=complete,
        diagnostics={
            "candidates": tuple(candidates),
            "candidate_count": len(candidates),
            "d_sl": 4.0,
            "front_extent": 0.0,
        },
    )
    if complete:
        return region
    return SemanticIntervalSet(
        inner_true=region.inner_true,
        outer_true=region.outer_true,
        complete=False,
        source=region.source,
        diagnostics=region.diagnostics,
    )


def _candidate(
    *, stop_s=10.0, lane_interval=(0.0, 20.0), truth_overlap=(6.0, 10.0)
):
    return {
        "lanelet_id": 1,
        "stop_s": float(stop_s),
        "s_star": float(stop_s - 2.0),
        "stop_window": (float(stop_s - 4.0), float(stop_s)),
        "lane_interval": tuple(map(float, lane_interval)),
        "certified_lane_interval": (
            float(lane_interval[0] + 0.1),
            float(lane_interval[1] - 0.1),
        ),
        "truth_overlap": tuple(map(float, truth_overlap)),
    }


def test_stop_line_piecewise_cases_and_lane_gate():
    candidate = _candidate(lane_interval=(6.0, 10.0))
    region = _stop_line_region((candidate,))
    builder, evaluator = _stop_line_builder(region)

    assert builder._stop_line_frame(evaluator, (7.0, 9.0)).domain == TRUE_DOMAIN
    assert builder._stop_line_frame(evaluator, (0.0, 1.0)).domain == FALSE_DOMAIN
    assert builder._stop_line_frame(evaluator, (5.0, 7.0)).domain == UNKNOWN_DOMAIN
    # The robustness is positive on part of this interval, but the certified
    # lane/lateral gate ends at 9.9 m.  Crossing it must never certify true.
    assert builder._stop_line_frame(evaluator, (9.8, 10.2)).domain == UNKNOWN_DOMAIN


def test_stop_line_multiple_windows_use_union_coverage():
    first = _candidate(stop_s=5.0, truth_overlap=(1.0, 5.0))
    second = _candidate(stop_s=9.0, truth_overlap=(5.0, 9.0))
    region = _stop_line_region((first, second), uncertainty=0.0)
    builder, evaluator = _stop_line_builder(region, uncertainty=0.0)

    assert builder._stop_line_frame(evaluator, (2.0, 8.0)).domain == TRUE_DOMAIN


def test_stop_line_no_candidate_and_incomplete_geometry():
    empty = _stop_line_region((), complete=True)
    builder, evaluator = _stop_line_builder(empty)
    assert builder._stop_line_frame(evaluator, (0.0, 1.0)).domain == FALSE_DOMAIN

    incomplete = _stop_line_region((), complete=False)
    builder, evaluator = _stop_line_builder(incomplete)
    assert builder._stop_line_frame(evaluator, (0.0, 1.0)).domain == UNKNOWN_DOMAIN


def test_standstill_uses_signed_velocity_piecewise_branches():
    evaluator = SimpleNamespace(config={"standstill_error": 0.5})
    builder = object.__new__(SemanticINPredicateRegionBuilder)

    cases = (
        ((-1.0, -0.6), FALSE_DOMAIN),
        ((0.6, 1.0), FALSE_DOMAIN),
        ((-0.4, 0.4), TRUE_DOMAIN),
        ((-0.6, -0.4), UNKNOWN_DOMAIN),
        ((0.4, 0.6), UNKNOWN_DOMAIN),
    )
    for index, (velocity, expected) in enumerate(cases):
        builder.reachable_velocity_by_time = {index: velocity}
        assert builder._standstill_frame(evaluator, index).domain == expected


def _speed_estimator(profile):
    estimator = VPPredicateEstimation()
    estimator._lane_speed_limit_profile = lambda _clcs=None: tuple(profile)
    return estimator


def test_lane_speed_position_segments_combine_domains():
    estimator = _speed_estimator(((0.0, 5.0, 10.0), (5.0, 10.0, 5.0)))

    domains, diagnostics = estimator._lane_speed_limit_segment_domains(
        2.0, 8.0, 4.0, 6.0
    )

    assert domains == (TRUE_DOMAIN, UNKNOWN_DOMAIN)
    assert diagnostics["segment_count"] == 2
    assert diagnostics["analysis_axis"] == "s,v"


def test_lane_speed_unlimited_and_projection_gap_are_distinct():
    unlimited = _speed_estimator(((0.0, 5.0, None),))
    domains, _ = unlimited._lane_speed_limit_segment_domains(
        1.0, 4.0, 20.0, 30.0
    )
    assert domains == (TRUE_DOMAIN,)

    gap = _speed_estimator(((0.0, 4.0, 10.0), (6.0, 10.0, 10.0)))
    domains, _ = gap._lane_speed_limit_segment_domains(
        2.0, 8.0, 1.0, 2.0
    )
    assert domains == (TRUE_DOMAIN, UNKNOWN_DOMAIN, TRUE_DOMAIN)


def test_lane_speed_can_certify_false_under_every_reachable_limit():
    estimator = _speed_estimator(((0.0, 5.0, 10.0), (5.0, 10.0, 5.0)))
    domains, _ = estimator._lane_speed_limit_segment_domains(
        2.0, 8.0, 11.0, 12.0
    )
    assert domains == (FALSE_DOMAIN, FALSE_DOMAIN)


def _causes_braking_builder(*, valid=True):
    target = SimpleNamespace(id=2)
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.other_id = 2
    builder.uncertainty = 0.1
    builder.world = SimpleNamespace(vehicle_by_id=lambda _target_id: target)
    builder._target_front_rear = lambda _target, _step: (0.0, None, -2.0)
    cells = {
        "s_lower": np.asarray([0.0]),
        "s_upper": np.asarray([10.0]),
        "rear_left": np.asarray([0.0]),
        "rear_right": np.asarray([10.0]),
        "valid": np.asarray([valid]),
    }
    builder._target_path_rear_cells = lambda _target: cells
    evaluator = SimpleNamespace(
        config={"d_br": 10.0, "a_br": -1.0},
        _scale_acc=lambda value: value,
        _scale_lon_dist=lambda value: value,
    )
    return builder, evaluator


def test_causes_braking_splits_mapping_cell_at_distance_maximum():
    builder, evaluator = _causes_braking_builder()

    estimate = builder._causes_braking_piecewise_estimate(
        evaluator, 0, (4.0, 6.0)
    )

    assert estimate is not None
    assert estimate.partition_points == (4.0, 5.0, 6.0)
    assert estimate.domain == TRUE_DOMAIN


def test_causes_braking_preserves_invalid_mapping_as_unknown_fallback():
    builder, evaluator = _causes_braking_builder(valid=False)

    assert (
        builder._causes_braking_piecewise_estimate(evaluator, 0, (4.0, 6.0))
        is None
    )
