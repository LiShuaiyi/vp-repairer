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


def _incoming_quantitative_builder(
    *,
    candidates=None,
    complete=True,
    target_feature=(8.0, 2.0, 0.0),
    uncertainty=0.1,
    fallback_region=None,
):
    evaluator = SimpleNamespace(_scale_lon_dist=lambda value: float(value))
    candidate = {
        "ego_lower": 0.0,
        "ego_upper": 10.0,
        "target_start": 0.0,
        "target_end": 10.0,
    }
    candidates = (candidate,) if candidates is None else tuple(candidates)
    target = SimpleNamespace(id=2)
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.other_id = 2
    builder.uncertainty = float(uncertainty)
    builder.world = SimpleNamespace(vehicle_by_id=lambda _target_id: target)
    builder._target_front_rear = lambda _target, _step: target_feature
    builder._incoming_left_of_geometry = lambda: {
        "candidates": candidates,
        "complete": bool(complete),
    }
    builder._region_cache = {}
    builder._timing = {"region_build": 0.0}
    if fallback_region is not None:
        builder._region_cache[("on_incoming_left_of", 0, 2)] = fallback_region
    return builder, evaluator, candidate


def test_incoming_quantitative_no_candidate_is_monitor_floor_false():
    builder, evaluator, _ = _incoming_quantitative_builder(candidates=())

    estimate = builder._on_incoming_left_of_frame(evaluator, 0, (2.0, 4.0))

    assert estimate.domain == FALSE_DOMAIN
    assert estimate.source == "on_incoming_left_of:quantitative"
    assert estimate.diagnostics["robustness_min"] == -1.0
    assert estimate.diagnostics["robustness_max"] == -1.0


def test_incoming_quantitative_unique_candidate_true_false_unknown():
    builder, evaluator, _ = _incoming_quantitative_builder()

    true_estimate = builder._on_incoming_left_of_frame(
        evaluator, 0, (2.0, 4.0)
    )
    false_estimate = builder._on_incoming_left_of_frame(
        evaluator, 0, (-3.0, -2.0)
    )
    unknown_estimate = builder._on_incoming_left_of_frame(
        evaluator, 0, (-0.05, 0.05)
    )

    assert true_estimate.domain == TRUE_DOMAIN
    assert false_estimate.domain == FALSE_DOMAIN
    assert unknown_estimate.domain == UNKNOWN_DOMAIN


def test_incoming_quantitative_adds_tent_peak_only_when_reachable():
    builder, evaluator, _ = _incoming_quantitative_builder()

    containing = builder._on_incoming_left_of_frame(evaluator, 0, (4.0, 6.0))
    excluding = builder._on_incoming_left_of_frame(evaluator, 0, (1.0, 2.0))

    assert containing.diagnostics["partition_points"] == [4.0, 5.0, 6.0]
    assert containing.diagnostics["tent_peak_reachable"] is True
    assert containing.diagnostics["robustness_max"] == 5.0
    assert excluding.diagnostics["partition_points"] == [1.0, 2.0]
    assert excluding.diagnostics["tent_peak_reachable"] is False
    assert excluding.diagnostics["robustness_max"] == 2.0


def test_incoming_quantitative_target_cap_and_negative_one_floor():
    capped_builder, evaluator, _ = _incoming_quantitative_builder(
        target_feature=(0.05, 5.0, 0.0)
    )
    floored_builder, evaluator, _ = _incoming_quantitative_builder()

    capped = capped_builder._on_incoming_left_of_frame(
        evaluator, 0, (2.0, 4.0)
    )
    floored = floored_builder._on_incoming_left_of_frame(
        evaluator, 0, (-5.0, -4.0)
    )

    assert capped.domain == UNKNOWN_DOMAIN
    assert capped.diagnostics["target_cap"] == 0.05
    assert capped.diagnostics["robustness_max"] == 0.05
    assert floored.domain == FALSE_DOMAIN
    assert floored.diagnostics["robustness_min"] == -1.0
    assert floored.diagnostics["robustness_max"] == -1.0


def test_incoming_quantitative_falls_back_for_multiple_or_failed_geometry():
    fallback = SemanticIntervalSet.from_nominal(
        ((0.0, 1.0),), uncertainty=0.0, source="incoming:fallback"
    )
    _, _, candidate = _incoming_quantitative_builder()
    multiple_builder, evaluator, _ = _incoming_quantitative_builder(
        candidates=(candidate, dict(candidate)), fallback_region=fallback
    )
    failed_builder, evaluator, _ = _incoming_quantitative_builder(
        complete=False, fallback_region=fallback
    )
    missing_builder, evaluator, _ = _incoming_quantitative_builder(
        fallback_region=fallback
    )
    missing_builder._incoming_left_of_geometry = lambda: None

    multiple = multiple_builder._on_incoming_left_of_frame(
        evaluator, 0, (0.2, 0.8)
    )
    failed = failed_builder._on_incoming_left_of_frame(
        evaluator, 0, (0.2, 0.8)
    )
    missing = missing_builder._on_incoming_left_of_frame(
        evaluator, 0, (0.2, 0.8)
    )

    assert multiple.domain == TRUE_DOMAIN
    assert multiple.source == "incoming:fallback"
    assert failed.domain == TRUE_DOMAIN
    assert failed.source == "incoming:fallback"
    assert missing.domain == TRUE_DOMAIN
    assert missing.source == "incoming:fallback"


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


def _causes_braking_endpoint_builder(
    rear_endpoints, *, target_acceleration=-2.0, uncertainty=0.1
):
    target_lane = object()
    target = SimpleNamespace(id=2, ref_path_lane=target_lane)
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.other_id = 2
    builder.uncertainty = float(uncertainty)
    builder.world = SimpleNamespace(vehicle_by_id=lambda _target_id: target)
    builder._target_front_rear = lambda _target, _step: (
        0.0,
        None,
        float(target_acceleration),
    )
    builder._ego_lane_feature_endpoints = (
        lambda reachable, path_lane, coordinate: (
            tuple(float(value) for value in rear_endpoints),
            {
                "analysis_mode": "monotone_endpoint",
                "mapping_assumption": "fixed_reference_path_lane_forward",
            },
        )
    )
    evaluator = SimpleNamespace(
        config={"d_br": 10.0, "a_br": -1.0},
        _scale_acc=lambda value: value,
        _scale_lon_dist=lambda value: value,
    )
    return builder, evaluator


def test_causes_braking_uses_reachable_endpoint_range_and_tent_peak():
    builder, evaluator = _causes_braking_endpoint_builder((4.0, 6.0))

    estimate = builder._causes_braking_frame(evaluator, 0, (2.0, 8.0))

    assert estimate.domain == TRUE_DOMAIN
    assert estimate.source.endswith("monotone_endpoint")
    assert estimate.diagnostics["distance_interval"] == [4.0, 6.0]
    assert estimate.diagnostics["distance_critical_reachable"] is True


def test_causes_braking_reachable_endpoint_range_certifies_false_and_unknown():
    false_builder, evaluator = _causes_braking_endpoint_builder((-2.0, -1.0))
    false_estimate = false_builder._causes_braking_frame(
        evaluator, 0, (2.0, 8.0)
    )
    assert false_estimate.domain == FALSE_DOMAIN

    unknown_builder, evaluator = _causes_braking_endpoint_builder((-1.0, 1.0))
    unknown_estimate = unknown_builder._causes_braking_frame(
        evaluator, 0, (2.0, 8.0)
    )
    assert unknown_estimate.domain == UNKNOWN_DOMAIN


def test_causes_braking_acceleration_boundary_remains_unknown():
    builder, evaluator = _causes_braking_endpoint_builder(
        (4.0, 6.0), target_acceleration=-1.0
    )

    estimate = builder._causes_braking_frame(evaluator, 0, (2.0, 8.0))

    assert estimate.domain == UNKNOWN_DOMAIN


def _longitudinal_feature_cells(front_values, rear_values=None, valid=None):
    front_values = np.asarray(front_values, dtype=float)
    rear_values = np.asarray(
        front_values - 1.0 if rear_values is None else rear_values,
        dtype=float,
    )
    count = len(front_values) - 1
    return {
        "s_lower": np.arange(count, dtype=float),
        "s_upper": np.arange(1, count + 1, dtype=float),
        "front_left": front_values[:-1],
        "front_right": front_values[1:],
        "rear_left": rear_values[:-1],
        "rear_right": rear_values[1:],
        "front_lower": np.minimum(front_values[:-1], front_values[1:]),
        "front_upper": np.maximum(front_values[:-1], front_values[1:]),
        "rear_lower": np.minimum(rear_values[:-1], rear_values[1:]),
        "rear_upper": np.maximum(rear_values[:-1], rear_values[1:]),
        "valid": (
            np.ones(count, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        ),
    }


def _endpoint_builder(cells, *, uncertainty=0.1):
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.ego = SimpleNamespace(id=1)
    builder.uncertainty = float(uncertainty)
    builder._target_path_rear_cells = lambda _ego, path_lane=None: cells
    return builder


def test_lane_feature_endpoint_mapping_is_nonlinear_and_cache_compatible():
    cells = _longitudinal_feature_cells(
        (0.0, 1.0, 4.0), (-1.0, 0.0, 3.0)
    )
    builder = _endpoint_builder(cells)
    endpoints, diagnostics = builder._ego_lane_feature_endpoints(
        (0.25, 1.75), path_lane=object(), coordinate="front"
    )
    assert endpoints == (0.25, 3.25)
    assert diagnostics["mapping_assumption"] == (
        "fixed_reference_path_lane_forward"
    )

    cells.pop("front_left")
    cells.pop("front_right")
    endpoints, _ = builder._ego_lane_feature_endpoints(
        (0.25, 1.75), path_lane=object(), coordinate="front"
    )
    assert endpoints == (0.25, 3.25)


def test_lane_feature_endpoint_projection_failure_is_unknown():
    builder = _endpoint_builder(
        _longitudinal_feature_cells(
            (0.0, 1.0, 2.0), valid=(True, False)
        )
    )
    endpoints, diagnostics = builder._ego_lane_feature_endpoints(
        (0.25, 1.75), path_lane=object(), coordinate="front"
    )
    assert endpoints is None
    assert diagnostics["unknown_reason"] == "endpoint_projection_failure"


def _vehicle(vehicle_id, lane, *, front=0.0, rear=0.0, velocity=1.0):
    return SimpleNamespace(
        id=int(vehicle_id),
        get_lane=lambda _step: lane,
        front_s=lambda _step, *args: float(front),
        rear_s=lambda _step, *args: float(rear),
        states_cr={0: SimpleNamespace(velocity=float(velocity))},
        vehicle_param={"a_min": -2.0, "t_react": 1.0},
    )


def _predicate_builder(*, ego_id, vehicles, endpoints, uncertainty=0.1):
    builder = object.__new__(SemanticINPredicateRegionBuilder)
    builder.ego_id = int(ego_id)
    builder.ego = vehicles[int(ego_id)]
    builder.world = SimpleNamespace(
        vehicle_by_id=lambda vehicle_id: vehicles[int(vehicle_id)]
    )
    builder.uncertainty = float(uncertainty)
    builder.reachable_velocity_by_time = {0: (0.0, 2.0)}
    builder._ego_lane_feature_endpoints = (
        lambda reachable, path_lane, coordinate: (
            tuple(map(float, endpoints)),
            {"analysis_mode": "monotone_endpoint", "coordinate": coordinate},
        )
    )
    builder._longitudinal_cell_region = lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            AssertionError("cell-region estimator must not be used")
        )
    )
    builder._constant_path_region = lambda value, source: SimpleNamespace(
        source=source
    )
    return builder


def test_in_front_of_endpoint_ranges_cover_both_argument_orders():
    lane = object()
    for threshold, expected in (
        (10.0, TRUE_DOMAIN),
        (1.0, FALSE_DOMAIN),
        (3.0, UNKNOWN_DOMAIN),
        (4.05, UNKNOWN_DOMAIN),
    ):
        vehicles = {
            1: _vehicle(1, lane),
            2: _vehicle(2, lane, rear=threshold),
        }
        builder = _predicate_builder(
            ego_id=1, vehicles=vehicles, endpoints=(2.0, 4.0)
        )
        estimate = builder._in_front_of_frame(0, (1, 2), (0.0, 1.0))
        assert estimate.domain == expected
        assert estimate.source == "in_front_of:monotone_endpoint"

    for threshold, expected in (
        (1.0, TRUE_DOMAIN),
        (7.0, FALSE_DOMAIN),
        (5.0, UNKNOWN_DOMAIN),
        (3.95, UNKNOWN_DOMAIN),
    ):
        vehicles = {
            1: _vehicle(1, lane, front=threshold),
            2: _vehicle(2, lane),
        }
        builder = _predicate_builder(
            ego_id=2, vehicles=vehicles, endpoints=(4.0, 6.0)
        )
        estimate = builder._in_front_of_frame(0, (1, 2), (0.0, 1.0))
        assert estimate.domain == expected


def test_safe_distance_velocity_envelope_covers_both_argument_orders():
    lane = object()
    for lead_rear, expected in (
        (10.0, TRUE_DOMAIN),
        (0.0, FALSE_DOMAIN),
        (5.0, UNKNOWN_DOMAIN),
    ):
        vehicles = {
            1: _vehicle(1, lane, velocity=1.0),
            2: _vehicle(2, lane, rear=lead_rear, velocity=1.0),
        }
        builder = _predicate_builder(
            ego_id=1, vehicles=vehicles, endpoints=(1.0, 3.0)
        )
        estimate = builder._safe_distance_frame(0, (1, 2), (0.0, 1.0))
        assert estimate.domain == expected
        assert estimate.diagnostics["safe_distance_max"] == 2.75
        assert estimate.source == "safe_distance:velocity_envelope_endpoint"

    for follow_front, expected in (
        (1.0, TRUE_DOMAIN),
        (10.0, FALSE_DOMAIN),
        (5.0, UNKNOWN_DOMAIN),
    ):
        vehicles = {
            1: _vehicle(1, lane, front=follow_front, velocity=1.0),
            2: _vehicle(2, lane, velocity=1.0),
        }
        builder = _predicate_builder(
            ego_id=2, vehicles=vehicles, endpoints=(5.0, 7.0)
        )
        estimate = builder._safe_distance_frame(0, (1, 2), (0.0, 1.0))
        assert estimate.domain == expected
        assert estimate.diagnostics["safe_distance_max"] == 1.25


def test_safe_distance_velocity_monotonicity_and_negative_rejection():
    safe = SemanticINPredicateRegionBuilder._safe_distance_value
    assert safe(2.0, 2.0, -2.0, -2.0, 1.0) > safe(
        1.0, 2.0, -2.0, -2.0, 1.0
    )
    assert safe(1.0, 2.0, -2.0, -2.0, 1.0) < safe(
        1.0, 1.0, -2.0, -2.0, 1.0
    )

    lane = object()
    vehicles = {
        1: _vehicle(1, lane),
        2: _vehicle(2, lane, rear=10.0),
    }
    builder = _predicate_builder(
        ego_id=1, vehicles=vehicles, endpoints=(1.0, 3.0)
    )
    builder.reachable_velocity_by_time = {0: (-0.01, 2.0)}
    estimate = builder._safe_distance_frame(0, (1, 2), (0.0, 1.0))
    assert estimate.domain == UNKNOWN_DOMAIN
    assert estimate.diagnostics["unknown_reason"] == (
        "negative_reachable_velocity"
    )


def test_endpoint_velocity_envelope_is_sound_against_dense_samples():
    safe = SemanticINPredicateRegionBuilder._safe_distance_value
    for mapping in (
        lambda value: value,
        lambda value: value * value,
        lambda value: np.sqrt(value + 1.0),
    ):
        positions = np.linspace(0.0, 2.0, 101)
        velocities = np.linspace(0.0, 2.0, 81)
        samples = np.asarray(
            [
                5.0 - mapping(position) - safe(
                    velocity, 1.0, -2.0, -2.0, 1.0
                )
                for position in positions
                for velocity in velocities
            ]
        )
        endpoint_features = (mapping(0.0), mapping(2.0))
        safe_values = (
            safe(0.0, 1.0, -2.0, -2.0, 1.0),
            safe(2.0, 1.0, -2.0, -2.0, 1.0),
        )
        robustness_min = 5.0 - max(endpoint_features) - max(safe_values)
        robustness_max = 5.0 - min(endpoint_features) - min(safe_values)
        if robustness_min > 0.1:
            assert np.all(samples > 0.0)
        elif robustness_max < -0.1:
            assert np.all(samples < 0.0)
        elif np.any(samples > 0.0) and np.any(samples < 0.0):
            assert robustness_min <= 0.1 and robustness_max >= -0.1
