"""Definition-driven robustness regions for intersection predicates.

The velocity planner changes the ego vehicle's longitudinal progress along a
fixed reference path.  For many intersection predicates, the monitor
definition can therefore be represented as a set of longitudinal intervals:

* ``inner_true`` is guaranteed to be part of the predicate's true set;
* ``outer_true`` is guaranteed to contain the predicate's complete true set.

For an ego reachable interval ``R`` this gives the conservative three-valued
fallback

``R subset inner_true -> {1}``, ``R disjoint outer_true -> {0}``, otherwise
``{0, 1}``.  Predicates with reliable nominal path geometry use their
continuous signed-distance robustness by default; the inner/outer regions are
retained for incomplete or ambiguous geometry.

Unlike the older experimental estimator, this module never fits a region to
the ego vehicle's recorded predicate trace.  It derives boundaries from the
monitor's map/topology definitions (stop lines, lanelet extents, conflict
geometry, and turning successors).  Predicate-cache reads are restricted to
fixed target-vehicle facts and topology/priority gates.

The builder is intentionally independent of :mod:`domain`; callers can enable
it experimentally without changing the baseline DomainDPLL implementation.
Unsupported or incomplete geometry always returns an unknown domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import time
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString

from commonroad.scenario.lanelet import LaneletType

from crrepairer.repairer.vp.predicate_regions import (
    ClosedInterval,
    as_closed_interval,
    merge_intervals,
)
from crrepairer.repairer.vp.predicate_piecewise import (
    FALSE_DOMAIN,
    TRUE_DOMAIN,
    UNKNOWN_DOMAIN,
    PiecewiseEstimate,
    TruthDomain,
    classify_robustness_range,
    classify_region_piecewise,
    estimate_piecewise_monotone_1d,
    piecewise_diagnostics,
)


def _domain(value: Optional[bool]) -> TruthDomain:
    if value is None:
        return UNKNOWN_DOMAIN
    return TRUE_DOMAIN if value else FALSE_DOMAIN


def _predicate_name(evaluator: Any) -> str:
    name = getattr(evaluator, "predicate_name", None)
    return str(getattr(name, "value", name or "")).strip().lower()


def _freeze_cache_value(value: Any) -> Any:
    """Return a stable, hashable signature for evaluator configuration."""
    if isinstance(value, Mapping):
        return tuple(sorted(
            (str(key), _freeze_cache_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(repr(_freeze_cache_value(item)) for item in value))
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _evaluator_semantic_key(evaluator: Any) -> Tuple[Any, ...]:
    """Identify repeated monitor evaluators by equivalent semantics."""
    evaluator_type = type(evaluator)
    return (
        evaluator_type.__module__,
        evaluator_type.__qualname__,
        _predicate_name(evaluator),
        tuple(getattr(evaluator, "agent_placeholders", ()) or ()),
        _freeze_cache_value(getattr(evaluator, "config", None)),
    )


def _finite_pair(value: Any) -> Optional[Tuple[float, float]]:
    try:
        lower, upper = float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return (lower, upper) if lower <= upper else (upper, lower)


def estimate_acceleration_predicate_frame(
    evaluator: Any,
    time_step: int,
    vehicle_ids: Sequence[int],
    *,
    ego_id: int,
    world: Any,
    reachable_acceleration_by_time: Mapping[int, Sequence[float]],
) -> FramePredicateEstimate:
    """Estimate acceleration predicates over the complete VP envelope.

    Acceleration robustness is affine, so its extrema occur at the reachable
    acceleration endpoints.  No longitudinal-distance uncertainty is applied
    on this independent physical axis; exact zero remains unknown through the
    shared strict-sign classifier.
    """
    name = _predicate_name(evaluator)
    reachable = _finite_pair(
        reachable_acceleration_by_time.get(int(time_step))
    )
    try:
        threshold = float(evaluator.config["a_abrupt"])
    except (AttributeError, KeyError, TypeError, ValueError):
        threshold = math.nan
    if reachable is None or not math.isfinite(threshold):
        return FramePredicateEstimate(
            UNKNOWN_DOMAIN, f"{name}:acceleration_unavailable"
        )

    ids = tuple(int(item) for item in vehicle_ids)
    lower, upper = reachable
    if name == "brakes_abruptly":
        if ids != (int(ego_id),):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "brakes_abruptly:non_ego"
            )
        robustness_min = threshold - upper
        robustness_max = threshold - lower
    elif name == "rel_brakes_abruptly" and len(ids) == 2:
        if ids.count(int(ego_id)) != 1:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "rel_brakes_abruptly:unsupported_arguments"
            )
        other_index = 1 if ids[0] == int(ego_id) else 0
        try:
            other_state = world.vehicle_by_id(ids[other_index]).states_cr[
                int(time_step)
            ]
            other_acceleration = float(other_state.acceleration)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            other_acceleration = math.nan
        if not math.isfinite(other_acceleration):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "rel_brakes_abruptly:target_acceleration_unavailable"
            )
        if ids[0] == int(ego_id):
            robustness_min = -upper + other_acceleration + threshold
            robustness_max = -lower + other_acceleration + threshold
        else:
            robustness_min = -other_acceleration + lower + threshold
            robustness_max = -other_acceleration + upper + threshold
    else:
        return FramePredicateEstimate(
            UNKNOWN_DOMAIN, f"unsupported_acceleration:{name}"
        )

    domain = classify_robustness_range(
        robustness_min,
        robustness_max,
        uncertainty=0.0,
    )
    return FramePredicateEstimate(
        domain,
        f"{name}:reachable_acceleration",
        diagnostics={
            "analysis_axis": "a",
            "acceleration_lower": float(lower),
            "acceleration_upper": float(upper),
            "robustness_min": float(robustness_min),
            "robustness_max": float(robustness_max),
        },
    )


def _project_points_to_s(
    clcs: Any,
    points: Any,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Project path points while retaining the scalar failure semantics.

    ``commonroad-clcs`` exposes a C++ batch conversion which is much cheaper
    than crossing the Python/C++ boundary once per 0.1 m reference-path
    sample.  A batch can fail when even one point is outside the projection
    domain, so failed chunks fall back to the former point-wise conversion.
    The returned array therefore has exactly the same useful samples as the
    scalar implementation; failed/non-finite projections are represented by
    ``nan``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.full(len(points), math.nan, dtype=float)
    points = np.ascontiguousarray(points[:, :2], dtype=float)
    result = np.full(len(points), math.nan, dtype=float)
    bulk = getattr(clcs, "convert_list_of_points_to_curvilinear_coords", None)
    chunk_size = max(1, int(chunk_size))
    # Scalar conversion is faster for very short paths and also avoids paying
    # any one-time batch/OpenMP setup cost for nearly stationary trajectories.
    if len(points) <= 16:
        bulk = None
    elif bulk is not None:
        try:
            projected = np.asarray(bulk(points, 1), dtype=float)
            if projected.shape == (len(points), 2):
                values = projected[:, 0]
                return np.where(np.isfinite(values), values, math.nan)
        except Exception:
            # Retry in chunks below.  Only a chunk containing an out-of-domain
            # sample then needs the scalar compatibility path.
            pass
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        chunk = points[start:end]
        if bulk is not None:
            try:
                projected = np.asarray(bulk(chunk, 1), dtype=float)
                if projected.shape == (len(chunk), 2):
                    values = projected[:, 0]
                    result[start:end] = np.where(
                        np.isfinite(values), values, math.nan
                    )
                    continue
            except Exception:
                pass
        for index, point in enumerate(chunk, start=start):
            try:
                value = clcs.convert_to_curvilinear_coords(
                    float(point[0]), float(point[1])
                )[0]
                value = float(value)
                if math.isfinite(value):
                    result[index] = value
            except Exception:
                continue
    return result


def _downsample_estimation_path(
    points: Any,
    max_spacing: float,
    *,
    sharp_turn_threshold: float = 0.08,
) -> np.ndarray:
    """Return a sparse geometry-preserving view of a dense VP path.

    Acceleration planning keeps its original 0.1 m reference path for CLCS,
    curvature, and trajectory generation.  Predicate estimation only needs a
    piecewise-monotone map, so retaining points roughly every ``max_spacing``
    metres is sufficient.  Vertices with an abrupt local heading change are
    retained independently of spacing.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) <= 2 or max_spacing <= 0.0:
        return points

    xy = points[:, :2]
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    finite_positive = np.isfinite(lengths) & (lengths > 1.0e-9)
    if not np.any(finite_positive):
        return points[[0, -1]]

    sharp = np.zeros(len(points), dtype=bool)
    if len(points) > 2:
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        heading_change = np.abs(
            np.arctan2(
                np.sin(np.diff(headings)),
                np.cos(np.diff(headings)),
            )
        )
        sharp[1:-1] = heading_change >= float(sharp_turn_threshold)

    keep = [0]
    accumulated = 0.0
    for index in range(1, len(points) - 1):
        step = float(lengths[index - 1])
        if math.isfinite(step):
            accumulated += max(0.0, step)
        if accumulated >= max_spacing or sharp[index]:
            keep.append(index)
            accumulated = 0.0
    keep.append(len(points) - 1)
    return points[np.asarray(keep, dtype=int)]


@dataclass(frozen=True)
class SemanticIntervalSet:
    """Conservative inner/outer approximation of a predicate true set.

    ``complete`` means that ``outer_true`` is known to cover *all* true
    positions on the repair path.  Guaranteed truth can still be inferred from
    an inner interval when ``complete`` is false, but guaranteed falsehood
    requires a complete outer approximation.
    """

    inner_true: Tuple[ClosedInterval, ...] = ()
    outer_true: Tuple[ClosedInterval, ...] = ()
    complete: bool = False
    source: str = "unknown"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inner = merge_intervals(self.inner_true)
        outer = merge_intervals(self.outer_true)
        object.__setattr__(self, "inner_true", inner)
        object.__setattr__(self, "outer_true", outer)

    @classmethod
    def unknown(cls, source: str, **diagnostics: Any) -> "SemanticIntervalSet":
        return cls(complete=False, source=source, diagnostics=diagnostics)

    @classmethod
    def empty(cls, source: str, **diagnostics: Any) -> "SemanticIntervalSet":
        return cls(complete=True, source=source, diagnostics=diagnostics)

    @classmethod
    def from_nominal(
        cls,
        intervals: Iterable[Sequence[float]],
        *,
        uncertainty: float,
        source: str,
        complete: bool = True,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> "SemanticIntervalSet":
        """Create nested bounds by shrinking/expanding nominal intervals."""
        uncertainty = max(0.0, float(uncertainty))
        nominal = merge_intervals(intervals)
        inner = []
        outer = []
        for interval in nominal:
            outer.append(
                ClosedInterval(
                    interval.lower - uncertainty,
                    interval.upper + uncertainty,
                )
            )
            lower = interval.lower + uncertainty
            upper = interval.upper - uncertainty
            if lower <= upper:
                inner.append(ClosedInterval(lower, upper))
        return cls(
            inner_true=tuple(inner),
            outer_true=tuple(outer),
            complete=bool(complete),
            source=source,
            diagnostics=dict(diagnostics or {}),
        )

    @property
    def exact(self) -> bool:
        """Whether the inner and outer interval sets coincide."""
        return self.complete and self.inner_true == self.outer_true

    @property
    def reason(self) -> Optional[str]:
        """Compatibility description for callers using ``CertifiedRegion``."""
        return None if self.complete else self.source

    def classify(
        self,
        reachable: Sequence[float] | float,
        upper: Optional[float] = None,
    ) -> TruthDomain:
        """Classify all positions in one reachable interval."""
        try:
            interval = as_closed_interval(
                (reachable, upper) if upper is not None else reachable
            )
        except (TypeError, ValueError):
            return UNKNOWN_DOMAIN
        if any(item.contains_interval(interval) for item in self.inner_true):
            return TRUE_DOMAIN
        if self.complete and all(
            item.is_disjoint(interval) for item in self.outer_true
        ):
            return FALSE_DOMAIN
        return UNKNOWN_DOMAIN

    def to_diagnostics(self) -> Dict[str, Any]:
        def pairs(items: Iterable[ClosedInterval]) -> list:
            return [[float(item.lower), float(item.upper)] for item in items]

        return {
            "source": self.source,
            "complete": bool(self.complete),
            "inner_true": pairs(self.inner_true),
            "outer_true": pairs(self.outer_true),
            **dict(self.diagnostics),
        }


@dataclass(frozen=True)
class FramePredicateEstimate:
    """Truth domain of one proposition at one temporal frame."""

    domain: TruthDomain
    source: str
    region: Optional[SemanticIntervalSet] = None
    gate_domain: TruthDomain = TRUE_DOMAIN
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> Optional[int]:
        if len(self.domain) != 1:
            return None
        return next(iter(self.domain))


@dataclass(frozen=True)
class _SignedDistanceGeometry:
    """Cached critical geometry for a union of longitudinal truth intervals."""

    components: Tuple[ClosedInterval, ...]
    component_midpoints: Tuple[float, ...]
    gap_midpoints: Tuple[float, ...]

    @classmethod
    def from_intervals(
        cls, intervals: Iterable[Sequence[float]]
    ) -> "_SignedDistanceGeometry":
        components = merge_intervals(intervals)
        component_midpoints = tuple(
            0.5 * (item.lower + item.upper) for item in components
        )
        gap_midpoints = tuple(
            0.5 * (left.upper + right.lower)
            for left, right in zip(components[:-1], components[1:])
        )
        return cls(components, component_midpoints, gap_midpoints)

    def signed_distance(self, progress: float) -> float:
        """Return ``max_i min(s-a_i, b_i-s)`` for non-empty components."""
        progress = float(progress)
        if not math.isfinite(progress) or not self.components:
            raise ValueError("signed-distance geometry is empty or non-finite")
        return max(
            min(progress - item.lower, item.upper - progress)
            for item in self.components
        )

    def critical_points(
        self, reachable: Sequence[float]
    ) -> Tuple[float, ...]:
        """Return endpoints and in-range component/gap extrema once each."""
        reachable_pair = _finite_pair(reachable)
        if reachable_pair is None:
            raise ValueError("reachable interval is unavailable")
        lower, upper = reachable_pair
        candidates = [lower, upper]
        candidates.extend(
            point
            for point in self.component_midpoints + self.gap_midpoints
            if lower <= point <= upper
        )
        points = []
        for point in sorted(candidates):
            if not points or point - points[-1] > 1.0e-9:
                points.append(float(point))
        return tuple(points)


def _scaled_longitudinal_uncertainty(
    evaluator: Any, uncertainty: float
) -> float:
    """Convert a metric longitudinal band to evaluator robustness units."""
    scale = evaluator._scale_lon_dist
    zero = float(scale(0.0))
    result = max(
        abs(float(scale(float(uncertainty))) - zero),
        abs(float(scale(-float(uncertainty))) - zero),
    )
    if not math.isfinite(result):
        raise ValueError("non-finite scaled longitudinal uncertainty")
    return result


def _signed_distance_robustness_range(
    geometry: _SignedDistanceGeometry,
    reachable: Sequence[float],
    evaluator: Any,
) -> Tuple[float, float, Tuple[float, ...]]:
    """Evaluate the exact extrema of a union signed-distance robustness."""
    points = geometry.critical_points(reachable)
    raw_values = tuple(geometry.signed_distance(point) for point in points)
    if not raw_values or not all(
        math.isfinite(value) for value in raw_values
    ):
        raise ValueError("non-finite signed-distance robustness")
    # Longitudinal robustness scaling is monotone, so transform only the two
    # extrema rather than every critical point.
    raw_min, raw_max = min(raw_values), max(raw_values)
    scaler = getattr(evaluator, "_scaler", None)
    constants = getattr(scaler, "_scale_constants", None)
    max_distance = getattr(constants, "MAX_LONG_DIST", None)
    if scaler is not None and getattr(scaler, "scale", None) is False:
        robustness_min, robustness_max = float(raw_min), float(raw_max)
    elif (
        scaler is not None
        and getattr(scaler, "scale", None) is True
        and max_distance is not None
        and math.isfinite(float(max_distance))
        and float(max_distance) > 0.0
    ):
        denominator = float(max_distance)

        def scale_known(value: float) -> float:
            return max(-1.0, min(1.0, float(value) / denominator))

        robustness_min = scale_known(raw_min)
        robustness_max = scale_known(raw_max)
    else:
        robustness_min = float(evaluator._scale_lon_dist(raw_min))
        robustness_max = float(evaluator._scale_lon_dist(raw_max))
    if not (
        math.isfinite(robustness_min) and math.isfinite(robustness_max)
    ):
        raise ValueError("non-finite scaled signed-distance robustness")
    return robustness_min, robustness_max, points


class _ProgressMap:
    """Monotone map from monitor lane progress to VP trajectory progress."""

    def __init__(self, lane_clcs: Any, trajectory_clcs: Any, ref_path: Any):
        points = np.asarray(ref_path, dtype=float)
        lane_by_ref = _project_points_to_s(lane_clcs, points)
        # Interstate VP can use the route/lane CLCS itself as the trajectory
        # coordinate system.  Reuse identical projections instead of calling
        # the same C++ object twice.
        trajectory_by_ref = (
            lane_by_ref.copy()
            if trajectory_clcs is lane_clcs
            else _project_points_to_s(trajectory_clcs, points)
        )
        self.reference_lane_s = lane_by_ref
        self.reference_trajectory_s = trajectory_by_ref
        valid = np.isfinite(lane_by_ref) & np.isfinite(trajectory_by_ref)
        samples = list(
            zip(lane_by_ref[valid].tolist(), trajectory_by_ref[valid].tolist())
        )
        if len(samples) < 2:
            raise ValueError("fewer than two shared lane/trajectory samples")

        # Sorting by lane progress also handles reference paths whose lane CLCS
        # orientation is opposite to the VP trajectory orientation.
        samples.sort(key=lambda item: item[0])
        lane_values = []
        trajectory_values = []
        for lane_s, trajectory_s in samples:
            if lane_values and abs(lane_s - lane_values[-1]) <= 1.0e-9:
                continue
            lane_values.append(lane_s)
            trajectory_values.append(trajectory_s)
        if len(lane_values) < 2:
            raise ValueError("shared progress map is degenerate")
        trajectory_delta = np.diff(trajectory_values)
        nonzero_delta = trajectory_delta[np.abs(trajectory_delta) > 1.0e-8]
        if len(nonzero_delta) == 0:
            raise ValueError("trajectory progress map is degenerate")
        direction = 1.0 if float(np.median(nonzero_delta)) > 0.0 else -1.0
        if any(
            direction * delta < -1.0e-5 for delta in trajectory_delta
        ):
            raise ValueError("lane/trajectory progress map is not monotone")

        self._lane = np.asarray(lane_values, dtype=float)
        self._trajectory = np.asarray(trajectory_values, dtype=float)
        self.direction = direction
        self.trajectory_bounds = tuple(
            sorted((float(np.min(self._trajectory)), float(np.max(self._trajectory))))
        )
        lane_step = np.diff(self._lane)
        trajectory_step = np.diff(self._trajectory)
        slopes = np.abs(trajectory_step / lane_step)
        self.scale = float(np.median(slopes[np.isfinite(slopes)]))

    def __call__(self, lane_s: float) -> float:
        lane_s = float(lane_s)
        if not math.isfinite(lane_s):
            raise ValueError("non-finite lane progress")
        if lane_s < self._lane[0]:
            slope = (
                (self._trajectory[1] - self._trajectory[0])
                / (self._lane[1] - self._lane[0])
            )
            return float(self._trajectory[0] + slope * (lane_s - self._lane[0]))
        if lane_s > self._lane[-1]:
            slope = (
                (self._trajectory[-1] - self._trajectory[-2])
                / (self._lane[-1] - self._lane[-2])
            )
            return float(self._trajectory[-1] + slope * (lane_s - self._lane[-1]))
        return float(np.interp(lane_s, self._lane, self._trajectory))

    def map_array(self, lane_s: Any) -> np.ndarray:
        """Vectorized equivalent of :meth:`__call__`, including extrapolation."""
        values = np.asarray(lane_s, dtype=float)
        result = np.interp(values, self._lane, self._trajectory)
        below = values < self._lane[0]
        if np.any(below):
            slope = (
                (self._trajectory[1] - self._trajectory[0])
                / (self._lane[1] - self._lane[0])
            )
            result[below] = self._trajectory[0] + slope * (
                values[below] - self._lane[0]
            )
        above = values > self._lane[-1]
        if np.any(above):
            slope = (
                (self._trajectory[-1] - self._trajectory[-2])
                / (self._lane[-1] - self._lane[-2])
            )
            result[above] = self._trajectory[-1] + slope * (
                values[above] - self._lane[-1]
            )
        result[~np.isfinite(values)] = math.nan
        return result


class SemanticINPredicateRegionBuilder:
    """Build monitor-definition-based robustness models for IN predicates.

    Parameters
    ----------
    repairer:
        Active VP repairer.  The builder uses its monitor world and reuses its
        conflict-geometry helper, but does not inspect the ego predicate cache.
    trajectory_clcs, ref_path:
        Fixed path used by velocity planning.
    reachable_by_time:
        Mapping from monitor time step to ego trajectory-progress interval.
    reachable_velocity_by_time:
        Optional velocity intervals used by ``in_standstill``.
    uncertainty:
        Longitudinal boundary band in metres.  It accounts for path sampling,
        projection, and strict monitor boundary conventions.
    """

    STATIC_REGION_NAMES = frozenset(
        {
            "stop_line_in_front",
            "at_traffic_sign_stop",
            "relevant_traffic_light",
            "on_lanelet_with_type_intersection",
        }
    )
    TURNING_NAMES = frozenset(
        {"turning_right", "turning_left", "going_straight"}
    )
    QUANTITATIVE_NAMES = frozenset(
        {
            "stop_line_in_front",
            "at_traffic_sign_stop",
            "relevant_traffic_light",
            "on_lanelet_with_type_intersection",
            "in_intersection_conflict_area",
            "in_front_of",
            "keeps_safe_distance_prec",
            "causes_braking_intersection",
            "in_standstill",
            "on_incoming_left_of",
        }
    )
    BOOLEAN_NAMES = frozenset(
        {
            "in_same_lane",
            "cut_in",
        }
    )
    ACCELERATION_NAMES = frozenset(
        {"brakes_abruptly", "rel_brakes_abruptly"}
    )

    @classmethod
    def supports_evaluator(cls, evaluator: Any) -> bool:
        """Return whether this builder has a sound per-frame estimator.

        Unsupported predicates remain unknown without constructing the
        comparatively expensive intersection geometry.
        """
        if evaluator is None:
            return False
        if getattr(evaluator, "_turning_ego", None) is not None:
            return True
        return _predicate_name(evaluator) in (
            cls.QUANTITATIVE_NAMES
            | cls.BOOLEAN_NAMES
            | cls.TURNING_NAMES
            | cls.ACCELERATION_NAMES
        )

    def __init__(
        self,
        repairer: Any,
        trajectory_clcs: Any,
        ref_path: Any,
        reachable_by_time: Optional[Mapping[int, Sequence[float]]] = None,
        *,
        lanelet_clcs: Any = None,
        reachable_velocity_by_time: Optional[
            Mapping[int, Sequence[float]]
        ] = None,
        reachable_acceleration_by_time: Optional[
            Mapping[int, Sequence[float]]
        ] = None,
        fixed_domain_cache: Optional[Dict[Any, Any]] = None,
        uncertainty: float = 0.10,
    ) -> None:
        started = time.perf_counter()
        self.repairer = repairer
        self.world = repairer.rule_monitor.world
        self.ego_id = int(repairer.config.repair.ego_id)
        other_id = getattr(repairer.rule_monitor, "other_id", None)
        self.other_id = None if other_id is None else int(other_id)
        self.ego = self.world.vehicle_by_id(self.ego_id)
        self.trajectory_clcs = trajectory_clcs
        full_ref_path = np.asarray(ref_path, dtype=float)
        repair_mode = getattr(repairer, "_vp_repair_mode", "deceleration")
        extend_acceleration_path = bool(
            getattr(
                getattr(repairer.config, "repair", None),
                "extend_acceleration_reference_path",
                True,
            )
        )
        sparse_acceleration_path = (
            repair_mode == "acceleration" and extend_acceleration_path
        )
        try:
            estimation_path_step = float(
                os.environ.get("CRREPAIR_VP_SEMANTIC_PATH_STEP", "0.5")
            )
        except ValueError:
            estimation_path_step = 0.5
        estimation_path_step = max(0.0, estimation_path_step)
        self.ref_path = (
            _downsample_estimation_path(
                full_ref_path,
                estimation_path_step,
            )
            if sparse_acceleration_path and estimation_path_step > 0.0
            else full_ref_path
        )
        self._full_ref_path_count = int(len(full_ref_path))
        self._estimation_ref_path_count = int(len(self.ref_path))
        self._estimation_path_step = (
            estimation_path_step if sparse_acceleration_path else 0.0
        )
        self.lanelet_clcs = lanelet_clcs or self.ego.ref_path_lane.clcs
        self.reachable_by_time = {
            int(step): pair
            for step, value in (reachable_by_time or {}).items()
            if (pair := _finite_pair(value)) is not None
        }
        self.reachable_velocity_by_time = {
            int(step): pair
            for step, value in (reachable_velocity_by_time or {}).items()
            if (pair := _finite_pair(value)) is not None
        }
        self.reachable_acceleration_by_time = {
            int(step): pair
            for step, value in (reachable_acceleration_by_time or {}).items()
            if (pair := _finite_pair(value)) is not None
        }
        self.uncertainty = max(0.0, float(uncertainty))
        self._progress = _ProgressMap(
            self.lanelet_clcs, trajectory_clcs, self.ref_path
        )
        self._region_cache: Dict[Any, SemanticIntervalSet] = {}
        # Fixed target trajectories and topology/priority gates do not depend
        # on whether the ego VP phase is deceleration or acceleration.  A
        # repair-call-scoped cache lets both builders reuse those exact monitor
        # evaluations without sharing any path-dependent spatial region.
        self._fixed_cache = (
            fixed_domain_cache if fixed_domain_cache is not None else {}
        )
        self._frame_cache: Dict[Any, FramePredicateEstimate] = {}
        self._evaluator_semantic_key_cache: Dict[int, Tuple[Any, ...]] = {}
        self._path_line = LineString(np.asarray(self.ref_path, dtype=float)[:, :2])
        self._same_lane_sample_context_cache: Optional[Mapping[str, Any]] = None
        self._same_lane_robustness_region_cache: Dict[Any, SemanticIntervalSet] = {}
        # Path samples expressed in a cutting vehicle's lane coordinates.
        # Projection depends only on that lane, not on the repair phase.
        self._cut_in_path_lateral_cache: Dict[Any, Optional[Mapping[str, Any]]] = {}
        self._cut_in_single_lane_cache: Dict[Any, TruthDomain] = {}
        self._turning_spatial_domain_cache: Dict[Any, TruthDomain] = {}
        self._lanelet_bounds_cache: Dict[int, Optional[Tuple[float, float]]] = {}
        self._signed_distance_geometry_cache: Dict[
            Any, Optional[_SignedDistanceGeometry]
        ] = {}
        self._scaled_longitudinal_uncertainty_cache: Dict[Any, float] = {}
        # These caches contain only definition-derived route geometry and
        # fixed target-trajectory features.  In particular, no entry is fitted
        # from the recorded ego predicate trace.
        self._incoming_geometry_cache: Optional[Any] = None
        self._target_longitudinal_cache: Dict[Any, Any] = {}
        self._target_path_rear_cells_cache: Dict[Any, Any] = {}
        self._ego_lane_feature_cache: Dict[Any, Optional[float]] = {}
        self._front_extent, self._rear_extent = self._vehicle_extents()
        self._timing = {
            "initialization": time.perf_counter() - started,
            "region_build": 0.0,
            "frame_classification": 0.0,
            "fixed_gate": 0.0,
        }
        self._counts: Dict[str, int] = {}

    def _semantic_evaluator_key(self, evaluator: Any) -> Tuple[Any, ...]:
        object_key = id(evaluator)
        cached = self._evaluator_semantic_key_cache.get(object_key)
        if cached is None:
            cached = _evaluator_semantic_key(evaluator)
            self._evaluator_semantic_key_cache[object_key] = cached
        return cached

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate_frame(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        vehicle_ids: Optional[Sequence[int]] = None,
        reachable: Optional[Sequence[float]] = None,
    ) -> FramePredicateEstimate:
        """Return ``{0}``, ``{1}``, or ``{0,1}`` for one frame."""
        vehicle_ids = tuple(
            int(item)
            for item in (
                vehicle_ids
                if vehicle_ids is not None
                else self._default_vehicle_ids(evaluator, prop_name)
            )
        )
        reachable_pair = _finite_pair(
            reachable
            if reachable is not None
            else self.reachable_by_time.get(int(time_step))
        )
        # The monitor can reuse one atomic PredicateNode in several temporal
        # propositions (for example IN5 c/g).  Temporal wrappers do not change
        # a frame predicate's semantics; only argument order and the explicit
        # not-oncoming composite marker do.  Share the exact frame result
        # across those aliases rather than evaluating it twice.
        name = _predicate_name(evaluator)
        frame_context = (
            name,
            "__1_0" in str(prop_name),
            "not_oncoming" in str(prop_name).lower(),
        )
        evaluator_key = (
            ("object", id(evaluator))
            if hasattr(evaluator, "_turning_ego")
            else self._semantic_evaluator_key(evaluator)
        )
        key = (
            evaluator_key,
            frame_context,
            int(time_step),
            vehicle_ids,
            reachable_pair,
        )
        if key in self._frame_cache:
            return self._frame_cache[key]
        started = time.perf_counter()
        result = self._estimate_frame_uncached(
            evaluator,
            str(prop_name),
            int(time_step),
            vehicle_ids,
            reachable_pair,
        )
        self._timing["frame_classification"] += time.perf_counter() - started
        self._frame_cache[key] = result
        self._counts[result.source] = self._counts.get(result.source, 0) + 1
        return result

    # Short alias convenient for domain.py.
    estimate = estimate_frame

    def estimate_domain(
        self,
        evaluator: Any,
        prop_name: str,
        time_steps: Iterable[int],
        vehicle_ids: Optional[Sequence[int]] = None,
        reachable_by_time: Optional[Mapping[int, Sequence[float]]] = None,
    ) -> TruthDomain:
        """Union per-frame possibilities over a proposition's active frames.

        A singleton is returned only when the predicate has the same guaranteed
        value at every active frame.  This is the invariant needed before
        DomainDPLL may omit the corresponding LP constraint.
        """
        possible = set()
        found_frame = False
        for time_step in self._normalize_time_steps(time_steps):
            found_frame = True
            reachable = None
            if reachable_by_time is not None:
                reachable = reachable_by_time.get(int(time_step))
            possible.update(
                self.estimate_frame(
                    evaluator,
                    prop_name,
                    int(time_step),
                    vehicle_ids,
                    reachable=reachable,
                ).domain
            )
            if possible == {0, 1}:
                return UNKNOWN_DOMAIN
        if not found_frame or not possible:
            return UNKNOWN_DOMAIN
        return frozenset(possible)

    def estimate_turning_spatial_domain(
        self,
        turning_evaluator: Any,
        time_steps: Iterable[int],
    ) -> TruthDomain:
        """Classify a shared ego-turning atom over one active time window.

        IN3/IN4/IN5 contain several composite predicates with the same
        ``turning_left``/``turning_right``/``going_straight`` ego atom but
        different target priority gates.  If the shared spatial atom is false
        over the complete active window, every such conjunction is false and
        none of the target gates needs to be evaluated.
        """
        steps = tuple(int(item) for item in self._normalize_time_steps(time_steps))
        name = _predicate_name(turning_evaluator)
        key = (name, steps)
        cached = self._turning_spatial_domain_cache.get(key)
        if cached is not None:
            return cached
        if not steps or name not in self.TURNING_NAMES:
            result = UNKNOWN_DOMAIN
        else:
            region = self._turning_region(turning_evaluator)
            possible = set()
            for step in steps:
                possible.update(
                    self._continuous_region_frame(
                        turning_evaluator,
                        f"turning:{name}",
                        region,
                        self.reachable_by_time.get(int(step)),
                    ).domain
                )
                if possible == {0, 1}:
                    break
            result = (
                frozenset(possible) if len(possible) == 1 else UNKNOWN_DOMAIN
            )
        self._turning_spatial_domain_cache[key] = result
        return result

    def estimate_proposition(
        self,
        proposition: Any,
        time_steps: Iterable[int],
        reachable_by_time: Optional[Mapping[int, Sequence[float]]] = None,
    ) -> TruthDomain:
        """Convenience wrapper for a monitor proposition node."""
        children = list(getattr(proposition, "children", ()) or ())
        if len(children) != 1:
            return UNKNOWN_DOMAIN
        evaluator = getattr(children[0], "evaluator", None)
        if evaluator is None:
            return UNKNOWN_DOMAIN
        return self.estimate_domain(
            evaluator,
            str(getattr(proposition, "name", "")),
            time_steps,
            reachable_by_time=reachable_by_time,
        )

    def region_for(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: Optional[int] = None,
    ) -> SemanticIntervalSet:
        """Expose a static spatial region without classifying reachability."""
        name = _predicate_name(evaluator)
        if hasattr(evaluator, "_turning_ego"):
            region = self._turning_region(evaluator._turning_ego)
            if time_step is None:
                return region
            gate = self._turning_gate_domain(evaluator, prop_name, int(time_step))
            if gate == FALSE_DOMAIN:
                return SemanticIntervalSet.empty("turning_composite:fixed_false")
            if gate == TRUE_DOMAIN:
                return region
            # If a fixed gate is unknown, truth cannot be certified inside the
            # spatial region, but falsehood outside its complete outer bound is
            # still valid.
            return SemanticIntervalSet(
                inner_true=(),
                outer_true=region.outer_true,
                complete=region.complete,
                source="turning_composite:unknown_gate",
            )
        if name == "stop_line_in_front":
            return self._cached_region(
                (name, id(evaluator)), lambda: self._stop_line_region(evaluator)
            )
        if name == "at_traffic_sign_stop":
            return self._cached_region((name,), self._stop_sign_region)
        if name == "relevant_traffic_light":
            return self._cached_region((name,), self._traffic_light_region)
        if name == "on_lanelet_with_type_intersection":
            return self._cached_region((name,), self._intersection_lanelet_region)
        if name == "in_intersection_conflict_area" and "__1_0" not in prop_name:
            return self._cached_region((name, self.other_id), self._ego_conflict_region)
        if name in self.TURNING_NAMES:
            return self._turning_region(evaluator)
        return SemanticIntervalSet.unknown(f"unsupported:{name or type(evaluator).__name__}")

    def get_diagnostics(self, *, include_regions: bool = False) -> Dict[str, Any]:
        result = {
            "timing": dict(self._timing),
            "counts": dict(self._counts),
            "cached_region_count": len(self._region_cache),
            "cached_fixed_gate_count": len(self._fixed_cache),
            "cached_frame_count": len(self._frame_cache),
            "cached_turning_spatial_domain_count": len(
                self._turning_spatial_domain_cache
            ),
            "cached_signed_distance_geometry_count": len(
                self._signed_distance_geometry_cache
            ),
            "cached_scaled_longitudinal_uncertainty_count": len(
                self._scaled_longitudinal_uncertainty_cache
            ),
            "cached_target_frame_count": len(self._target_longitudinal_cache),
            "cached_target_path_mapping_count": len(
                self._target_path_rear_cells_cache
            ),
            "uncertainty": float(self.uncertainty),
            "front_extent": float(self._front_extent),
            "rear_extent": float(self._rear_extent),
            "full_ref_path_count": self._full_ref_path_count,
            "estimation_ref_path_count": self._estimation_ref_path_count,
            "estimation_path_step": self._estimation_path_step,
        }
        if include_regions:
            result["regions"] = {
                str(key): value.to_diagnostics()
                for key, value in self._region_cache.items()
            }
        return result

    def ego_conflict_region(self) -> SemanticIntervalSet:
        """Return the cached path-specific outer/inner conflict bounds."""
        return self._cached_region(
            ("in_intersection_conflict_area", self.other_id),
            self._ego_conflict_region,
        )

    def same_lane_monitor_region(
        self, target_id: int, time_step: int
    ) -> SemanticIntervalSet:
        """Return monitor-aligned longitudinal bounds for ``in_same_lane``.

        The inexpensive domain estimator uses an orientation-independent
        circumcircle cover.  That cover is sound but deliberately too broad
        to generate a useful ``not in_same_lane`` LP bound for adjacent-lane
        traffic.  Constraint extraction needs the sharper robustness-zero
        boundary derived from the monitor's complete rectangular occupancy.
        It is built lazily and cached per target assignment/time step, so the
        extra work is paid only when SAT actually selects the negative
        literal.
        """
        return self._same_lane_robustness_region(
            int(target_id), int(time_step)
        )

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Compact timing/cache diagnostics for batch-result accounting."""
        return self.get_diagnostics()

    # ------------------------------------------------------------------
    # Dispatch and fixed gates
    # ------------------------------------------------------------------
    def _estimate_frame_uncached(
        self,
        evaluator: Any,
        prop_name: str,

        time_step: int,
        vehicle_ids: Tuple[int, ...],
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        name = _predicate_name(evaluator)
        if evaluator is None:
            return FramePredicateEstimate(UNKNOWN_DOMAIN, "missing_evaluator")

        if name in self.ACCELERATION_NAMES:
            return estimate_acceleration_predicate_frame(
                evaluator,
                time_step,
                vehicle_ids,
                ego_id=self.ego_id,
                world=self.world,
                reachable_acceleration_by_time=(
                    self.reachable_acceleration_by_time
                ),
            )

        if name == "in_same_lane":
            return self._same_lane_frame(
                evaluator, time_step, vehicle_ids, reachable
            )

        if name == "in_front_of":
            return self._in_front_of_frame(time_step, vehicle_ids, reachable)

        if name == "keeps_safe_distance_prec":
            return self._safe_distance_frame(time_step, vehicle_ids, reachable)

        if name == "cut_in":
            return self._cut_in_frame(evaluator, time_step, vehicle_ids, reachable)

        if hasattr(evaluator, "_turning_ego"):
            return self._turning_composite_frame(
                evaluator, prop_name, time_step, reachable
            )

        if name == "in_intersection_conflict_area" and "__1_0" in prop_name:
            fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
            return FramePredicateEstimate(fixed, "fixed_target_conflict")

        if name in {
            "at_traffic_sign_stop",
            "relevant_traffic_light",
            "on_lanelet_with_type_intersection",
        }:
            if vehicle_ids != (self.ego_id,):
                fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
                return FramePredicateEstimate(fixed, "fixed_non_ego_predicate")
            region = self.region_for(evaluator, prop_name, time_step)
            return self._continuous_region_frame(
                evaluator, name, region, reachable
            )

        if name == "in_intersection_conflict_area":
            if (
                len(vehicle_ids) != 2
                or vehicle_ids[0] != self.ego_id
                or self.other_id is None
                or vehicle_ids[1] != self.other_id
            ):
                fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
                return FramePredicateEstimate(fixed, "fixed_target_conflict")
            region = self.region_for(evaluator, prop_name, time_step)
            return self._continuous_region_frame(
                evaluator, name, region, reachable
            )

        if name == "in_standstill":
            return self._standstill_frame(evaluator, time_step)

        if name == "on_incoming_left_of":
            return self._on_incoming_left_of_frame(
                evaluator, time_step, reachable
            )

        if name == "causes_braking_intersection":
            return self._causes_braking_frame(
                evaluator, time_step, reachable
            )

        if name == "stop_line_in_front":
            return self._stop_line_frame(evaluator, reachable)

        region = self.region_for(evaluator, prop_name, time_step)
        if region.source.startswith("unsupported:"):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, region.source, region=region
            )
        result, diagnostics = self._classify_region_estimate(region, reachable)
        return FramePredicateEstimate(
            result, region.source, region=region, diagnostics=diagnostics
        )

    def _region_fallback_frame(
        self,
        predicate_name: str,
        region: SemanticIntervalSet,
        reachable: Optional[Tuple[float, float]],
        *,
        reason: str,
    ) -> FramePredicateEstimate:
        """Classify with the retained Boolean inner/outer implementation."""
        domain, diagnostics = self._classify_region_estimate(region, reachable)
        return FramePredicateEstimate(
            domain,
            "region_fallback",
            region=region,
            diagnostics={
                "predicate": str(predicate_name),
                "fallback_reason": str(reason),
                "fallback_source": region.source,
                **diagnostics,
            },
        )

    def _longitudinal_uncertainty(self, evaluator: Any) -> float:
        """Return the cached metric-to-robustness uncertainty conversion."""
        key = (self._semantic_evaluator_key(evaluator), float(self.uncertainty))
        cache = self._scaled_longitudinal_uncertainty_cache
        if key not in cache:
            cache[key] = _scaled_longitudinal_uncertainty(
                evaluator, self.uncertainty
            )
        return cache[key]

    def _continuous_region_frame(
        self,
        evaluator: Any,
        predicate_name: str,
        region: SemanticIntervalSet,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Estimate a lanelet/conflict truth region by signed distance."""
        if region.complete and not region.outer_true:
            return FramePredicateEstimate(
                FALSE_DOMAIN,
                "constant_false_topology",
                region=region,
                diagnostics={
                    "predicate": str(predicate_name),
                    "topology_source": region.source,
                },
            )
        if not region.complete:
            return self._region_fallback_frame(
                predicate_name, region, reachable, reason="incomplete_geometry"
            )
        nominal = region.diagnostics.get("nominal_intervals")
        if nominal is None:
            return self._region_fallback_frame(
                predicate_name, region, reachable, reason="nominal_geometry_missing"
            )
        try:
            geometry_key = (
                "signed_distance",
                str(predicate_name),
                id(region),
            )
            if geometry_key not in self._signed_distance_geometry_cache:
                self._signed_distance_geometry_cache[geometry_key] = (
                    _SignedDistanceGeometry.from_intervals(nominal)
                )
            geometry = self._signed_distance_geometry_cache[geometry_key]
            if geometry is None or not geometry.components:
                return FramePredicateEstimate(
                    FALSE_DOMAIN,
                    "constant_false_topology",
                    region=region,
                    diagnostics={
                        "predicate": str(predicate_name),
                        "topology_source": region.source,
                    },
                )
            robustness_min, robustness_max, points = (
                _signed_distance_robustness_range(
                    geometry, reachable, evaluator
                )
            )
            scaled_uncertainty = self._longitudinal_uncertainty(evaluator)
            domain = classify_robustness_range(
                robustness_min,
                robustness_max,
                uncertainty=scaled_uncertainty,
            )
            if str(predicate_name) == "in_intersection_conflict_area":
                # The monitor assigns the complete oriented vehicle shape to
                # conflict lanelets.  The nominal longitudinal interval is
                # exact only away from the lateral/rotation ambiguity band
                # certified by the retained inner/outer construction.  Do
                # not let a one-dimensional signed distance collapse that
                # band to a singleton: this is precisely a geometry-
                # ambiguity fallback, not a nominal-value override.
                fallback_domain, _ = self._classify_region_estimate(
                    region, reachable
                )
                if domain != fallback_domain:
                    return self._region_fallback_frame(
                        predicate_name,
                        region,
                        reachable,
                        reason="conflict_occupancy_ambiguity",
                    )
        except Exception as exc:
            return self._region_fallback_frame(
                predicate_name,
                region,
                reachable,
                reason=type(exc).__name__,
            )
        diagnostics = {
            "predicate": str(predicate_name),
            "nominal_source": region.source,
            "component_count": len(geometry.components),
            "robustness_min": float(robustness_min),
            "robustness_max": float(robustness_max),
            "scaled_uncertainty": float(scaled_uncertainty),
        }
        if os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
            diagnostics.update(
                {
                    "component_midpoints": list(geometry.component_midpoints),
                    "gap_midpoints": list(geometry.gap_midpoints),
                    "critical_points": list(points),
                }
            )
        return FramePredicateEstimate(
            domain,
            "continuous_signed_distance",
            region=region,
            diagnostics=diagnostics,
        )

    def _turning_composite_frame(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        quantitative = self._turning_composite_quantitative_estimate(
            evaluator, prop_name, time_step, reachable
        )
        if quantitative is not None:
            return quantitative
        return self._turning_composite_region_fallback(
            evaluator, prop_name, time_step, reachable
        )

    def _turning_composite_quantitative_estimate(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> Optional[FramePredicateEstimate]:
        """Estimate a turning conjunction from one ego tent and fixed caps."""
        region = self._turning_region(evaluator._turning_ego)
        if region.complete and not region.outer_true:
            return FramePredicateEstimate(
                FALSE_DOMAIN,
                "constant_false_topology",
                region=region,
                diagnostics={
                    "predicate": _predicate_name(evaluator),
                    "turning_source": region.source,
                },
            )
        nominal = region.diagnostics.get("nominal_intervals")
        if (
            not region.complete
            or nominal is None
            or reachable is None
            or self.other_id is None
        ):
            return None
        try:
            turning_name = _predicate_name(evaluator._turning_ego)
            geometry_key = (
                "turning_signed_distance",
                turning_name,
                id(region),
            )
            if geometry_key not in self._signed_distance_geometry_cache:
                self._signed_distance_geometry_cache[geometry_key] = (
                    _SignedDistanceGeometry.from_intervals(nominal)
                )
            geometry = self._signed_distance_geometry_cache[geometry_key]
            if geometry is None or not geometry.components:
                return FramePredicateEstimate(
                    FALSE_DOMAIN,
                    "constant_false_topology",
                    region=region,
                    diagnostics={"turning_type": turning_name},
                )
            ego_lower, ego_upper, points = _signed_distance_robustness_range(
                geometry, reachable, evaluator._turning_ego
            )
            scaled_uncertainty = self._longitudinal_uncertainty(
                evaluator._turning_ego
            )

            gate_values = {
                "target_turning": self._fixed_robustness_value(
                    evaluator._turning_target,
                    time_step,
                    (self.other_id,),
                )
            }
            if hasattr(evaluator, "_same_priority"):
                gate_values["same_priority"] = self._fixed_robustness_value(
                    evaluator._same_priority,
                    time_step,
                    (self.ego_id, self.other_id),
                    cache_across_time=True,
                )
            elif hasattr(evaluator, "_target_has_priority"):
                gate_values["target_has_priority"] = (
                    self._fixed_robustness_value(
                        evaluator._target_has_priority,
                        time_step,
                        (self.other_id, self.ego_id),
                        cache_across_time=True,
                    )
                )
            else:
                return None

            if hasattr(evaluator, "_on_oncoming_of"):
                oncoming = self._fixed_robustness_value(
                    evaluator._on_oncoming_of,
                    time_step,
                    (self.other_id, self.ego_id),
                    cache_across_time=True,
                )
                if oncoming is None:
                    return None
                composite_name = (
                    f"{_predicate_name(evaluator)} {str(prop_name).lower()}"
                )
                if "not_oncoming" in composite_name:
                    oncoming = -oncoming
                    gate_values["not_oncoming"] = oncoming
                else:
                    gate_values["oncoming"] = oncoming

            if any(value is None for value in gate_values.values()):
                return None
            finite_gate_values = {
                key: float(value) for key, value in gate_values.items()
            }
            if not all(
                math.isfinite(value) for value in finite_gate_values.values()
            ):
                return None
            fixed_cap = min(finite_gate_values.values())
            if fixed_cap < 0.0:
                return FramePredicateEstimate(
                    FALSE_DOMAIN,
                    "constant_false_gate",
                    region=region,
                    gate_domain=FALSE_DOMAIN,
                    diagnostics={
                        "turning_type": turning_name,
                        "fixed_components": finite_gate_values,
                        "fixed_cap": float(fixed_cap),
                    },
                )
            robustness_min = min(ego_lower, fixed_cap)
            robustness_max = min(ego_upper, fixed_cap)
            domain = classify_robustness_range(
                robustness_min,
                robustness_max,
                uncertainty=scaled_uncertainty,
            )
        except Exception:
            return None

        diagnostics = {
            "turning_type": turning_name,
            "component_count": len(geometry.components),
            "fixed_cap": float(fixed_cap),
            "robustness_min": float(robustness_min),
            "robustness_max": float(robustness_max),
            "scaled_uncertainty": float(scaled_uncertainty),
        }
        if os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
            diagnostics.update(
                {
                    "component_midpoints": list(geometry.component_midpoints),
                    "gap_midpoints": list(geometry.gap_midpoints),
                    "critical_points": list(points),
                    "ego_turning_robustness_lower": float(ego_lower),
                    "ego_turning_robustness_upper": float(ego_upper),
                    "fixed_components": finite_gate_values,
                }
            )
        return FramePredicateEstimate(
            domain,
            "continuous_turning_composite",
            region=region,
            gate_domain=TRUE_DOMAIN,
            diagnostics=diagnostics,
        )

    def _turning_composite_region_fallback(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Retained Boolean/three-valued turning-composite implementation."""
        region = self._turning_region(evaluator._turning_ego)
        spatial, spatial_diagnostics = self._classify_region_estimate(
            region, reachable
        )
        if spatial == FALSE_DOMAIN:
            return FramePredicateEstimate(
                FALSE_DOMAIN,
                "region_fallback",
                region=region,
                gate_domain=UNKNOWN_DOMAIN,
                diagnostics={
                    "fallback_source": "turning_composite:spatial_false",
                    "spatial_domain": [0],
                    "gate_skipped": True,
                    **spatial_diagnostics,
                },
            )
        gate = self._turning_gate_domain(
            evaluator, prop_name, time_step, reachable
        )
        if gate == FALSE_DOMAIN or spatial == FALSE_DOMAIN:
            combined = FALSE_DOMAIN
        elif gate == TRUE_DOMAIN and spatial == TRUE_DOMAIN:
            combined = TRUE_DOMAIN
        else:
            combined = UNKNOWN_DOMAIN
        return FramePredicateEstimate(
            combined,
            "region_fallback",
            region=region,
            gate_domain=gate,
            diagnostics={
                "fallback_source": "turning_composite",
                "spatial_domain": sorted(spatial),
                **spatial_diagnostics,
            },
        )

    def _turning_gate_domain(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        reachable: Optional[Tuple[float, float]] = None,
    ) -> TruthDomain:
        if self.other_id is None:
            return UNKNOWN_DOMAIN

        # Evaluate the cheapest/shared fixed facts first.  False is absorbing
        # for conjunction, so later priority/oncoming cache lookups can be
        # skipped without changing the three-valued result.
        fixed_domain = self._fixed_boolean_domain
        gate_parts = [
            fixed_domain(
                evaluator._turning_target, time_step, (self.other_id,)
            )
        ]
        if gate_parts[-1] == FALSE_DOMAIN:
            return FALSE_DOMAIN
        if hasattr(evaluator, "_same_priority"):
            gate_parts.append(
                self._topology_region_domain(
                    evaluator._same_priority,
                    time_step,
                    (self.ego_id, self.other_id),
                    reachable,
                )
            )
        elif hasattr(evaluator, "_target_has_priority"):
            gate_parts.append(
                self._topology_region_domain(
                    evaluator._target_has_priority,
                    time_step,
                    (self.other_id, self.ego_id),
                    reachable,
                )
            )
        else:
            gate_parts.append(UNKNOWN_DOMAIN)
        if gate_parts[-1] == FALSE_DOMAIN:
            return FALSE_DOMAIN
        if hasattr(evaluator, "_on_oncoming_of"):
            oncoming = self._topology_region_domain(
                evaluator._on_oncoming_of,
                time_step,
                (self.other_id, self.ego_id),
                reachable,
            )
            if "not_oncoming" in prop_name.lower():
                oncoming = self._negate_domain(oncoming)
            gate_parts.append(oncoming)

        return self._and_domains(gate_parts)

    def _topology_region_domain(
        self,
        evaluator: Any,
        time_step: int,
        vehicle_ids: Sequence[int],
        reachable: Optional[Tuple[float, float]],
    ) -> TruthDomain:
        """Classify a route-topology Boolean as a path-wide semantic region.

        The underlying monitor predicates (same/has priority and oncoming)
        depend on fixed incoming/successor topology, not ego progress.  A true
        relation therefore owns the complete longitudinal path; a false one
        owns the empty set.  The result still goes through the common
        reachable-region classifier instead of being injected as a sampled
        monitor truth value.
        """
        key = (
            "topology_constant",
            self._semantic_evaluator_key(evaluator),
            tuple(int(item) for item in vehicle_ids),
        )

        def build() -> SemanticIntervalSet:
            try:
                value = bool(
                    evaluator.evaluate_boolean(
                        self.world, int(time_step), list(vehicle_ids)
                    )
                )
            except Exception as exc:
                return SemanticIntervalSet.unknown(
                    "topology_constant:error", error=type(exc).__name__
                )
            if not value:
                return SemanticIntervalSet.empty("topology_constant:false")
            return SemanticIntervalSet(
                complete=True,
                source="topology_constant:true",
                diagnostics={"constant_true": True},
            )

        return self._classify_region(self._cached_region(key, build), reachable)

    def _fixed_boolean_domain(
        self, evaluator: Any, time_step: int, vehicle_ids: Sequence[int]
    ) -> TruthDomain:
        """Evaluate an immutable Boolean/topology gate exactly.

        Turning-priority composites are Boolean in critical-hybrid mode.  The
        target trajectory, priority relation, and oncoming topology are not
        changed by VP, so their monitor Boolean value at a frame is a sound
        fixed gate and avoids importing continuous robustness conventions into
        the complete conjunction.
        """
        name = _predicate_name(evaluator)
        key = (
            "boolean",
            self._semantic_evaluator_key(evaluator),
            int(time_step),
            tuple(vehicle_ids),
        )
        if key in self._fixed_cache:
            return self._fixed_cache[key]
        started = time.perf_counter()
        result = UNKNOWN_DOMAIN
        try:
            value = evaluator.evaluate_boolean(
                self.world, int(time_step), list(vehicle_ids)
            )
            result = TRUE_DOMAIN if bool(value) else FALSE_DOMAIN
        except Exception:
            result = UNKNOWN_DOMAIN
        self._timing["fixed_gate"] += time.perf_counter() - started
        self._fixed_cache[key] = result
        return result

    def _fixed_domain(
        self, evaluator: Any, time_step: int, vehicle_ids: Sequence[int]
    ) -> TruthDomain:
        """Evaluate a fixed target/topology fact, with cache-first lookup."""
        name = _predicate_name(evaluator)
        key = (
            "robustness",
            self._semantic_evaluator_key(evaluator),
            int(time_step),
            tuple(vehicle_ids),
        )
        if key in self._fixed_cache:
            return self._fixed_cache[key]
        started = time.perf_counter()
        result = UNKNOWN_DOMAIN
        if evaluator is not None and vehicle_ids:
            value = None
            try:
                vehicle = self.world.vehicle_by_id(int(vehicle_ids[0]))
                value = vehicle.predicate_cache.get_robustness(
                    int(time_step),
                    evaluator.predicate_name,
                    tuple(int(item) for item in vehicle_ids[1:]),
                )
            except Exception:
                value = None
            if value is None:
                try:
                    value = evaluator.evaluate_robustness(
                        self.world, int(time_step), list(vehicle_ids)
                    )
                except Exception:
                    value = None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isnan(numeric):
                if numeric > 0.0:
                    result = TRUE_DOMAIN
                elif numeric < 0.0:
                    result = FALSE_DOMAIN
                else:
                    # Robustness zero lies exactly on a monitor boundary.  In
                    # particular, some Boolean evaluators use >= 0 while SAT
                    # proposition construction uses a strict sign test.  It is
                    # unsafe to freeze either polarity here.
                    result = UNKNOWN_DOMAIN
        self._timing["fixed_gate"] += time.perf_counter() - started
        self._fixed_cache[key] = result
        return result

    def _fixed_robustness_value(
        self,
        evaluator: Any,
        time_step: int,
        vehicle_ids: Sequence[int],
        *,
        cache_across_time: bool = False,
    ) -> Optional[float]:
        """Evaluate and cache one target/topology robustness scalar."""
        ids = tuple(int(item) for item in vehicle_ids)
        semantic_key = self._semantic_evaluator_key(evaluator)
        key = (
            "robustness_value",
            semantic_key,
            int(time_step),
            ids,
        )
        if key in self._fixed_cache:
            cached = self._fixed_cache[key]
            return None if cached is None else float(cached)
        static_key = ("robustness_value_static", semantic_key, ids)
        if cache_across_time and static_key in self._fixed_cache:
            cached = self._fixed_cache[static_key]
            self._fixed_cache[key] = cached
            return None if cached is None else float(cached)
        started = time.perf_counter()
        value = None
        if evaluator is not None and ids:
            try:
                vehicle = self.world.vehicle_by_id(ids[0])
                value = vehicle.predicate_cache.get_robustness(
                    int(time_step), evaluator.predicate_name, ids[1:]
                )
            except Exception:
                value = None
            if value is None:
                try:
                    value = evaluator.evaluate_robustness(
                        self.world, int(time_step), list(ids)
                    )
                except Exception:
                    value = None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        result = numeric if math.isfinite(numeric) else None
        self._timing["fixed_gate"] += time.perf_counter() - started
        self._fixed_cache[key] = result
        if cache_across_time:
            # Priority and oncoming evaluators depend only on immutable
            # incoming/successor topology.  Keep the required per-frame alias
            # above while sharing the underlying scalar across all frames and
            # equivalent composite evaluator instances.
            self._fixed_cache[static_key] = result
        return result

    @staticmethod
    def _and_domains(domains: Iterable[TruthDomain]) -> TruthDomain:
        domains = tuple(domains)
        if any(item == FALSE_DOMAIN for item in domains):
            return FALSE_DOMAIN
        if domains and all(item == TRUE_DOMAIN for item in domains):
            return TRUE_DOMAIN
        return UNKNOWN_DOMAIN

    @staticmethod
    def _negate_domain(domain: TruthDomain) -> TruthDomain:
        return frozenset(1 - value for value in domain)

    @staticmethod
    def _classify_region(
        region: SemanticIntervalSet,
        reachable: Optional[Sequence[float]],
    ) -> TruthDomain:
        """Classify geometry while preserving path/topology constants.

        A missing reachable interval normally means ``unknown``.  Constant
        map facts are different: an empty complete true set proves false and
        a region explicitly tagged ``constant_true`` proves true independently
        of longitudinal progress.  This matters when a non-monotone recorded
        trajectory deliberately disables ``reachable_by_time``.
        """
        if bool(region.diagnostics.get("constant_true", False)):
            return TRUE_DOMAIN
        if region.complete and not region.outer_true:
            return FALSE_DOMAIN
        if reachable is None:
            return UNKNOWN_DOMAIN
        return classify_region_piecewise(region, reachable).domain

    @staticmethod
    def _classify_region_estimate(
        region: SemanticIntervalSet,
        reachable: Optional[Sequence[float]],
    ) -> Tuple[TruthDomain, Dict[str, Any]]:
        """Return a Boolean-region domain with uniform compact diagnostics."""
        if bool(region.diagnostics.get("constant_true", False)):
            return TRUE_DOMAIN, {
                "analysis_mode": "piecewise_boolean_region",
                "analysis_axis": "s",
                "partition_points": [],
                "segment_count": 0,
            }
        if region.complete and not region.outer_true:
            return FALSE_DOMAIN, {
                "analysis_mode": "piecewise_boolean_region",
                "analysis_axis": "s",
                "partition_points": [],
                "segment_count": 0,
            }
        if reachable is None:
            return UNKNOWN_DOMAIN, {
                "analysis_mode": "piecewise_boolean_region",
                "analysis_axis": "s",
                "partition_points": [],
                "segment_count": 0,
            }
        estimate = classify_region_piecewise(region, reachable)
        return estimate.domain, piecewise_diagnostics(
            estimate,
            analysis_mode="piecewise_boolean_region",
            analysis_axis="s",
            include_segments=bool(
                os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG")
            ),
        )

    # ------------------------------------------------------------------
    # Definition-driven static regions
    # ------------------------------------------------------------------
    def _cached_region(self, key: Any, build: Any) -> SemanticIntervalSet:
        if key in self._region_cache:
            return self._region_cache[key]
        started = time.perf_counter()
        try:
            region = build()
        except Exception as exc:
            region = SemanticIntervalSet.unknown(
                f"geometry_error:{key}", error=type(exc).__name__
            )
        self._timing["region_build"] += time.perf_counter() - started
        self._region_cache[key] = region
        return region

    def _stop_line_region(self, evaluator: Any) -> SemanticIntervalSet:
        intervals = []
        candidates = []
        route_stop_line_count = 0
        route_ids = set(self.ego.ref_path_lane.contained_lanelets)
        network = self.world.road_network.lanelet_network
        d_sl = float(getattr(evaluator, "config", {}).get("d_sl", 0.0))
        if d_sl < 0.0:
            return SemanticIntervalSet.unknown("stop_line", invalid_d_sl=d_sl)
        for lanelet_id in route_ids:
            lanelet = network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is None:
                continue
            route_stop_line_count += 1
            try:
                stop_lane_s = min(
                    self.lanelet_clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.start
                    )[0],
                    self.lanelet_clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.end
                    )[0],
                )
                stop_s = self._progress(stop_lane_s)
            except Exception:
                return SemanticIntervalSet.unknown(
                    "stop_line", failed_lanelet=int(lanelet_id)
                )
            stop_window = (
                stop_s - self._front_extent - d_sl,
                stop_s - self._front_extent,
            )
            s_star = stop_s - self._front_extent - 0.5 * d_sl
            lane_bounds = self._lanelet_truth_bounds(lanelet_id)
            if lane_bounds is None:
                return SemanticIntervalSet.unknown(
                    "stop_line", failed_lanelet=int(lanelet_id)
                )
            overlap = self._intersection(stop_window, lane_bounds)
            if overlap is not None:
                intervals.append(overlap)
                lane_inner = (
                    float(lane_bounds[0]) + self.uncertainty,
                    float(lane_bounds[1]) - self.uncertainty,
                )
                if lane_inner[0] > lane_inner[1]:
                    lane_inner = None
                candidates.append(
                    {
                        "lanelet_id": int(lanelet_id),
                        "stop_s": float(stop_s),
                        "s_star": float(s_star),
                        "stop_window": tuple(map(float, stop_window)),
                        "lane_interval": tuple(map(float, lane_bounds)),
                        "certified_lane_interval": lane_inner,
                        "truth_overlap": tuple(map(float, overlap)),
                    }
                )
        return SemanticIntervalSet.from_nominal(
            intervals,
            uncertainty=self.uncertainty,
            source="stop_line",
            diagnostics={
                "candidate_count": len(candidates),
                "route_stop_line_count": int(route_stop_line_count),
                "region_interval_count": len(intervals),
                "d_sl": float(d_sl),
                "front_extent": float(self._front_extent),
                "piecewise_monotone": True,
                "candidates": tuple(candidates),
            },
        )

    @staticmethod
    def _stop_line_robustness_bounds(
        reachable: Sequence[float],
        *,
        stop_s: float,
        front_extent: float,
        d_sl: float,
        s_star: float,
    ) -> Tuple[float, float, int]:
        """Bound fixed-offset stop-line robustness at monotone endpoints.

        On the left of ``s_star`` the robustness is increasing; on the right
        it is decreasing.  The estimate is exact under the VP path model's
        fixed longitudinal front-offset assumption; callers retain the
        configured uncertainty band for occupancy/orientation mismatch.
        """
        def robustness(s_value: float) -> float:
            delta_s = float(stop_s) - (float(s_value) + float(front_extent))
            return min(float(d_sl) - delta_s, delta_s)

        estimate = estimate_piecewise_monotone_1d(
            reachable,
            (s_star,),
            robustness,
            uncertainty=0.0,
        )
        return (
            min(segment.robustness_min for segment in estimate.segments),
            max(segment.robustness_max for segment in estimate.segments),
            len(estimate.segments),
        )

    def _stop_line_piecewise_estimate(
        self,
        reachable: Sequence[float],
        *,
        stop_s: float,
        front_extent: float,
        d_sl: float,
        s_star: float,
    ) -> PiecewiseEstimate:
        """Estimate one stop-line candidate under a certified lane gate."""

        def robustness(s_value: float) -> float:
            delta_s = float(stop_s) - (float(s_value) + float(front_extent))
            return min(float(d_sl) - delta_s, delta_s)

        return estimate_piecewise_monotone_1d(
            reachable,
            (s_star,),
            robustness,
            uncertainty=self.uncertainty,
        )

    def _stop_line_frame(
        self,
        evaluator: Any,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Estimate ``stop_line_in_front`` from certified piecewise geometry.

        A single candidate uses quantitative endpoint bounds only when its
        lane/lateral gate is certified throughout the reachable interval.
        Multiple candidates and intervals crossing a gate boundary use the
        union of certified Boolean truth regions instead; this accounts for
        different stop-line windows jointly covering the reachable interval.
        """
        region = self._cached_region(
            ("stop_line_in_front", id(evaluator)),
            lambda: self._stop_line_region(evaluator),
        )
        if reachable is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "stop_line:piecewise", region=region
            )
        if not region.complete:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "stop_line:piecewise", region=region
            )

        candidates = tuple(region.diagnostics.get("candidates", ()))
        d_sl = float(region.diagnostics.get("d_sl", math.nan))
        front_extent = float(
            region.diagnostics.get("front_extent", self._front_extent)
        )
        reachable_interval = tuple(
            sorted((float(reachable[0]), float(reachable[1])))
        )
        estimate = None
        analysis_mode = "piecewise_boolean_region"
        if not candidates:
            # No relevant stop line exists on the fixed reference path.
            piecewise_domain = FALSE_DOMAIN
        elif len(candidates) == 1:
            candidate = candidates[0]
            certified_lane = candidate.get("certified_lane_interval")
            gate_certified = (
                certified_lane is not None
                and float(certified_lane[0]) <= reachable_interval[0]
                and reachable_interval[1] <= float(certified_lane[1])
            )
            if gate_certified:
                analysis_mode = "piecewise_monotone"
                estimate = self._stop_line_piecewise_estimate(
                    reachable_interval,
                    stop_s=float(candidate["stop_s"]),
                    front_extent=front_extent,
                    d_sl=d_sl,
                    s_star=float(candidate["s_star"]),
                )
            else:
                estimate = classify_region_piecewise(region, reachable_interval)
            piecewise_domain = estimate.domain
        else:
            estimate = classify_region_piecewise(region, reachable_interval)
            piecewise_domain = estimate.domain

        diagnostics = {
            "piecewise_domain": sorted(piecewise_domain),
            "candidate_count": len(candidates),
        }
        if estimate is not None:
            diagnostics.update(
                piecewise_diagnostics(
                    estimate,
                    analysis_mode=analysis_mode,
                    analysis_axis="s",
                    include_segments=bool(
                        os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG")
                    ),
                )
            )

        return FramePredicateEstimate(
            piecewise_domain,
            "stop_line:piecewise",
            region=region,
            diagnostics=diagnostics,
        )

    def _stop_sign_region(self) -> SemanticIntervalSet:
        try:
            from crmonitor.predicates import utils as predicate_utils
        except ImportError:
            return SemanticIntervalSet.unknown("stop_sign:monitor_utils_missing")
        stop_sign_lanelet_ids = []
        evaluator_stop_id = None
        # Avoid constructing another evaluator solely to get the enum value.
        try:
            from commonroad.scenario.traffic_sign import TrafficSignIDGermany

            evaluator_stop_id = TrafficSignIDGermany.STOP
        except ImportError:
            return SemanticIntervalSet.unknown("stop_sign:enum_missing")
        for lanelet_id in set(self.ego.ref_path_lane.contained_lanelets):
            try:
                elements = predicate_utils.traffic_sign(
                    lanelet_id, evaluator_stop_id, self.world.road_network
                )
            except Exception:
                return SemanticIntervalSet.unknown(
                    "stop_sign", failed_lanelet=int(lanelet_id)
                )
            if elements is not None:
                stop_sign_lanelet_ids.append(int(lanelet_id))
        if not stop_sign_lanelet_ids:
            return SemanticIntervalSet.empty("stop_sign:route_constant_false")
        # The Boolean evaluator checks current vehicle occupancy of the
        # sign-carrying lanelet.  Merely having such a lanelet somewhere on
        # the route does not make the predicate globally true.
        return self._lanelet_union_region(
            stop_sign_lanelet_ids, "stop_sign:boolean_lanelet_occupancy"
        )

    def _traffic_light_region(self) -> SemanticIntervalSet:
        road_network = self.world.road_network
        try:
            reachable_ids = set(
                road_network.get_reach_suc_cache(self.ego.lanelets_dir[0])
            )
        except Exception:
            return SemanticIntervalSet.unknown("relevant_traffic_light")
        ids = []
        route_ids = set(self.ego.ref_path_lane.contained_lanelets)
        network = road_network.lanelet_network
        for lanelet_id in route_ids.intersection(reachable_ids):
            lanelet = network.find_lanelet_by_id(lanelet_id)
            if not lanelet.traffic_lights:
                continue
            # The monitor currently supports one light per lanelet.  If the
            # map violates that assumption, stay conservative.
            if len(lanelet.traffic_lights) != 1:
                return SemanticIntervalSet.unknown(
                    "relevant_traffic_light", ambiguous_lanelet=int(lanelet_id)
                )
            light = network.find_traffic_light_by_id(
                next(iter(lanelet.traffic_lights))
            )
            if bool(light.active):
                ids.append(lanelet_id)
        return self._lanelet_union_region(ids, "relevant_traffic_light")

    def _intersection_lanelet_region(self) -> SemanticIntervalSet:
        network = self.world.road_network.lanelet_network
        ids = []
        for lanelet_id in set(self.ego.ref_path_lane.contained_lanelets):
            lanelet = network.find_lanelet_by_id(lanelet_id)
            if any(
                str(getattr(item, "value", item)).lower() == "intersection"
                for item in lanelet.lanelet_type
            ):
                ids.append(lanelet_id)
        return self._lanelet_union_region(ids, "intersection_lanelet")

    def _ego_conflict_region(self) -> SemanticIntervalSet:
        if self.other_id is None:
            return SemanticIntervalSet.unknown("ego_conflict:no_target")
        target = self.world.vehicle_by_id(self.other_id)
        try:
            incoming_ego = self.ego.incoming_intersection
            incoming_target = target.incoming_intersection
            adjacent = self.world.road_network.adjacent_lanelets(
                incoming_ego.incoming_lanelets
            )
            if adjacent.intersection(incoming_target.incoming_lanelets):
                return SemanticIntervalSet.empty("ego_conflict:adjacent_incoming")
        except Exception:
            return SemanticIntervalSet.unknown("ego_conflict:incoming_topology")

        return self._ego_conflict_boolean_region(target)

    def _ego_conflict_boolean_region(self, target: Any) -> SemanticIntervalSet:
        """Bound the monitor's lanelet-assignment conflict predicate.

        ``PredInIntersectionConflictArea.evaluate_boolean`` first selects the
        target-route intersection lanelets and removes ego-direction
        lanelets.  It is true only while the ego shape is assigned to one of
        the remaining lanelets.  We reproduce that topology exactly and use
        two geometric bounds along the fixed VP path:

        * center inside the selected polygons is a sufficient truth region;
        * center outside their circumradius buffer is a sufficient false
          region for every ego orientation.

        The band between them deliberately stays unknown.  This avoids both
        fitting recorded predicate samples and claiming an exact Boolean flip
        from the legacy finite-trajectory offset-curve approximation.
        """
        network = self.world.road_network.lanelet_network
        ego_direction_ids = set(int(item) for item in self.ego.lanelets_dir)
        conflict_ids = []
        conflict_shapes = []
        try:
            for lanelet_id in target.ref_path_lane.contained_lanelets:
                lanelet_id = int(lanelet_id)
                if lanelet_id in ego_direction_ids:
                    continue
                lanelet = network.find_lanelet_by_id(lanelet_id)
                if LaneletType.INTERSECTION not in lanelet.lanelet_type:
                    continue
                conflict_ids.append(lanelet_id)
                conflict_shapes.append(lanelet.polygon.shapely_object)
        except Exception:
            return SemanticIntervalSet.unknown(
                "ego_conflict:boolean_lanelet_geometry"
            )

        if not conflict_shapes:
            return SemanticIntervalSet.empty(
                "ego_conflict:boolean_no_conflict_lanelet",
                conflict_lanelet_count=0,
            )

        try:
            conflict_polygon = shapely.unary_union(conflict_shapes)
            path_line = LineString(np.asarray(self.ref_path, dtype=float)[:, :2])
            half_length = 0.5 * float(self.ego.shape.length)
            half_width = 0.5 * float(self.ego.shape.width)
            circumradius = math.hypot(half_length, half_width)
            inner_nominal = self._path_polygon_intervals(
                path_line, conflict_polygon
            )
            outer_nominal = self._path_polygon_intervals(
                path_line, conflict_polygon.buffer(circumradius)
            )
            occupancy_nominal = merge_intervals(
                (
                    float(item.lower) - self._front_extent,
                    float(item.upper) + self._rear_extent,
                )
                for item in inner_nominal
            )
        except Exception:
            return SemanticIntervalSet.unknown(
                "ego_conflict:boolean_lanelet_geometry"
            )

        # Shrink sufficient-truth cells and expand the necessary-truth cover
        # by the numerical/projection uncertainty band.
        inner = SemanticIntervalSet.from_nominal(
            inner_nominal,
            uncertainty=self.uncertainty,
            source="ego_conflict:boolean_inner",
        ).inner_true
        outer = SemanticIntervalSet.from_nominal(
            outer_nominal,
            uncertainty=self.uncertainty,
            source="ego_conflict:boolean_outer",
        ).outer_true
        return SemanticIntervalSet(
            inner_true=inner,
            outer_true=outer,
            complete=True,
            source="ego_conflict:boolean_lanelet_assignment",
            diagnostics={
                "conflict_lanelet_count": len(conflict_ids),
                "circumradius": float(circumradius),
                "inner_component_count": len(inner_nominal),
                "outer_component_count": len(outer_nominal),
                "conflict_lanelet_ids": tuple(sorted(conflict_ids)),
                # Nominal occupancy follows the fixed ego path through the
                # selected conflict polygons.  The legacy Boolean inner/outer
                # regions above intentionally remain unchanged as fallback.
                # A path which misses the nominal polygon but intersects its
                # vehicle-size cover has no reliable one-dimensional nominal
                # occupancy interval.  Omitting the field deliberately sends
                # that ambiguous geometry through the old conservative
                # inner/outer classifier instead of claiming constant false.
                **(
                    {
                        "nominal_intervals": tuple(
                            (float(item.lower), float(item.upper))
                            for item in occupancy_nominal
                        )
                    }
                    if occupancy_nominal or not outer_nominal
                    else {}
                ),
            },
        )

    def _path_polygon_intervals(
        self, path_line: Any, polygon: Any
    ) -> Tuple[ClosedInterval, ...]:
        """Project every connected path/polygon intersection to VP progress."""
        intersection = path_line.intersection(polygon)
        intervals = []

        def append_geometry(geometry: Any) -> None:
            if geometry is None or geometry.is_empty:
                return
            geometry_type = geometry.geom_type
            if geometry_type in ("LineString", "LinearRing"):
                coordinates = list(geometry.coords)
                if not coordinates:
                    return
                endpoints = (coordinates[0], coordinates[-1])
                progress = []
                for point in endpoints:
                    progress.append(
                        float(
                            self.trajectory_clcs.convert_to_curvilinear_coords(
                                float(point[0]), float(point[1])
                            )[0]
                        )
                    )
                intervals.append(tuple(sorted(progress)))
                return
            if geometry_type == "Point":
                progress = float(
                    self.trajectory_clcs.convert_to_curvilinear_coords(
                        float(geometry.x), float(geometry.y)
                    )[0]
                )
                intervals.append((progress, progress))
                return
            for part in getattr(geometry, "geoms", ()):
                append_geometry(part)

        append_geometry(intersection)
        return merge_intervals(intervals)

    def _turning_region(self, turning_evaluator: Any) -> SemanticIntervalSet:
        name = _predicate_name(turning_evaluator)
        return self._cached_region(
            ("turning", name),
            lambda: self._turning_region_uncached(name),
        )

    def _turning_region_uncached(self, name: str) -> SemanticIntervalSet:
        incoming = getattr(self.ego, "incoming_intersection", None)
        if incoming is None:
            return SemanticIntervalSet.unknown(f"{name}:no_incoming")
        if name == "turning_right":
            ids = incoming.successors_right
        elif name == "turning_left":
            ids = incoming.successors_left
        elif name == "going_straight":
            ids = incoming.successors_straight
        else:
            return SemanticIntervalSet.unknown(f"turning:unsupported:{name}")
        route_ids = set(self.ego.ref_path_lane.contained_lanelets)
        selected_ids = route_ids.intersection(set(ids))
        if selected_ids:
            return self._lanelet_union_region(selected_ids, f"turning:{name}")

        # IN turning+priority predicates are Boolean conjunctions.  On a
        # fixed route which contains none of the selected turn's successor
        # lanelets, the ego turning atom cannot become true merely by changing
        # longitudinal progress.
        return SemanticIntervalSet.empty(
            f"turning:{name}:boolean_nonactual_route"
        )

    def _lanelet_union_region(
        self, lanelet_ids: Iterable[int], source: str
    ) -> SemanticIntervalSet:
        lanelet_ids = tuple(sorted(set(int(item) for item in lanelet_ids)))
        intervals = []
        for lanelet_id in lanelet_ids:
            bounds = self._lanelet_truth_bounds(lanelet_id)
            if bounds is None:
                return SemanticIntervalSet.unknown(
                    source, failed_lanelet=int(lanelet_id)
                )
            intervals.append(bounds)
        nominal = merge_intervals(intervals)
        return SemanticIntervalSet.from_nominal(
            nominal,
            uncertainty=self.uncertainty,
            source=source,
            diagnostics={
                "lanelet_count": len(lanelet_ids),
                "lanelet_ids": lanelet_ids,
                # This is the direct lanelet/path geometry, not either of the
                # uncertainty-expanded Boolean inner/outer approximations.
                "nominal_intervals": tuple(
                    (float(item.lower), float(item.upper)) for item in nominal
                ),
            },
        )

    # ------------------------------------------------------------------
    # Definition-driven target-time-dependent regions
    # ------------------------------------------------------------------
    def _on_incoming_left_of_frame(
        self,
        evaluator: Any,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Estimate the monitor's continuous incoming-distance robustness.

        The current experiment maps contain one intersection, so normally
        there is at most one topology-compatible candidate.  Its ego distance
        is the tent

        ``min(s-L, U-s)``

        on the fixed VP path.  The target distance is immutable at one frame
        and caps this tent exactly as in the monitor.  Geometry or numerical
        failures retain the former certified Boolean-region implementation.
        """
        quantitative = self._on_incoming_left_of_quantitative_estimate(
            evaluator, int(time_step), reachable
        )
        if quantitative is not None:
            return quantitative
        return self._on_incoming_left_of_region_estimate(
            evaluator, int(time_step), reachable
        )

    def _on_incoming_left_of_quantitative_estimate(
        self,
        evaluator: Any,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> Optional[FramePredicateEstimate]:
        """Return the exact single-candidate robustness range when available."""
        try:
            geometry = self._incoming_left_of_geometry()
            if geometry is None or not bool(geometry.get("complete", False)):
                return None
            candidates = tuple(geometry.get("candidates", ()))
            if len(candidates) > 1:
                return None

            distance_zero = float(evaluator._scale_lon_dist(0.0))
            distance_uncertainty = max(
                abs(
                    float(evaluator._scale_lon_dist(self.uncertainty))
                    - distance_zero
                ),
                abs(
                    float(evaluator._scale_lon_dist(-self.uncertainty))
                    - distance_zero
                ),
            )
            if not math.isfinite(distance_uncertainty):
                return None

            # The monitor initializes robustness to -1.  With complete
            # topology and no matching left-of candidate this value is exact
            # and independent of ego progress and target state.
            if not candidates:
                domain = classify_robustness_range(
                    -1.0, -1.0, uncertainty=distance_uncertainty
                )
                return FramePredicateEstimate(
                    domain,
                    "on_incoming_left_of:quantitative",
                    diagnostics={
                        "analysis_mode": "piecewise_monotone",
                        "analysis_axis": "s",
                        "candidate_count": 0,
                        "partition_points": [],
                        "segment_count": 0,
                        "robustness_min": -1.0,
                        "robustness_max": -1.0,
                        "robustness_floor": -1.0,
                    },
                )

            reachable_pair = _finite_pair(reachable)
            if reachable_pair is None or self.other_id is None:
                return None
            candidate = candidates[0]
            lower = float(candidate["ego_lower"])
            upper = float(candidate["ego_upper"])
            lower, upper = sorted((lower, upper))
            target_start = float(candidate["target_start"])
            target_end = float(candidate["target_end"])
            target = self.world.vehicle_by_id(self.other_id)
            target_feature = self._target_front_rear(target, int(time_step))
            if target_feature is None:
                return None
            front_target, rear_target, _ = target_feature
            front_target = float(front_target)
            rear_target = float(rear_target)
            values = (
                lower,
                upper,
                target_start,
                target_end,
                front_target,
                rear_target,
                float(reachable_pair[0]),
                float(reachable_pair[1]),
            )
            if not all(math.isfinite(value) for value in values):
                return None

            target_distance = min(
                front_target - target_start,
                target_end - rear_target,
            )
            target_cap = float(evaluator._scale_lon_dist(target_distance))
            if not math.isfinite(target_cap):
                return None

            reachable_lower, reachable_upper = sorted(reachable_pair)
            peak = 0.5 * (lower + upper)

            def robustness(progress: float) -> float:
                ego_distance = min(
                    float(progress) - lower,
                    upper - float(progress),
                )
                ego_robustness = float(
                    evaluator._scale_lon_dist(ego_distance)
                )
                return max(-1.0, min(ego_robustness, target_cap))

            partition_points = [reachable_lower]
            evaluation_points = [reachable_lower, reachable_upper]
            peak_reachable = reachable_lower <= peak <= reachable_upper
            if reachable_lower < peak < reachable_upper:
                partition_points.append(peak)
                evaluation_points.append(peak)
            partition_points.append(reachable_upper)
            robustness_values = [robustness(point) for point in evaluation_points]
            if not robustness_values or not all(
                math.isfinite(value) for value in robustness_values
            ):
                return None
            robustness_min = min(robustness_values)
            robustness_max = max(robustness_values)
            domain = classify_robustness_range(
                robustness_min,
                robustness_max,
                uncertainty=distance_uncertainty,
            )
            return FramePredicateEstimate(
                domain,
                "on_incoming_left_of:quantitative",
                diagnostics={
                    "analysis_mode": "piecewise_monotone",
                    "analysis_axis": "s",
                    "candidate_count": 1,
                    "partition_points": [
                        float(point) for point in partition_points
                    ],
                    "segment_count": max(0, len(partition_points) - 1),
                    "reachable_interval": [
                        float(reachable_lower),
                        float(reachable_upper),
                    ],
                    "ego_incoming_interval": [lower, upper],
                    "tent_peak": peak,
                    "tent_peak_reachable": bool(peak_reachable),
                    "target_distance": target_distance,
                    "target_cap": target_cap,
                    "robustness_min": robustness_min,
                    "robustness_max": robustness_max,
                    "robustness_floor": -1.0,
                },
            )
        except Exception:
            # Quantitative estimation is an optimization over the existing
            # certified Boolean-region path.  Any unavailable monitor/map
            # input must therefore preserve the established fallback rather
            # than changing the predicate domain or aborting repair.
            return None

    def _on_incoming_left_of_region_estimate(
        self,
        evaluator: Any,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Former Boolean truth-region estimator retained as fallback."""
        key = ("on_incoming_left_of", int(time_step), self.other_id)
        region = self._region_cache.get(key)
        if region is None:
            started = time.perf_counter()
            region = self._build_on_incoming_left_of_region(int(time_step))
            self._timing["region_build"] += time.perf_counter() - started
            self._region_cache[key] = region
        domain, diagnostics = self._classify_region_estimate(region, reachable)
        return FramePredicateEstimate(
            domain, region.source, region=region, diagnostics=diagnostics
        )

    def _build_on_incoming_left_of_region(
        self, time_step: int
    ) -> SemanticIntervalSet:
        geometry = self._incoming_left_of_geometry()
        if geometry is None:
            return SemanticIntervalSet.unknown(
                "on_incoming_left_of:geometry_unavailable"
            )
        candidates = geometry["candidates"]
        complete = bool(geometry["complete"])
        if not candidates and complete:
            return SemanticIntervalSet.empty(
                "on_incoming_left_of:topology_constant_false"
            )

        inner = []
        outer = []
        target = (
            None
            if self.other_id is None
            else self.world.vehicle_by_id(self.other_id)
        )
        target_feature = self._target_front_rear(target, int(time_step))
        front_target = rear_target = None
        if target_feature is not None:
            front_target, rear_target, _ = target_feature

        target_true = target_false = target_boundary = 0
        for candidate in candidates:
            base = candidate["ego_region"]
            if front_target is None or rear_target is None:
                gate = None
            else:
                distance = min(
                    float(front_target) - float(candidate["target_start"]),
                    float(candidate["target_end"]) - float(rear_target),
                )
                if distance > self.uncertainty:
                    gate = True
                    target_true += 1
                elif distance < -self.uncertainty:
                    gate = False
                    target_false += 1
                else:
                    # The monitor uses a non-negative boundary while parts of
                    # the SAT abstraction use a strict sign.  Keep the band
                    # unknown instead of freezing either polarity.
                    gate = None
                    target_boundary += 1
            if gate is True:
                inner.extend(base.inner_true)
                outer.extend(base.outer_true)
            elif gate is None:
                outer.extend(base.outer_true)

        return SemanticIntervalSet(
            inner_true=tuple(inner),
            outer_true=tuple(outer),
            complete=complete,
            source="on_incoming_left_of",
            diagnostics={
                "candidate_count": len(candidates),
                "target_gate_true_count": target_true,
                "target_gate_false_count": target_false,
                "target_gate_boundary_count": target_boundary,
            },
        )

    def _incoming_left_of_geometry(self) -> Optional[Mapping[str, Any]]:
        """Build the route/topology part of ``on_incoming_left_of`` once."""
        if self._incoming_geometry_cache is not None:
            return self._incoming_geometry_cache
        if self.other_id is None:
            return None
        # The monitor defines front/rear in its lane orientation.  If that
        # orientation opposes VP progress, a simple translated interval would
        # swap the vehicle ends; leave the predicate unknown rather than use
        # an unsound bound.
        if getattr(self._progress, "direction", 1.0) < 0.0:
            self._incoming_geometry_cache = {
                "candidates": (),
                "complete": False,
            }
            return self._incoming_geometry_cache
        try:
            from crmonitor.predicates import utils as predicate_utils

            road_network = self.world.road_network
            target = self.world.vehicle_by_id(self.other_id)
            ego_possible = self._possible_route_lanelets(self.ego)
            target_possible = self._possible_route_lanelets(target)
            candidates = []
            complete = True
            for intersection in road_network.lanelet_network.intersections:
                ego_status, ego_data = self._incoming_geometry_for_vehicle(
                    self.ego,
                    intersection,
                    ego_possible,
                    predicate_utils,
                )
                target_status, target_data = self._incoming_geometry_for_vehicle(
                    target,
                    intersection,
                    target_possible,
                    predicate_utils,
                )
                if "ambiguous" in {ego_status, target_status}:
                    complete = False
                    continue
                if ego_status != "ok" or target_status != "ok":
                    continue
                ego_incoming, ego_start, ego_end = ego_data
                target_incoming, target_start, target_end = target_data
                if target_incoming.incoming_id != ego_incoming.left_of:
                    continue
                ego_start_s = self._progress(float(ego_start))
                ego_end_s = self._progress(float(ego_end))
                nominal = tuple(
                    sorted(
                        (
                            ego_start_s - self._front_extent,
                            ego_end_s + self._rear_extent,
                        )
                    )
                )
                candidates.append(
                    {
                        "ego_region": SemanticIntervalSet.from_nominal(
                            [nominal],
                            uncertainty=self.uncertainty,
                            source="on_incoming_left_of:ego_incoming",
                        ),
                        "ego_lower": float(nominal[0]),
                        "ego_upper": float(nominal[1]),
                        "target_start": float(target_start),
                        "target_end": float(target_end),
                    }
                )
            self._incoming_geometry_cache = {
                "candidates": tuple(candidates),
                "complete": complete,
            }
        except Exception:
            # Keep a failed construction retry-free, but do not turn it into a
            # false proof.
            self._incoming_geometry_cache = {
                "candidates": (),
                "complete": False,
            }
        return self._incoming_geometry_cache

    def _possible_route_lanelets(self, vehicle: Any) -> FrozenSet[int]:
        road_network = self.world.road_network
        lanelets_dir = tuple(int(item) for item in vehicle.lanelets_dir)
        if not lanelets_dir:
            raise ValueError("vehicle has no directed route lanelets")
        result = set(lanelets_dir)
        result.update(
            int(item)
            for item in road_network.lanelet_reach_pre(lanelets_dir[0])
        )
        result.update(
            int(item)
            for item in road_network.lanelet_reach_suc(lanelets_dir[-1])
        )
        return frozenset(result)

    def _incoming_geometry_for_vehicle(
        self,
        vehicle: Any,
        intersection: Any,
        possible_lanelets: FrozenSet[int],
        predicate_utils: Any,
    ) -> Tuple[str, Optional[Tuple[Any, float, float]]]:
        incomings = [
            incoming
            for incoming in intersection.incomings
            if set(incoming.incoming_lanelets).intersection(possible_lanelets)
        ]
        if not incomings:
            return "absent", None
        if len(incomings) > 1:
            route_ids = set(vehicle.ref_path_lane.contained_lanelets)
            incomings = [
                incoming
                for incoming in incomings
                if set(incoming.incoming_lanelets).intersection(route_ids)
            ]
            if len(incomings) != 1:
                return "ambiguous", None
        incoming = incomings[0]
        incoming_ids = set(incoming.incoming_lanelets).intersection(
            possible_lanelets
        )
        successor_ids = set(incoming.successors_right).union(
            incoming.successors_straight, incoming.successors_left
        )
        successor_ids.intersection_update(possible_lanelets)
        if not incoming_ids or not successor_ids:
            return "ambiguous", None
        try:
            start_s = predicate_utils.get_lanelets_start_s(
                vehicle.ref_path_lane, incoming_ids, self.world.road_network
            )
            end_s = predicate_utils.get_lanelets_end_s(
                vehicle.ref_path_lane, successor_ids, self.world.road_network
            )
            if not (math.isfinite(start_s) and math.isfinite(end_s)):
                return "ambiguous", None
        except Exception:
            return "ambiguous", None
        return "ok", (incoming, float(start_s), float(end_s))

    def _causes_braking_frame(
        self,
        evaluator: Any,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        """Estimate directly over the complete ego reachable interval.

        The fixed reference path is assumed lane-forward, consistently with
        ``in_front_of`` and safe-distance estimation.  Therefore the ego rear
        coordinate on the target lane attains its extrema at the two reachable
        path endpoints.  The distance robustness is a concave tent whose only
        internal maximum is at ``distance=d_br/2``; its exact range follows
        from those endpoint extrema without iterating over mapping cells.
        """
        reachable_pair = _finite_pair(reachable)
        if self.other_id is None or reachable_pair is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "causes_braking_intersection:endpoint_unknown",
                diagnostics={
                    "analysis_mode": "monotone_endpoint",
                    "unknown_reason": (
                        "target_unavailable"
                        if self.other_id is None
                        else "reachable_interval_unavailable"
                    ),
                },
            )
        try:
            d_br = float(evaluator.config["d_br"])
            a_br = float(evaluator.config["a_br"])
            target = self.world.vehicle_by_id(self.other_id)
            target_feature = self._target_front_rear(target, int(time_step))
            if target_feature is None:
                raise ValueError("target state unavailable")
            front_target, _, acceleration_target = target_feature
            front_target = float(front_target)
            acceleration_target = float(acceleration_target)
            if not all(
                math.isfinite(value)
                for value in (d_br, a_br, front_target, acceleration_target)
            ):
                raise ValueError("non-finite predicate input")
            distance_zero = float(evaluator._scale_lon_dist(0.0))
            distance_uncertainty = max(
                abs(
                    float(evaluator._scale_lon_dist(self.uncertainty))
                    - distance_zero
                ),
                abs(
                    float(evaluator._scale_lon_dist(-self.uncertainty))
                    - distance_zero
                ),
            )
            acceleration_robustness = float(
                evaluator._scale_acc(a_br - acceleration_target)
            )
            if d_br < 0.0 or acceleration_robustness < -distance_uncertainty:
                return FramePredicateEstimate(
                    FALSE_DOMAIN,
                    "causes_braking_intersection:constant_false",
                    diagnostics={
                        "analysis_mode": "monotone_endpoint",
                        "target_acceleration": acceleration_target,
                        "acceleration_robustness": acceleration_robustness,
                    },
                )
            rear_endpoints, diagnostics = self._ego_lane_feature_endpoints(
                reachable_pair,
                path_lane=target.ref_path_lane,
                coordinate="rear",
            )
        except Exception as exc:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "causes_braking_intersection:endpoint_unknown",
                diagnostics={
                    "analysis_mode": "monotone_endpoint",
                    "unknown_reason": type(exc).__name__,
                },
            )
        diagnostics = dict(diagnostics)
        if rear_endpoints is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "causes_braking_intersection:endpoint_unknown",
                diagnostics=diagnostics,
            )

        distance_lower, distance_upper = sorted(
            float(value) - front_target for value in rear_endpoints
        )

        def distance_robustness(distance: float) -> float:
            return min(float(distance), d_br - float(distance))

        raw_endpoint_robustness = (
            distance_robustness(distance_lower),
            distance_robustness(distance_upper),
        )
        raw_distance_min = min(raw_endpoint_robustness)
        distance_peak = 0.5 * d_br
        raw_distance_max = (
            distance_peak
            if distance_lower <= distance_peak <= distance_upper
            else max(raw_endpoint_robustness)
        )
        distance_min = float(evaluator._scale_lon_dist(raw_distance_min))
        distance_max = float(evaluator._scale_lon_dist(raw_distance_max))
        robustness_min = min(distance_min, acceleration_robustness)
        robustness_max = min(distance_max, acceleration_robustness)
        domain = classify_robustness_range(
            robustness_min,
            robustness_max,
            uncertainty=distance_uncertainty,
        )
        diagnostics.update(
            {
                "distance_interval": [distance_lower, distance_upper],
                "distance_critical_point": distance_peak,
                "distance_critical_reachable": bool(
                    distance_lower <= distance_peak <= distance_upper
                ),
                "target_acceleration": acceleration_target,
                "acceleration_robustness": acceleration_robustness,
                "robustness_min": robustness_min,
                "robustness_max": robustness_max,
            }
        )
        return FramePredicateEstimate(
            domain,
            "causes_braking_intersection:monotone_endpoint",
            diagnostics=diagnostics,
        )

    def _constant_path_region(
        self, value: bool, source: str
    ) -> SemanticIntervalSet:
        if not value:
            return SemanticIntervalSet.empty(source)
        lower, upper = self._progress.trajectory_bounds
        coverage = ClosedInterval(
            float(lower) - self.uncertainty,
            float(upper) + self.uncertainty,
        )
        return SemanticIntervalSet(
            inner_true=(coverage,),
            outer_true=(coverage,),
            complete=True,
            source=source,
        )

    def _same_lane_frame(
        self,
        evaluator: Any,
        time_step: int,
        vehicle_ids: Tuple[int, ...],
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        if self.ego_id not in vehicle_ids or len(vehicle_ids) != 2:
            fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
            return FramePredicateEstimate(fixed, "same_lane:fixed")
        target_id = (
            vehicle_ids[1] if vehicle_ids[0] == self.ego_id else vehicle_ids[0]
        )
        region = self._same_lane_region(int(target_id), int(time_step))
        domain, diagnostics = self._classify_region_estimate(region, reachable)
        return FramePredicateEstimate(
            domain, region.source, region=region, diagnostics=diagnostics
        )

    def _same_lane_region(
        self, target_id: int, time_step: int
    ) -> SemanticIntervalSet:
        target = self.world.vehicle_by_id(int(target_id))
        try:
            target_lanelets = frozenset(
                int(item) for item in target.lanelet_assignment[int(time_step)]
            )
        except Exception:
            return SemanticIntervalSet.unknown("same_lane:target_assignment_missing")
        key = ("same_lane", target_lanelets)
        cached = self._region_cache.get(key)
        if cached is not None:
            return cached

        road_network = self.world.road_network
        network = road_network.lanelet_network
        try:
            target_lanes = road_network.find_lanes_by_lanelets(target_lanelets)
            matching_ids = sorted({
                int(lanelet_id)
                for lane in target_lanes
                for lanelet_id in lane.contained_lanelets
            })
            shapes = [
                network.find_lanelet_by_id(lanelet_id).polygon.shapely_object
                for lanelet_id in matching_ids
            ]
            if not shapes:
                region = SemanticIntervalSet.empty(
                    "same_lane:boolean_no_matching_lane"
                )
            else:
                polygon = shapely.unary_union(shapes)
                half_length = 0.5 * float(self.ego.shape.length)
                half_width = 0.5 * float(self.ego.shape.width)
                circumradius = math.hypot(half_length, half_width)
                # A centre point inside the polygon is not sufficient: the
                # monitor assigns lanes from the complete vehicle occupancy.
                # The circumcircle gives orientation-independent bounds.  If
                # the centre lies in the eroded target-lane union, the whole
                # ego occupancy is inside it (guaranteed same lane).  If the
                # centre lies outside the buffered union, the occupancy cannot
                # touch any target lanelet (guaranteed different lane).
                inner_polygon = polygon.buffer(-circumradius)
                outer_polygon = polygon.buffer(circumradius)
                inner_nominal = self._path_polygon_intervals(
                    self._path_line, inner_polygon
                )
                outer_nominal = self._path_polygon_intervals(
                    self._path_line, outer_polygon
                )
                inner = SemanticIntervalSet.from_nominal(
                    inner_nominal,
                    uncertainty=self.uncertainty,
                    source="same_lane:boolean_inner",
                ).inner_true
                outer = SemanticIntervalSet.from_nominal(
                    outer_nominal,
                    uncertainty=self.uncertainty,
                    source="same_lane:boolean_outer",
                ).outer_true
                region = SemanticIntervalSet(
                    inner_true=inner,
                    outer_true=outer,
                    complete=True,
                    source="same_lane:critical_lane_boundary",
                    diagnostics={
                        "target_lanelet_count": len(target_lanelets),
                        "matching_lanelet_count": len(matching_ids),
                        "circumradius": float(circumradius),
                    },
                )
        except Exception:
            region = SemanticIntervalSet.unknown("same_lane:geometry_unavailable")
        self._region_cache[key] = region
        return region

    def _same_lane_sample_context(self) -> Optional[Mapping[str, Any]]:
        cached = self._same_lane_sample_context_cache
        if cached is not None:
            return cached
        points = np.asarray(self.ref_path, dtype=float)
        trajectory_s = np.asarray(
            self._progress.reference_trajectory_s, dtype=float
        )
        finite = np.isfinite(trajectory_s)
        if len(points) != len(trajectory_s) or np.count_nonzero(finite) < 2:
            return None
        points = points[finite, :2]
        trajectory_s = trajectory_s[finite]
        order = np.argsort(trajectory_s)
        points = points[order]
        trajectory_s = trajectory_s[order]
        keep = np.concatenate(
            ([True], np.diff(trajectory_s) > 1.0e-8)
        )
        points = points[keep]
        trajectory_s = trajectory_s[keep]
        if len(points) < 2:
            return None

        tangents = np.empty_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
        tangent_norm = np.linalg.norm(tangents, axis=1)
        valid = tangent_norm > 1.0e-9
        orientations = np.arctan2(tangents[:, 1], tangents[:, 0])

        cached = {
            "points": points,
            "trajectory_s": trajectory_s,
            "orientations": orientations,
            "valid": valid,
        }
        self._same_lane_sample_context_cache = cached
        return cached

    def _same_lane_robustness_region(
        self, target_id: int, time_step: int
    ) -> SemanticIntervalSet:
        context = self._same_lane_sample_context()
        if context is None:
            return SemanticIntervalSet.unknown(
                "same_lane:sample_context_unavailable"
            )
        target = self.world.vehicle_by_id(int(target_id))
        try:
            target_lanelets = frozenset(
                int(item) for item in target.lanelet_assignment[int(time_step)]
            )
            target_lanes = self.world.road_network.find_lanes_by_lanelets(
                target_lanelets
            )
            target_lane_union = frozenset(
                int(item)
                for lane in target_lanes
                for item in lane.contained_lanelets
            )
        except Exception:
            return SemanticIntervalSet.unknown(
                "same_lane:target_assignment_missing"
            )
        key = tuple(sorted(target_lane_union))
        cached = self._same_lane_robustness_region_cache.get(key)
        if cached is not None:
            return cached

        network = self.world.road_network.lanelet_network
        try:
            target_polygon = shapely.unary_union(
                [
                    network.find_lanelet_by_id(item).polygon.shapely_object
                    for item in target_lane_union
                ]
            )
            points = np.asarray(context["points"], dtype=float)
            orientations = np.asarray(context["orientations"], dtype=float)
            local_vertices = np.asarray(
                self.ego.shape.vertices[:-1], dtype=float
            )
            cosine = np.cos(orientations)[:, None]
            sine = np.sin(orientations)[:, None]
            local_x = local_vertices[:, 0][None, :]
            local_y = local_vertices[:, 1][None, :]
            vertices = np.empty(
                (len(points), len(local_vertices), 2), dtype=float
            )
            vertices[:, :, 0] = (
                local_x * cosine - local_y * sine + points[:, 0, None]
            )
            vertices[:, :, 1] = (
                local_x * sine + local_y * cosine + points[:, 1, None]
            )
            occupancies = shapely.polygons(vertices)
            clearance = np.asarray(
                shapely.distance(occupancies, target_polygon), dtype=float
            )
        except Exception:
            return SemanticIntervalSet.unknown(
                "same_lane:occupancy_distance_unavailable"
            )

        s_values = np.asarray(context["trajectory_s"], dtype=float)
        s_lower = s_values[:-1]
        s_upper = s_values[1:]
        valid = (
            np.asarray(context["valid"][:-1], dtype=bool)
            & np.asarray(context["valid"][1:], dtype=bool)
            & np.isfinite(clearance[:-1])
            & np.isfinite(clearance[1:])
        )
        displacement = np.linalg.norm(points[1:] - points[:-1], axis=1)
        angle_delta = np.abs(
            np.arctan2(
                np.sin(
                    np.asarray(context["orientations"][1:])
                    - np.asarray(context["orientations"][:-1])
                ),
                np.cos(
                    np.asarray(context["orientations"][1:])
                    - np.asarray(context["orientations"][:-1])
                ),
            )
        )
        radius = math.hypot(
            0.5 * float(self.ego.shape.length),
            0.5 * float(self.ego.shape.width),
        )
        error = displacement + radius * angle_delta + self.uncertainty
        # Distance between compact sets is 1-Lipschitz under rigid-body
        # translation.  ``radius * angle_delta`` bounds the additional corner
        # displacement under rotation, so clearance larger than the complete
        # cell motion proves that no intermediate rectangular occupancy can
        # intersect a lanelet belonging to the target's lane.
        proven_false = (
            valid
            & (clearance[:-1] > error)
            & (clearance[1:] > error)
        )
        cells = {
            "s_lower": s_lower,
            "s_upper": s_upper,
        }
        region = SemanticIntervalSet(
            inner_true=(),
            outer_true=self._cell_mask_to_intervals(
                cells, np.logical_not(proven_false)
            ),
            complete=True,
            source="same_lane:occupancy_boundary",
            diagnostics={
                "sample_count": int(len(s_values)),
                "valid_cell_count": int(np.count_nonzero(valid)),
                "false_cell_count": int(np.count_nonzero(proven_false)),
                "target_lanelet_count": int(len(target_lane_union)),
            },
        )
        self._same_lane_robustness_region_cache[key] = region
        return region


    def _longitudinal_cell_region(
        self,
        cells: Mapping[str, Any],
        coordinate: str,
        relation: str,
        threshold_inner: float,
        threshold_outer: float,
        source: str,
    ) -> SemanticIntervalSet:
        valid = np.asarray(cells["valid"], dtype=bool)
        lower = np.asarray(cells[f"{coordinate}_lower"], dtype=float)
        upper = np.asarray(cells[f"{coordinate}_upper"], dtype=float)
        width = np.asarray(cells["s_upper"], dtype=float) - np.asarray(
            cells["s_lower"], dtype=float
        )
        error = width + self.uncertainty
        if relation == "below":
            inner_mask = valid & (upper + error < float(threshold_inner))
            outer_mask = (~valid) | (
                lower - error <= float(threshold_outer)
            )
        elif relation == "above":
            inner_mask = valid & (lower - error > float(threshold_inner))
            outer_mask = (~valid) | (
                upper + error >= float(threshold_outer)
            )
        else:
            return SemanticIntervalSet.unknown(f"{source}:invalid_relation")
        return SemanticIntervalSet(
            inner_true=self._cell_mask_to_intervals(cells, inner_mask),
            outer_true=self._cell_mask_to_intervals(cells, outer_mask),
            complete=True,
            source=source,
            diagnostics={
                "cell_count": int(len(valid)),
                "invalid_cell_count": int(np.count_nonzero(~valid)),
            },
        )

    def _ego_lane_feature_at_progress(
        self,
        progress: float,
        *,
        path_lane: Any,
        coordinate: str,
    ) -> Optional[float]:
        """Evaluate an ego footprint endpoint at one VP path progress.

        The sampled path map is used only as a numerical coordinate converter
        at the requested endpoint.  Its cells are not predicate truth regions.
        Forward monotonicity is the fixed-reference-path model assumption.
        """
        if coordinate not in {"front", "rear"}:
            return None
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(progress):
            return None

        cache = getattr(self, "_ego_lane_feature_cache", None)
        if cache is None:
            cache = {}
            self._ego_lane_feature_cache = cache
        cache_key = (id(path_lane), str(coordinate), round(progress, 10))
        if cache_key in cache:
            return cache[cache_key]

        cells = self._target_path_rear_cells(self.ego, path_lane=path_lane)
        if cells is None:
            cache[cache_key] = None
            return None
        try:
            lower = np.asarray(cells["s_lower"], dtype=float)
            upper = np.asarray(cells["s_upper"], dtype=float)
            valid = np.asarray(cells["valid"], dtype=bool)
            left_key = f"{coordinate}_left"
            right_key = f"{coordinate}_right"
            if left_key in cells and right_key in cells:
                left = np.asarray(cells[left_key], dtype=float)
                right = np.asarray(cells[right_key], dtype=float)
            else:
                # Compatibility with the compact RG mapping cache.  Under the
                # lane-forward model, its extrema are the ordered endpoints.
                left = np.asarray(cells[f"{coordinate}_lower"], dtype=float)
                right = np.asarray(cells[f"{coordinate}_upper"], dtype=float)
        except (KeyError, TypeError, ValueError):
            cache[cache_key] = None
            return None

        tolerance = 1.0e-8 * max(1.0, abs(progress))
        # Cells are ordered by path progress.  Only the insertion cell and its
        # immediate neighbours can contain an endpoint (two cells may meet at
        # a shared boundary), so avoid a full-array scan for every frame.
        insertion = int(np.searchsorted(upper, progress, side="left"))
        candidates = range(
            max(0, insertion - 1),
            min(len(upper), insertion + 2),
        )
        values = []
        for index in candidates:
            if not valid[index]:
                continue
            if progress < float(lower[index]) - tolerance:
                continue
            if progress > float(upper[index]) + tolerance:
                continue
            cell_lower = float(lower[index])
            cell_upper = float(upper[index])
            span = cell_upper - cell_lower
            if not math.isfinite(span) or span <= 1.0e-12:
                continue
            fraction = min(1.0, max(0.0, (progress - cell_lower) / span))
            value = float(left[index]) + fraction * (
                float(right[index]) - float(left[index])
            )
            if math.isfinite(value):
                values.append(value)
        if not values:
            cache[cache_key] = None
            return None

        # At a shared cell boundary both adjacent interpolants may contribute.
        # A disagreement larger than the numerical projection band is treated
        # as failed coordinate evaluation, never as a singleton truth value.
        projection_tolerance = max(
            1.0e-6, min(1.0e-2, 0.1 * float(self.uncertainty))
        )
        if max(values) - min(values) > projection_tolerance:
            cache[cache_key] = None
            return None
        result = float(sum(values) / len(values))
        cache[cache_key] = result
        return result

    def _ego_lane_feature_endpoints(
        self,
        reachable: Sequence[float],
        *,
        path_lane: Any,
        coordinate: str,
    ) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
        """Map exactly the two reachable-position endpoints."""
        reachable_pair = _finite_pair(reachable)
        diagnostics: Dict[str, Any] = {
            "analysis_mode": "monotone_endpoint",
            "analysis_axis": "s",
            "coordinate": str(coordinate),
            "mapping_assumption": "fixed_reference_path_lane_forward",
        }
        if reachable_pair is None:
            diagnostics["unknown_reason"] = "reachable_interval_unavailable"
            return None, diagnostics
        feature_values = tuple(
            self._ego_lane_feature_at_progress(
                progress,
                path_lane=path_lane,
                coordinate=coordinate,
            )
            for progress in reachable_pair
        )
        if any(value is None for value in feature_values):
            diagnostics["unknown_reason"] = "endpoint_projection_failure"
            return None, diagnostics
        result = tuple(float(value) for value in feature_values)
        diagnostics.update(
            {
                "endpoint_progress": [float(value) for value in reachable_pair],
                "endpoint_features": [float(value) for value in result],
                "projection_uncertainty": float(self.uncertainty),
            }
        )
        return result, diagnostics

    def _in_front_of_frame(
        self,
        time_step: int,
        vehicle_ids: Tuple[int, ...],
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        if len(vehicle_ids) != 2 or self.ego_id not in vehicle_ids:
            return FramePredicateEstimate(UNKNOWN_DOMAIN, "in_front_of:fixed")
        reachable_pair = _finite_pair(reachable)
        if reachable_pair is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "in_front_of:endpoint_unknown",
                diagnostics={
                    "analysis_mode": "monotone_endpoint",
                    "unknown_reason": "reachable_interval_unavailable",
                },
            )
        rear_id, front_id = vehicle_ids
        try:
            if rear_id == self.ego_id:
                rear_vehicle = self.ego
                lane = rear_vehicle.get_lane(int(time_step))
                front_vehicle = self.world.vehicle_by_id(int(front_id))
                threshold = front_vehicle.rear_s(int(time_step), lane)
                coordinate = "front"
                direction = "decreasing"
            else:
                rear_vehicle = self.world.vehicle_by_id(int(rear_id))
                lane = rear_vehicle.get_lane(int(time_step))
                threshold = rear_vehicle.front_s(int(time_step))
                coordinate = "rear"
                direction = "increasing"
        except Exception:
            lane = None
            threshold = None
        if lane is None or threshold is None or not math.isfinite(float(threshold)):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "in_front_of:path_mapping_unavailable"
            )
        features, diagnostics = self._ego_lane_feature_endpoints(
            reachable_pair,
            path_lane=lane,
            coordinate=coordinate,
        )
        diagnostics = dict(diagnostics)
        diagnostics["robustness_direction"] = direction
        if features is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "in_front_of:endpoint_unknown",
                diagnostics=diagnostics,
            )
        if rear_id == self.ego_id:
            robustness = tuple(float(threshold) - value for value in features)
        else:
            robustness = tuple(value - float(threshold) for value in features)
        robustness_min = min(robustness)
        robustness_max = max(robustness)
        domain = classify_robustness_range(
            robustness_min,
            robustness_max,
            uncertainty=self.uncertainty,
        )
        diagnostics.update(
            {
                "robustness_endpoints": [float(value) for value in robustness],
                "robustness_min": float(robustness_min),
                "robustness_max": float(robustness_max),
            }
        )
        return FramePredicateEstimate(
            domain,
            "in_front_of:monotone_endpoint",
            diagnostics=diagnostics,
        )

    @staticmethod
    def _safe_distance_value(
        v_follow: float,
        v_lead: float,
        a_min_lead: float,
        a_min_follow: float,
        t_react_follow: float,
    ) -> float:
        return (
            (float(v_lead) ** 2) / (-2.0 * abs(float(a_min_lead)))
            - (float(v_follow) ** 2) / (-2.0 * abs(float(a_min_follow)))
            + float(v_follow) * float(t_react_follow)
        )

    def _safe_distance_frame(
        self,
        time_step: int,
        vehicle_ids: Tuple[int, ...],
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        if len(vehicle_ids) != 2 or self.ego_id not in vehicle_ids:
            return FramePredicateEstimate(UNKNOWN_DOMAIN, "safe_distance:fixed")
        reachable_pair = _finite_pair(reachable)
        if reachable_pair is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "safe_distance:endpoint_unknown",
                diagnostics={
                    "analysis_mode": "velocity_envelope_endpoint",
                    "unknown_reason": "reachable_interval_unavailable",
                },
            )
        follow_id, lead_id = vehicle_ids
        try:
            follow = self.world.vehicle_by_id(int(follow_id))
            lead = self.world.vehicle_by_id(int(lead_id))
            if lead.get_lane(int(time_step)) is None:
                region = self._constant_path_region(
                    True, "safe_distance:lead_lane_missing"
                )
                return FramePredicateEstimate(
                    TRUE_DOMAIN, region.source, region=region
                )
            velocity_interval = self.reachable_velocity_by_time.get(
                int(time_step)
            )
            if velocity_interval is None:
                raise ValueError("reachable velocity missing")
            velocity_pair = _finite_pair(velocity_interval)
            if velocity_pair is None:
                raise ValueError("reachable velocity is non-finite")
            v_lower, v_upper = velocity_pair
            if v_lower < 0.0:
                return FramePredicateEstimate(
                    UNKNOWN_DOMAIN,
                    "safe_distance:velocity_unknown",
                    diagnostics={
                        "analysis_mode": "velocity_envelope_endpoint",
                        "unknown_reason": "negative_reachable_velocity",
                        "velocity_interval": [float(v_lower), float(v_upper)],
                    },
                )
            a_min_follow = float(follow.vehicle_param.get("a_min"))
            a_min_lead = float(lead.vehicle_param.get("a_min"))
            t_react = float(follow.vehicle_param.get("t_react"))
            if not all(
                math.isfinite(value)
                for value in (a_min_follow, a_min_lead, t_react)
            ):
                raise ValueError("safe-distance parameter is non-finite")
            if follow_id == self.ego_id:
                fixed_velocity = float(lead.states_cr[int(time_step)].velocity)
                safe_values = [
                    self._safe_distance_value(
                        velocity,
                        fixed_velocity,
                        a_min_lead,
                        a_min_follow,
                        t_react,
                    )
                    for velocity in (v_lower, v_upper)
                ]
                threshold = float(lead.rear_s(int(time_step)))
                lane = follow.get_lane(int(time_step))
                coordinate = "front"
                direction = "decreasing"
            else:
                fixed_velocity = float(follow.states_cr[int(time_step)].velocity)
                safe_values = [
                    self._safe_distance_value(
                        fixed_velocity,
                        velocity,
                        a_min_lead,
                        a_min_follow,
                        t_react,
                    )
                    for velocity in (v_lower, v_upper)
                ]
                threshold = float(follow.front_s(int(time_step)))
                lane = follow.get_lane(int(time_step))
                coordinate = "rear"
                direction = "increasing"
        except Exception:
            lane = None
            threshold = None
        if (
            lane is None
            or threshold is None
            or not math.isfinite(float(threshold))
            or not math.isfinite(float(fixed_velocity))
            or not all(math.isfinite(float(value)) for value in safe_values)
        ):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "safe_distance:path_mapping_unavailable"
            )
        safe_min = min(float(value) for value in safe_values)
        safe_max = max(float(value) for value in safe_values)
        features, diagnostics = self._ego_lane_feature_endpoints(
            reachable_pair,
            path_lane=lane,
            coordinate=coordinate,
        )
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "analysis_mode": "velocity_envelope_endpoint",
                "robustness_direction": direction,
                "velocity_interval": [float(v_lower), float(v_upper)],
                "safe_distance_endpoints": [
                    float(safe_values[0]),
                    float(safe_values[1]),
                ],
                "safe_distance_min": float(safe_min),
                "safe_distance_max": float(safe_max),
            }
        )
        if features is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN,
                "safe_distance:endpoint_unknown",
                diagnostics=diagnostics,
            )
        if follow_id == self.ego_id:
            robustness_min = float(threshold) - max(features) - safe_max
            robustness_max = float(threshold) - min(features) - safe_min
        else:
            robustness_min = min(features) - float(threshold) - safe_max
            robustness_max = max(features) - float(threshold) - safe_min
        domain = classify_robustness_range(
            robustness_min,
            robustness_max,
            uncertainty=self.uncertainty,
        )
        diagnostics.update(
            {
                "robustness_min": float(robustness_min),
                "robustness_max": float(robustness_max),
            }
        )
        return FramePredicateEstimate(
            domain,
            "safe_distance:velocity_envelope_endpoint",
            diagnostics=diagnostics,
        )

    def _cut_in_frame(
        self,
        evaluator: Any,
        time_step: int,
        vehicle_ids: Tuple[int, ...],
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        if len(vehicle_ids) != 2 or self.ego_id not in vehicle_ids:
            fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
            return FramePredicateEstimate(fixed, "cut_in:other_only")
        cutting_id, cutted_id = vehicle_ids
        if cutting_id != self.ego_id and cutted_id == self.ego_id:
            single_lane = self._cut_in_single_lane_domain(
                int(cutting_id), int(time_step)
            )
            if single_lane == TRUE_DOMAIN:
                return FramePredicateEstimate(
                    FALSE_DOMAIN, "cut_in:cutting_vehicle_single_lane"
                )
            try:
                cutting = self.world.vehicle_by_id(int(cutting_id))
                cutting_lane = cutting.get_lane(int(time_step))
                cutting_lat = cutting.get_lat_state(int(time_step), cutting_lane)
                theta = float(cutting_lat.theta)
                cutting_d = float(cutting_lat.d)
                epsilon = float(getattr(evaluator, "eps", 1.0e-5))
            except Exception:
                return FramePredicateEstimate(
                    UNKNOWN_DOMAIN, "cut_in:target_state_unavailable"
                )
            if -epsilon <= theta <= epsilon:
                return FramePredicateEstimate(
                    FALSE_DOMAIN, "cut_in:fixed_orientation_false"
                )

            same_lane = self._same_lane_region(int(cutting_id), int(time_step))
            lateral = self._cut_in_lateral_region(
                cutting_lane, cutting_d, theta > epsilon
            )
            region = self._intersect_semantic_regions(
                same_lane, lateral, "cut_in:reachable_lane_lateral"
            )
            domain, region_diagnostics = self._classify_region_estimate(
                region, reachable
            )
            return FramePredicateEstimate(
                domain,
                region.source,
                region=region,
                diagnostics={
                    "same_lane_source": same_lane.source,
                    "lateral_source": lateral.source,
                    **region_diagnostics,
                },
            )

        # In this argument order the ego itself is the cutting vehicle, so
        # both lateral coordinate and heading depend on path progress.
        return FramePredicateEstimate(
            UNKNOWN_DOMAIN, "cut_in:ego_is_cutting_vehicle"
        )

    def _cut_in_single_lane_domain(
        self, cutting_id: int, time_step: int
    ) -> TruthDomain:
        """Evaluate the other-only ``single_lane`` gate by assignment signature."""
        try:
            cutting = self.world.vehicle_by_id(int(cutting_id))
            assignment = frozenset(
                int(item) for item in cutting.lanelet_assignment[int(time_step)]
            )
        except Exception:
            return UNKNOWN_DOMAIN
        key = (int(cutting_id), assignment)
        cached = self._cut_in_single_lane_cache.get(key)
        if cached is not None:
            return cached
        try:
            lanes = self.world.road_network.find_lanes_by_lanelets(assignment)
            result = TRUE_DOMAIN if len(lanes) == 1 else FALSE_DOMAIN
        except Exception:
            result = UNKNOWN_DOMAIN
        self._cut_in_single_lane_cache[key] = result
        return result

    def _cut_in_lateral_region(
        self, cutting_lane: Any, cutting_d: float, ego_must_be_above: bool
    ) -> SemanticIntervalSet:
        """Bound the lateral-side term of ``cut_in(other, ego)`` over path s."""
        if cutting_lane is None or not math.isfinite(float(cutting_d)):
            return self._unknown_path_region("cut_in:lateral_lane_missing")
        lane_key = tuple(sorted(
            int(item)
            for item in getattr(cutting_lane, "contained_lanelets", ())
        )) or (id(cutting_lane),)
        context = self._cut_in_path_lateral_cache.get(lane_key)
        if lane_key not in self._cut_in_path_lateral_cache:
            context = self._build_cut_in_lateral_context(cutting_lane)
            self._cut_in_path_lateral_cache[lane_key] = context
        if context is None:
            return self._unknown_path_region(
                "cut_in:lateral_projection_unavailable"
            )

        d_lower = np.asarray(context["d_lower"], dtype=float)
        d_upper = np.asarray(context["d_upper"], dtype=float)
        valid = np.asarray(context["valid"], dtype=bool)
        error = np.asarray(context["error"], dtype=float)
        threshold = float(cutting_d)
        if ego_must_be_above:
            inner_mask = valid & (d_lower - error > threshold)
            outer_mask = (~valid) | (d_upper + error > threshold)
        else:
            inner_mask = valid & (d_upper + error < threshold)
            outer_mask = (~valid) | (d_lower - error < threshold)
        return SemanticIntervalSet(
            inner_true=self._cell_mask_to_intervals(context, inner_mask),
            outer_true=self._cell_mask_to_intervals(context, outer_mask),
            complete=True,
            source="cut_in:lateral_critical_boundary",
            diagnostics={
                "lanelet_count": len(lane_key),
                "cell_count": int(len(valid)),
                "invalid_cell_count": int(np.count_nonzero(~valid)),
            },
        )

    def _build_cut_in_lateral_context(
        self, cutting_lane: Any
    ) -> Optional[Mapping[str, Any]]:
        sample = self._same_lane_sample_context()
        if sample is None:
            return None
        points = np.asarray(sample["points"], dtype=float)
        s_values = np.asarray(sample["trajectory_s"], dtype=float)
        try:
            bulk = getattr(
                cutting_lane.clcs,
                "convert_list_of_points_to_curvilinear_coords",
                None,
            )
            if bulk is None:
                raise AttributeError("bulk CLCS projection unavailable")
            projected = np.asarray(bulk(points, 1), dtype=float)
            if projected.shape != (len(points), 2):
                raise ValueError("unexpected CLCS projection shape")
            lateral = projected[:, 1]
        except Exception:
            # Projection failures make only affected cells unknown; never
            # substitute the nominal monitor truth value.
            lateral = np.full(len(points), math.nan, dtype=float)
            for index, point in enumerate(points):
                try:
                    lateral[index] = float(
                        cutting_lane.clcs.convert_to_curvilinear_coords(
                            float(point[0]), float(point[1])
                        )[1]
                    )
                except Exception:
                    continue

        displacement = np.linalg.norm(points[1:] - points[:-1], axis=1)
        valid = (
            np.isfinite(s_values[:-1])
            & np.isfinite(s_values[1:])
            & np.isfinite(lateral[:-1])
            & np.isfinite(lateral[1:])
        )
        return {
            "s_lower": np.minimum(s_values[:-1], s_values[1:]),
            "s_upper": np.maximum(s_values[:-1], s_values[1:]),
            "d_lower": np.minimum(lateral[:-1], lateral[1:]),
            "d_upper": np.maximum(lateral[:-1], lateral[1:]),
            "error": displacement + self.uncertainty,
            "valid": valid,
        }

    @staticmethod
    def _intersect_interval_sets(
        left: Sequence[ClosedInterval], right: Sequence[ClosedInterval]
    ) -> Tuple[ClosedInterval, ...]:
        intersections = []
        for lhs in left:
            for rhs in right:
                lower = max(float(lhs.lower), float(rhs.lower))
                upper = min(float(lhs.upper), float(rhs.upper))
                if lower <= upper:
                    intersections.append((lower, upper))
        return merge_intervals(intersections)

    def _intersect_semantic_regions(
        self,
        left: SemanticIntervalSet,
        right: SemanticIntervalSet,
        source: str,
    ) -> SemanticIntervalSet:
        return SemanticIntervalSet(
            inner_true=self._intersect_interval_sets(
                left.inner_true, right.inner_true
            ),
            outer_true=self._intersect_interval_sets(
                left.outer_true, right.outer_true
            ),
            complete=bool(left.complete and right.complete),
            source=source,
            diagnostics={"left_source": left.source, "right_source": right.source},
        )


    def _target_front_rear(
        self, target: Any, time_step: int
    ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        if target is None:
            return None
        key = (int(target.id), int(time_step))
        if key in self._target_longitudinal_cache:
            return self._target_longitudinal_cache[key]
        result = None
        try:
            front_s = target.front_s(int(time_step), target.ref_path_lane)
            rear_s = target.rear_s(int(time_step), target.ref_path_lane)
            lon_state = target.get_lon_state(int(time_step), target.ref_path_lane)
            acceleration = getattr(lon_state, "a", None)
            result = (
                self._finite_or_none(front_s),
                self._finite_or_none(rear_s),
                self._finite_or_none(acceleration),
            )
        except Exception:
            result = None
        self._target_longitudinal_cache[key] = result
        return result

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _reachable_path_slice(self, trajectory_values: np.ndarray) -> slice:
        """Limit dynamic predicate mapping to the VP-reachable path span.

        Static predicate boundaries still use the full sparse progress map.
        ``causes_braking`` only needs cells which can intersect an ego
        reachable interval; two neighbouring samples and a vehicle-sized
        margin keep interpolation valid near both ends.  The omitted path tails
        are later represented as invalid outer cells, so they cannot create a
        false guaranteed-false result.
        """
        if not self.reachable_by_time or len(trajectory_values) < 3:
            return slice(None)
        finite = np.isfinite(trajectory_values)
        if np.count_nonzero(finite) < 2:
            return slice(None)
        reachable_lower = min(
            float(pair[0]) for pair in self.reachable_by_time.values()
        )
        reachable_upper = max(
            float(pair[1]) for pair in self.reachable_by_time.values()
        )
        margin = max(2.0, self._front_extent, self._rear_extent) + 1.0
        selected = np.flatnonzero(
            finite
            & (trajectory_values >= reachable_lower - margin)
            & (trajectory_values <= reachable_upper + margin)
        )
        if len(selected) < 2:
            return slice(None)
        start = max(0, int(selected[0]) - 2)
        stop = min(len(trajectory_values), int(selected[-1]) + 3)
        return slice(start, stop)

    def _target_path_rear_cells(
        self, target: Any, path_lane: Any = None
    ) -> Optional[Mapping[str, Any]]:
        """Map VP path cells to ego front/rear progress on one lane once."""
        target_lane = path_lane if path_lane is not None else target.ref_path_lane
        key = (int(target.id), id(target_lane))
        if key in self._target_path_rear_cells_cache:
            return self._target_path_rear_cells_cache[key]
        points = np.asarray(self.ref_path, dtype=float)
        if len(points) < 2:
            self._target_path_rear_cells_cache[key] = None
            return None

        half_length = 0.5 * float(self.ego.shape.length)
        half_width = 0.5 * float(self.ego.shape.width)
        # The progress map has already projected these exact reference-path
        # samples into both CLCSs.  Reuse those values instead of repeating a
        # second full path scan for every target mapping.
        trajectory_values = np.asarray(
            self._progress.reference_trajectory_s, dtype=float
        ).copy()
        missing_trajectory = ~np.isfinite(trajectory_values)
        if np.any(missing_trajectory):
            lane_values = np.asarray(
                self._progress.reference_lane_s, dtype=float
            )
            recoverable = missing_trajectory & np.isfinite(lane_values)
            if np.any(recoverable):
                trajectory_values[recoverable] = self._progress.map_array(
                    lane_values[recoverable]
                )

        reachable_slice = self._reachable_path_slice(trajectory_values)
        points = points[reachable_slice]
        trajectory_values = trajectory_values[reachable_slice]
        if len(points) < 2:
            self._target_path_rear_cells_cache[key] = None
            return None

        tangents = np.empty_like(points[:, :2])
        tangents[0] = points[1, :2] - points[0, :2]
        tangents[-1] = points[-1, :2] - points[-2, :2]
        if len(points) > 2:
            tangents[1:-1] = points[2:, :2] - points[:-2, :2]

        if target_lane.clcs is self.lanelet_clcs:
            # The progress map already projected these exact samples.
            target_center_values = np.asarray(
                self._progress.reference_lane_s, dtype=float
            )[reachable_slice].copy()
        else:
            target_center_values = _project_points_to_s(
                target_lane.clcs, points
            )
        missing_target = ~np.isfinite(target_center_values)
        large_step_clcs = getattr(target_lane, "clcs_large_step", None)
        if np.any(missing_target) and large_step_clcs is not None:
            target_center_values[missing_target] = _project_points_to_s(
                large_step_clcs, points[missing_target]
            )

        tangent_norm = np.linalg.norm(tangents, axis=1)
        geometry_valid = (
            np.isfinite(target_center_values)
            & np.isfinite(tangent_norm)
            & (tangent_norm > 1.0e-9)
        )
        rear_values = np.full(len(points), math.nan, dtype=float)
        front_values = np.full(len(points), math.nan, dtype=float)
        if np.any(geometry_valid):
            ego_orientation = np.arctan2(tangents[:, 1], tangents[:, 0])
            try:
                target_orientation = np.asarray(
                    target_lane.orientation(target_center_values[geometry_valid]),
                    dtype=float,
                )
                if target_orientation.shape != (
                    int(np.count_nonzero(geometry_valid)),
                ):
                    raise ValueError("lane orientation did not preserve shape")
            except Exception:
                target_orientation = np.asarray(
                    [
                        target_lane.orientation(float(value))
                        for value in target_center_values[geometry_valid]
                    ],
                    dtype=float,
                )
            theta = ego_orientation[geometry_valid] - target_orientation
            longitudinal_extent = (
                np.abs(np.cos(theta)) * half_length
                + np.abs(np.sin(theta)) * half_width
            )
            rear_values[geometry_valid] = (
                target_center_values[geometry_valid] - longitudinal_extent
            )
            front_values[geometry_valid] = (
                target_center_values[geometry_valid] + longitudinal_extent
            )

        finite_trajectory = np.isfinite(trajectory_values)
        samples = list(
            zip(
                trajectory_values[finite_trajectory].tolist(),
                rear_values[finite_trajectory].tolist(),
                front_values[finite_trajectory].tolist(),
            )
        )

        samples.sort(key=lambda item: item[0])
        deduplicated = []
        for trajectory_s, rear_target_s, front_target_s in samples:
            if deduplicated and abs(trajectory_s - deduplicated[-1][0]) <= 1.0e-8:
                if not math.isfinite(deduplicated[-1][1]) and math.isfinite(rear_target_s):
                    deduplicated[-1] = (
                        trajectory_s,
                        rear_target_s,
                        front_target_s,
                    )
                continue
            deduplicated.append((trajectory_s, rear_target_s, front_target_s))
        if len(deduplicated) < 2:
            self._target_path_rear_cells_cache[key] = None
            return None

        s_values = np.asarray([item[0] for item in deduplicated], dtype=float)
        rear_values = np.asarray([item[1] for item in deduplicated], dtype=float)
        front_values = np.asarray([item[2] for item in deduplicated], dtype=float)
        s_lower = s_values[:-1]
        s_upper = s_values[1:]
        rear_left = rear_values[:-1]
        rear_right = rear_values[1:]
        front_left = front_values[:-1]
        front_right = front_values[1:]
        path_lower, path_upper = self._progress.trajectory_bounds
        if float(path_lower) < float(s_values[0]) - 1.0e-8:
            s_lower = np.concatenate(([float(path_lower)], s_lower))
            s_upper = np.concatenate(([float(s_values[0])], s_upper))
            rear_left = np.concatenate(([math.nan], rear_left))
            rear_right = np.concatenate(([math.nan], rear_right))
            front_left = np.concatenate(([math.nan], front_left))
            front_right = np.concatenate(([math.nan], front_right))
        if float(path_upper) > float(s_values[-1]) + 1.0e-8:
            s_lower = np.concatenate((s_lower, [float(s_values[-1])]))
            s_upper = np.concatenate((s_upper, [float(path_upper)]))
            rear_left = np.concatenate((rear_left, [math.nan]))
            rear_right = np.concatenate((rear_right, [math.nan]))
            front_left = np.concatenate((front_left, [math.nan]))
            front_right = np.concatenate((front_right, [math.nan]))
        valid = (
            np.isfinite(rear_left)
            & np.isfinite(rear_right)
            & np.isfinite(front_left)
            & np.isfinite(front_right)
        )
        positive_steps = np.diff(s_values)
        median_step = float(np.median(positive_steps[positive_steps > 1.0e-9]))
        max_step = max(2.0, 4.0 * median_step)
        valid &= (s_upper - s_lower) <= max_step
        result = {
            "s_lower": s_lower,
            "s_upper": s_upper,
            "rear_lower": np.minimum(rear_left, rear_right),
            "rear_upper": np.maximum(rear_left, rear_right),
            "rear_left": rear_left,
            "rear_right": rear_right,
            "front_lower": np.minimum(front_left, front_right),
            "front_upper": np.maximum(front_left, front_right),
            "valid": valid,
            "mapped_point_count": int(len(points)),
        }
        self._target_path_rear_cells_cache[key] = result
        return result

    @staticmethod
    def _cell_mask_to_intervals(
        cells: Mapping[str, Any], mask: np.ndarray
    ) -> Tuple[ClosedInterval, ...]:
        selected = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(selected) == 0:
            return ()
        lower = np.asarray(cells["s_lower"], dtype=float)[selected]
        upper = np.asarray(cells["s_upper"], dtype=float)[selected]
        # ``s_lower`` is sorted by construction.  Find exactly the same merge
        # groups as ``merge_intervals`` without allocating one Python object
        # per selected 0.1 m cell at every temporal frame.
        running_upper = np.maximum.accumulate(upper)
        group_start = np.concatenate(
            (
                np.asarray([True]),
                lower[1:] > running_upper[:-1] + 1.0e-9,
            )
        )
        starts = np.flatnonzero(group_start)
        ends = np.concatenate((starts[1:] - 1, [len(lower) - 1]))
        return tuple(
            ClosedInterval(float(lower[start]), float(running_upper[end]))
            for start, end in zip(starts, ends)
        )

    def _unknown_path_region(self, source: str) -> SemanticIntervalSet:
        lower, upper = self._progress.trajectory_bounds
        coverage = ClosedInterval(
            float(lower) - self.uncertainty,
            float(upper) + self.uncertainty,
        )
        return SemanticIntervalSet(
            inner_true=(),
            outer_true=(coverage,),
            complete=True,
            source=source,
            diagnostics={"path_wide_unknown": True},
        )

    # ------------------------------------------------------------------
    # Geometry primitives
    # ------------------------------------------------------------------
    def _vehicle_extents(self) -> Tuple[float, float]:
        fallback = max(0.0, float(self.ego.shape.length) / 2.0)
        try:
            first_time = min(self.ego.states_cr)
            state = self.ego.states_cr[first_time]
            center_lane = self.lanelet_clcs.convert_to_curvilinear_coords(
                float(state.position[0]), float(state.position[1])
            )[0]
            front_lane = self.ego.front_s(first_time, self.ego.ref_path_lane)
            rear_lane = self.ego.rear_s(first_time, self.ego.ref_path_lane)
            if front_lane is None or rear_lane is None:
                return fallback, fallback
            center_s = self._progress(center_lane)
            front_s = self._progress(float(front_lane))
            rear_s = self._progress(float(rear_lane))
            return abs(front_s - center_s), abs(center_s - rear_s)
        except Exception:
            return fallback, fallback

    def _lanelet_truth_bounds(
        self, lanelet_id: int
    ) -> Optional[Tuple[float, float]]:
        lanelet_id = int(lanelet_id)
        if lanelet_id in self._lanelet_bounds_cache:
            return self._lanelet_bounds_cache[lanelet_id]
        result = None
        try:
            lanelet = self.world.road_network.lanelet_network.find_lanelet_by_id(
                lanelet_id
            )
            vertices = np.asarray(lanelet.center_vertices, dtype=float)
            candidates = []
            # Project a few points at both boundaries.  This avoids depending
            # on polygon vertex ordering and is cheaper than scanning the path.
            for point in (vertices[0], vertices[-1]):
                lane_s = self.lanelet_clcs.convert_to_curvilinear_coords(
                    float(point[0]), float(point[1])
                )[0]
                candidates.append(self._progress(lane_s))
            start_s, end_s = sorted(candidates)
            result = (
                float(start_s - self._front_extent),
                float(end_s + self._rear_extent),
            )
        except Exception:
            result = None
        self._lanelet_bounds_cache[lanelet_id] = result
        return result

    @staticmethod
    def _intersection(
        left: Sequence[float], right: Sequence[float]
    ) -> Optional[Tuple[float, float]]:
        lower = max(float(left[0]), float(right[0]))
        upper = min(float(left[1]), float(right[1]))
        return (lower, upper) if lower <= upper else None

    def _standstill_frame(
        self, evaluator: Any, time_step: int
    ) -> FramePredicateEstimate:
        velocity = self.reachable_velocity_by_time.get(int(time_step))
        if velocity is None:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "standstill:velocity_reachability_missing"
            )
        try:
            epsilon = float(evaluator.config["standstill_error"])
        except Exception:
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, "standstill:invalid_threshold"
            )
        if epsilon < 0.0:
            return FramePredicateEstimate(
                FALSE_DOMAIN, "standstill:negative_threshold"
            )

        def robustness(velocity_value: float) -> float:
            return min(
                float(velocity_value) + epsilon,
                epsilon - float(velocity_value),
            )

        estimate = estimate_piecewise_monotone_1d(
            velocity,
            (0.0,),
            robustness,
            # The reachable velocity interval is already analytic.  Keep the
            # shared zero-robustness convention explicit without adding a
            # longitudinal geometry uncertainty to the velocity axis.
            uncertainty=0.0,
        )
        diagnostics = piecewise_diagnostics(
            estimate,
            analysis_mode="piecewise_monotone",
            analysis_axis="v",
            include_segments=bool(
                os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG")
            ),
        )
        return FramePredicateEstimate(
            estimate.domain,
            "standstill:piecewise_velocity",
            diagnostics=diagnostics,
        )

    def _default_vehicle_ids(
        self, evaluator: Any, prop_name: str
    ) -> Tuple[int, ...]:
        arity = int(getattr(evaluator, "arity", 0) or 0)
        if arity == 1:
            return (self.ego_id,)
        if arity == 2 and self.other_id is not None:
            if "__1_0" in str(prop_name):
                return (self.other_id, self.ego_id)
            return (self.ego_id, self.other_id)
        return ()

    @staticmethod
    def _normalize_time_steps(time_steps: Iterable[int]) -> Iterable[int]:
        """Accept a range/list or the repairer's temporal-interval object."""
        count = getattr(time_steps, "count", None)
        start = getattr(time_steps, "start", None)
        end = getattr(time_steps, "end", None)
        if count is not None and start is not None and end is not None:
            if int(count) <= 0:
                return ()
            return range(int(start), int(end) + 1)
        return time_steps


# Concise public aliases for callers and experiments.
INPredicateRegionBuilder = SemanticINPredicateRegionBuilder
INSemanticRegionEstimator = SemanticINPredicateRegionBuilder
IntervalSetEstimate = SemanticIntervalSet
CertifiedRegion = SemanticIntervalSet
IntervalSet = Tuple[ClosedInterval, ...]


def build_semantic_in_predicate_region_builder(
    repairer: Any,
    trajectory_clcs: Any,
    ref_path: Any,
    reachable_by_time: Mapping[int, Sequence[float]],
    **kwargs: Any,
) -> SemanticINPredicateRegionBuilder:
    """Factory retained to keep domain integration declarative."""
    return SemanticINPredicateRegionBuilder(
        repairer,
        trajectory_clcs,
        ref_path,
        reachable_by_time,
        **kwargs,
    )


__all__ = [
    "FALSE_DOMAIN",
    "TRUE_DOMAIN",
    "UNKNOWN_DOMAIN",
    "TruthDomain",
    "SemanticIntervalSet",
    "CertifiedRegion",
    "IntervalSet",
    "IntervalSetEstimate",
    "FramePredicateEstimate",
    "SemanticINPredicateRegionBuilder",
    "INPredicateRegionBuilder",
    "INSemanticRegionEstimator",
    "build_semantic_in_predicate_region_builder",
    "estimate_acceleration_predicate_frame",
]
