"""Definition-driven truth regions for intersection predicates.

The velocity planner changes the ego vehicle's longitudinal progress along a
fixed reference path.  For many intersection predicates, the monitor
definition can therefore be represented as a set of longitudinal intervals:

* ``inner_true`` is guaranteed to be part of the predicate's true set;
* ``outer_true`` is guaranteed to contain the predicate's complete true set.

For an ego reachable interval ``R`` this gives the sound three-valued test

``R subset inner_true -> {1}``, ``R disjoint outer_true -> {0}``, otherwise
``{0, 1}``.

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
from shapely.geometry import LineString

from commonroad.scenario.lanelet import LaneletType

from crrepairer.repairer.vp.predicate_regions import (
    ClosedInterval,
    as_closed_interval,
    merge_intervals,
)


TruthDomain = FrozenSet[int]
FALSE_DOMAIN: TruthDomain = frozenset((0,))
TRUE_DOMAIN: TruthDomain = frozenset((1,))
UNKNOWN_DOMAIN: TruthDomain = frozenset((0, 1))


def _domain(value: Optional[bool]) -> TruthDomain:
    if value is None:
        return UNKNOWN_DOMAIN
    return TRUE_DOMAIN if value else FALSE_DOMAIN


def _predicate_name(evaluator: Any) -> str:
    name = getattr(evaluator, "predicate_name", None)
    return str(getattr(name, "value", name or "")).strip().lower()


def _finite_pair(value: Any) -> Optional[Tuple[float, float]]:
    try:
        lower, upper = float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return (lower, upper) if lower <= upper else (upper, lower)


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


class _ProgressMap:
    """Monotone map from monitor lane progress to VP trajectory progress."""

    def __init__(self, lane_clcs: Any, trajectory_clcs: Any, ref_path: Any):
        points = np.asarray(ref_path, dtype=float)
        lane_by_ref = _project_points_to_s(lane_clcs, points)
        trajectory_by_ref = _project_points_to_s(trajectory_clcs, points)
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
    """Build monitor-definition-based regions for IN predicates.

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
            "relevant_traffic_light",
            "on_lanelet_with_type_intersection",
            "causes_braking_intersection",
            "in_standstill",
        }
    )
    BOOLEAN_NAMES = frozenset(
        {
            "at_traffic_sign_stop",
            "on_incoming_left_of",
            "in_intersection_conflict_area",
        }
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
        self.uncertainty = max(0.0, float(uncertainty))
        self._progress = _ProgressMap(
            self.lanelet_clcs, trajectory_clcs, self.ref_path
        )
        self._region_cache: Dict[Any, SemanticIntervalSet] = {}
        self._fixed_cache: Dict[Any, TruthDomain] = {}
        self._frame_cache: Dict[Any, FramePredicateEstimate] = {}
        self._turning_spatial_domain_cache: Dict[Any, TruthDomain] = {}
        self._lanelet_bounds_cache: Dict[int, Optional[Tuple[float, float]]] = {}
        # These caches contain only definition-derived route geometry and
        # fixed target-trajectory features.  In particular, no entry is fitted
        # from the recorded ego predicate trace.
        self._incoming_geometry_cache: Optional[Any] = None
        self._target_longitudinal_cache: Dict[Any, Any] = {}
        self._target_path_rear_cells_cache: Dict[Any, Any] = {}
        self._front_extent, self._rear_extent = self._vehicle_extents()
        self._timing = {
            "initialization": time.perf_counter() - started,
            "region_build": 0.0,
            "frame_classification": 0.0,
            "fixed_gate": 0.0,
        }
        self._counts: Dict[str, int] = {}

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
        key = (
            id(evaluator),
            str(prop_name),
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
                    self._classify_region(
                        region, self.reachable_by_time.get(int(step))
                    )
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

        if hasattr(evaluator, "_turning_ego"):
            return self._turning_composite_frame(
                evaluator, prop_name, time_step, reachable
            )

        if name == "in_intersection_conflict_area" and "__1_0" in prop_name:
            fixed = self._fixed_domain(evaluator, time_step, vehicle_ids)
            return FramePredicateEstimate(fixed, "fixed_target_conflict")

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

        region = self.region_for(evaluator, prop_name, time_step)
        if region.source.startswith("unsupported:"):
            return FramePredicateEstimate(
                UNKNOWN_DOMAIN, region.source, region=region
            )
        result = self._classify_region(region, reachable)
        return FramePredicateEstimate(result, region.source, region=region)

    def _turning_composite_frame(
        self,
        evaluator: Any,
        prop_name: str,
        time_step: int,
        reachable: Optional[Tuple[float, float]],
    ) -> FramePredicateEstimate:
        region = self._turning_region(evaluator._turning_ego)
        spatial = self._classify_region(region, reachable)
        if spatial == FALSE_DOMAIN:
            return FramePredicateEstimate(
                FALSE_DOMAIN,
                "turning_composite:spatial_false",
                region=region,
                gate_domain=UNKNOWN_DOMAIN,
                diagnostics={"spatial_domain": [0], "gate_skipped": True},
            )
        gate = self._turning_gate_domain(evaluator, prop_name, time_step)
        if gate == FALSE_DOMAIN or spatial == FALSE_DOMAIN:
            combined = FALSE_DOMAIN
        elif gate == TRUE_DOMAIN and spatial == TRUE_DOMAIN:
            combined = TRUE_DOMAIN
        else:
            combined = UNKNOWN_DOMAIN
        return FramePredicateEstimate(
            combined,
            "turning_composite",
            region=region,
            gate_domain=gate,
            diagnostics={"spatial_domain": sorted(spatial)},
        )

    def _turning_gate_domain(
        self, evaluator: Any, prop_name: str, time_step: int
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
                fixed_domain(
                    evaluator._same_priority,
                    time_step,
                    (self.ego_id, self.other_id),
                )
            )
        elif hasattr(evaluator, "_target_has_priority"):
            gate_parts.append(
                fixed_domain(
                    evaluator._target_has_priority,
                    time_step,
                    (self.other_id, self.ego_id),
                )
            )
        else:
            gate_parts.append(UNKNOWN_DOMAIN)
        if gate_parts[-1] == FALSE_DOMAIN:
            return FALSE_DOMAIN
        if hasattr(evaluator, "_on_oncoming_of"):
            oncoming = fixed_domain(
                evaluator._on_oncoming_of,
                time_step,
                (self.other_id, self.ego_id),
            )
            if "not_oncoming" in prop_name.lower():
                oncoming = self._negate_domain(oncoming)
            gate_parts.append(oncoming)

        return self._and_domains(gate_parts)

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
        key = ("boolean", name, id(evaluator), int(time_step), tuple(vehicle_ids))
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
        key = (name, id(evaluator), int(time_step), tuple(vehicle_ids))
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
        return region.classify(reachable)

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
        route_ids = set(self.ego.ref_path_lane.contained_lanelets)
        network = self.world.road_network.lanelet_network
        d_sl = float(getattr(evaluator, "config", {}).get("d_sl", 0.0))
        if d_sl < 0.0:
            return SemanticIntervalSet.unknown("stop_line", invalid_d_sl=d_sl)
        for lanelet_id in route_ids:
            lanelet = network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is None:
                continue
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
            lane_bounds = self._lanelet_truth_bounds(lanelet_id)
            if lane_bounds is None:
                return SemanticIntervalSet.unknown(
                    "stop_line", failed_lanelet=int(lanelet_id)
                )
            overlap = self._intersection(stop_window, lane_bounds)
            if overlap is not None:
                intervals.append(overlap)
        return SemanticIntervalSet.from_nominal(
            intervals,
            uncertainty=self.uncertainty,
            source="stop_line",
            diagnostics={"candidate_count": len(intervals)},
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
        intervals = []
        for lanelet_id in set(int(item) for item in lanelet_ids):
            bounds = self._lanelet_truth_bounds(lanelet_id)
            if bounds is None:
                return SemanticIntervalSet.unknown(
                    source, failed_lanelet=int(lanelet_id)
                )
            intervals.append(bounds)
        return SemanticIntervalSet.from_nominal(
            intervals,
            uncertainty=self.uncertainty,
            source=source,
            diagnostics={"lanelet_count": len(intervals)},
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
        """Classify ``on_incoming_left_of`` without an ego trace fit.

        For one intersection, the monitor's ego distance is

        ``min(front_ego - incoming_start, incoming_end - rear_ego)``.

        Its non-negative set is therefore a longitudinal interval on the
        fixed VP path.  The target distance is evaluated from the immutable
        target trajectory at ``time_step`` and acts as a per-frame gate.
        """
        key = ("on_incoming_left_of", int(time_step), self.other_id)
        region = self._region_cache.get(key)
        if region is None:
            started = time.perf_counter()
            region = self._build_on_incoming_left_of_region(int(time_step))
            self._timing["region_build"] += time.perf_counter() - started
            self._region_cache[key] = region
        domain = self._classify_region(region, reachable)
        return FramePredicateEstimate(domain, region.source, region=region)

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
        key = ("causes_braking_intersection", id(evaluator), int(time_step))
        region = self._region_cache.get(key)
        if region is None:
            started = time.perf_counter()
            region = self._build_causes_braking_region(evaluator, int(time_step))
            self._timing["region_build"] += time.perf_counter() - started
            self._region_cache[key] = region
        domain = self._classify_region(region, reachable)
        return FramePredicateEstimate(domain, region.source, region=region)

    def _build_causes_braking_region(
        self, evaluator: Any, time_step: int
    ) -> SemanticIntervalSet:
        if self.other_id is None:
            return SemanticIntervalSet.unknown("causes_braking:no_target")
        try:
            d_br = float(evaluator.config["d_br"])
            a_br = float(evaluator.config["a_br"])
        except Exception:
            return SemanticIntervalSet.unknown("causes_braking:invalid_config")
        if d_br < 0.0:
            return SemanticIntervalSet.empty("causes_braking:negative_distance")

        target = self.world.vehicle_by_id(self.other_id)
        target_feature = self._target_front_rear(target, int(time_step))
        if target_feature is None:
            return self._unknown_path_region("causes_braking:target_state_missing")
        front_target, _, acceleration_target = target_feature
        if front_target is None or acceleration_target is None:
            return self._unknown_path_region("causes_braking:target_state_missing")

        acceleration_band = max(1.0e-9, abs(a_br) * 1.0e-9)
        if acceleration_target > a_br + acceleration_band:
            return SemanticIntervalSet.empty(
                "causes_braking:target_not_braking",
                target_acceleration=float(acceleration_target),
            )
        acceleration_certain = acceleration_target < a_br - acceleration_band

        cells = self._target_path_rear_cells(target)
        if cells is None:
            return SemanticIntervalSet.unknown(
                "causes_braking:path_target_mapping_unavailable"
            )
        valid = cells["valid"]
        rear_lower = cells["rear_lower"]
        rear_upper = cells["rear_upper"]
        band_lower = float(front_target)
        band_upper = float(front_target) + d_br
        distance_band = self.uncertainty

        outer_mask = np.logical_not(valid).copy()
        outer_mask |= valid & (
            (rear_upper >= band_lower - distance_band)
            & (rear_lower <= band_upper + distance_band)
        )
        if acceleration_certain and band_lower + distance_band <= band_upper - distance_band:
            inner_mask = valid & (
                (rear_lower >= band_lower + distance_band)
                & (rear_upper <= band_upper - distance_band)
            )
        else:
            # At a_target == a_br, monitor robustness is exactly zero.  Its
            # Boolean and SAT sign conventions differ, so only the outer set
            # may be certified in this boundary band.
            inner_mask = np.zeros_like(valid, dtype=bool)

        inner = self._cell_mask_to_intervals(cells, inner_mask)
        outer = self._cell_mask_to_intervals(cells, outer_mask)
        return SemanticIntervalSet(
            inner_true=inner,
            outer_true=outer,
            complete=True,
            source="causes_braking_intersection",
            diagnostics={
                "cell_count": int(len(valid)),
                "mapped_point_count": int(
                    cells.get("mapped_point_count", len(valid) + 1)
                ),
                "invalid_cell_count": int(np.count_nonzero(~valid)),
                "target_acceleration": float(acceleration_target),
                "acceleration_gate_certain": bool(acceleration_certain),
            },
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

    def _target_path_rear_cells(self, target: Any) -> Optional[Mapping[str, Any]]:
        """Map VP path cells to ego-rear progress on the target lane once."""
        key = (int(target.id), id(target.ref_path_lane))
        if key in self._target_path_rear_cells_cache:
            return self._target_path_rear_cells_cache[key]
        points = np.asarray(self.ref_path, dtype=float)
        if len(points) < 2:
            self._target_path_rear_cells_cache[key] = None
            return None

        target_lane = target.ref_path_lane
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

        target_center_values = _project_points_to_s(target_lane.clcs, points)
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

        finite_trajectory = np.isfinite(trajectory_values)
        samples = list(
            zip(
                trajectory_values[finite_trajectory].tolist(),
                rear_values[finite_trajectory].tolist(),
            )
        )

        samples.sort(key=lambda item: item[0])
        deduplicated = []
        for trajectory_s, rear_target_s in samples:
            if deduplicated and abs(trajectory_s - deduplicated[-1][0]) <= 1.0e-8:
                if not math.isfinite(deduplicated[-1][1]) and math.isfinite(rear_target_s):
                    deduplicated[-1] = (trajectory_s, rear_target_s)
                continue
            deduplicated.append((trajectory_s, rear_target_s))
        if len(deduplicated) < 2:
            self._target_path_rear_cells_cache[key] = None
            return None

        s_values = np.asarray([item[0] for item in deduplicated], dtype=float)
        rear_values = np.asarray([item[1] for item in deduplicated], dtype=float)
        s_lower = s_values[:-1]
        s_upper = s_values[1:]
        rear_left = rear_values[:-1]
        rear_right = rear_values[1:]
        path_lower, path_upper = self._progress.trajectory_bounds
        if float(path_lower) < float(s_values[0]) - 1.0e-8:
            s_lower = np.concatenate(([float(path_lower)], s_lower))
            s_upper = np.concatenate(([float(s_values[0])], s_upper))
            rear_left = np.concatenate(([math.nan], rear_left))
            rear_right = np.concatenate(([math.nan], rear_right))
        if float(path_upper) > float(s_values[-1]) + 1.0e-8:
            s_lower = np.concatenate((s_lower, [float(s_values[-1])]))
            s_upper = np.concatenate((s_upper, [float(path_upper)]))
            rear_left = np.concatenate((rear_left, [math.nan]))
            rear_right = np.concatenate((rear_right, [math.nan]))
        valid = np.isfinite(rear_left) & np.isfinite(rear_right)
        positive_steps = np.diff(s_values)
        median_step = float(np.median(positive_steps[positive_steps > 1.0e-9]))
        max_step = max(2.0, 4.0 * median_step)
        valid &= (s_upper - s_lower) <= max_step
        result = {
            "s_lower": s_lower,
            "s_upper": s_upper,
            "rear_lower": np.minimum(rear_left, rear_right),
            "rear_upper": np.maximum(rear_left, rear_right),
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
        lower, upper = velocity
        # Monitor boolean semantics use strict inequalities.  Equality remains
        # false, so no geometric uncertainty band is needed for this exact
        # velocity interval test.
        if lower > -epsilon and upper < epsilon:
            result = TRUE_DOMAIN
        elif upper <= -epsilon or lower >= epsilon:
            result = FALSE_DOMAIN
        else:
            result = UNKNOWN_DOMAIN
        return FramePredicateEstimate(result, "standstill:velocity_interval")

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
]
