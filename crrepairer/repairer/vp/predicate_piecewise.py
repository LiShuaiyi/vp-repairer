"""Generic one-dimensional piecewise predicate-domain estimation.

Predicate-specific code remains responsible for providing a complete set of
critical points (continuous robustness) or certified inner/outer truth
regions (Boolean geometry).  The helpers in this module only partition one
reachable interval, classify every resulting cell, and combine the cell
domains.  They deliberately do not sample monitor values or infer missing
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, FrozenSet, Iterable, Sequence, Tuple


TruthDomain = FrozenSet[int]
FALSE_DOMAIN: TruthDomain = frozenset((0,))
TRUE_DOMAIN: TruthDomain = frozenset((1,))
UNKNOWN_DOMAIN: TruthDomain = frozenset((0, 1))


@dataclass(frozen=True)
class MonotoneSegmentEstimate:
    lower: float
    upper: float
    robustness_lower: float
    robustness_upper: float
    robustness_min: float
    robustness_max: float
    domain: TruthDomain


@dataclass(frozen=True)
class PiecewiseEstimate:
    domain: TruthDomain
    partition_points: Tuple[float, ...]
    segments: Tuple[MonotoneSegmentEstimate, ...]


def _combine_domains(domains: Iterable[TruthDomain]) -> TruthDomain:
    """Return the truth values possible on at least one partition cell."""
    possible = set()
    for domain in domains:
        possible.update(domain)
    if not possible:
        return UNKNOWN_DOMAIN
    return frozenset(possible)


def _partition_points(
    reachable: Sequence[float],
    critical_points: Iterable[float],
    *,
    tolerance: float,
) -> Tuple[float, ...]:
    try:
        lower, upper = sorted((float(reachable[0]), float(reachable[1])))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("reachable must contain two finite endpoints") from exc
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("reachable endpoints must be finite")

    scale = max(1.0, abs(lower), abs(upper))
    absolute_tolerance = max(0.0, float(tolerance)) * scale
    interior = []
    for point in critical_points:
        try:
            point = float(point)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(point):
            continue
        if point <= lower + absolute_tolerance:
            continue
        if point >= upper - absolute_tolerance:
            continue
        interior.append(point)

    points = [lower]
    for point in sorted(interior):
        if point - points[-1] > absolute_tolerance:
            points.append(point)
    if upper - points[-1] > absolute_tolerance:
        points.append(upper)
    elif len(points) == 1:
        # A point reachable set is represented by one degenerate segment.
        points.append(upper)
    else:
        points[-1] = upper
    return tuple(points)


def _classify_robustness_range(
    robustness_min: float,
    robustness_max: float,
    *,
    uncertainty: float,
) -> TruthDomain:
    """Classify a robustness interval with an explicit unknown boundary band."""
    uncertainty = max(0.0, float(uncertainty))
    if robustness_min > uncertainty:
        return TRUE_DOMAIN
    if robustness_max < -uncertainty:
        return FALSE_DOMAIN
    return UNKNOWN_DOMAIN


def estimate_piecewise_monotone_1d(
    reachable: Sequence[float],
    critical_points: Iterable[float],
    robustness: Callable[[float], float],
    *,
    uncertainty: float,
    point_tolerance: float = 1.0e-9,
) -> PiecewiseEstimate:
    """Bound a piecewise-monotone robustness function from segment endpoints.

    Completeness of ``critical_points`` and monotonicity between adjacent
    points are preconditions supplied by the predicate-specific caller.
    Robustness at zero and throughout the configured uncertainty band remains
    unknown, so Boolean boundary conventions are never silently changed.
    """
    points = _partition_points(
        reachable, critical_points, tolerance=point_tolerance
    )
    segments = []
    for lower, upper in zip(points[:-1], points[1:]):
        value_lower = float(robustness(lower))
        value_upper = float(robustness(upper))
        if not (math.isfinite(value_lower) and math.isfinite(value_upper)):
            domain = UNKNOWN_DOMAIN
            robustness_min = -math.inf
            robustness_max = math.inf
        else:
            robustness_min = min(value_lower, value_upper)
            robustness_max = max(value_lower, value_upper)
            domain = _classify_robustness_range(
                robustness_min,
                robustness_max,
                uncertainty=uncertainty,
            )
        segments.append(
            MonotoneSegmentEstimate(
                lower=float(lower),
                upper=float(upper),
                robustness_lower=value_lower,
                robustness_upper=value_upper,
                robustness_min=float(robustness_min),
                robustness_max=float(robustness_max),
                domain=domain,
            )
        )
    return PiecewiseEstimate(
        domain=_combine_domains(segment.domain for segment in segments),
        partition_points=points,
        segments=tuple(segments),
    )


def classify_region_piecewise(
    region: Any,
    reachable: Sequence[float],
    *,
    point_tolerance: float = 1.0e-9,
) -> PiecewiseEstimate:
    """Classify a certified Boolean truth region cell by cell.

    ``region`` is intentionally a small duck-typed protocol: it must expose
    ``inner_true``, ``outer_true``, ``complete``, and ``classify``.  Keeping
    the helper independent avoids coupling generic interval analysis to map
    construction.  Closed adjacent cells conservatively share boundaries;
    any inner/outer uncertainty band therefore remains unknown.
    """
    boundaries = []
    for interval in tuple(region.inner_true) + tuple(region.outer_true):
        boundaries.extend((float(interval.lower), float(interval.upper)))
    points = _partition_points(
        reachable, boundaries, tolerance=point_tolerance
    )
    segments = []
    for lower, upper in zip(points[:-1], points[1:]):
        domain = region.classify((lower, upper))
        # Boolean regions do not invent a quantitative robustness.  The
        # numerical fields merely encode the certified Boolean range for
        # compact, uniform diagnostics.
        if domain == TRUE_DOMAIN:
            robustness_min = robustness_max = 1.0
        elif domain == FALSE_DOMAIN:
            robustness_min = robustness_max = -1.0
        else:
            robustness_min, robustness_max = -1.0, 1.0
        segments.append(
            MonotoneSegmentEstimate(
                lower=float(lower),
                upper=float(upper),
                robustness_lower=float(robustness_min),
                robustness_upper=float(robustness_max),
                robustness_min=float(robustness_min),
                robustness_max=float(robustness_max),
                domain=domain,
            )
        )
    return PiecewiseEstimate(
        domain=_combine_domains(segment.domain for segment in segments),
        partition_points=points,
        segments=tuple(segments),
    )


def piecewise_diagnostics(
    estimate: PiecewiseEstimate,
    *,
    analysis_mode: str,
    analysis_axis: str,
    include_segments: bool = False,
) -> dict:
    """Build compact, consistent diagnostics for a piecewise estimate."""
    diagnostics = {
        "analysis_mode": str(analysis_mode),
        "analysis_axis": str(analysis_axis),
        "partition_points": [float(point) for point in estimate.partition_points],
        "segment_count": len(estimate.segments),
    }
    if include_segments:
        diagnostics["segments"] = [
            {
                "lower": float(segment.lower),
                "upper": float(segment.upper),
                "robustness_min": float(segment.robustness_min),
                "robustness_max": float(segment.robustness_max),
                "domain": sorted(segment.domain),
            }
            for segment in estimate.segments
        ]
    return diagnostics
