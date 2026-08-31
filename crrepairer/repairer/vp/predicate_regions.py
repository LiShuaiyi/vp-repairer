"""Small one-dimensional interval primitives used by VP predicate bounds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple, Union


DEFAULT_TOLERANCE = 1.0e-9


@dataclass(frozen=True, order=True)
class ClosedInterval:
    """A finite, closed one-dimensional interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError("Interval endpoints must be finite.")
        if lower > upper:
            raise ValueError(
                "Interval lower endpoint exceeds upper endpoint: "
                f"{lower} > {upper}."
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains_interval(
        self,
        other: "IntervalLike",
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        candidate = as_closed_interval(other)
        tolerance = _validate_tolerance(tolerance)
        return (
            candidate.lower >= self.lower - tolerance
            and candidate.upper <= self.upper + tolerance
        )

    def is_disjoint(
        self,
        other: "IntervalLike",
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        candidate = as_closed_interval(other)
        tolerance = _validate_tolerance(tolerance)
        return (
            self.upper < candidate.lower - tolerance
            or candidate.upper < self.lower - tolerance
        )


IntervalLike = Union[ClosedInterval, Tuple[float, float], Sequence[float]]


def _validate_tolerance(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Tolerance must be a finite non-negative number.")
    return value


def as_closed_interval(interval: IntervalLike) -> ClosedInterval:
    """Normalize a two-element interval-like object."""
    if isinstance(interval, ClosedInterval):
        return interval
    if len(interval) != 2:
        raise ValueError(f"Expected two interval endpoints, got {interval!r}.")
    return ClosedInterval(float(interval[0]), float(interval[1]))


def merge_intervals(
    intervals: Iterable[IntervalLike],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[ClosedInterval, ...]:
    """Merge overlapping or numerically touching closed intervals."""
    tolerance = _validate_tolerance(tolerance)
    normalized = sorted(as_closed_interval(interval) for interval in intervals)
    if not normalized:
        return ()

    merged = []
    current_lower = normalized[0].lower
    current_upper = normalized[0].upper
    for interval in normalized[1:]:
        if interval.lower <= current_upper + tolerance:
            current_upper = max(current_upper, interval.upper)
            continue
        merged.append(ClosedInterval(current_lower, current_upper))
        current_lower = interval.lower
        current_upper = interval.upper
    merged.append(ClosedInterval(current_lower, current_upper))
    return tuple(merged)


__all__ = [
    "ClosedInterval",
    "IntervalLike",
    "as_closed_interval",
    "merge_intervals",
]
