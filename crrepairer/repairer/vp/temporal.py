"""Temporal-window expansion for velocity-planning constraints.

The monitor pastifies bounded future operators before exposing proposition
names to the repairer.  Consequently, a proposition is evaluated at a delayed
anchor while its leaf predicate still refers to an earlier trajectory state.
This module maps those delayed proposition anchors back to the trajectory time
steps on which VP constraints must be imposed.

VP deliberately uses a conservative approximation here: existential temporal
operators are expanded in exactly the same way as universal operators.  Thus
every sampled point in the shifted interval is constrained.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Sequence, Tuple


_TEMPORAL_PREFIX = re.compile(
    r"^\s*(once|eventually|historically|globally|always|previous|prev|pre)\b",
    re.IGNORECASE,
)
_INTERVAL = re.compile(r"^\[\s*([^,]+?)\s*,\s*([^\]]+?)\s*\]")


@dataclass(frozen=True)
class TemporalExpansion:
    """Conservative temporal support of one monitor proposition."""

    leaf_expression: str
    operators: Tuple[str, ...]
    offsets: Optional[Tuple[int, ...]]

    @property
    def is_unbounded(self) -> bool:
        return self.offsets is None


@dataclass(frozen=True)
class TemporalConstraintInterval:
    """Closed interval of trajectory frames carrying a leaf constraint."""

    start: Optional[int]
    end: Optional[int]

    @property
    def is_empty(self) -> bool:
        return self.start is None or self.end is None

    @property
    def count(self) -> int:
        if self.is_empty:
            return 0
        return self.end - self.start + 1

    def contains(self, time_step: int) -> bool:
        return (
            not self.is_empty
            and self.start <= time_step <= self.end
        )


def _strip_outer_parentheses(expression: str) -> str:
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        encloses_all = True
        for index, character in enumerate(expression):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(expression) - 1:
                    encloses_all = False
                    break
            if depth < 0:
                encloses_all = False
                break
        if not encloses_all or depth != 0:
            break
        expression = expression[1:-1].strip()
    return expression


def _parse_bound(value: str) -> Fraction:
    value = value.strip().lower()
    for suffix in ("seconds", "second", "secs", "sec", "s"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break
    return Fraction(value)


def _seconds_to_sample_interval(
    lower: Fraction,
    upper: Fraction,
    dt: float,
) -> Tuple[int, ...]:
    if dt <= 0:
        raise ValueError(f"Scenario dt must be positive, got {dt}.")
    if lower < 0 or upper < lower:
        raise ValueError(f"Invalid temporal interval [{lower}, {upper}].")

    # Use a small tolerance because decimal scenario sampling periods (notably
    # 0.1 and 0.2) are not exactly representable as binary floats.
    lower_step = math.ceil(float(lower) / dt - 1e-9)
    upper_step = math.floor(float(upper) / dt + 1e-9)
    if lower_step > upper_step:
        raise ValueError(
            f"Temporal interval [{lower}, {upper}] contains no samples for dt={dt}."
        )
    return tuple(range(lower_step, upper_step + 1))


def _extract_call(expression: str, prefix_end: int) -> Tuple[Optional[str], str]:
    remainder = expression[prefix_end:].lstrip()
    interval = None
    if remainder.startswith("["):
        match = _INTERVAL.match(remainder)
        if match is None:
            raise ValueError(f"Malformed temporal interval in {expression!r}.")
        interval = (match.group(1), match.group(2))
        remainder = remainder[match.end() :].lstrip()

    if not remainder.startswith("("):
        raise ValueError(f"Temporal operator has no parenthesized operand: {expression!r}.")

    depth = 0
    closing_index = None
    for index, character in enumerate(remainder):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                closing_index = index
                break
        if depth < 0:
            break
    if closing_index is None or remainder[closing_index + 1 :].strip():
        raise ValueError(f"Unbalanced temporal operand in {expression!r}.")
    return interval, remainder[1:closing_index]


@lru_cache(maxsize=256)
def expand_temporal_expression(expression: str, dt: float) -> TemporalExpansion:
    """Expand nested unary temporal prefixes into conservative sample offsets.

    Past operators contribute negative offsets and future operators positive
    offsets.  Nested intervals are combined by a Minkowski sum.  An unbounded
    operator yields ``offsets=None`` so callers can preserve the existing full
    planning-horizon behavior.
    """

    current = _strip_outer_parentheses(expression)
    offsets = {0}
    operators = []
    unbounded = False

    while True:
        match = _TEMPORAL_PREFIX.match(current)
        if match is None:
            break
        operator = match.group(1).lower()
        interval, operand = _extract_call(current, match.end())
        operators.append(operator)

        if operator in {"previous", "prev", "pre"}:
            operator_offsets: Sequence[int] = (-1,)
        elif interval is None:
            # Unbounded once/historically/always cannot be represented by a
            # finite offset set without the concrete trajectory bounds.
            unbounded = True
            operator_offsets = (0,)
        else:
            lower = _parse_bound(interval[0])
            upper = _parse_bound(interval[1])
            samples = _seconds_to_sample_interval(lower, upper, dt)
            direction = 1 if operator in {"eventually", "globally", "always"} else -1
            operator_offsets = tuple(direction * sample for sample in samples)

        offsets = {
            existing + operator_offset
            for existing in offsets
            for operator_offset in operator_offsets
        }
        current = _strip_outer_parentheses(operand)

    return TemporalExpansion(
        leaf_expression=current,
        operators=tuple(operators),
        offsets=None if unbounded else tuple(sorted(offsets)),
    )


def constraint_time_interval(
    expression: str,
    dt: float,
    trajectory_start: int,
    planning_start: int,
    trajectory_end: int,
    future_time_step: int,
) -> Tuple[TemporalConstraintInterval, TemporalExpansion, int]:
    """Return leaf-predicate frames required by the outer all-time rule.

    Each source rule time is shifted by ``future_time_step`` by RTAMT's
    pastification.  Since both the outer all-time anchor set and every sampled
    temporal interval are contiguous, the union of all shifted windows is
    computed directly from its endpoints.  There is no per-anchor expansion.
    Only modifiable trajectory frames are returned.  The final integer is the
    number of logical anchor/offset pairs represented by the closed-form
    interval, useful for diagnostics.
    """

    expansion = expand_temporal_expression(expression, dt)
    modifiable_start = max(trajectory_start, planning_start)
    if modifiable_start > trajectory_end:
        return TemporalConstraintInterval(None, None), expansion, 0
    if expansion.offsets is None:
        interval = TemporalConstraintInterval(modifiable_start, trajectory_end)
        return interval, expansion, interval.count

    # The traffic rule is required at every original trajectory time.  RTAMT
    # may expose a future-time formula later through pastification, but that
    # monitor delay must not shorten the source anchor interval.  We therefore
    # retain all source anchors and clip only the resulting leaf-predicate
    # interval to the finite VP trajectory.
    last_source = trajectory_end
    if trajectory_start > last_source:
        return TemporalConstraintInterval(None, None), expansion, 0

    first_delayed_anchor = trajectory_start + int(future_time_step)
    last_delayed_anchor = last_source + int(future_time_step)
    first_leaf = first_delayed_anchor + min(expansion.offsets)
    last_leaf = last_delayed_anchor + max(expansion.offsets)
    first_leaf = max(first_leaf, modifiable_start)
    last_leaf = min(last_leaf, trajectory_end)
    if first_leaf > last_leaf:
        interval = TemporalConstraintInterval(None, None)
    else:
        interval = TemporalConstraintInterval(first_leaf, last_leaf)

    anchor_count = last_source - trajectory_start + 1
    pair_count = anchor_count * len(expansion.offsets)
    return interval, expansion, pair_count
