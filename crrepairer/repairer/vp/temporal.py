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
from typing import Iterable, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class TemporalConstraintSteps:
    """Exact (possibly disjoint) leaf frames selected by gated anchors."""

    steps: frozenset[int]

    @property
    def start(self) -> Optional[int]:
        return min(self.steps) if self.steps else None

    @property
    def end(self) -> Optional[int]:
        return max(self.steps) if self.steps else None

    @property
    def is_empty(self) -> bool:
        return not self.steps

    @property
    def count(self) -> int:
        return len(self.steps)

    def contains(self, time_step: int) -> bool:
        return int(time_step) in self.steps


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


def constraint_steps_for_anchors(
    expression: str,
    dt: float,
    source_anchors: Iterable[int],
    planning_start: int,
    trajectory_end: int,
    future_time_step: int,
) -> Tuple[TemporalConstraintSteps, TemporalExpansion, int]:
    """Map implication-active anchors to exact VP leaf frames.

    ``future_time_step`` is nonzero for original source-rule anchors and zero
    when callers pass already-delayed pastified-monitor evaluation anchors.
    """

    expansion = expand_temporal_expression(expression, dt)
    anchors = tuple(sorted({int(anchor) for anchor in source_anchors}))
    if not anchors or planning_start > trajectory_end:
        return TemporalConstraintSteps(frozenset()), expansion, 0

    if expansion.offsets is None:
        # An unbounded temporal child extends from every active anchor through
        # the available trajectory tail.
        steps = {
            step
            for anchor in anchors
            for step in range(
                max(planning_start, anchor + int(future_time_step)),
                trajectory_end + 1,
            )
        }
        return TemporalConstraintSteps(frozenset(steps)), expansion, len(steps)

    steps = {
        anchor + int(future_time_step) + int(offset)
        for anchor in anchors
        for offset in expansion.offsets
        if planning_start
        <= anchor + int(future_time_step) + int(offset)
        <= trajectory_end
    }
    pair_count = len(anchors) * len(expansion.offsets)
    return TemporalConstraintSteps(frozenset(steps)), expansion, pair_count


@dataclass(frozen=True)
class TemporalTruthNode:
    """Small AST for three-valued temporal truth propagation."""

    kind: str
    value: str = ""
    children: Tuple["TemporalTruthNode", ...] = ()
    offsets: Optional[Tuple[int, ...]] = ()


def _find_top_level_operator(expression: str, tokens: Sequence[str]):
    """Return the rightmost top-level Boolean operator and its span."""
    depth = 0
    matches = []
    lowered = expression.lower()
    index = 0
    while index < len(expression):
        character = expression[index]
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth -= 1
            index += 1
            continue
        if depth == 0:
            for token in tokens:
                if not lowered.startswith(token, index):
                    continue
                if token.isalpha():
                    before = lowered[index - 1] if index else " "
                    after_index = index + len(token)
                    after = lowered[after_index] if after_index < len(lowered) else " "
                    if (before.isalnum() or before == "_") or (
                        after.isalnum() or after == "_"
                    ):
                        continue
                matches.append((index, index + len(token), token))
                index += len(token)
                break
            else:
                index += 1
            continue
        index += 1
    return matches[-1] if matches else None


@lru_cache(maxsize=512)
def parse_temporal_truth_expression(
    expression: str, dt: float
) -> TemporalTruthNode:
    """Parse temporal/Boolean proposition syntax without flattening nesting."""
    current = _strip_outer_parentheses(str(expression).strip())
    match = _TEMPORAL_PREFIX.match(current)
    if match is not None:
        operator = match.group(1).lower()
        interval, operand = _extract_call(current, match.end())
        if operator in {"previous", "prev", "pre"}:
            offsets = (-1,)
        elif interval is None:
            offsets = None
        else:
            offsets = _seconds_to_sample_interval(
                _parse_bound(interval[0]), _parse_bound(interval[1]), dt
            )
            if operator in {"once", "historically"}:
                offsets = tuple(-item for item in offsets)
        return TemporalTruthNode(
            kind=operator,
            children=(parse_temporal_truth_expression(operand, dt),),
            offsets=offsets,
        )

    lowered = current.lower()
    if lowered.startswith("not("):
        _, operand = _extract_call(current, 3)
        return TemporalTruthNode(
            kind="not",
            children=(parse_temporal_truth_expression(operand, dt),),
        )
    if current.startswith("!"):
        return TemporalTruthNode(
            kind="not",
            children=(parse_temporal_truth_expression(current[1:], dt),),
        )

    # Lowest precedence first.  Rightmost splitting preserves left-associative
    # And/Or while implication is evaluated by its Boolean truth table.
    for kind, tokens in (
        ("implies", ("implies", "->")),
        ("or", ("or", "||")),
        ("and", ("and", "&&")),
    ):
        found = _find_top_level_operator(current, tokens)
        if found is None:
            continue
        start, end, _ = found
        left = current[:start]
        right = current[end:]
        if not left.strip() or not right.strip():
            continue
        return TemporalTruthNode(
            kind=kind,
            children=(
                parse_temporal_truth_expression(left, dt),
                parse_temporal_truth_expression(right, dt),
            ),
        )
    return TemporalTruthNode(kind="atom", value=current.strip())


_FALSE_MASK = 0b01
_TRUE_MASK = 0b10
_UNKNOWN_MASK = _FALSE_MASK | _TRUE_MASK


def _truth_domain_to_mask(domain) -> int:
    """Encode a conservative subset of {0, 1} as two bits."""
    mask = 0
    for value in domain:
        value = int(value)
        if value == 0:
            mask |= _FALSE_MASK
        elif value == 1:
            mask |= _TRUE_MASK
        else:
            return _UNKNOWN_MASK
    return mask or _UNKNOWN_MASK


def _truth_mask_to_domain(mask: int):
    if mask == _FALSE_MASK:
        return frozenset({0})
    if mask == _TRUE_MASK:
        return frozenset({1})
    return frozenset({0, 1})


def _build_binary_truth_table(kind: str):
    table = [[_UNKNOWN_MASK] * 4 for _ in range(4)]
    for left in (_FALSE_MASK, _TRUE_MASK, _UNKNOWN_MASK):
        for right in (_FALSE_MASK, _TRUE_MASK, _UNKNOWN_MASK):
            result = 0
            for lhs in (0, 1):
                if not left & (1 << lhs):
                    continue
                for rhs in (0, 1):
                    if not right & (1 << rhs):
                        continue
                    if kind == "and":
                        value = int(bool(lhs) and bool(rhs))
                    elif kind == "or":
                        value = int(bool(lhs) or bool(rhs))
                    else:
                        value = int((not bool(lhs)) or bool(rhs))
                    result |= 1 << value
            table[left][right] = result or _UNKNOWN_MASK
    return tuple(tuple(row) for row in table)


_NOT_MASK = (0, _TRUE_MASK, _FALSE_MASK, _UNKNOWN_MASK)
_BOOLEAN_MASK_TABLES = {
    kind: _build_binary_truth_table(kind)
    for kind in ("and", "or", "implies")
}


def evaluate_temporal_truth_nodes(
    roots: Sequence[TemporalTruthNode],
    dt: float,
    source_anchors: Iterable[int],
    trace_start: int,
    trace_end: int,
    atomic_domain,
):
    """Batch-lift pre-resolved temporal ASTs using shared caches."""
    roots = tuple(roots)
    trace_start = int(trace_start)
    trace_end = int(trace_end)
    anchors = tuple(
        anchor
        for anchor in sorted({int(item) for item in source_anchors})
        if trace_start <= anchor <= trace_end
    )
    cache = {}
    atomic_cache = {}

    def aggregate(kind, masks):
        existential = kind in {"once", "eventually"}
        seen = False
        can_be_true = not existential
        can_be_false = existential
        for mask in masks:
            seen = True
            if existential:
                can_be_true = can_be_true or bool(mask & _TRUE_MASK)
                can_be_false = can_be_false and bool(mask & _FALSE_MASK)
                if can_be_true and not can_be_false:
                    return _TRUE_MASK
            else:
                can_be_true = can_be_true and bool(mask & _TRUE_MASK)
                can_be_false = can_be_false or bool(mask & _FALSE_MASK)
                if can_be_false and not can_be_true:
                    return _FALSE_MASK
        if not seen:
            return _FALSE_MASK if existential else _TRUE_MASK
        result = 0
        if can_be_true:
            result |= _TRUE_MASK
        if can_be_false:
            result |= _FALSE_MASK
        return result or _UNKNOWN_MASK

    def evaluate(node, anchor):
        key = (node, int(anchor))
        if key in cache:
            return cache[key]
        if node.kind == "atom":
            atomic_key = (node.value, int(anchor))
            if atomic_key not in atomic_cache:
                atomic_cache[atomic_key] = _truth_domain_to_mask(
                    atomic_domain(node.value, int(anchor))
                )
            result = atomic_cache[atomic_key]
        elif node.kind == "not":
            result = _NOT_MASK[evaluate(node.children[0], anchor)]
        elif node.kind in {"and", "or", "implies"}:
            result = _BOOLEAN_MASK_TABLES[node.kind][
                evaluate(node.children[0], anchor)
            ][evaluate(node.children[1], anchor)]
        else:
            child = node.children[0]
            if node.offsets is None:
                if node.kind in {"once", "historically"}:
                    times = range(trace_start, min(trace_end, int(anchor)) + 1)
                elif node.kind in {"eventually", "globally", "always"}:
                    times = range(max(trace_start, int(anchor)), trace_end + 1)
                else:
                    times = ()
            else:
                times = tuple(
                    int(anchor) + int(offset)
                    for offset in node.offsets
                    if trace_start <= int(anchor) + int(offset) <= trace_end
                )
            if node.kind in {"previous", "prev", "pre"}:
                result = evaluate(child, times[0]) if times else _UNKNOWN_MASK
            else:
                result = aggregate(
                    node.kind,
                    (evaluate(child, time_step) for time_step in times),
                )
        cache[key] = result
        return result

    results = []
    for root in roots:
        possible = 0
        for anchor in anchors:
            possible |= evaluate(root, anchor)
            if possible == _UNKNOWN_MASK:
                break
        results.append(_truth_mask_to_domain(possible or _UNKNOWN_MASK))
    return tuple(results)


def evaluate_temporal_truth_domains(
    expressions: Sequence[str],
    dt: float,
    source_anchors: Iterable[int],
    trace_start: int,
    trace_end: int,
    atomic_domain,
):
    """Batch-lift expression strings through shared temporal AST caches."""
    return evaluate_temporal_truth_nodes(
        tuple(
            parse_temporal_truth_expression(str(expression), float(dt))
            for expression in expressions
        ),
        dt,
        source_anchors,
        trace_start,
        trace_end,
        atomic_domain,
    )


def evaluate_temporal_truth_domain(
    expression: str,
    dt: float,
    source_anchors: Iterable[int],
    trace_start: int,
    trace_end: int,
    atomic_domain,
):
    """Backward-compatible single-expression temporal truth lifting."""
    return evaluate_temporal_truth_domains(
        (expression,),
        dt,
        source_anchors,
        trace_start,
        trace_end,
        atomic_domain,
    )[0]
