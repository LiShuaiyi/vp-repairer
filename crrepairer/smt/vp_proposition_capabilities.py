"""VP proposition capabilities shared by abstraction, domains, and constraints.

The repairer must classify propositions by the predicate action that velocity
planning can execute, rather than by rule-specific substring checks.  Keeping
the registry here also lets the propositional abstraction decide whether a
Boolean operand can be decomposed into independently executable VP literals.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Iterable, Optional


class VPConstraintKind(str, Enum):
    STOP_LINE_UPPER = "stop_line_upper"
    STOP_LINE_BEFORE_REGION = "stop_line_before_region"
    STANDSTILL_VELOCITY = "standstill_velocity"
    OUTSIDE_EGO_CONFLICT = "outside_ego_conflict"
    OUTSIDE_CAUSES_BRAKING = "outside_causes_braking"
    OUTSIDE_INTERSECTION = "outside_intersection"
    INSIDE_SEMANTIC_REGION = "inside_semantic_region"
    OUTSIDE_SEMANTIC_REGION = "outside_semantic_region"


_PREDICATE_CAPABILITIES = {
    "stop_line_in_front": VPConstraintKind.STOP_LINE_UPPER,
    "in_standstill": VPConstraintKind.STANDSTILL_VELOCITY,
}

_NEGATED_PREDICATE_CAPABILITIES = {
    "in_intersection_conflict_area": VPConstraintKind.OUTSIDE_EGO_CONFLICT,
    "causes_braking_intersection": VPConstraintKind.OUTSIDE_CAUSES_BRAKING,
    "on_lanelet_with_type_intersection": VPConstraintKind.OUTSIDE_INTERSECTION,
}

ADDITIONAL_CONSTRAINT_EXTRACTION_ENV = (
    "CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION"
)
_ADDITIONAL_CONSTRAINT_KINDS = frozenset(
    {
        VPConstraintKind.OUTSIDE_CAUSES_BRAKING,
        VPConstraintKind.OUTSIDE_INTERSECTION,
        VPConstraintKind.INSIDE_SEMANTIC_REGION,
        VPConstraintKind.OUTSIDE_SEMANTIC_REGION,
    }
)

CORE_CONSTRAINT_KINDS = frozenset(
    {
        VPConstraintKind.STOP_LINE_UPPER,
        VPConstraintKind.STOP_LINE_BEFORE_REGION,
        VPConstraintKind.STANDSTILL_VELOCITY,
        VPConstraintKind.OUTSIDE_EGO_CONFLICT,
    }
)
DEFERRED_CONSTRAINT_KINDS = _ADDITIONAL_CONSTRAINT_KINDS


def is_deferred_constraint_kind(kind: Optional[VPConstraintKind]) -> bool:
    """Return whether a constraint action belongs to experimental Tier 2."""
    return kind in DEFERRED_CONSTRAINT_KINDS

# Quantitative estimation by itself is not an executable capability.  The
# experimental registry is deliberately explicit by predicate and polarity;
# every entry below has a constraint-ready longitudinal true region.
TURNING_PRIORITY_COMPOSITE = "turning_priority_composite"
_EGO_UNARY_ADDITIONAL_PREDICATES = frozenset(
    {
        "at_traffic_sign_stop",
        "relevant_traffic_light",
        "on_lanelet_with_type_intersection",
        "turning_left",
        "turning_right",
        "going_straight",
    }
)
_EGO_FIRST_BINARY_ADDITIONAL_PREDICATES = frozenset(
    {
        "in_intersection_conflict_area",
        "on_incoming_left_of",
        "causes_braking_intersection",
    }
)
_ADDITIONAL_POSITIVE_PREDICATE_CAPABILITIES = {
    name: VPConstraintKind.INSIDE_SEMANTIC_REGION
    for name in (
        "at_traffic_sign_stop",
        "relevant_traffic_light",
        "on_lanelet_with_type_intersection",
        "in_intersection_conflict_area",
        "turning_left",
        "turning_right",
        "going_straight",
        "on_incoming_left_of",
        "causes_braking_intersection",
        TURNING_PRIORITY_COMPOSITE,
    )
}
_ADDITIONAL_NEGATIVE_PREDICATE_CAPABILITIES = {
    name: VPConstraintKind.OUTSIDE_SEMANTIC_REGION
    for name in (
        "at_traffic_sign_stop",
        "relevant_traffic_light",
        "turning_left",
        "turning_right",
        "going_straight",
        "on_incoming_left_of",
        TURNING_PRIORITY_COMPOSITE,
    )
}


def additional_constraint_extraction_enabled() -> bool:
    """Whether experimental semantic-region IN constraints are extracted.

    They are disabled by default to avoid their additional predicate-estimate,
    SAT-search, and constraint-extraction cost.  This also preserves the
    original experiment setup.  Set
    ``CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION=1`` for the A/B setup.
    """
    value = os.environ.get(ADDITIONAL_CONSTRAINT_EXTRACTION_ENV, "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

_RULE_CAPABILITIES = {
    "R_IN1": frozenset(
        (
            *_PREDICATE_CAPABILITIES.values(),
            VPConstraintKind.STOP_LINE_BEFORE_REGION,
            *_ADDITIONAL_CONSTRAINT_KINDS,
        )
    ),
    "R_IN3": frozenset(
        (*_NEGATED_PREDICATE_CAPABILITIES.values(), *_ADDITIONAL_CONSTRAINT_KINDS)
    ),
    "R_IN3_hand_draft": frozenset(
        (*_NEGATED_PREDICATE_CAPABILITIES.values(), *_ADDITIONAL_CONSTRAINT_KINDS)
    ),
    "R_IN4": frozenset(
        (*_NEGATED_PREDICATE_CAPABILITIES.values(), *_ADDITIONAL_CONSTRAINT_KINDS)
    ),
    "R_IN5": frozenset(
        (*_NEGATED_PREDICATE_CAPABILITIES.values(), *_ADDITIONAL_CONSTRAINT_KINDS)
    ),
}

# Only conjunctions whose leaves all occur here may be relaxed by distributing
# their common temporal prefix.  The condition is structural and independent
# of a traffic-rule identifier.
DECOMPOSABLE_VP_PREDICATES = frozenset(_PREDICATE_CAPABILITIES)

_INDEX_SUFFIX = re.compile(r"__\d+(?:_\d+)*$")


def predicate_base_name(name: object) -> str:
    """Return a stable atomic predicate identity from a monitor node/name."""
    value = getattr(name, "value", name)
    text = str(value)
    return _INDEX_SUFFIX.sub("", text)


def proposition_predicate_names(proposition) -> frozenset[str]:
    """Return explicit leaf identities carried by a proposition node."""
    result = set()
    for child in getattr(proposition, "children", ()):
        evaluator = getattr(child, "evaluator", None)
        if getattr(evaluator, "_turning_ego", None) is not None:
            result.add(TURNING_PRIORITY_COMPOSITE)
            continue
        candidates = (
            getattr(child, "base_name", None),
            getattr(child, "name", None),
            getattr(evaluator, "predicate_name", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            base_name = predicate_base_name(candidate)
            if base_name in _registered_predicate_names():
                result.add(base_name)
                break

    # PropositionNode.children is populated after ttv calculation.  This
    # conservative token fallback supports earlier domain construction while
    # still matching complete predicate identifiers, not arbitrary substrings.
    if not result:
        expression = str(getattr(proposition, "name", proposition))
        tokens = set(re.findall(r"[A-Za-z_]\w*(?:__\d+(?:_\d+)*)?", expression))
        result.update(
            base_name
            for token in tokens
            if (base_name := predicate_base_name(token))
            in _registered_predicate_names()
        )
    return frozenset(result)


def _registered_predicate_names() -> frozenset[str]:
    return frozenset(
        _PREDICATE_CAPABILITIES
        | _NEGATED_PREDICATE_CAPABILITIES
        | _ADDITIONAL_POSITIVE_PREDICATE_CAPABILITIES
        | _ADDITIONAL_NEGATIVE_PREDICATE_CAPABILITIES
    )


def proposition_predicate_identity(proposition) -> Optional[str]:
    """Return the unique structurally identified predicate capability."""
    names = proposition_predicate_names(proposition)
    if len(names) != 1:
        return None
    return next(iter(names))


def proposition_constraint_kind(
    proposition,
    desired_value: Optional[bool] = None,
) -> Optional[VPConstraintKind]:
    """Return the unique executable positive VP action of a proposition.

    The registered bounds implement positive atomic predicates.  A selected
    negated literal, or a temporal proposition whose leaf is internally
    negated (for example ``previous(not(stop_line_in_front))``), is not the
    same action and must not be classified as extractable.
    """
    negative = (
        str(getattr(proposition, "alphabet", "")).startswith("~")
        if desired_value is None
        else not bool(desired_value)
    )
    expression = str(getattr(proposition, "name", proposition))
    name = proposition_predicate_identity(proposition)
    if name is None:
        return None
    if (
        name in _EGO_UNARY_ADDITIONAL_PREDICATES
        and f"{name}__0" not in expression
    ):
        return None
    if (
        name in _EGO_FIRST_BINARY_ADDITIONAL_PREDICATES
        and f"{name}__0_1" not in expression
    ):
        # Only ego-first occurrences are controllable by ego longitudinal VP.
        return None
    if re.search(r"\bnot\s*\(", expression, flags=re.IGNORECASE):
        # A past predicate derived from the negation of a controllable
        # stop-line atom is not a fixed fact.  In a deceleration phase VP can
        # realize it by remaining on the before-region side.  Classify this
        # structurally rather than tying the action to a particular rule.
        if (
            not negative
            and name == "stop_line_in_front"
            and re.match(
                r"^\s*(previous|prev|pre)\s*\(\s*not\s*\(",
                expression,
                flags=re.IGNORECASE,
            )
        ):
            return VPConstraintKind.STOP_LINE_BEFORE_REGION
        return None
    if negative:
        kind = _NEGATED_PREDICATE_CAPABILITIES.get(name)
        if (
            kind in _ADDITIONAL_CONSTRAINT_KINDS
            and not additional_constraint_extraction_enabled()
        ):
            return None
        if kind is not None:
            return kind
        kind = _ADDITIONAL_NEGATIVE_PREDICATE_CAPABILITIES.get(name)
        if kind is not None and additional_constraint_extraction_enabled():
            return kind
        return None
    kind = _PREDICATE_CAPABILITIES.get(name)
    if kind is not None:
        return kind
    kind = _ADDITIONAL_POSITIVE_PREDICATE_CAPABILITIES.get(name)
    if kind is not None and additional_constraint_extraction_enabled():
        return kind
    return None


def has_constraint_kind(proposition, kinds: Iterable[VPConstraintKind]) -> bool:
    return proposition_constraint_kind(proposition) in frozenset(kinds)


def supported_constraint_kinds(rules: Iterable[str]) -> frozenset[VPConstraintKind]:
    """Return registered VP actions for the active rule set."""
    result = set()
    for rule in rules:
        result.update(_RULE_CAPABILITIES.get(rule, ()))
    if not additional_constraint_extraction_enabled():
        result.difference_update(_ADDITIONAL_CONSTRAINT_KINDS)
    return frozenset(result)
