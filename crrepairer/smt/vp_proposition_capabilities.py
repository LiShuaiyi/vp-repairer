"""VP proposition capabilities shared by abstraction, domains, and constraints.

The repairer must classify propositions by the predicate action that velocity
planning can execute, rather than by rule-specific substring checks.  Keeping
the registry here also lets the propositional abstraction decide whether a
Boolean operand can be decomposed into independently executable VP literals.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable, Optional


class VPConstraintKind(str, Enum):
    STOP_LINE_UPPER = "stop_line_upper"
    STANDSTILL_VELOCITY = "standstill_velocity"


_PREDICATE_CAPABILITIES = {
    "stop_line_in_front": VPConstraintKind.STOP_LINE_UPPER,
    "in_standstill": VPConstraintKind.STANDSTILL_VELOCITY,
}

_RULE_CAPABILITIES = {
    "R_IN1": frozenset(_PREDICATE_CAPABILITIES.values()),
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
        candidates = (
            getattr(child, "base_name", None),
            getattr(child, "name", None),
            getattr(getattr(child, "evaluator", None), "predicate_name", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            base_name = predicate_base_name(candidate)
            if base_name in _PREDICATE_CAPABILITIES:
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
            if (base_name := predicate_base_name(token)) in _PREDICATE_CAPABILITIES
        )
    return frozenset(result)


def proposition_constraint_kind(proposition) -> Optional[VPConstraintKind]:
    """Return the unique executable positive VP action of a proposition.

    The registered bounds implement positive atomic predicates.  A selected
    negated literal, or a temporal proposition whose leaf is internally
    negated (for example ``previous(not(stop_line_in_front))``), is not the
    same action and must not be classified as extractable.
    """
    if str(getattr(proposition, "alphabet", "")).startswith("~"):
        return None
    expression = str(getattr(proposition, "name", proposition))
    if re.search(r"\bnot\s*\(", expression, flags=re.IGNORECASE):
        return None
    names = proposition_predicate_names(proposition)
    if len(names) != 1:
        return None
    return _PREDICATE_CAPABILITIES[next(iter(names))]


def has_constraint_kind(proposition, kinds: Iterable[VPConstraintKind]) -> bool:
    return proposition_constraint_kind(proposition) in frozenset(kinds)


def supported_constraint_kinds(rules: Iterable[str]) -> frozenset[VPConstraintKind]:
    """Return registered VP actions for the active rule set."""
    result = set()
    for rule in rules:
        result.update(_RULE_CAPABILITIES.get(rule, ()))
    return frozenset(result)
