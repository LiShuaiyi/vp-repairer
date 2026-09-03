"""History-aware lexicographic A* using the released Halder search objects.

The released node identity is ``(t, s, v)``.  That is sufficient only when
every future rule cost is Markovian in the kinematic state.  This adapter uses
all rule transfer memories as part of the graph-state identity while retaining
the authors' vehicle model, first cubic transition, priority queue, rule-cost
integration, and first-popped-goal termination.
"""

from __future__ import annotations

import math
from typing import Any


def freeze_memory(value: Any):
    """Convert arbitrary rule memory into a stable, hashable node-key value."""
    if value is None or isinstance(value, (str, bytes, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return value
    if isinstance(value, tuple):
        return tuple(freeze_memory(item) for item in value)
    if isinstance(value, list):
        return ("list", tuple(freeze_memory(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (freeze_memory(key), freeze_memory(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(freeze_memory(item)) for item in value)))
    try:
        hash(value)
    except TypeError:
        return (type(value).__qualname__, repr(value))
    return value


def node_key(node):
    """Kinematic lattice state augmented by every rule's transfer memory."""
    return (
        node.state.t,
        node.state.s,
        node.state.v,
        tuple(freeze_memory(memory) for memory in node.transfer_values),
    )


def execute_history_aware_astar(planner, priority_queue_cls, better, max_expansions):
    """Run the released lexicographic A* with a complete graph-state key."""
    frontier = priority_queue_cls(
        planner.nof_active_rules,
        [0.0] * planner.nof_active_rules,
        lambda node: node.g_h_violation,
        planner.compareCompleteViolation,
    )
    initial = planner._initial_node
    initial.id = node_key(initial)
    planner._nodes = {initial.id: initial}
    frontier.append(initial)
    explored = set()
    planner._parent_nodes = []
    node_cls = type(initial)

    def raw_children(parent):
        if math.isclose(parent.state.t, 0.0):
            # The released method implements its dense cubic first edge.  It
            # creates fresh nodes already, so use a scratch dictionary to keep
            # its kinematic-only keys out of the strict graph index.
            strict_nodes = planner._nodes
            planner._nodes = {}
            try:
                return planner.getInitialChildren(parent)
            finally:
                planner._nodes = strict_nodes

        children = []
        for action in planner._actions:
            state = planner._vehicle_model.propagate(
                parent.state,
                action,
                planner._dt,
                planner._ds,
                planner._dv,
                planner._initial_state.s,
            )
            if not (planner._initial_state.s <= state.s <= planner._station_horizon[1]):
                continue
            if not (planner._velocity_horizon[0] <= state.v <= planner._velocity_horizon[1]):
                continue
            children.append((node_cls(state.t, state.s, state.v), state))
        return children

    while frontier:
        parent = frontier.pop()
        planner._parent_nodes.append(parent)
        if math.isclose(parent.state.t, planner._time_horizon[1]):
            return parent
        if len(planner._parent_nodes) >= max_expansions:
            raise RuntimeError(
                f"history-aware A* reached max_expansions={max_expansions}"
            )
        explored.add(parent)

        for candidate, state in raw_children(parent):
            g_cost, instant, heuristic, memories = planner._rulebook.evaluateRules(
                parent, candidate, state
            )
            planner.rule_evaluation_count += planner.nof_active_rules
            candidate.update(
                state=state,
                predecessor=parent,
                g_violation=g_cost,
                i_violation=instant,
                h_violation=heuristic,
                transfer_values=memories,
            )
            candidate.id = node_key(candidate)
            existing = planner._nodes.get(candidate.id)
            if existing is None:
                planner._nodes[candidate.id] = candidate
                frontier.append(candidate)
            elif existing in frontier and better(
                tuple(candidate.g_violation), tuple(existing.g_violation)
            ):
                frontier.remove(existing)
                existing.update(
                    state=state,
                    predecessor=parent,
                    g_violation=g_cost,
                    i_violation=instant,
                    h_violation=heuristic,
                    transfer_values=memories,
                )
                frontier.append(existing)
            # With nonnegative edge costs and h=0, an explored label cannot be
            # improved.  This matches the released implementation's policy.
    return None
