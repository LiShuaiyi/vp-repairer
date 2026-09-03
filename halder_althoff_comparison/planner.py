"""Lexicographic lattice search for minimum-violation velocity planning."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from math import ceil, floor, inf
from typing import Any, Dict, List, Sequence, Tuple

from .environment import Environment
from .model import LatticeConfig, State
from .rules import Rule


@dataclass(frozen=True)
class PlanResult:
    states: tuple[State, ...]
    actions: tuple[float, ...]
    violation_costs: tuple[float, ...]
    rule_names: tuple[str, ...]
    expanded_nodes: int
    generated_nodes: int
    runtime_s: float
    search_limited: bool = False
    rule_evaluations: int = 0
    rule_evaluation_time_s: float = 0.0

    @property
    def compliant(self) -> bool:
        # Comfort is deliberately a final tie-break objective, not a traffic
        # requirement whose negative robustness makes a plan non-compliant.
        return all(
            cost <= 1e-12
            for name, cost in zip(self.rule_names, self.violation_costs)
            if name != "acceleration_comfort"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "states": [state.as_dict() for state in self.states],
            "actions": list(self.actions),
            "violation_costs": dict(zip(self.rule_names, self.violation_costs)),
            "compliant_on_lattice_encoding": self.compliant,
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "runtime_s": self.runtime_s,
            "search_limited": self.search_limited,
            "rule_evaluations": self.rule_evaluations,
            "rule_evaluation_time_s": self.rule_evaluation_time_s,
        }


@dataclass
class _Label:
    state: State
    costs: Tuple[float, ...]
    memories: Tuple[Any, ...]
    parent: int | None
    action: float | None


class MinimumViolationPlanner:
    """Dijkstra/A* with lexicographically ordered rule-violation vectors.

    The heuristic is the all-zero vector, an admissible lower bound because
    every incremental violation cost is non-negative.  Consequently, the
    returned terminal label is lexicographically optimal on this lattice.
    """

    def __init__(
        self,
        config: LatticeConfig,
        environment: Environment,
        rules: Sequence[Rule],
    ):
        self.config = config
        self.environment = environment
        self.rules = tuple(rules)
        expected = config.horizon_steps + 1
        if environment.samples != expected:
            raise ValueError(f"environment has {environment.samples}, expected {expected}")

    def _s_index(self, s: float, origin: float) -> int:
        return round((s - origin) / self.config.ds)

    def _s_value(self, index: int, origin: float) -> float:
        return origin + index * self.config.ds

    def _v_index(self, v: float) -> int:
        return round((v - self.config.v_min) / self.config.dv)

    def _v_value(self, index: int) -> float:
        return self.config.v_min + index * self.config.dv

    def _memory_key(self, memory: Any):
        if memory is None or isinstance(memory, (str, int, float, bool)):
            return memory
        if isinstance(memory, tuple):
            return tuple(self._memory_key(item) for item in memory)
        return repr(memory)

    def _node_key(self, label: _Label, s_origin: float):
        return (
            label.state.k,
            self._s_index(label.state.s, s_origin),
            self._v_index(label.state.v),
            tuple(self._memory_key(memory) for memory in label.memories),
        )

    def _initial_label(self, initial: State) -> _Label:
        memories = []
        costs = []
        for rule in self.rules:
            margin, memory = rule.step(
                None, initial, rule.initial_memory(), self.environment
            )
            memories.append(memory)
            costs.append(max(0.0, -margin) * self.config.dt)
        return _Label(initial, tuple(costs), tuple(memories), None, None)

    def _successors(self, label: _Label, s_origin: float):
        cfg = self.config
        state = label.state
        min_v = max(cfg.v_min, state.v + cfg.a_min * cfg.dt)
        max_v = min(cfg.v_max, state.v + cfg.a_max * cfg.dt)
        first = ceil((min_v - cfg.v_min) / cfg.dv - 1e-10)
        last = floor((max_v - cfg.v_min) / cfg.dv + 1e-10)
        for v_index in range(first, last + 1):
            next_v = self._v_value(v_index)
            acceleration = (next_v - state.v) / cfg.dt
            raw_s = state.s + state.v * cfg.dt + 0.5 * acceleration * cfg.dt**2
            s_index = self._s_index(raw_s, s_origin)
            next_s = self._s_value(s_index, s_origin)
            if abs(next_s - raw_s) > cfg.transition_tolerance:
                continue
            if not (cfg.s_min - 1e-9 <= next_s <= cfg.s_max + 1e-9):
                continue
            next_state = State(state.k + 1, next_s, next_v, acceleration)
            costs = list(label.costs)
            memories = []
            for index, rule in enumerate(self.rules):
                margin, memory = rule.step(
                    state, next_state, label.memories[index], self.environment
                )
                costs[index] += max(0.0, -margin) * cfg.dt
                memories.append(memory)
            yield next_state, tuple(costs), tuple(memories), acceleration

    def plan(self, initial_s: float, initial_v: float) -> PlanResult:
        start = time.perf_counter()
        cfg = self.config
        if not (cfg.s_min <= initial_s <= cfg.s_max):
            raise ValueError("initial_s is outside lattice bounds")
        if not (cfg.v_min <= initial_v <= cfg.v_max):
            raise ValueError("initial_v is outside lattice bounds")
        snapped_v = self._v_value(self._v_index(initial_v))
        if abs(snapped_v - initial_v) > 1e-7:
            raise ValueError(
                "initial_v is not on the velocity grid; choose compatible v_min/dv "
                "or explicitly preprocess the initial state"
            )

        initial = State(0, float(initial_s), snapped_v, 0.0)
        labels: List[_Label] = [self._initial_label(initial)]
        queue = []
        counter = itertools.count()
        heapq.heappush(queue, (labels[0].costs, next(counter), 0))
        best = {self._node_key(labels[0], initial_s): labels[0].costs}
        expanded = 0
        generated = 1

        while queue and expanded < cfg.max_expansions:
            costs, _, label_id = heapq.heappop(queue)
            label = labels[label_id]
            key = self._node_key(label, initial_s)
            if costs != best.get(key):
                continue
            if label.state.k == cfg.horizon_steps:
                return self._result(labels, label_id, expanded, generated, start, False)
            expanded += 1
            for state, child_costs, memories, action in self._successors(label, initial_s):
                child = _Label(state, child_costs, memories, label_id, action)
                child_key = self._node_key(child, initial_s)
                if child_costs >= best.get(child_key, tuple(inf for _ in self.rules)):
                    continue
                best[child_key] = child_costs
                labels.append(child)
                child_id = len(labels) - 1
                generated += 1
                heapq.heappush(queue, (child_costs, next(counter), child_id))

        if expanded >= cfg.max_expansions:
            raise RuntimeError(
                f"search reached max_expansions={cfg.max_expansions} before the horizon"
            )
        raise RuntimeError("lattice has no trajectory reaching the requested horizon")

    def _result(self, labels, final_id, expanded, generated, start, limited):
        states = []
        actions = []
        label = labels[final_id]
        final_costs = label.costs
        while True:
            states.append(label.state)
            if label.action is not None:
                actions.append(label.action)
            if label.parent is None:
                break
            label = labels[label.parent]
        states.reverse()
        actions.reverse()
        return PlanResult(
            states=tuple(states),
            actions=tuple(actions),
            violation_costs=final_costs,
            rule_names=tuple(rule.name for rule in self.rules),
            expanded_nodes=expanded,
            generated_nodes=generated,
            runtime_s=time.perf_counter() - start,
            search_limited=limited,
        )
