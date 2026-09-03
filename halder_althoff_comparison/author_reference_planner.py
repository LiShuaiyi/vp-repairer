"""Adapter around the search engine released with Halder & Althoff (ITSC 2022).

The author checkout is treated as read-only.  This module supplies only API
compatibility and a small rulebook facade for the project's RG/IN rules.  The
state lattice, cubic first transition, vehicle model, node representation and
lexicographic A* implementation are executed directly from the released code.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

from .model import State
from .planner import PlanResult
from .history_aware_search import execute_history_aware_astar


DEFAULT_AUTHOR_SOURCE = Path(
    "/home/shuaiyi/Downloads/Halder-2022-ITSC/software_data/"
    "minimum-violation-velocity-planner-master"
)


def _load_author_planner(source: Path):
    """Load the unmodified release despite two CommonRoad symbol moves."""
    source = source.resolve()
    if not (source / "minimum_violation_velocity_planner").is_dir():
        raise FileNotFoundError(f"Halder source package not found under {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import commonroad.scenario.traffic_sign_interpreter as interpreter
    if not hasattr(interpreter, "TrafficSigInterpreter"):
        interpreter.TrafficSigInterpreter = interpreter.TrafficSignInterpreter

    import commonroad.scenario.traffic_sign as traffic_sign
    import commonroad.scenario.traffic_light as traffic_light
    if not hasattr(traffic_sign, "TrafficLight"):
        traffic_sign.TrafficLight = traffic_light.TrafficLight
        traffic_sign.TrafficLightState = traffic_light.TrafficLightState

    from minimum_violation_velocity_planner.velocity_planner.velocity_planner import (
        VelocityPlanner,
    )
    return VelocityPlanner


@dataclass(frozen=True)
class _RuleTag:
    tag: str


class ProjectRulebook:
    """Author-compatible, current-value integration rulebook.

    Rules are deliberately evaluated on every explored edge.  This corresponds
    to ``integration_strategy=2`` and the non-minimized A* path in the release.
    """

    def __init__(self, rules, environment, dt):
        self.rules = tuple(rules)
        self.environment = environment
        self.dt = float(dt)
        self._tags = tuple(_RuleTag(rule.name) for rule in self.rules)
        self.evaluation_count = 0
        self.evaluation_time_s = 0.0

    def nofActiveRules(self):
        return len(self.rules)

    def getActiveRules(self):
        return self._tags

    def getTotalRuleEvaluationTime(self):
        return self.evaluation_time_s

    def _state(self, value):
        return State(
            k=max(0, min(self.environment.samples - 1, round(value.t / self.dt))),
            s=float(value.s),
            v=float(value.v),
            a=float(value.a),
        )

    def _one(self, rule_index, previous, current, memory):
        start = time.perf_counter()
        margin, new_memory = self.rules[rule_index].step(
            previous, current, memory, self.environment
        )
        self.evaluation_time_s += time.perf_counter() - start
        self.evaluation_count += 1
        return max(0.0, -float(margin)), 0.0, new_memory

    def evaluateInitialRules(self, state):
        current = self._state(state)
        instantaneous, heuristic, memories = [], [], []
        for index, rule in enumerate(self.rules):
            value, h_value, memory = self._one(
                index, None, current, rule.initial_memory()
            )
            instantaneous.append(value)
            heuristic.append(h_value)
            memories.append(memory)
        return [0.0] * len(self.rules), instantaneous, heuristic, memories

    def evaluateRules(self, parent, child, updated_child_state):
        current = self._state(updated_child_state)
        if parent.id[0] == 0:
            delta_s = updated_child_state.s - parent.state.s
            parent.state.a = -(
                2.0
                * (
                    2.0 * self.dt * parent.state.v
                    - 3.0 * delta_s
                    + self.dt * updated_child_state.v
                )
                / self.dt**2
            )
            parent_g, parent_i, _, parent_memory = self.evaluateInitialRules(parent.state)
            # The project monitor evaluates the first state as well.  The
            # released current-value integrator otherwise drops a0 of the
            # cubic initial edge, allowing an unobserved abrupt brake at k=0.
            parent_g = [self.dt * value for value in parent_i]
        else:
            parent_g = parent.g_violation
            parent_i = parent.i_violation
            parent_memory = parent.transfer_values
        previous = self._state(parent.state)

        new_g, new_i, new_h, new_memory = [], [], [], []
        for index in range(len(self.rules)):
            value, heuristic, memory = self._one(
                index, previous, current, parent_memory[index]
            )
            # Released planner uses current-value (box) integration here.
            new_g.append(parent_g[index] + self.dt * value)
            new_i.append(value)
            new_h.append(heuristic)
            new_memory.append(memory)
        return new_g, new_i, new_h, new_memory

    def evaluateRuleAtLevel(self, parent, child, updated_child_state, level):
        current = self._state(updated_child_state)
        if parent.id[0] == 0:
            delta_s = updated_child_state.s - parent.state.s
            parent.state.a = -(
                2.0
                * (
                    2.0 * self.dt * parent.state.v
                    - 3.0 * delta_s
                    + self.dt * updated_child_state.v
                )
                / self.dt**2
            )
            parent_g, parent_i, _, parent_memory = self.evaluateInitialRules(parent.state)
            parent_g = [self.dt * value for value in parent_i]
        else:
            parent_g = parent.g_violation
            parent_i = parent.i_violation
            parent_memory = parent.transfer_values
        previous = self._state(parent.state)
        value, heuristic, memory = self._one(
            level, previous, current, parent_memory[level]
        )
        return parent_g[level] + self.dt * value, value, heuristic, memory


def plan_with_author_engine(
    scenario,
    planning_problem,
    reference_lane,
    config,
    environment,
    rules,
    author_source=DEFAULT_AUTHOR_SOURCE,
    planner="A_STAR",
    history_aware=True,
):
    """Execute the released graph search and return the local result schema."""
    VelocityPlanner = _load_author_planner(Path(author_source))
    vehicle = parameters_vehicle2()
    grid = (
        config.dt,
        config.dv,
        config.ds,
        config.dv / config.dt,
        (0.0, config.horizon_steps * config.dt),
        (config.s_min, config.s_max),
        (config.v_min, min(config.v_max, vehicle.longitudinal.v_max)),
    )
    params = {
        "general": {"state_rounding_precision": 8},
        "vehicle_model": {
            "v_min_extension": 0.0,
            "v_max_extension": 0.0,
            "a_min_extension": 0.0,
            "a_max_extension": 0.0,
        },
    }
    rulebook = ProjectRulebook(rules, environment, config.dt)
    vp = VelocityPlanner(
        scenario,
        planning_problem,
        reference_lane.clcs,
        rulebook,
        grid,
        vehicle,
        params,
    )
    if history_aware:
        from minimum_violation_velocity_planner.common.priority_queue import (
            PriorityQueueLex,
        )
        from minimum_violation_velocity_planner.common.utils import (
            isLexicographicallyBetter,
        )

        vp.executeAStar = lambda verbose=False: execute_history_aware_astar(
            vp,
            PriorityQueueLex,
            isLexicographicallyBetter,
            config.max_expansions,
        )
    started = time.perf_counter()
    vp.plan(planner, verbose=False)
    runtime = time.perf_counter() - started
    states = tuple(
        State(
            k=round(value.t / config.dt),
            s=float(value.s),
            v=float(value.v),
            a=float(value.a),
        )
        for value in vp.optimal_trajectory
    )
    actions = tuple(state.a for state in states[1:])
    return PlanResult(
        states=states,
        actions=actions,
        violation_costs=tuple(float(x) for x in vp.optimal_violations[-1]),
        rule_names=tuple(rule.name for rule in rules),
        expanded_nodes=len(vp._parent_nodes),
        generated_nodes=len(vp._nodes),
        runtime_s=runtime,
        rule_evaluations=rulebook.evaluation_count,
        rule_evaluation_time_s=rulebook.evaluation_time_s,
    )
