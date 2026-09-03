"""Incremental quantitative semantics for the supported RG/IN rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Dict, Mapping, Sequence, Tuple

from .environment import Environment, LeadPrediction
from .model import State


def _scaled(value: float, maximum: float) -> float:
    """Match the monitor's default clipped robustness normalization."""
    return max(-1.0, min(1.0, value / maximum))


class Rule:
    name = "rule"

    def initial_memory(self) -> Any:
        return None

    def step(
        self, previous: State | None, current: State, memory: Any, env: Environment
    ) -> Tuple[float, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class VehicleConstraintsRule(Rule):
    v_min: float
    v_max: float
    a_min: float
    a_max: float
    name: str = "vehicle_constraints"

    def step(self, previous, current, memory, env):
        margin = min(
            current.v - self.v_min,
            self.v_max - current.v,
            current.a - self.a_min,
            self.a_max - current.a,
        )
        return margin, None


@dataclass(frozen=True)
class SafeDistanceRule(Rule):
    ego_length: float = 4.5
    reaction_time: float = 0.4
    ego_deceleration: float = 10.0
    other_deceleration: float = 10.5
    name: str = "R_G1"

    def safe_distance(self, ego_v: float, lead_v: float) -> float:
        return max(
            0.0,
            lead_v * lead_v / (-2.0 * self.other_deceleration)
            - ego_v * ego_v / (-2.0 * self.ego_deceleration)
            + ego_v * self.reaction_time,
        )

    def lead_margin(self, state: State, lead: LeadPrediction, env: Environment) -> float:
        gap = float(lead.rear_s[state.k]) - (state.s + 0.5 * self.ego_length)
        return gap - self.safe_distance(state.v, float(lead.velocity[state.k]))

    def step(self, previous, current, memory, env):
        leads = list(env.relevant_leads(current.k, current.s + 0.5 * self.ego_length))
        if not leads:
            return inf, None
        return min(self.lead_margin(current, lead, env) for lead in leads), None


@dataclass(frozen=True)
class UnnecessaryBrakingRule(SafeDistanceRule):
    abrupt_threshold: float = -2.0
    max_longitudinal_distance: float = 200.0
    max_acceleration: float = 10.5
    name: str = "R_G2"

    def step(self, previous, current, memory, env):
        # Robustness of antecedent "brakes abruptly": threshold - acceleration.
        abrupt = _scaled(
            self.abrupt_threshold - current.a, self.max_acceleration
        )
        if abrupt <= 0.0:
            return -abrupt, None
        # R_G2 quantifies an existential target but conjoins `precedes`, whose
        # monitor semantics selects only the nearest same-lane vehicle ahead.
        # Letting any farther lead justify ego braking is too permissive.
        leads = list(
            env.relevant_leads(current.k, current.s + 0.5 * self.ego_length)
        )
        if not leads:
            consequent = -inf
        else:
            lead = min(leads, key=lambda item: float(item.rear_s[current.k]))
            unsafe = -_scaled(
                self.lead_margin(current, lead, env),
                self.max_longitudinal_distance,
            )
            comparable_braking = _scaled(
                current.a
                - (float(lead.acceleration[current.k]) + self.abrupt_threshold),
                self.max_acceleration,
            )
            consequent = max(unsafe, comparable_braking)
        # Quantitative implication: max(-antecedent, consequent).
        return max(-abrupt, consequent), None


@dataclass(frozen=True)
class SpeedLimitRule(Rule):
    name: str = "R_G3"

    def step(self, previous, current, memory, env):
        return env.effective_speed_limit(current.k) - current.v, None


@dataclass(frozen=True)
class StopLineRule(Rule):
    stop_line_s: float
    ego_length: float = 4.5
    standstill_tolerance: float = 0.01
    approach_distance: float = 1.0
    stop_duration_s: float = 3.0
    history_s: float = 3.0
    name: str = "R_IN1"

    def initial_memory(self):
        # Minimal future-equivalence state for
        # once(historically[0, stop_duration](stop_line and standstill)).
        # Keeping the raw floating-point trace here prevents A* from merging
        # paths that have exactly the same future obligations.
        return (0, False)

    def step(self, previous, current, memory, env):
        required = max(1, round(self.stop_duration_s / env.dt) + 1)
        consecutive, completed = memory
        front = current.s + 0.5 * self.ego_length
        distance = self.stop_line_s - front
        stop_line_margin = min(distance, self.approach_distance - distance)
        qualifies = (
            stop_line_margin >= 0.0
            and self.standstill_tolerance - abs(current.v) >= 0.0
        )
        prior_consecutive = consecutive
        if completed:
            # `once` remains true forever.  Canonicalizing the memory also
            # makes all already-satisfied paths merge at the same lattice node.
            new_memory = (required, True)
        else:
            consecutive = min(required, consecutive + 1) if qualifies else 0
            completed = consecutive >= required
            new_memory = (required, True) if completed else (consecutive, False)

        if previous is None:
            return inf, new_memory
        prev_front = previous.s + 0.5 * self.ego_length
        crossed = prev_front <= self.stop_line_s < front
        if not crossed:
            return inf, new_memory
        if completed:
            return 0.0, new_memory

        # The finite-state memory preserves the exact Boolean STL language.
        # For a violating crossing, use the missing fraction of the required
        # stop duration as a bounded quantitative search cost.  The crossing
        # sample itself is already beyond the line, hence prior_consecutive is
        # the useful duration immediately before it.
        deficit = max(1, required - prior_consecutive) / required
        return -float(deficit), new_memory


@dataclass(frozen=True)
class IntersectionYieldRule(Rule):
    name: str

    def step(self, previous, current, memory, env):
        signal = env.intersections.get(self.name)
        if signal is None:
            raise ValueError(f"missing intersection_rules.{self.name} signal")
        if not signal.blocked(current.k):
            return inf, None
        lower = signal.s_enter - signal.clearance
        upper = signal.s_exit + signal.clearance
        # Robustness of s outside [lower, upper]. Boundary contact is neutral.
        return max(lower - current.s, current.s - upper), None


@dataclass(frozen=True)
class AccelerationComfortRule(Rule):
    name: str = "acceleration_comfort"

    def step(self, previous, current, memory, env):
        return -(current.a * current.a), None


@dataclass(frozen=True)
class TrajectoryTrackingRule(Rule):
    """Lowest-priority repair objective preserving the supplied motion."""

    position_scale: float = 1.0
    velocity_scale: float = 1.0
    name: str = "trajectory_tracking"

    def step(self, previous, current, memory, env):
        ds = (current.s - float(env.original_s[current.k])) / self.position_scale
        dv = (current.v - float(env.original_v[current.k])) / self.velocity_scale
        return -(ds * ds + dv * dv), None


def build_rulebook(
    order: Sequence[str], parameters: Mapping[str, Any], lattice
) -> tuple[Rule, ...]:
    rules = []
    for name in order:
        params: Dict[str, Any] = dict(parameters.get(name, {}))
        if name == "vehicle_constraints":
            params = {
                "v_min": lattice.v_min,
                "v_max": lattice.v_max,
                "a_min": lattice.a_min,
                "a_max": lattice.a_max,
                **params,
            }
            rule = VehicleConstraintsRule(**params)
        elif name == "R_G1":
            rule = SafeDistanceRule(**params)
        elif name == "R_G2":
            rule = UnnecessaryBrakingRule(**params)
        elif name == "R_G3":
            rule = SpeedLimitRule(**params)
        elif name == "R_IN1":
            rule = StopLineRule(**params)
        elif name in {"R_IN3", "R_IN3_hand_draft", "R_IN4", "R_IN5"}:
            rule = IntersectionYieldRule(name=name, **params)
        elif name == "acceleration_comfort":
            rule = AccelerationComfortRule(**params)
        elif name == "trajectory_tracking":
            rule = TrajectoryTrackingRule(**params)
        else:
            raise ValueError(f"unsupported rule {name!r}")
        rules.append(rule)
    if not rules:
        raise ValueError("rule_order must not be empty")
    return tuple(rules)
