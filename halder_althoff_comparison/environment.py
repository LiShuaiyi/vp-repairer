"""Exogenous signals queried by the fixed-path rule encodings."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def broadcast(value: Any, count: int, name: str) -> tuple:
    if value is None:
        return tuple(None for _ in range(count))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return tuple(value for _ in range(count))
    if len(value) == 1:
        return tuple(value[0] for _ in range(count))
    if len(value) != count:
        raise ValueError(f"{name} needs 1 or {count} samples, got {len(value)}")
    return tuple(value)


@dataclass(frozen=True)
class LeadPrediction:
    vehicle_id: str
    rear_s: tuple
    velocity: tuple
    acceleration: tuple
    same_lane: tuple


@dataclass(frozen=True)
class IntersectionSignal:
    s_enter: float
    s_exit: float
    priority_active: tuple
    priority_terms: tuple
    incoming_left: tuple
    relevant_traffic_light: tuple
    target_in_conflict: tuple
    ego_lookahead_steps: int
    target_clearance_steps: int
    intersection_intervals: tuple
    causes_braking_intervals: tuple
    clearance: float = 0.0

    def blocked(self, k: int) -> bool:
        if not bool(self.priority_active[k]):
            return False
        last = len(self.target_in_conflict) - 1
        future_end = min(last, k + self.ego_lookahead_steps)
        future_target = any(
            bool(self.target_in_conflict[j]) for j in range(k, future_end + 1)
        )
        past_start = max(0, k - self.target_clearance_steps)
        recent_target = any(
            bool(self.target_in_conflict[j]) for j in range(past_start, k + 1)
        )
        return future_target or recent_target

    def antecedent(self, k: int) -> bool:
        terms = (
            any(bool(term[k]) for term in self.priority_terms)
            if self.priority_terms else bool(self.priority_active[k])
        )
        return terms and bool(self.incoming_left[k]) and not bool(self.relevant_traffic_light[k])

    def ego_in_conflict(self, s: float) -> bool:
        return self.s_enter - self.clearance <= s <= self.s_exit + self.clearance

    def ego_on_intersection(self, s: float) -> bool:
        return any(lower <= s <= upper for lower, upper in self.intersection_intervals)

    def causes_braking(self, k: int, s: float) -> bool:
        interval = self.causes_braking_intervals[k]
        return interval is not None and interval[0] <= s <= interval[1]

    def target_conflict_in_window(self, k: int) -> bool:
        end = min(len(self.target_in_conflict) - 1, k + self.ego_lookahead_steps)
        return any(bool(self.target_in_conflict[j]) for j in range(k, end + 1))


class Environment:
    def __init__(self, data: Mapping[str, Any], samples: int, dt: float):
        self.samples = samples
        self.dt = dt
        original = data.get("original_trajectory", {})
        self.original_s = broadcast(original.get("s", 0.0), samples, "original_trajectory.s")
        self.original_v = broadcast(original.get("v", 0.0), samples, "original_trajectory.v")
        raw_limits = data.get("speed_limits", {})
        self.speed_limits = {
            key: broadcast(value, samples, f"speed_limits.{key}")
            for key, value in raw_limits.items()
        }
        self.leads: List[LeadPrediction] = []
        for index, lead in enumerate(data.get("lead_vehicles", [])):
            name = str(lead.get("id", index))
            self.leads.append(
                LeadPrediction(
                    vehicle_id=name,
                    rear_s=broadcast(lead["rear_s"], samples, f"lead {name}.rear_s"),
                    velocity=broadcast(lead.get("velocity", 0.0), samples, f"lead {name}.velocity"),
                    acceleration=broadcast(lead.get("acceleration", 0.0), samples, f"lead {name}.acceleration"),
                    same_lane=broadcast(lead.get("same_lane", True), samples, f"lead {name}.same_lane"),
                )
            )
        self.intersections: Dict[str, IntersectionSignal] = {}
        for rule_name, raw in data.get("intersection_rules", {}).items():
            enter, exit_ = (float(x) for x in raw["conflict_interval"])
            if enter > exit_:
                enter, exit_ = exit_, enter
            priority_terms = tuple(
                broadcast(term, samples, f"{rule_name}.priority_terms[{index}]")
                for index, term in enumerate(raw.get("priority_terms", ()))
            )
            intersection_intervals = tuple(
                tuple(sorted(float(value) for value in interval))
                for interval in raw.get("intersection_intervals", ((enter, exit_),))
            )
            raw_braking = raw.get("causes_braking_intervals")
            if raw_braking is None:
                causes_braking_intervals = tuple(None for _ in range(samples))
            elif len(raw_braking) != samples:
                raise ValueError(
                    f"{rule_name}.causes_braking_intervals needs {samples} samples, got {len(raw_braking)}"
                )
            else:
                causes_braking_intervals = tuple(
                    None if interval is None else tuple(sorted(float(value) for value in interval))
                    for interval in raw_braking
                )
            self.intersections[rule_name] = IntersectionSignal(
                s_enter=enter,
                s_exit=exit_,
                priority_active=broadcast(raw.get("priority_active", True), samples, f"{rule_name}.priority_active"),
                priority_terms=priority_terms,
                incoming_left=broadcast(raw.get("incoming_left", True), samples, f"{rule_name}.incoming_left"),
                relevant_traffic_light=broadcast(raw.get("relevant_traffic_light", False), samples, f"{rule_name}.relevant_traffic_light"),
                target_in_conflict=broadcast(raw.get("target_in_conflict", False), samples, f"{rule_name}.target_in_conflict"),
                ego_lookahead_steps=max(0, round(float(raw.get("ego_lookahead_s", 1.0)) / dt)),
                target_clearance_steps=max(0, round(float(raw.get("target_clearance_s", 0.6)) / dt)),
                intersection_intervals=intersection_intervals,
                causes_braking_intervals=causes_braking_intervals,
                clearance=float(raw.get("clearance", 0.0)),
            )

    def effective_speed_limit(self, k: int) -> float:
        values = [
            float(signal[k])
            for signal in self.speed_limits.values()
            if signal[k] is not None
        ]
        return min(values) if values else inf

    def relevant_leads(self, k: int, ego_front_s: float) -> Iterable[LeadPrediction]:
        return (
            lead
            for lead in self.leads
            if bool(lead.same_lane[k]) and float(lead.rear_s[k]) >= ego_front_s
        )
