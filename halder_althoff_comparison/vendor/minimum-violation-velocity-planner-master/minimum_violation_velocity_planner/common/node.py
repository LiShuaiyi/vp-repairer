from typing import List

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class Node:
    def __init__(self, t: float, s: float, v: float, state: LongitudinalState = None, predecessor=None,
                 g_violation: List = None, h_violation: List = None, i_violation: List = None,
                 g_h_violation: List = None, transfer_values: List = None):
        self.id = (t, s, v)
        self.state = state
        self.predecessor = predecessor
        self.g_violation = g_violation
        self.h_violation = h_violation
        self.i_violation = i_violation
        self.g_h_violation = g_h_violation
        self.transfer_values = transfer_values

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def update(self, state: LongitudinalState=None, predecessor=None, g_violation: List=None, h_violation: List=None, i_violation: List=None, transfer_values:List=None):
        if state:
            self.state = state
        if predecessor:
            self.predecessor = predecessor
        if g_violation:
            self.g_violation = g_violation
        if h_violation:
            self.h_violation = h_violation
        if i_violation:
            self.i_violation = i_violation
        if h_violation and g_violation:
            self.g_h_violation = [g + h for g, h in zip(g_violation, h_violation)]
        if transfer_values:
            self.transfer_values = transfer_values

    def updateViolationAtLevel(self,  rule_level: int, g_violation_score: float=None, h_violation_score: float=None, i_violation_score: float=None, transfer_value:float=None):
        if g_violation_score is not None:
            self.g_violation[rule_level] = g_violation_score
        if h_violation_score is not None:
            self.h_violation[rule_level] = h_violation_score
        if i_violation_score is not None:
            self.i_violation[rule_level] = i_violation_score
        if h_violation_score is not None and g_violation_score is not None:
            self.g_h_violation[rule_level] = g_violation_score + h_violation_score
        if transfer_value is not None:
            self.transfer_values[rule_level] = transfer_value
