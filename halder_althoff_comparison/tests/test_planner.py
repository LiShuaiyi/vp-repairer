import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from halder_althoff_comparison.environment import Environment
from halder_althoff_comparison.io import load_problem
from halder_althoff_comparison.model import LatticeConfig, State
from halder_althoff_comparison.planner import MinimumViolationPlanner
from halder_althoff_comparison.history_aware_search import freeze_memory, node_key
from halder_althoff_comparison.rules import (
    IntersectionYieldRule,
    SafeDistanceRule,
    SpeedLimitRule,
    StopLineRule,
    build_rulebook,
)


class RuleTests(unittest.TestCase):
    def test_safe_distance_is_negative_when_too_close(self):
        env = Environment(
            {"lead_vehicles": [{"rear_s": 8.0, "velocity": 0.0}]}, 2, 0.5
        )
        margin, _ = SafeDistanceRule(ego_length=4.0).step(
            None, State(0, 4.0, 8.0), None, env
        )
        self.assertLess(margin, 0.0)

    def test_g2_only_immediate_predecessor_can_justify_braking(self):
        env = Environment(
            {
                "lead_vehicles": [
                    {"id": "near", "rear_s": 10.0, "acceleration": 0.0},
                    {"id": "far", "rear_s": 20.0, "acceleration": -10.0},
                ]
            },
            1,
            0.2,
        )
        rule = build_rulebook(
            ["R_G2"], {"R_G2": {"ego_length": 4.0}},
            LatticeConfig(0.2, 0.1, 1.0, 1, 0, 100, 0, 20, -10, 5),
        )[0]
        margin, _ = rule.step(None, State(0, 0.0, 0.0, -3.0), None, env)
        self.assertLess(margin, 0.0)

    def test_g2_uses_monitor_robustness_scaling(self):
        env = Environment(
            {"lead_vehicles": [{"rear_s": 100.0, "acceleration": 0.0}]},
            1,
            0.2,
        )
        rule = build_rulebook(
            ["R_G2"], {},
            LatticeConfig(0.2, 0.1, 1.0, 1, 0, 200, 0, 20, -10, 5),
        )[0]
        # False antecedent robustness is -scale_acc(a_abrupt - a).
        margin, _ = rule.step(None, State(0, 0.0, 0.0, 0.0), None, env)
        self.assertAlmostEqual(margin, 2.0 / 10.5)

    def test_intersection_temporal_expansion(self):
        env = Environment(
            {
                "intersection_rules": {
                    "R_IN4": {
                        "conflict_interval": [10, 12],
                        "target_in_conflict": [False, False, True, False, False],
                        "ego_lookahead_s": 1.0,
                        "target_clearance_s": 0.5,
                    }
                }
            },
            5,
            0.5,
        )
        rule = IntersectionYieldRule("R_IN4")
        self.assertLess(rule.step(None, State(1, 11, 1), None, env)[0], 0)
        self.assertLess(rule.step(None, State(3, 11, 1), None, env)[0], 0)
        self.assertTrue(math.isinf(rule.step(None, State(4, 11, 1), None, env)[0]))

    def test_stop_line_requires_prior_stop(self):
        env = Environment({}, 4, 1.0)
        rule = StopLineRule(5.0, ego_length=0.0, stop_duration_s=1.0)
        memory = rule.initial_memory()
        previous = None
        margins = []
        for state in (State(0, 3, 1), State(1, 4, 0), State(2, 4, 0), State(3, 6, 2)):
            margin, memory = rule.step(previous, state, memory, env)
            margins.append(margin)
            previous = state
        self.assertGreaterEqual(margins[-1], 0.0)

    def test_stop_line_once_is_latched_and_memory_is_finite(self):
        env = Environment({}, 7, 1.0)
        rule = StopLineRule(5.0, ego_length=0.0, stop_duration_s=1.0)
        memory = rule.initial_memory()
        previous = None
        margins = []
        states = (
            State(0, 4, 0),
            State(1, 4, 0),  # completes the required interval
            State(2, 3, 1),  # leaves the stop region without crossing
            State(3, 2, 2),
            State(4, 4, 1),
            State(5, 6, 2),  # crosses much later
        )
        for state in states:
            margin, memory = rule.step(previous, state, memory, env)
            margins.append(margin)
            previous = state
        self.assertEqual(memory, (2, True))
        self.assertGreaterEqual(margins[-1], 0.0)

    def test_stop_line_memory_merges_equivalent_histories(self):
        env = Environment({}, 3, 1.0)
        rule = StopLineRule(5.0, ego_length=0.0, stop_duration_s=2.0)
        left = rule.initial_memory()
        right = rule.initial_memory()
        _, left = rule.step(None, State(0, 0, 3), left, env)
        _, left = rule.step(State(0, 0, 3), State(1, 4, 0), left, env)
        _, right = rule.step(None, State(0, 2, 2), right, env)
        _, right = rule.step(State(0, 2, 2), State(1, 4, 0), right, env)
        self.assertEqual(left, right)


class PlannerTests(unittest.TestCase):
    def test_rule_memory_can_be_used_as_generic_state_key(self):
        first = freeze_memory({"history": [(1.0, 0.0)], "seen": {3, 2}})
        second = freeze_memory({"seen": {2, 3}, "history": [(1.0, 0.0)]})
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

        state = SimpleNamespace(t=1.0, s=2.0, v=3.0)
        left = SimpleNamespace(state=state, transfer_values=[None, (1, 0)])
        right = SimpleNamespace(state=state, transfer_values=[None, (2, 0)])
        self.assertNotEqual(node_key(left), node_key(right))

    def test_lexicographic_priority_beats_comfort(self):
        cfg = LatticeConfig(1.0, 0.5, 1.0, 3, 0, 20, 0, 5, -1, 1)
        env = Environment({"speed_limits": {"lane": 1.0}}, 4, 1.0)
        rules = build_rulebook(
            ["vehicle_constraints", "R_G3", "acceleration_comfort"], {}, cfg
        )
        result = MinimumViolationPlanner(cfg, env, rules).plan(0.0, 3.0)
        self.assertEqual(result.states[-1].v, 1.0)
        self.assertGreater(result.violation_costs[1], 0.0)  # unavoidable at k=0/1
        self.assertEqual(len(result.states), 4)

    def test_demo_loads_and_plans(self):
        root = Path(__file__).parents[1]
        cfg, env, rules, s0, v0, _ = load_problem(root / "examples/rg_in_demo.json")
        result = MinimumViolationPlanner(cfg, env, rules).plan(s0, v0)
        self.assertEqual(len(result.states), cfg.horizon_steps + 1)
        self.assertEqual(tuple(r.name for r in rules), result.rule_names)


if __name__ == "__main__":
    unittest.main()
