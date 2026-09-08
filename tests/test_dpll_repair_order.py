from types import SimpleNamespace

from crrepairer.smt.sat_solver.dpll import DPLL


def node(alphabet, name, robustness, historical_min, placeholders=(0, 1)):
    predicate = SimpleNamespace(
        base_name=name,
        agent_placeholders=placeholders,
    )
    return SimpleNamespace(
        alphabet=alphabet,
        name=name,
        children=[predicate],
        ttv_value=robustness,
        ttv_h_min=historical_min,
    )


def test_dpll_prefers_encodable_in_literal_and_repair_polarity():
    nodes = [
        node("a", "unsupported_priority", -0.001, -0.001),
        node("b", "in_intersection_conflict_area", 0.2, -0.4),
        node("c", "in_standstill", -0.3, -0.3),
    ]

    literals = DPLL.get_literal(["a b c"], nodes, 4)

    assert literals[:2] == ["c", "~b"]
    assert literals[-1] == "a"


def test_dpll_retains_cnf_order_without_robustness_metadata():
    assert DPLL.get_literal(["~b a"], None, 0) == ["~b", "a"]


def test_conflict_avoidance_polarity_does_not_follow_current_negative_value():
    conflict = node(
        "j", "in_intersection_conflict_area", -0.2, -0.4,
        placeholders=(0, 1),
    )

    assert DPLL.get_literal(["j"], [conflict], 4) == ["~j"]


def test_previous_stop_line_fact_is_not_ranked_as_repairable():
    previous = node(
        "a", "stop_line_in_front", -0.01, -0.01,
    )
    previous.name = "previous(not(stop_line_in_front__0))"
    standstill = node("f", "in_standstill", -0.5, -0.5)

    assert DPLL.get_literal(["a f"], [previous, standstill], 4) == ["f", "a"]


def test_direct_stop_line_avoidance_precedes_historical_stop_window():
    direct = node("b", "stop_line_in_front", -0.01, -0.01)
    historical_line = node("e", "stop_line_in_front", -0.02, -0.02)
    historical_line.name = "once(historically[0,3](stop_line_in_front__0))"
    historical_stop = node("f", "in_standstill", -0.001, -0.001)
    historical_stop.name = "once(historically[0,3](in_standstill__0))"

    assert DPLL.get_literal(
        ["b e f"], [direct, historical_line, historical_stop], 4
    ) == ["b", "f", "e"]


def test_dpll_defers_priority_without_fixing_its_truth_value():
    ordinary = node("a", "in_intersection", -0.8, -0.8)
    priority = node(
        "p", "turning_right_ego_target_same_priority__0_1", -0.001, -0.001
    )

    literals = DPLL.get_literal(["p a"], [priority, ordinary], 4)

    # Robustness alone would put p first.  Ordering it last is only a search
    # heuristic: its current repair polarity remains present and unrestricted.
    assert literals == ["a", "p"]
