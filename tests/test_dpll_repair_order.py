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


def test_plain_dpll_orders_only_by_legacy_robustness():
    nodes = [
        node("a", "unsupported_priority", -0.001, -0.001),
        node("b", "in_intersection_conflict_area", 0.2, -0.4),
        node("c", "in_standstill", -0.3, -0.3),
    ]

    literals = DPLL.get_literal(["a b c"], nodes, 4)

    assert literals == ["a", "c", "b"]


def test_dpll_retains_cnf_order_without_robustness_metadata():
    assert DPLL.get_literal(["~b a"], None, 0) == ["~b", "a"]


def test_plain_dpll_does_not_rewrite_literal_polarity():
    conflict = node(
        "j", "in_intersection_conflict_area", -0.2, -0.4,
        placeholders=(0, 1),
    )

    assert DPLL.get_literal(["j"], [conflict], 4) == ["j"]


def test_plain_dpll_does_not_special_case_previous_stop_line_facts():
    previous = node(
        "a", "stop_line_in_front", -0.01, -0.01,
    )
    previous.name = "previous(not(stop_line_in_front__0))"
    standstill = node("f", "in_standstill", -0.5, -0.5)

    assert DPLL.get_literal(["a f"], [previous, standstill], 4) == ["a", "f"]


def test_plain_dpll_does_not_special_case_temporal_stop_line_literals():
    direct = node("b", "stop_line_in_front", -0.01, -0.01)
    historical_line = node("e", "stop_line_in_front", -0.02, -0.02)
    historical_line.name = "once(historically[0,3](stop_line_in_front__0))"
    historical_stop = node("f", "in_standstill", -0.001, -0.001)
    historical_stop.name = "once(historically[0,3](in_standstill__0))"

    assert DPLL.get_literal(
        ["b e f"], [direct, historical_line, historical_stop], 4
    ) == ["f", "b", "e"]


def test_plain_dpll_does_not_defer_priority_predicates():
    ordinary = node("a", "in_intersection", -0.8, -0.8)
    priority = node(
        "p", "turning_right_ego_target_same_priority__0_1", -0.001, -0.001
    )

    literals = DPLL.get_literal(["p a"], [priority, ordinary], 4)

    assert literals == ["p", "a"]


def test_domain_guidance_keeps_vp_aware_order_and_polarity():
    nodes = [
        node("a", "unsupported_priority", -0.001, -0.001),
        node("b", "in_intersection_conflict_area", 0.2, -0.4),
        node("c", "in_standstill", -0.3, -0.3),
    ]

    literals = DPLL.get_domain_guided_literal(["a b c"], nodes, 4)

    assert literals[:2] == ["c", "~b"]
    assert literals[-1] == "a"
