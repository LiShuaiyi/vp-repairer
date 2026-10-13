from types import SimpleNamespace

import numpy as np
import pytest

from crrepairer.repairer.vp.constraints import (
    UnsupportedVPCandidateError,
    VPConstraintExtraction,
    _STRICT_BOUND_TOLERANCE,
)
from crrepairer.repairer.vp.predicate_regions import ClosedInterval
from crrepairer.repairer.vp.semantic_predicate_regions import (
    FALSE_DOMAIN,
    FramePredicateEstimate,
    SemanticIntervalSet,
    TRUE_DOMAIN,
)
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.smt.vp_proposition_capabilities import (
    VPConstraintKind,
    is_deferred_constraint_kind,
    proposition_constraint_kind,
    supported_constraint_kinds,
)


def _prop(name, alphabet="a", *, predicate_name=None, evaluator=None):
    if evaluator is None and predicate_name is not None:
        evaluator = SimpleNamespace(predicate_name=predicate_name)
    child = SimpleNamespace(
        name=predicate_name,
        base_name=predicate_name,
        evaluator=evaluator,
    )
    return SimpleNamespace(
        name=name,
        alphabet=alphabet,
        children=[] if evaluator is None else [child],
        ttv_value=-1.0,
    )


def _extractor(region, mode="deceleration"):
    extractor = object.__new__(VPConstraintExtraction)
    extractor._vp_repair_mode = mode
    extractor._semantic_constraint_region = lambda **_kwargs: (
        "at_traffic_sign_stop",
        region,
    )
    return extractor


def _apply(region, kind, *, mode="deceleration", start_s=0.0):
    extractor = _extractor(region, mode)
    lower = np.asarray([-np.inf])
    upper = np.asarray([np.inf])
    result = extractor._apply_semantic_region_constraint(
        proposition=_prop("at_traffic_sign_stop__0", predicate_name="at_traffic_sign_stop"),
        constraint_kind=kind,
        time_step=1,
        start_s=start_s,
        trajectory_s_min_cap=lower,
        trajectory_s_max_cap=upper,
        index=0,
    )
    return result, lower, upper


def test_positive_semantic_region_uses_single_inner_component():
    region = SemanticIntervalSet(
        inner_true=(ClosedInterval(2.0, 4.0),),
        outer_true=(ClosedInterval(1.0, 5.0),),
        complete=True,
        source="test",
    )
    result, lower, upper = _apply(
        region, VPConstraintKind.INSIDE_SEMANTIC_REGION
    )
    assert lower.tolist() == [2.0]
    assert upper.tolist() == [4.0]
    assert result["branch"] == "inside_region"
    assert result["bound_tightened"] is True


@pytest.mark.parametrize(
    ("mode", "expected_lower", "expected_upper"),
    [
        ("deceleration", -np.inf, 1.0 - _STRICT_BOUND_TOLERANCE),
        ("acceleration", 5.0 + _STRICT_BOUND_TOLERANCE, np.inf),
    ],
)
def test_negative_semantic_region_uses_outer_complement(
    mode, expected_lower, expected_upper
):
    region = SemanticIntervalSet(
        inner_true=(ClosedInterval(2.0, 4.0),),
        outer_true=(ClosedInterval(1.0, 5.0),),
        complete=True,
        source="test",
    )
    result, lower, upper = _apply(
        region,
        VPConstraintKind.OUTSIDE_SEMANTIC_REGION,
        mode=mode,
    )
    assert lower[0] == pytest.approx(expected_lower)
    assert upper[0] == pytest.approx(expected_upper)
    assert result["bound_tightened"] is True


def test_negative_region_already_after_adds_no_redundant_bound():
    region = SemanticIntervalSet.from_nominal(
        ((1.0, 5.0),), uncertainty=0.0, source="test"
    )
    result, lower, upper = _apply(
        region,
        VPConstraintKind.OUTSIDE_SEMANTIC_REGION,
        mode="deceleration",
        start_s=6.0,
    )
    assert result["branch"] == "already_after_region"
    assert result["bound_tightened"] is False
    assert np.isneginf(lower[0]) and np.isposinf(upper[0])


def test_empty_and_incomplete_semantic_regions_are_handled_soundly():
    empty = SemanticIntervalSet.empty("fixed_false")
    result, _, _ = _apply(empty, VPConstraintKind.OUTSIDE_SEMANTIC_REGION)
    assert result["branch"] == "fixed_false_no_constraint"
    with pytest.raises(UnsupportedVPCandidateError, match="fixed false"):
        _apply(empty, VPConstraintKind.INSIDE_SEMANTIC_REGION)

    incomplete = SemanticIntervalSet(
        inner_true=(ClosedInterval(2.0, 4.0),),
        outer_true=(ClosedInterval(1.0, 5.0),),
        complete=False,
        source="incomplete",
    )
    with pytest.raises(UnsupportedVPCandidateError, match="Incomplete"):
        _apply(incomplete, VPConstraintKind.OUTSIDE_SEMANTIC_REGION)


def test_disconnected_semantic_region_is_rejected_for_both_polarities():
    region = SemanticIntervalSet(
        inner_true=(ClosedInterval(1.0, 2.0), ClosedInterval(4.0, 5.0)),
        outer_true=(ClosedInterval(0.5, 2.5), ClosedInterval(3.5, 5.5)),
        complete=True,
        source="disconnected",
    )
    for kind in (
        VPConstraintKind.INSIDE_SEMANTIC_REGION,
        VPConstraintKind.OUTSIDE_SEMANTIC_REGION,
    ):
        with pytest.raises(UnsupportedVPCandidateError, match="disconnected"):
            _apply(region, kind)


def test_feature_flag_off_preserves_baseline_capabilities(monkeypatch):
    monkeypatch.delenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", raising=False)
    stop = _prop("stop_line_in_front__0", predicate_name="stop_line_in_front")
    conflict = _prop(
        "in_intersection_conflict_area__0_1",
        alphabet="~b",
        predicate_name="in_intersection_conflict_area",
    )
    additional = _prop(
        "at_traffic_sign_stop__0", predicate_name="at_traffic_sign_stop"
    )
    assert proposition_constraint_kind(stop) == VPConstraintKind.STOP_LINE_UPPER
    assert proposition_constraint_kind(conflict) == VPConstraintKind.OUTSIDE_EGO_CONFLICT
    assert proposition_constraint_kind(additional) is None
    assert VPConstraintKind.INSIDE_SEMANTIC_REGION not in supported_constraint_kinds(("R_IN1",))


@pytest.mark.parametrize("rule", ["R_G1", "R_G2", "R_G3"])
def test_additional_intersection_constraints_do_not_enable_rg_fallback(
    monkeypatch, rule
):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    assert not any(
        is_deferred_constraint_kind(kind)
        for kind in supported_constraint_kinds((rule,))
    )


@pytest.mark.parametrize(
    ("predicate_name", "positive_kind", "negative_kind", "expression"),
    [
        ("at_traffic_sign_stop", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_SEMANTIC_REGION, "at_traffic_sign_stop__0"),
        ("relevant_traffic_light", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_SEMANTIC_REGION, "relevant_traffic_light__0"),
        ("on_lanelet_with_type_intersection", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_INTERSECTION, "on_lanelet_with_type_intersection__0"),
        ("in_intersection_conflict_area", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_EGO_CONFLICT, "in_intersection_conflict_area__0_1"),
        ("turning_left", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_SEMANTIC_REGION, "turning_left__0"),
        ("on_incoming_left_of", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_SEMANTIC_REGION, "on_incoming_left_of__0_1"),
        ("causes_braking_intersection", VPConstraintKind.INSIDE_SEMANTIC_REGION, VPConstraintKind.OUTSIDE_CAUSES_BRAKING, "causes_braking_intersection__0_1"),
    ],
)
def test_feature_flag_on_registers_explicit_polarities(
    monkeypatch, predicate_name, positive_kind, negative_kind, expression
):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    assert proposition_constraint_kind(
        _prop(expression, "a", predicate_name=predicate_name)
    ) == positive_kind
    assert proposition_constraint_kind(
        _prop(expression, "~a", predicate_name=predicate_name)
    ) == negative_kind


@pytest.mark.parametrize(
    ("predicate_name", "expression"),
    [
        ("in_intersection_conflict_area", "in_intersection_conflict_area__1_0"),
        ("on_incoming_left_of", "on_incoming_left_of__1_0"),
        ("causes_braking_intersection", "causes_braking_intersection__1_0"),
        ("turning_left", "turning_left__1"),
        ("relevant_traffic_light", "relevant_traffic_light__1"),
    ],
)
def test_other_vehicle_predicates_are_not_registered(
    monkeypatch, predicate_name, expression
):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    prop = _prop(
        expression,
        predicate_name=predicate_name,
    )
    assert proposition_constraint_kind(prop) is None


def test_turning_composite_is_identified_from_evaluator_structure(monkeypatch):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    evaluator = SimpleNamespace(
        predicate_name="opaque_composite_name",
        _turning_ego=SimpleNamespace(predicate_name="turning_left"),
    )
    prop = _prop("opaque__0_1", evaluator=evaluator)
    assert proposition_constraint_kind(prop) == VPConstraintKind.INSIDE_SEMANTIC_REGION
    prop.alphabet = "~a"
    assert proposition_constraint_kind(prop) == VPConstraintKind.OUTSIDE_SEMANTIC_REGION


@pytest.mark.parametrize(
    "predicate_name",
    [
        "at_traffic_sign_stop",
        "relevant_traffic_light",
        "on_lanelet_with_type_intersection",
        "in_intersection_conflict_area",
        "turning_left",
        "turning_right",
        "going_straight",
    ],
)
def test_static_predicates_reuse_region_for(predicate_name):
    expected = SemanticIntervalSet.from_nominal(
        ((1.0, 2.0),), uncertainty=0.0, source=predicate_name
    )
    calls = []
    builder = SimpleNamespace(
        region_for=lambda evaluator, prop_name, time_step: (
            calls.append((evaluator, prop_name, time_step)) or expected
        )
    )
    extractor = object.__new__(VPConstraintExtraction)
    extractor._ensure_semantic_in_region_builder = lambda: (builder, {})
    expression = (
        "in_intersection_conflict_area__0_1"
        if predicate_name == "in_intersection_conflict_area"
        else f"{predicate_name}__0"
    )
    prop = _prop(expression, predicate_name=predicate_name)

    identity, region = extractor._semantic_constraint_region(
        proposition=prop, time_step=3
    )

    assert identity == predicate_name
    assert region is expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("gate_domain", "expected_empty"),
    [(TRUE_DOMAIN, False), (FALSE_DOMAIN, True)],
)
def test_turning_composite_fixed_gate_controls_constraint_region(
    gate_domain, expected_empty
):
    turning_region = SemanticIntervalSet.from_nominal(
        ((2.0, 4.0),), uncertainty=0.0, source="turning"
    )
    evaluator = SimpleNamespace(
        predicate_name="opaque_composite",
        _turning_ego=SimpleNamespace(predicate_name="turning_left"),
    )
    prop = _prop("opaque__0_1", evaluator=evaluator)
    estimate = FramePredicateEstimate(
        FALSE_DOMAIN if gate_domain == FALSE_DOMAIN else TRUE_DOMAIN,
        "continuous_turning_composite",
        region=turning_region,
        gate_domain=gate_domain,
    )
    builder = SimpleNamespace(
        reachable_by_time={3: (0.0, 5.0)},
        estimate_frame=lambda *_args, **_kwargs: estimate,
    )
    extractor = object.__new__(VPConstraintExtraction)
    extractor._ensure_semantic_in_region_builder = lambda: (builder, {})

    identity, region = extractor._semantic_constraint_region(
        proposition=prop, time_step=3
    )

    assert identity == "turning_priority_composite"
    assert (not region.outer_true) is expected_empty


@pytest.mark.parametrize(
    "predicate_name",
    ["on_incoming_left_of", "causes_braking_intersection"],
)
def test_dynamic_predicates_reuse_estimator_region(predicate_name):
    expected = SemanticIntervalSet.from_nominal(
        ((3.0, 6.0),), uncertainty=0.0, source=predicate_name
    )
    estimate = FramePredicateEstimate(
        TRUE_DOMAIN, f"{predicate_name}:quantitative", region=expected
    )
    builder = SimpleNamespace(
        reachable_by_time={4: (0.0, 8.0)},
        estimate_frame=lambda *_args, **_kwargs: estimate,
    )
    extractor = object.__new__(VPConstraintExtraction)
    extractor._ensure_semantic_in_region_builder = lambda: (builder, {})
    prop = _prop(
        f"{predicate_name}__0_1", predicate_name=predicate_name
    )

    identity, region = extractor._semantic_constraint_region(
        proposition=prop, time_step=4
    )

    assert identity == predicate_name
    assert region is expected


def test_causes_braking_additional_extractor_adds_a_real_bound(monkeypatch):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    region = SemanticIntervalSet.from_nominal(
        ((3.0, 6.0),),
        uncertainty=0.0,
        source="causes_braking_intersection:monotone_region",
    )
    builder = SimpleNamespace(
        reachable_by_time={4: (0.0, 8.0)},
        estimate_frame=lambda *_args, **_kwargs: FramePredicateEstimate(
            TRUE_DOMAIN,
            "causes_braking_intersection:monotone_endpoint",
            region=region,
        ),
    )
    extractor = object.__new__(VPConstraintExtraction)
    extractor._vp_repair_mode = "deceleration"
    extractor._ensure_semantic_in_region_builder = lambda: (builder, {})
    proposition = _prop(
        "causes_braking_intersection__0_1",
        alphabet="~a",
        predicate_name="causes_braking_intersection",
    )
    lower = np.asarray([-np.inf])
    upper = np.asarray([np.inf])

    result = extractor._apply_semantic_region_constraint(
        proposition=proposition,
        constraint_kind=proposition_constraint_kind(proposition),
        time_step=4,
        start_s=0.0,
        trajectory_s_min_cap=lower,
        trajectory_s_max_cap=upper,
        index=0,
    )

    assert result["branch"] == "before_region_upper"
    assert result["bound_tightened"] is True
    assert upper[0] == pytest.approx(3.0 - _STRICT_BOUND_TOLERANCE)


def test_final_domain_sort_respects_explicit_tiers():
    solver = object.__new__(SATSolver)
    solver._solver_mode = "domain_dpll"
    solver._formula = "a & b & c"
    solver._repair_literals = ["c", "b", "a"]
    solver._repair_literal_tiers = {"a": 0, "b": 1, "c": 2}
    solver._prop_nodes = [
        SimpleNamespace(alphabet="a", ttv_value=-1.0),
        SimpleNamespace(alphabet="b", ttv_value=1.0),
        SimpleNamespace(alphabet="c", ttv_value=-1.0),
    ]
    solver._dpll_solver = SimpleNamespace(
        _tv_time_step=0,
        _assign_cnf=lambda _formula: [],
        get_literal=lambda *_args: ["c", "b", "a"],
    )
    solver.sort_domain_repair_literals()
    assert solver._repair_literals == ["a", "b", "c"]


def test_temporal_selector_inherits_parent_tier_before_final_sort():
    solver = object.__new__(SATSolver)
    solver._solver_mode = "domain_dpll"
    solver._formula = "a & b & c"
    solver._repair_literals = ["a", "c", "b"]
    solver._repair_literal_tiers = {"a": 0, "c": 1, "b": 2}
    solver._domain_dict = {}
    solver._admissible_polarities = {}
    solver._expanded_decomposition_groups = set()
    solver._prop_nodes = [
        SimpleNamespace(alphabet="a", ttv_value=-1.0),
        SimpleNamespace(alphabet="b", ttv_value=-1.0),
        SimpleNamespace(alphabet="c", ttv_value=1.0),
    ]
    solver._dpll_solver = SimpleNamespace(
        _tv_time_step=0,
        _prop_nodes=solver._prop_nodes,
        _assign_cnf=lambda _formula: [],
        get_literal=lambda *_args: ["d", "c", "a"],
    )
    parent = SimpleNamespace(
        alphabet="b",
        name="once(region__0)",
        ttv_value=-1.0,
        ttv_h_min=-1.0,
    )
    member = SimpleNamespace(
        alphabet="b",
        name="region__0",
        source_rule="R_IN3_hand_draft",
        children=[],
        ttv_value=-1.0,
        ttv_h_min=-1.0,
    )

    selectors = solver.add_once_time_selectors(
        {
            "once(region__0)": {
                "parent": parent,
                "children": (member,),
                "candidates": ((1, 1),),
            }
        }
    )
    selector = selectors["once(region__0)"][0][0]
    assert solver._repair_literal_tiers[selector] == 2

    solver.sort_domain_repair_literals()

    assert solver._repair_literals == ["a", "c", selector]


def test_explicit_tiers_preserve_historical_order_within_one_tier():
    solver = object.__new__(SATSolver)
    solver._solver_mode = "domain_dpll"
    solver._formula = "a & b"
    solver._repair_literals = ["a", "b"]
    solver._repair_literal_tiers = {"a": 0, "b": 0}
    solver._prop_nodes = [
        SimpleNamespace(alphabet="a", ttv_value=-1.0),
        SimpleNamespace(alphabet="b", ttv_value=1.0),
    ]
    solver._dpll_solver = SimpleNamespace(
        _tv_time_step=0,
        _assign_cnf=lambda _formula: [],
        get_literal=lambda *_args: ["b", "a"],
    )
    solver.sort_domain_repair_literals()
    assert solver._repair_literals == ["a", "b"]


def test_sort_without_tiers_preserves_historical_current_polarity_key():
    solver = object.__new__(SATSolver)
    solver._solver_mode = "domain_dpll"
    solver._formula = "a & b"
    solver._repair_literals = ["a", "b"]
    solver._repair_literal_tiers = {}
    solver._prop_nodes = [
        SimpleNamespace(alphabet="a", ttv_value=1.0),
        SimpleNamespace(alphabet="b", ttv_value=-1.0),
    ]
    solver._dpll_solver = SimpleNamespace(
        _tv_time_step=0,
        _assign_cnf=lambda _formula: [],
        get_literal=lambda *_args: ["a", "b"],
    )
    solver.sort_domain_repair_literals()
    assert solver._repair_literals == ["b", "a"]


def test_deferred_phase_requires_at_least_one_deferred_literal():
    solver = object.__new__(SATSolver)
    solver._formula = "a & (b | c)"
    assert solver.require_any_literals(["~b", "c", "~b"]) is True
    assert solver._formula == "(a & (b | c)) & (~b | c)"
    assert solver.require_any_literals([]) is False


def test_deferred_precheck_rejects_unreachable_positive_region(monkeypatch):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    extractor = object.__new__(VPConstraintExtraction)
    extractor._vp_repair_mode = "deceleration"
    extractor._temporal_constraint_steps = lambda *_args, **_kwargs: {
        id(proposition): SimpleNamespace(count=1, steps=frozenset({1}))
    }
    extractor._semantic_constraint_region = lambda **_kwargs: (
        "at_traffic_sign_stop",
        SemanticIntervalSet.from_nominal(
            ((4.0, 6.0),), uncertainty=0.0, source="test"
        ),
    )
    proposition = _prop(
        "at_traffic_sign_stop__0",
        alphabet="a",
        predicate_name="at_traffic_sign_stop",
    )
    builder = SimpleNamespace(reachable_by_time={1: (0.0, 3.0)})

    verdict = extractor._deferred_semantic_polarity_precheck(
        proposition=proposition,
        desired_positive=True,
        all_states=(),
        builder=builder,
        start_s=0.0,
    )

    assert verdict == {
        "executable": False,
        "actionable": False,
        "reason": "positive_region_unreachable",
    }


def test_deferred_precheck_uses_curvature_aware_acceleration_cap(monkeypatch):
    monkeypatch.setenv("CRREPAIR_VP_ENABLE_ADDITIONAL_CONSTRAINT_EXTRACTION", "1")
    extractor = object.__new__(VPConstraintExtraction)
    extractor._vp_repair_mode = "acceleration"
    extractor._temporal_constraint_steps = lambda *_args, **_kwargs: {
        id(proposition): SimpleNamespace(count=1, steps=frozenset({1}))
    }
    extractor._semantic_constraint_region = lambda **_kwargs: (
        "at_traffic_sign_stop",
        SemanticIntervalSet.from_nominal(
            ((2.0, 5.0),), uncertainty=0.0, source="test"
        ),
    )
    proposition = _prop(
        "at_traffic_sign_stop__0",
        alphabet="~a",
        predicate_name="at_traffic_sign_stop",
    )
    builder = SimpleNamespace(reachable_by_time={1: (0.0, 10.0)})

    verdict = extractor._deferred_semantic_polarity_precheck(
        proposition=proposition,
        desired_positive=False,
        all_states=(),
        builder=builder,
        start_s=0.0,
        phase_reachable_upper={1: 4.0},
    )

    assert verdict == {
        "executable": False,
        "actionable": False,
        "reason": "after_region_unreachable",
    }
