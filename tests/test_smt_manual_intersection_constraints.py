from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from crrepairer.smt.monitor_wrapper import PropositionNode
from crrepairer.smt.t_solver.rule_constraints import RuleConstraintsManual


def make_constraints(selected, *, dt=0.2, tc=0, end=20):
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    constraints._sel_prop_full = [selected]
    constraints._world_state = SimpleNamespace(dt=dt)
    constraints._tc_obj = SimpleNamespace(
        tc_time_step=tc, tv_time_step=2, N=end
    )
    constraints._rule_monitor = SimpleNamespace(_rules=[selected.source_rule])
    return constraints


def test_historical_standstill_uses_one_complete_contiguous_window():
    proposition = PropositionNode(
        name="once(historically[0,3](in_standstill__0))",
        alphabet="f",
        source_rule="R_IN1",
    )
    constraints = make_constraints(proposition, dt=0.2, end=20)

    selected_steps = [
        step
        for step in range(1, 21)
        if constraints._historical_witness_window_contains(proposition, step)
    ]

    # [0, 3s] contains 16 samples at dt=0.2 and the chosen witness ends at N.
    assert selected_steps == list(range(5, 21))


def test_negated_ego_conflict_literal_forces_avoidance_for_in_rules():
    proposition = PropositionNode(
        name="once[1,1](in_intersection_conflict_area__0_1)",
        alphabet="~j",
        source_rule="R_IN4",
    )
    constraints = make_constraints(proposition)
    constraints.s_circle_center_front = 10.0
    constraints._ego_vehicle = SimpleNamespace(shape=SimpleNamespace(length=4.5))
    constraints._veh_config = SimpleNamespace(wheelbase=2.5, wb_ra=1.2)

    assert constraints._selected_negative_ego_conflict(proposition)
    constraints._avoid_conflict_upper_bound = None
    lower, upper = constraints.ConstrAvoidIntersectionConflictArea()
    assert lower == -np.inf
    assert upper == 10.0 - 1.2 - constraints._CONFLICT_ENTRY_MARGIN


def test_conflict_avoidance_does_not_require_backward_motion_from_outside_start():
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    constraints.s_circle_center_front = 60.48
    constraints._avoid_conflict_upper_bound = None
    constraints._veh_config = SimpleNamespace(wb_ra=1.42)
    constraints._start_time_step = 0
    constraints._world_state = object()
    constraints._ego_id = 1
    constraints._other_id = 2
    constraints._ego_vehicle = SimpleNamespace(states_cr={0: object()})
    predicate = SimpleNamespace(
        evaluator=SimpleNamespace(
            evaluate_robustness=lambda world, step, ids: -0.01
        )
    )

    with patch(
        "crrepairer.smt.t_solver.rule_constraints.convert_pos_curvilinear",
        return_value=(59.36, 0.0),
    ):
        lower, upper = constraints.ConstrAvoidIntersectionConflictArea(predicate)

    assert lower == -np.inf
    assert upper == pytest.approx(59.36 + constraints._CONFLICT_ENTRY_MARGIN)


def test_conflict_avoidance_keeps_geometric_bound_if_start_is_inside():
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    constraints.s_circle_center_front = 60.48
    constraints._avoid_conflict_upper_bound = None
    constraints._veh_config = SimpleNamespace(wb_ra=1.42)
    constraints._start_time_step = 0
    constraints._world_state = object()
    constraints._ego_id = 1
    constraints._other_id = 2
    constraints._ego_vehicle = SimpleNamespace(states_cr={0: object()})
    predicate = SimpleNamespace(
        evaluator=SimpleNamespace(
            evaluate_robustness=lambda world, step, ids: 0.01
        )
    )

    with patch(
        "crrepairer.smt.t_solver.rule_constraints.convert_pos_curvilinear",
        return_value=(59.36, 0.0),
    ):
        _, upper = constraints.ConstrAvoidIntersectionConflictArea(predicate)

    assert upper == pytest.approx(
        60.48 - 1.42 - constraints._CONFLICT_ENTRY_MARGIN
    )


def test_explicit_not_wrapper_flips_predicate_assignment():
    proposition = PropositionNode(
        name="previous(not(stop_line_in_front__0))",
        alphabet="a",
        source_rule="R_IN1",
    )
    assert RuleConstraintsManual._predicate_assignment(proposition, proposition) < 0
    proposition.alphabet = "~a"
    assert RuleConstraintsManual._predicate_assignment(proposition, proposition) > 0


def test_negative_stop_line_predicate_is_not_encoded_as_positive_upper_bound():
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    assert constraints.ConstrStopLine(0, -1.0) == [-np.inf, np.inf]


def test_stop_line_constraint_uses_qp_clcs():
    class OffsetClcs:
        def __init__(self, offset):
            self.offset = offset

        def convert_to_curvilinear_coords(self, x, y):
            return x + self.offset, y

    stop_line = SimpleNamespace(start=(5.0, 0.0), end=(6.0, 0.0))
    lanelet = SimpleNamespace(stop_line=stop_line)
    lanelet_network = SimpleNamespace(
        find_lanelet_by_id=lambda lanelet_id: lanelet
    )
    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    constraints._rule_monitor = SimpleNamespace(
        world=SimpleNamespace(
            road_network=SimpleNamespace(lanelet_network=lanelet_network)
        )
    )
    constraints._ego_vehicle = SimpleNamespace(
        lanelets_dir=[1],
        circle_radius=2.0,
        shape=SimpleNamespace(length=4.5, width=2.0),
        initial_state=SimpleNamespace(position=np.array([0.0, 0.0]), orientation=0.0),
        states_cr={0: SimpleNamespace(position=np.array([0.0, 0.0]), orientation=0.0)},
        front_s=lambda time_step, lane: -96.0,
        ref_path_lane=SimpleNamespace(clcs=OffsetClcs(-100.0)),
    )
    constraints._tc_obj = SimpleNamespace(tc_time_step=0)
    constraints._veh_config = SimpleNamespace(
        CLCS=OffsetClcs(10.0), wb_ra=1.0
    )

    _, upper = constraints.ConstrStopLine(0, 1.0)

    assert upper == pytest.approx(
        9.0 + 1.0 - constraints._STOP_LINE_MARGIN
    )


def test_conflict_point_uses_qp_clcs_not_monitor_lane_clcs():
    class OffsetClcs:
        def __init__(self, offset):
            self.offset = offset

        def convert_to_curvilinear_coords(self, x, y):
            return x + self.offset, y

    constraints = RuleConstraintsManual.__new__(RuleConstraintsManual)
    constraints._veh_config = SimpleNamespace(CLCS=OffsetClcs(10.0))
    ego = SimpleNamespace(
        ref_path_lane=SimpleNamespace(clcs=OffsetClcs(-20.0))
    )

    assert constraints._conflict_point_qp_progress((5.0, 0.0), ego) == 15.0
