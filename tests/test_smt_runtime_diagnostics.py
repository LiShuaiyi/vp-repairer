from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from z3 import unsat

import crrepairer.smt.t_solver.t_solver as t_solver_module
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.smt.t_solver.qp_planner_repair import QPPlannerRepair
from crrepairer.smt.t_solver.t_solver import TSolver
from crrepairer.smt.sat_solver.sat_solver import SATSolver


def test_smt_runtime_includes_terminal_sat_solve():
    repairer = object.__new__(SMTTrajectoryRepairer)
    repairer.rule_monitor = SimpleNamespace(tv_time_step=1)
    repairer.sat_solver = SimpleNamespace(solve=Mock(return_value=unsat))
    repairer.sat_reasoning_time = 0.0
    repairer.nr_iter = 0
    repairer._tv = -1

    with patch(
        "crrepairer.repairer.smt_repairer.time.perf_counter",
        side_effect=[10.0, 10.25],
    ):
        assert repairer.repair() is None

    assert repairer.sat_reasoning_time == pytest.approx(0.25)


def test_fixed_miqp_configuration_refuses_silent_qp_substitution():
    config = SimpleNamespace(
        repair=SimpleNamespace(planner=2, allow_planner_fallback=False),
        planning_problem=object(),
    )
    with (
        patch.object(t_solver_module, "TC", return_value=object()),
        patch.object(
            t_solver_module,
            "MIQPPlannerRepair",
            side_effect=t_solver_module.GurobiError(10009, "missing license"),
        ),
    ):
        with pytest.raises(RuntimeError, match="refusing to substitute QP"):
            TSolver(object(), object(), config)


def test_qp_failed_longitudinal_solve_keeps_elapsed_phase_times():
    planner = object.__new__(QPPlannerRepair)
    planner._start_time_step = 0
    planner._tc_object = object()
    planner._rule_monitor = object()
    planner.sel_proposition = []
    planner.full_proposition = []
    planner._qp_configuration = object()
    planner._planning_problem = SimpleNamespace(
        initial_state=SimpleNamespace(velocity=10.0)
    )
    planner._rule_constraints = SimpleNamespace(
        reset=Mock(), longitudinal_constraints=Mock(return_value=object())
    )
    planner.construct_s_reference = Mock(return_value=object())
    planner.step = Mock()
    planner.longitudinal_trajectory_planning = Mock(
        return_value=(None, "user_limit")
    )
    planner.reach_set_time = 0.0
    planner.opti_plan_time = 0.0

    with patch(
        "crrepairer.smt.t_solver.qp_planner_repair.time.time",
        side_effect=[0.0, 2.0, 10.0, 13.0],
    ):
        assert planner.plan() is None

    assert planner.reach_set_time == pytest.approx(2.0)
    assert planner.opti_plan_time == pytest.approx(3.0)


def test_failed_negative_conflict_branch_returns_single_literal_core():
    solver = object.__new__(TSolver)
    solver._sel_prop = [
        SimpleNamespace(
            alphabet="~l",
            name="once[1,1](in_intersection_conflict_area__0_1)",
            ttv_h_min=-0.1,
        ),
        SimpleNamespace(alphabet="a", name="unrelated__0", ttv_h_min=-0.2),
    ]
    solver._compliant_maneuvers = [object()]

    assert solver.failed_semantic_core() == ["~l"]


def test_sat_solver_can_block_a_semantic_core_instead_of_whole_model():
    solver = object.__new__(SATSolver)
    solver._formula = "(~l | a)"
    solver._dpll_model = ["~l", "a", "b"]

    solver.update_formula(blocking_literals=["~l"])

    assert solver._formula == "(~l | a) & l"


def test_fixed_path_fallback_is_limited_to_standstill_longitudinal_results():
    stationary = SimpleNamespace(
        states=[SimpleNamespace(v=0.0), SimpleNamespace(v=0.03)]
    )
    moving = SimpleNamespace(
        states=[SimpleNamespace(v=0.0), SimpleNamespace(v=0.2)]
    )

    assert QPPlannerRepair._is_standstill_longitudinal_trajectory(stationary)
    assert not QPPlannerRepair._is_standstill_longitudinal_trajectory(moving)


def test_in1_historical_standstill_starts_prefix_repair_at_initial_state():
    solver = object.__new__(TSolver)
    solver._sel_prop = [
        SimpleNamespace(
            source_rule="R_IN1",
            alphabet="f",
            name="once(historically[0,3](in_standstill__0))",
        )
    ]
    solver._compliant_maneuvers = [object()]
    solver._tc_obj = SimpleNamespace(
        _selected_conflict_avoidance_predicates=[],
        _tc=-float("inf"),
        dT=0.2,
        generate=Mock(side_effect=AssertionError("legacy TC must not run")),
        ego_vehicle=SimpleNamespace(
            initial_state=SimpleNamespace(time_step=0)
        ),
    )

    assert solver.search_tc(use_dummy_tc=False) == 0.0
    assert solver._tc_obj._tc == 0.0


def test_in1_direct_stop_line_avoidance_starts_at_initial_state():
    solver = object.__new__(TSolver)
    solver._sel_prop = [
        SimpleNamespace(
            source_rule="R_IN1",
            alphabet="b",
            name="stop_line_in_front__0",
        )
    ]
    solver._compliant_maneuvers = [object()]
    solver._tc_obj = SimpleNamespace(
        _selected_conflict_avoidance_predicates=[],
        _tc=-float("inf"),
        dT=0.2,
        generate=Mock(side_effect=AssertionError("legacy TC must not run")),
        ego_vehicle=SimpleNamespace(initial_state=SimpleNamespace(time_step=0)),
    )

    assert solver.search_tc(use_dummy_tc=False) == 0.0
    assert solver._tc_obj._tc == 0.0


def test_rg2_direct_abrupt_braking_avoidance_starts_at_initial_state():
    solver = object.__new__(TSolver)
    solver._sel_prop = [
        SimpleNamespace(
            source_rule="R_G2",
            alphabet="~a",
            name="brakes_abruptly__0",
        )
    ]
    solver._compliant_maneuvers = [object()]
    solver._tc_obj = SimpleNamespace(
        _selected_conflict_avoidance_predicates=[],
        _tc=-float("inf"),
        dT=0.2,
        generate=Mock(side_effect=AssertionError("legacy TC must not run")),
        ego_vehicle=SimpleNamespace(initial_state=SimpleNamespace(time_step=0)),
    )

    assert solver.search_tc(use_dummy_tc=False) == 0.0
    assert solver._tc_obj._tc == 0.0


def test_rg1_direct_safe_distance_starts_at_initial_state():
    solver = object.__new__(TSolver)
    solver._sel_prop = [
        SimpleNamespace(
            source_rule="R_G1",
            alphabet="a",
            name="keeps_safe_distance_prec__0_1",
        )
    ]
    solver._compliant_maneuvers = [object()]
    solver._tc_obj = SimpleNamespace(
        _selected_conflict_avoidance_predicates=[],
        _tc=-float("inf"),
        dT=0.2,
        generate=Mock(side_effect=AssertionError("legacy TC must not run")),
        ego_vehicle=SimpleNamespace(initial_state=SimpleNamespace(time_step=0)),
    )

    assert solver.search_tc(use_dummy_tc=False) == 0.0
    assert solver._tc_obj._tc == 0.0
