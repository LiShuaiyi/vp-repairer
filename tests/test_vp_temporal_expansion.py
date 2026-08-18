import pytest

from crrepairer.repairer.vp.temporal import (
    constraint_time_interval,
    expand_temporal_expression,
)


def test_eventually_is_universalized_by_closed_form_interval_union():
    interval, expansion, pair_count = constraint_time_interval(
        expression="eventually[1,2](p)",
        dt=1.0,
        trajectory_start=0,
        planning_start=1,
        trajectory_end=10,
        future_time_step=0,
    )

    assert expansion.offsets == (1, 2)
    assert (interval.start, interval.end) == (1, 10)
    assert pair_count == 22


def test_pastified_point_once_cancels_monitor_future_delay():
    interval, expansion, _ = constraint_time_interval(
        expression="once[1,1](in_intersection_conflict_area__0_1)",
        dt=0.2,
        trajectory_start=0,
        planning_start=1,
        trajectory_end=10,
        future_time_step=5,
    )

    assert expansion.offsets == (-5,)
    assert (interval.start, interval.end) == (1, 10)


def test_nested_past_windows_use_minkowski_sum_before_union():
    interval, expansion, _ = constraint_time_interval(
        expression=(
            "once[0,3/5](once[2/5,2/5]("
            "in_intersection_conflict_area__0_1))"
        ),
        dt=0.2,
        trajectory_start=0,
        planning_start=1,
        trajectory_end=10,
        future_time_step=5,
    )

    assert expansion.offsets == (-5, -4, -3, -2)
    assert (interval.start, interval.end) == (1, 10)


def test_unbounded_temporal_operator_preserves_full_planning_horizon():
    interval, expansion, _ = constraint_time_interval(
        expression="once(historically[0,3](p))",
        dt=0.2,
        trajectory_start=0,
        planning_start=1,
        trajectory_end=10,
        future_time_step=0,
    )

    assert expansion.is_unbounded
    assert (interval.start, interval.end) == (1, 10)


def test_fractional_interval_must_contain_a_sample():
    with pytest.raises(ValueError, match="contains no samples"):
        expand_temporal_expression("once[0.51,0.52](p)", dt=0.2)
