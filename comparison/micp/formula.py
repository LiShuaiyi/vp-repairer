import numpy as np

from stlpy.STL import LinearPredicate, NonlinearPredicate
from stlpy.benchmarks.common import inside_rectangle_formula, outside_rectangle_formula
from matplotlib.patches import Rectangle, Circle
from functools import reduce
from commonroad_qp_planner.utils import (
    calculate_safe_distance,
    derivative_safe_distance,
)


def keeps_speed_limit(speed_limit, v_index, d, name=None):
    a = np.zeros((1, d))
    a[:, v_index] = 1

    # create predicate a*y >= b for the speed limit
    below_speed_limit = LinearPredicate(-a, -speed_limit)

    if name is not None:
        below_speed_limit.name = name
    return below_speed_limit


def braking_formula(a_index, d, name=None):
    a = np.zeros((1, d))
    a[:, a_index] = 1

    # create predicate a*y >= b for the speed limit
    v_braking = LinearPredicate(-a, 0)

    if name is not None:
        v_braking.name = name
    return v_braking


def not_braking_formula(a_index, d, name=None):
    a = np.zeros((1, d))
    a[:, a_index] = 1

    # create predicate a*y >= b for the speed limit
    v_braking = LinearPredicate(a, 0)

    if name is not None:
        v_braking.name = name
    return v_braking


def braking_abruptly_formula(a_index, d, name=None):
    a = np.zeros((1, d))
    a[:, a_index] = 1

    # create predicate a*y >= b for the speed limit
    braking_abruptly = LinearPredicate(-a, 2)

    if name is not None:
        braking_abruptly.name = name
    return braking_abruptly


def relative_braking_abruptly_formula(a_target, a_index, d, name=None):
    a = np.zeros((1, d))
    a[:, a_index] = 1

    a_abrupt = -2

    braking_abruptly_relative = LinearPredicate(-a, -(a_abrupt + a_target))

    if name is not None:
        braking_abruptly_relative.name = name
    return braking_abruptly_relative


def not_braking_abruptly_formula(a_index, d, name=None):
    a = np.zeros((1, d))
    a[:, a_index] = 1

    # create predicate a*y >= b for the speed limit
    braking_abruptly = LinearPredicate(a, -2)

    if name is not None:
        braking_abruptly.name = name
    return braking_abruptly


def in_same_lane_formula(bounds, y1_index, y2_index, d, name=None):
    return inside_rectangle_formula(bounds, y1_index, y2_index, d, name)


def not_in_same_lane_formula(bounds, y1_index, y2_index, d, name=None):
    return outside_rectangle_formula(bounds, y1_index, y2_index, d, name)


def in_front_of_formula(interval, index, d, length, wheelbase):
    # Convert tuple to list
    interval_list = list(interval)

    # Perform the operation
    interval_list[1] -= (1 / 2 * length/2 + wheelbase/2)

    # Convert back to tuple if necessary
    interval = tuple(interval_list)
    return inside_interval_formula(interval, index, d, "in_front_of")


def not_in_front_of_formula(interval, index, d, length, wheelbase):
    # Convert tuple to list
    interval_list = list(interval)

    # Perform the operation
    interval_list[1] -= (1 / 2 * length/2 + wheelbase/2)

    # Convert back to tuple if necessary
    interval = tuple(interval_list)
    return outside_interval_formula(interval, index, d, "not_in_front_of")


def keeps_safe_distance_formula(rear_l, velocity_l, position_index, velocity_index,
                                d, length, wheelbase, name=None):
    # Define the predicate function g(y) >= 0
    def g(y):
        d_safe = (
                (velocity_l ** 2) / (-2 * 10.5)
                - (y[velocity_index] ** 2) / (-2 * 10)
                + y[velocity_index] * 0.4
        )
        # return rear_l - (y[position_index]) - d_safe
        return rear_l - (y[position_index] + 1/2 * length + wheelbase/2) - d_safe
    return NonlinearPredicate(g, d, name)


def linearized_keeps_safe_distance_formula(rear_l, velocity_l, position_index, velocity_index, d,
                                           length, wheelbase, name=None):
    velocity_samples = np.linspace(0, 25, 5)
    safe_formula = []
    for i in range(len(velocity_samples)):
        safe_distance_0 = calculate_safe_distance(
            velocity_samples[i],
            velocity_l,
            -10.5,
            -10,
            0.4
        )
        safe_distance_der_0 = derivative_safe_distance(
            velocity_samples[i], -10, 0.4
        )
        right_hand = - rear_l + length/2 + wheelbase/2 + safe_distance_0 - safe_distance_der_0 * velocity_samples[i]
        a = np.zeros((1, d))
        a[:, position_index] = -1
        a[:, velocity_index] = -safe_distance_der_0
        safe_distance_sub = LinearPredicate(a, right_hand)
        safe_formula.append(safe_distance_sub)
    return reduce(lambda x, y: x & y, safe_formula)


def inside_interval_formula(interval, index, d, name=None):
    """
    Create an STL formula representing being inside a region/an interval:

    ::

                |                   |
                |                   |
                |                   |
                y_min              y_max
    :param interval:   Tuple ``(y_min, y_max)`` containing the bounds of the interval.
    :param index:    index of the variable
    :param d:        dimension of the overall signal
    :param name:     (optional) string describing this formula

    :return inside_interval:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    # unpack the interval
    y_min, y_max = interval

    # create predicate a*y >= b for each side
    a = np.zeros((1, d))
    a[:, index] = 1

    if y_max != np.inf:
        right = LinearPredicate(-a, -y_max)
    else:
        right = None
    if y_min != -np.inf:
        left = LinearPredicate(a, y_min)
    else:
        left = None
    if left is not None and right is not None:
        # Take the conjunction across the interval
        inside_interval = right & left
    elif left is None:
        inside_interval = right
    elif right is None:
        inside_interval = left
    else:
        inside_interval = None

    return inside_interval


def outside_interval_formula(interval, index, d, name=None):
    """
    Create an STL formula representing being outside a region/an interval:

    ::

                |                   |
                |                   |
                |                   |
                y_min              y_max
    :param interval:   Tuple ``(y_min, y_max)`` containing the bounds of the interval.
    :param index:    index of the variable
    :param d:        dimension of the overall signal
    :param name:     (optional) string describing this formula

    :return inside_interval:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    # unpack the interval
    y_min, y_max = interval

    # create predicate a*y >= b for each side
    a = np.zeros((1, d))
    a[:, index] = 1
    if y_max != np.inf:
        right = LinearPredicate(a, y_max)
    else:
        right = None
    if y_min != -np.inf:
        left = LinearPredicate(-a, -y_min)
    else:
        left = None
    if left is not None and right is not None:
        # Take the conjunction across the interval
        outside_interval = right | left
    elif left is None:
        outside_interval = right
    elif right is None:
        outside_interval = left
    else:
        outside_interval = None

    return outside_interval
