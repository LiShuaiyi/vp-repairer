import numpy as np

from stlpy.STL import LinearPredicate, NonlinearPredicate
from stlpy.benchmarks.common import inside_rectangle_formula
from matplotlib.patches import Rectangle, Circle


def in_same_lane_formula(bounds, y1_index, y2_index, d, name=None):
    return inside_rectangle_formula(bounds, y1_index, y2_index, d, name)


def in_front_of_formula(interval, index, d):
    return inside_rectangle_formula(interval, index, d, "in_front_of")


def keeps_safe_distance_formula(rear_l, velocity_l, position_index, velocity_index,
                                d, length, wheelbase, name=None):
    # Define the predicate function g(y) >= 0
    def g(y):
        d_safe = (
                (velocity_l ** 2) / (-2 * 10.5)
                - (y[velocity_index] ** 2) / (-2 * 10)
                + y[velocity_index] * 0.4
        )
        return rear_l - (y[position_index] + 1/2 * length + wheelbase) - d_safe
    return NonlinearPredicate(g, d, name)


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
    int_min = LinearPredicate(a, y_min)
    int_max = LinearPredicate(-a, -y_max)

    # Take the conjunction across the interval
    inside_interval = int_min & int_max

    # set the names
    if name is not None:
        int_min.name = "greater than " + name
        int_max.name = "smaller than " + name
        inside_interval.name = name

    return inside_interval
