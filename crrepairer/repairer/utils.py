from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    # todo: complete the function
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2)
    goal_velocity = Interval(0, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region