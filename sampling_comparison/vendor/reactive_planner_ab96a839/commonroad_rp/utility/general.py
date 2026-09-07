from typing import Tuple, Optional
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle


def load_scenario_and_planning_problem(path_scenario, idx_planning_problem: Optional[int] = None)\
        -> Tuple[Scenario, PlanningProblem, PlanningProblemSet]:
    """
    Loads a scenario and planning problem from the configuration.
    :param path_scenario: full path to scenario XML file
    :param idx_planning_problem: index of the planning problem (if none provided, first planning problem is returned)
    :return: scenario and planning problem and planning problem set
    """
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(lanelet_assignment=True)
    if idx_planning_problem is not None:
        try:
            planning_problem = planning_problem_set.find_planning_problem_by_id(idx_planning_problem)
        except KeyError:
            raise KeyError(f"<ReactivePlannerConfiguration.update()>:"
                           f"Planning Problem with ID: {idx_planning_problem} does not exist!")
    else:
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    return scenario, planning_problem, planning_problem_set


def retrieve_desired_velocity_from_pp(planning_problem: PlanningProblem):
    """Simple approach which retrieves average velocity from goal configuration"""
    goal = planning_problem.goal
    init_state = planning_problem.initial_state

    if hasattr(goal.state_list[0], 'velocity'):
        if goal.state_list[0].velocity.start > 0:
            desired_velocity = (planning_problem.goal.state_list[0].velocity.start + planning_problem.goal.state_list[
                0].velocity.end) / 2
        else:
            desired_velocity = (planning_problem.goal.state_list[0].velocity.end) / 2
    else:
        desired_velocity = init_state.velocity

    return desired_velocity


def shift_orientation(trajectory: Trajectory, interval_start=-np.pi, interval_end=np.pi):
    for state in trajectory.state_list:
        while state.orientation < interval_start:
            state.orientation += 2 * np.pi
        while state.orientation > interval_end:
            state.orientation -= 2 * np.pi
    return trajectory

def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region
