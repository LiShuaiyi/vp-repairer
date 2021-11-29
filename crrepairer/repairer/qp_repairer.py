from commonroad_qp_planner.qp_planner import QPPlanner
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import set_up
from stl_crmonitor.crmonitor.common.world_state import WorldState

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle

from typing import List
import yaml
import os
import numpy as np


class QPRepairer(QPPlanner):
    def __init__(self,
                 world_state: WorldState,
                 cut_off_time_step: int):
        self._scenario = world_state.scenario
        self._ego_vehicle = self._scenario.obstacle_by_id(world_state.ego_vehicle.id)
        # remove the existing ego vehicle from the scenario to avoid the conflict
        self._scenario.remove_obstacle(self._ego_vehicle)
        self._planning_problem = world_state.planning_problem
        self._initial_trajectory: Trajectory = self._ego_vehicle.prediction.trajectory
        self._cut_off_state = self._initial_trajectory.state_at_time_step(cut_off_time_step)
        self._settings = self.config_settings()
        self._reformulate_planning_problem()
        self._time_horizon = len(self._initial_trajectory.state_list) - cut_off_time_step
        self._vehicle_configuration: PlanningConfigurationVehicle = set_up(self._settings,
                                                                           self._scenario,
                                                                           self._planning_problem)
        self._planning_problem.initial_state = self._cut_off_state
        super().__init__(world_state.scenario,
                         self._planning_problem,
                         self._time_horizon,
                         self._vehicle_configuration)

    def _reformulate_planning_problem(self,):
        if not hasattr(self._planning_problem, "initial_state"):
            raise ValueError("<QPRepairer>: the initial state needs to be specified")
        self._planning_problem.initial_state = self._ego_vehicle.initial_state
        self._planning_problem.goal = update_goal_state(self._initial_trajectory)

    def config_settings(self):
        config_file = 'config_' + str(self._scenario.scenario_id) + '.yaml'
        config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                   "../../config"))

        with open(os.path.join(config_dir, config_file), 'r') as stream:
            try:
                settings = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
        return settings


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2)
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region
