from commonroad_qp_planner.qp_planner import QPPlanner
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.initialization import set_up
from stl_crmonitor.crmonitor.common.world_state import WorldState

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import State

from typing import List
import yaml
import os


class QPRepairer(QPPlanner):
    def __init__(self,
                 world_state: WorldState,
                 cut_off_time_step: int):
        self._scenario = world_state.scenario
        self._ego_vehicle = world_state.ego_vehicle
        self._initial_trajectory: List[State] = self._ego_vehicle.state_list_cr
        # remove the existing ego vehicle from the scenario to avoid the conflicts
        if self._scenario.obstacle_by_id(self._ego_vehicle.id):
            self._scenario.remove_obstacle(self._scenario.obstacle_by_id(self._ego_vehicle.id))
        # todo: check the time steps
        self._cut_off_state = self._initial_trajectory[cut_off_time_step]
        self._planning_problem = self._reformulate_planning_problem(world_state.planning_problem,
                                                                    self._cut_off_state)
        self._time_horizon = len(self._initial_trajectory) - cut_off_time_step
        self._settings = self.config_settings()
        self._vehicle_configuration: PlanningConfigurationVehicle = set_up(self._settings,
                                                                           self._scenario,
                                                                           self._planning_problem)
        super().__init__(world_state.scenario,
                         self._planning_problem,
                         self._time_horizon,
                         self._vehicle_configuration)

    @staticmethod
    def _reformulate_planning_problem(planning_problem: PlanningProblem,
                                      cut_off_state: State):
        planning_problem.initial_state = cut_off_state
        return planning_problem

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
