import logging
from typing import Dict, Tuple, Union, List

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class ReachGoal(Rule):
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, ccosy: CurvilinearCoordinateSystem,
                 time_step_map: Dict, time_horizon: List):
        super().__init__("REACH_GOAL", "The ego shall be in the goal area at the last time step.")

        #TODO: MAybe unify with bussstop rule??

        self.scenario = scenario
        self.planning_problem = planning_problem
        self.ccosy = ccosy
        self.time_step_map = time_step_map

        self.time_horizon = time_horizon[1]

        # Check if goal area exists
        if len(self.planning_problem.goal.state_list) == 0 or not hasattr(self.planning_problem.goal.state_list[0],
                                                                          'position'):
            logging.warning('This scenario does not contain a goal area.')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # Violation Metric
        #Todo: the time intervall for the goal state should also be considered, right?
        if self.time_step_map[state2.t] == self.time_horizon:
            position = self.ccosy.convert_to_cartesian_coords(state2.s, 0.0)

            # TODO: Is this loop really necessary ???
            for goal_state in self.planning_problem.goal.state_list:
                if not hasattr(goal_state, 'position') or goal_state.position.contains_point(position):
                    return 0.0, 0.0, None
                else:
                    return 1.0, 0.0, None
        return 0.0, 0.0, None
