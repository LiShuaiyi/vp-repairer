from typing import List, Union, Iterable

from commonroad_repair.crrepairer.abstraction.monitor import STLRuleMonitor

# CommonRoad STL monitor
from stl_crmonitor.crmonitor.common.world_state import WorldState
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode

# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem


class RuleAbstracter:
    """
    Wrapper class to wrap rule monitors with abstraction functionalities
    """

    def __init__(self,
                 scenario: Scenario,
                 planning_problem: PlanningProblem,
                 vehicle_id: int,
                 rule_str: Union[str, Iterable[str]]):
        self._world_state = self.construct_world_state(scenario,
                                                       planning_problem,
                                                       vehicle_id)
        self._rule_monitor = STLRuleMonitor(self._world_state, rule_str)
        self._vehicle_id = vehicle_id
        # if there is no other id
        if self._rule_monitor.other_id is None or not isinstance(self._rule_monitor.other_id, int):
            self._other_id = vehicle_id
        else:
            self._other_id = self._rule_monitor.other_id

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @staticmethod
    def construct_world_state(scenario: Scenario,
                              planning_problem: PlanningProblem,
                              ego_id: int) -> WorldState:
        """
        Constructs world state
        """
        world_state = WorldState.create_from_scenario(scenario,
                                                      ego_id,
                                                      planning_problem=planning_problem)
        return world_state

    @property
    def rule_monitor(self) -> STLRuleMonitor:
        return self._rule_monitor

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def propositions(self) -> List[PropositionNode]:
        # propositions
        return self.rule_monitor.proposition_nodes

    @property
    def other_veh_id(self):
        return self._other_id

    @property
    def sat_encoding(self):
        """
        Retrieves SAT encodings from the rule monitor
        """
        return self._rule_monitor.sat_formula
