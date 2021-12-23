import math
from typing import List, Dict, Union, Iterable

from abstraction.monitor import STLRuleMonitor, MonitorType
# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import PredicateNode, PropositionNode
# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem


class RuleAbstracter:
    """
    Wrapper class to wrap rule monitors with evaluation functionalities
    """

    def __init__(self,
                 scenario: Scenario,
                 planning_problem: PlanningProblem,
                 vehicle_id: int,
                 rule_str: Union[str, Iterable[str]],
                 time_horizon: float = None):
        self._world_state = self.construct_world_state(scenario,
                                                       planning_problem,
                                                       vehicle_id)
        self._rule_monitor = STLRuleMonitor(self._world_state, rule_str)
        self._prop_rob_ttv = self._rule_monitor.prop_robust_ttv
        self._vehicle_id = vehicle_id
        # if there is no other id
        if self._rule_monitor.other_id is None or not isinstance(self._rule_monitor.other_id, int):
            self._other_id = vehicle_id
        else:
            self._other_id = self._rule_monitor.other_id
        self._predicate_nodes = self._rule_monitor.predicate_nodes

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @staticmethod
    def construct_world_state(scenario: Scenario,
                              planning_problem: PlanningProblem,
                              ego_id: int) -> WorldState:
        world_state = WorldState.create_from_scenario(scenario, ego_id,
                                                      planning_problem=planning_problem)
        return world_state

    @property
    def rule_monitor(self):
        return self._rule_monitor

    @property
    def prop_robust_ttv(self):
        return self._prop_rob_ttv

    @property
    def prop_robust_all(self):
        return self._rule_monitor.prop_robust_all

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

    def select_predicates(self) -> List[PredicateNode]:
        """
        Selects the predicates to be repaired
        """
        # select the unvisited predicates within the least robust proposition at time step TTV.
        sel_prop_node = self.select_proposition()
        if sel_prop_node is not None:
            return sel_prop_node.children
        return None

    def select_proposition(self) -> PropositionNode:
        """
        Selects the proposition to be repaired
        """
        # select the unvisited predicates within the least robust proposition at time step TTV.
        prop_rob_min = self._prop_rob_ttv[self._prop_rob_ttv.robustness.abs()
                                          == self._prop_rob_ttv.robustness.abs().min()].alphabet.values
        sel_prop_node = next((prop_node for prop_node in self.propositions
                             if prop_node.alphabet == prop_rob_min), None)
        self.propositions.remove(sel_prop_node)  # todo: check whether this is correct
        return sel_prop_node
