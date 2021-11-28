import math
from typing import List, Dict, Union, Iterable

from lazy_smt.monitor import STLRuleMonitor, MonitorType
# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import PredicateNode
# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario


class RuleAbstracter:
    """
    Wrapper class to wrap rule monitors with evaluation functionalities
    """

    def __init__(self,
                 ttv: int,
                 scenario: Scenario,
                 vehicle_id: int,
                 rule_str: Union[str, Iterable[str]],):
        assert ttv != math.inf and ttv != - math.inf, "Provided TTV = {} is invalid".format(ttv)
        self._ttv = ttv
        self._world_state = self.construct_world_state(scenario, vehicle_id)
        self._rule_monitor = STLRuleMonitor(self._world_state, rule_str)
        self._prop_robustness = self._rule_monitor.rob_abstraction
        self._prop_robust_ttv = None
        self._prop_nodes = self.initialize_prop_rob()
        self._predicate_nodes = self._rule_monitor.predicate_nodes

    @staticmethod
    def construct_world_state(scenario: Scenario,
                              ego_id: int) -> WorldState:
        world_state = WorldState.create_from_scenario(scenario, ego_id)
        return world_state

    @property
    def rule_monitor(self):
        return self._rule_monitor

    @property
    def prop_robust_ttv(self):
        return self._prop_robust_ttv

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def propositions(self):
        # propositions
        return self._prop_nodes

    @property
    def prop_robustness(self):
        # the robustness of propositions
        return self._prop_robustness

    @property
    def sat_encoding(self):
        """
        Retrieves SAT encodings from the rule monitor
        """
        return self._rule_monitor.sat_formula

    def initialize_prop_rob(self):
        prop_nodes = self._rule_monitor.abstraction_nodes
        if self._prop_robustness is None:
            raise "the robustness of abstractions hasn't been specified"
        self._prop_robust_ttv = self._prop_robustness.query('time_step == @self._ttv')
        # assign the robustness at ttv
        for node in prop_nodes:
            node.ttv_value = self._prop_robust_ttv.query('alphabet == @node.alphabet')["robustness"].values[0]
        return prop_nodes

    def select_predicates(self) -> List[PredicateNode]:
        """
        Selects the predicates to be repaired
        """
        # select the unvisited predicates within the least robust abstraction at time step TTV.
        abs_rob_min = self._prop_robust_ttv[self._prop_robust_ttv.robustness.abs()
                                            == self._prop_robust_ttv.robustness.abs().min()].abstraction.values
        sel_abs_node = next((abs_node for abs_node in self._prop_nodes if abs_node.name == abs_rob_min), None)
        if sel_abs_node is not None:
            return sel_abs_node.children
        return None
