import math
from typing import List, Dict, Union, Iterable

from lazy_smt.monitor import STLRuleMonitor, MonitorType
# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import PredicateNode
# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario


class RuleEncoder:
    """
    Wrapper class to wrap rule monitors with evaluation functionalities
    """

    def __init__(self,
                 ttv: int,
                 scenario: Scenario,
                 vehicle_id: int,
                 rule_str: Union[str, Iterable[str]],
                 monitor_type: MonitorType = MonitorType.STL):
        assert ttv != math.inf and ttv != - math.inf, "Provided TTV = {} is invalid".format(ttv)
        self._ttv = ttv
        self._world_state = self.construct_world_state(scenario, vehicle_id)
        self._rule_monitor = STLRuleMonitor(self._world_state, rule_str, monitor_type)
        self._abs_robustness = self._rule_monitor.rob_abstraction
        self._abs_robust_ttv = None
        self._abstraction_nodes = self.initialize_abs_rob()
        self._predicate_nodes = self._rule_monitor.predicate_nodes

    @staticmethod
    def construct_world_state(scenario: Scenario,
                              ego_id: int) -> WorldState:
        world_state = WorldState.create_from_scenario(scenario, ego_id)
        return world_state

    @property
    def abs_robust_ttv(self):
        return self._abs_robust_ttv

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def prop_abs(self):
        # propositional abstraction
        return self._abstraction_nodes

    @property
    def abs_robustness(self):
        # the robustness of propositional abstractions
        return self._abs_robustness

    @property
    def sat_encoding(self):
        """
        Retrieves SAT encodings from the rule monitor
        """
        return self._rule_monitor.sat_formula

    def initialize_abs_rob(self):
        abs_nodes = self._rule_monitor.abstraction_nodes
        if self._abs_robustness is None:
            raise "the robustness of abstractions hasn't been specified"
        self._abs_robust_ttv = self._abs_robustness.query('time_step == @self._ttv')
        # assign the robustness at ttv
        for node in abs_nodes:
            node.ttv_value = self._abs_robust_ttv.query('abstraction == @node.name')["robustness"].values[0]
        return abs_nodes

    def select_predicates(self) -> List[PredicateNode]:
        """
        Selects the predicates to be repaired
        """
        # select the unvisited predicates within the least robust abstraction at time step TTV.
        abs_rob_min = self._abs_robust_ttv[self._abs_robust_ttv.robustness.abs()
                                           == self._abs_robust_ttv.robustness.abs().min()].abstraction.values
        sel_abs_node = next((abs_node for abs_node in self._abstraction_nodes if abs_node.name == abs_rob_min), None)
        if sel_abs_node is not None:
            return sel_abs_node.children
        return None
