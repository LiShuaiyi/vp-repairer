import abc
import math
import os.path
from typing import Iterable, Union, Tuple, Any
from enum import Enum
import numpy as np

from stl_crmonitor.crmonitor.evaluation.evaluation import RuleSetEvaluator
from stl_crmonitor.crmonitor.common.world_state import WorldState
# from mtl_crmonitor.common.commonroad_evaluation import CommonRoadObstacleEvaluation
from commonroad.scenario.scenario import Scenario


class MonitorType(Enum):
    MTL = "metric temporal logic"
    STL = "signal temporal logic"


class STLRuleMonitor:
    def __init__(self,
                 world_state,
                 rules: Union[str, Iterable[str]],):
        self._rule_eval = RuleSetEvaluator.create_from_config(rules,
                                                              dt=world_state.dt)
        self.world_state: WorldState = world_state
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self.evaluate_initially()
        self._tv, self._other_id = self._cal_tv_initial()
        self._prop_nodes = self._initialize_prop_rob()

    @property
    def tv_time_step(self) -> float:
        return self._tv

    @property
    def other_id(self) -> int:
        return self._other_id

    @property
    def type(self):
        return MonitorType.STL

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def proposition_nodes(self):
        return self._prop_nodes

    @property
    def predicate_nodes(self):
        return self._rule_eval.predicate_nodes

    @property
    def sat_formula(self):
        return self._rule_eval.sat_formula

    @property
    def prop_robust_all(self):
        return self.rob_abstraction.query('other_id == @self._other_id')

    @property
    def prop_robust_ttv(self):
        return self.prop_robust_all.query('time_step == @self._tv')

    def _initialize_prop_rob(self):
        # obtain the id of violation-relevant vehicle
        prop_nodes = self._rule_eval.proposition_nodes
        # assign the robustness at ttv
        if self._tv is math.inf:
            return None
        for node in prop_nodes:
            node.ttv_value = self.prop_robust_ttv.query('alphabet == @node.alphabet')["robustness"].values[0]
        return prop_nodes

    def evaluate_initially(self):
        """
        Evaluate whether the ego vehicle disobeys traffic rules
        """
        return self._rule_eval.\
            evaluate_incremental(self.world_state,
                                 to_pandas=True)

    def evaluate_consecutively(self):
        self._rule_eval.switch_to_boolean()
        self.rob_rule, self.rob_predicate = self._rule_eval.\
            evaluate_consecutively(self.world_state,
                                   )

    def query_rule_rob_all(self):
        if self.rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return self.rob_rule['robustness'].values, self.rob_rule["other_ids"].values

    def _cal_tv_initial(self) -> Tuple[float, Any]:
        # calculate the time-to-violation: detect violation time using STL monitor
        evaluated_robustness, evaluated_ids = self.query_rule_rob_all()
        if evaluated_robustness[0] < 0:
            return -math.inf, evaluated_ids[0][0]  # all violated
        tv = np.argmax(evaluated_robustness < 0)
        if tv == 0:
            return math.inf, None  # no violation
        return tv, evaluated_ids[tv][0]


class MTLRuleMonitor:
    def __init__(self,
                 scenario: Scenario,
                 ego_id: int,
                 rule_set: Union[str, Iterable[str]]):
        self.rule_eval = CommonRoadObstacleEvaluation(os.path.dirname(__file__) + "/../../config/")
        self.rule_eval.activated_traffic_rule_sets = rule_set
        assert self.rule_eval.simulation_param["evaluation_mode"] == "test", "<MTLRuleMonitor>: the given evaluation " \
                                                                             "mode {} is invalid".\
            format(self.rule_eval.simulation_param["evaluation_mode"])
        self.rule_eval.update_eval_dict()
        self._scenario = scenario
        self._ego_id = ego_id

    def evaluate_initially(self):
        """
        Evaluate the rule violation initially - if violated, return the corresponding rule-relevant vehicle (if existed)
        """
        eval_result = self.rule_eval.evaluate_scenario(self._scenario)
        ego_result = None
        for veh_id, evaluation in eval_result:
            if veh_id == self._ego_id:
                ego_result = evaluation
                break
        violation_boolean = False
        violation_veh = list()
        for rule_str, result in ego_result.items():
            if not result:
                violation_veh.append(int(rule_str[-4:]))
                violation_boolean = True
        return violation_boolean, violation_veh