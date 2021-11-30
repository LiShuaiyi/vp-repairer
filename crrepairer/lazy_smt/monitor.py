import abc
import os.path
from typing import Iterable, Union
from enum import Enum

from stl_crmonitor.crmonitor.evaluation.evaluation import RuleSetEvaluator
from stl_crmonitor.crmonitor.common.world_state import WorldState
from mtl_crmonitor.common.commonroad_evaluation import CommonRoadObstacleEvaluation
from commonroad.scenario.scenario import Scenario


class MonitorType(Enum):
    MTL = "metric temporal logic"
    STL = "signal temporal logic"


class STLRuleMonitor:
    def __init__(self,
                 world_state,
                 rules: Union[str, Iterable[str]],):
        self._rule_eval = RuleSetEvaluator.create_from_config(rules)
        self.world_state: WorldState = world_state
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self.evaluate_initially()

    @property
    def type(self):
        return MonitorType.STL

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def proposition_nodes(self):
        return self._rule_eval.proposition_nodes

    @property
    def predicate_nodes(self):
        return self._rule_eval.predicate_nodes

    @property
    def sat_formula(self):
        return self._rule_eval.sat_formula

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