from typing import Iterable, Union
from enum import Enum

from crmonitor.evaluation.evaluation import RuleSetEvaluator


class MonitorType(Enum):
    MTL = "metric temporal logic"
    STL = "signal temporal logic"


class RuleMonitor:
    def __init__(self,
                 world_state,
                 rules: Union[str, Iterable[str]],
                 monitor_type: MonitorType = MonitorType.STL,
                 to_pandas: bool = True):
        self._rule_eval = RuleSetEvaluator.create_from_config(rules)
        self._to_pandas = to_pandas
        self.world_state = world_state
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self.evaluate_initially()
        if self._to_pandas:
            self.rob_rule, self.rob_predicate = self._rule_eval.result2pandas(self.rob_rule, self.rob_predicate)
            self.rob_abstraction = self._rule_eval.abstraction2pandas(self.rob_abstraction)

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def abstraction_nodes(self):
        return self._rule_eval.abstraction_nodes

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
                                 to_pandas=False)

    def evaluate_consecutively(self):
        self.rob_rule, self.rob_predicate = self._rule_eval.\
            evaluate_consecutively(self.world_state,
                                   self.rob_rule,
                                   self.rob_predicate,)

    def query_rule_rob_all(self):
        if self.rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return self.rob_rule['robustness'].values
