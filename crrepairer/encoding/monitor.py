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
                 monitor_type: MonitorType = MonitorType.STL):
        self._rule_eval = RuleSetEvaluator.create_from_config(rules)
        self.world_state = world_state
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self.evaluate_initially()

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def abstraction_nodes(self):
        return self._rule_eval.abstraction_nodes

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
        rob_rule, _ = self._rule_eval.result2pandas(self.rob_rule, self.rob_predicate)
        if rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return rob_rule['robustness'].values
