from typing import Iterable, Union

from crmonitor.evaluation.evaluation import RuleSetEvaluator


class RuleMonitor:
    def __init__(self, world_state, rules: Union[str, Iterable[str]] = ("R_G1", "R_G2", "R_G3")):
        self._rule_eval = RuleSetEvaluator.create_from_config(rules)
        self.world_state = world_state
        self.rob_rule = None
        self.rob_predicate = None
        self.rob_abstraction = None

    def evaluate_initially(self):
        """
        Evaluate whether the ego vehicle disobeys traffic rules
        """
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self._rule_eval.\
            evaluate_incremental(self.world_state,
                                 to_pandas=False)

    def evaluate_consecutively(self):
        self.rob_rule, self.rob_predicate = self._rule_eval.\
            evaluate_consecutively(self.world_state,
                                   self.rob_rule,
                                   self.rob_predicate,)

    def query_rule_rob_all(self):
        rob_rule, _ = self._rule_eval.result2pandas(self.rob_rule, self.rob_predicate)
        print(rob_rule)
        if rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return rob_rule['robustness'].values
