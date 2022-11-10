import functools
import math
from typing import Iterable, Union, Tuple, Any, List
from enum import Enum
import numpy as np
import dataclasses
from dataclasses import dataclass
import pandas as pd

from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.common.world import World
from crmonitor.monitor.rule import PredicateNode

# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem


def flatten_nested_dict(data, path=tuple()):
    entries = []
    for key, val in data.items():
        if isinstance(val, dict):
            entries.extend(flatten_nested_dict(val, path + (key,)))
        else:
            entries.append(path + (key, val))
    return entries


def pandas_from_nested_dict(data, level_names):
    entries = flatten_nested_dict(data)
    return pd.DataFrame(entries, columns=level_names)


@dataclass
class PropositionNode:
    name: str
    alphabet: str
    ttv_value: float
    children: List[PredicateNode] = dataclasses.field(default_factory=list)


class MonitorType(Enum):
    """
    Type of temporal logic used in the traffic rule monitor
    """
    MTL = "metric temporal logic"
    STL = "signal temporal logic"


class STLRuleMonitor:
    def __init__(self,
                 scenario: Scenario,
                 vehicle_id: int,
                 rules: Union[str, Iterable[str]], ):
        self._world: World = World.create_from_scenario(scenario)
        self._vehicle_id = vehicle_id
        self._rules = rules
        # todo: now only one rule is supported
        self._rule_eval = RuleEvaluator.create_from_config(self._world,
                                                           self._world.vehicle_by_id(self._vehicle_id),
                                                           rules)
        self.rob_rule, self.rob_predicate, self.rob_abstraction = self.evaluate_initially()
        # obtain the time-to-violation
        self._tv, self._other_id = self._cal_tv_initial()
        self._prop_nodes = self._initialize_prop_rob()
        print("# =========== Traffic Rule Monitor ========== #")
        print("\tthe ego vehicle (id: {})'s initial\n\ttrajectory violates traffic rule {}".
              format(self._vehicle_id, self._rules))
        print('\tw.r.t vehicle {} at time step {}.'.format(self.other_id, self.tv_time_step))
        print("# =========================================== #")

    @property
    def tv_time_step(self) -> Union[int, float]:
        return self._tv

    @property
    def other_id(self) -> int:
        if self._other_id is None or not isinstance(self._other_id, int):
            return self._vehicle_id
        else:
            return self._other_id

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @property
    def world(self) -> World:
        return self._world

    @property
    def type(self):
        return MonitorType.STL

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def proposition_nodes(self) -> List[PropositionNode]:
        return self._prop_nodes

    @property
    def sat_formula(self):
        return self._rule_eval.sat_formula

    @property
    @functools.lru_cache(128)
    def prop_robust_all(self):
        return self.rob_abstraction.query('other_id == @self._other_id')

    @property
    @functools.lru_cache(128)
    def prop_robust_ttv(self):
        return self.prop_robust_all.query('time_step == @self._tv')


    def _initialize_prop_rob(self):
        """
        Construct 'nodes' for propositions for better backward compatibility.

        Returns:
        prop_nodes (List[PropositionNode]): List of proposition nodes
        """
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        def retrieve_preds(node, liste):
            # Method for retrieving PredicateNodes, which are to be used in determining maneuvers
            for child in node.children:
                if hasattr(child, 'latest_value'):
                    liste.append(child)
                else:
                    retrieve_preds(child, liste)

        if self._tv in (math.inf, -math.inf):
            return None
        all_props = self.prop_robust_ttv()
        prop_nodes = []
        pred_nodes = []
        retrieve_preds(self._rule_eval._rule, pred_nodes)
        for idx, row in all_props.iterrows():
            proposition = PropositionNode(row["prop_name"], alphabet[idx - all_props.index[0]], row["robustness"])
            for pred in pred_nodes:
                if pred.name in row["rule_name"]:
                    proposition.children.append(pred)
            prop_nodes.append(proposition)

        return prop_nodes

    def evaluate_initially(self):
        """
        Evaluate whether the ego vehicle disobeys traffic rules

        Update: Now uses the get_propositions method of the STLRuleEvaluator to obtain the rule, predicate, and
        proposition robustness values. The values are only obtained for the highest non-conforming vehicle at
        each time step, the id of which is kept for better backward compatibility and higher verbosity.

        Returns:
        df_rule (pd.Dataframe): DF constructed of the rule robustness at each timestep
        df_pred (pd.Dataframe): DF constructed of each predicate robustness at each timestep for given other_id
        df_prop (pd.Dataframe): DF constructed of each proposition robustness at each timestep for given other_id
        """
        rule_robustness = {}
        predicate_robustness = {}
        proposition_robustness = {}
        other_ids_values = {}

        #TODO: Incorporate support for multiple rules.

        while self._rule_eval.current_time() <= self._rule_eval.ego_vehicle.end_time:
            t = self._rule_eval.current_time()
            rule_robustness[t] = {}
            predicate_robustness[t] = {}
            proposition_robustness[t] = {}
            other_ids_values[t] = {}
            for rule in self._rules:
                rul = self._rule_eval.update()
                pred = self._rule_eval.get_predicates()
                prop, other, time = self._rule_eval.get_propositions()
                other_ids_values[t][rule.name] = other
                rule_robustness[t][rule.name] = rul

                proposition_robustness[t][rule.name][other] = {}
                for prop_name in prop.keys():
                    proposition_robustness[t][rule.name][other][prop_name] = prop[prop_name]

                predicate_robustness[t][rule.name] = {}
                for full_name in pred.keys():
                    predicate_robustness[t][rule.name][full_name] = pred[full_name]
                other_ids_values[t][rule.name] = other

        df_rule = pandas_from_nested_dict(rule_robustness,
                                          ["time_step", "rule_name", "robustness"])
        df_pred = pandas_from_nested_dict(predicate_robustness,
                                          ["time_step", "rule_name", "full_name", "robustness"])
        df_prop = pandas_from_nested_dict(proposition_robustness,
                                          ["time_step", "rule_name", "other_id", "prop_name", "robustness"])
        df_ids = pandas_from_nested_dict(other_ids_values,
                                         ["time_step", "rule_name", "other_ids"])
        df_rule = df_rule.merge(df_ids, on=["time_step", "rule_name"])

        return df_rule, df_pred, df_prop

    def evaluate_consecutively(self):
        """
        Evaluate the updated vehicle states (boolean assignments) in order to speed up the evaluation progress
        """
        self._rule_eval.switch_to_boolean()
        world_state = copy.copy(self._world)
        time_begin = world_state.time_step
        self._rule_eval.reset(world_state.vehicle_by_id(self._vehicle_id), world_state, 0)
        return self.evaluate_initially()

    def query_rule_rob_all(self):
        """
        Queries the robustness value and the other vehicle id with the minimum robustness
        """
        if self.rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return self.rob_rule['robustness'].values, self.rob_rule["other_ids"].values

    def _cal_tv_initial(self) -> Tuple[Union[int, float], Any]:
        # calculate the time-to-violation: detect violation time using STL monitor
        evaluated_robustness, evaluated_ids = self.query_rule_rob_all()
        if evaluated_robustness[0] < 0:
            if evaluated_ids[0] is ():
                return -math.inf, None
            return -math.inf, evaluated_ids[0][0]  # all violated
        tv = np.argmax(evaluated_robustness < 0)
        if tv == 0:
            return math.inf, None  # no violation
        if evaluated_ids[tv] is () or self._rules == 'R_G2':  # R_G2: we focus on the ego vehicle
            return int(tv), self._world.ego_vehicle.id
        return int(tv), evaluated_ids[tv][0]

# Currently, MTL monitor is not supported
# class MTLRuleMonitor:
#     def __init__(self,
#                  scenario: Scenario,
#                  ego_id: int,
#                  rule_set: Union[str, Iterable[str]]):
#         self.rule_eval = CommonRoadObstacleEvaluation(os.path.dirname(__file__) + "/../../config/")
#         self.rule_eval.activated_traffic_rule_sets = rule_set
#         assert self.rule_eval.simulation_param["evaluation_mode"] == "test", "<MTLRuleMonitor>: the given evaluation " \
#                                                                              "mode {} is invalid".\
#             format(self.rule_eval.simulation_param["evaluation_mode"])
#         self.rule_eval.update_eval_dict()
#         self._scenario = scenario
#         self._ego_id = ego_id
#
#     def evaluate_initially(self):
#         """
#         Evaluate the rule violation initially - if violated, return the corresponding rule-relevant vehicle (if existed)
#         """
#         eval_result = self.rule_eval.evaluate_scenario(self._scenario)
#         ego_result = None
#         for veh_id, evaluation in eval_result:
#             if veh_id == self._ego_id:
#                 ego_result = evaluation
#                 break
#         violation_boolean = False
#         violation_veh = list()
#         for rule_str, result in ego_result.items():
#             if not result:
#                 violation_veh.append(int(rule_str[-4:]))
#                 violation_boolean = True
#         return violation_boolean, violation_veh
