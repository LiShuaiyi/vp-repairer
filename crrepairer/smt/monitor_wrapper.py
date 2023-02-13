import functools
import math
from typing import Iterable, Union, Tuple, Any, List
from enum import Enum
import numpy as np
import dataclasses
from dataclasses import dataclass
import copy
from difflib import SequenceMatcher, get_close_matches

from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.common.world import World
from crmonitor.monitor.rule import PredicateNode

# CommonRoad Toolbox
from commonroad.scenario.scenario import Scenario


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
        self._rules = [rules] if isinstance(rules, str) else rules 
        # todo: now only one rule is supported
        # todo: create multiple rule evaluators
        self._rule_eval = []
        for rule in self._rules:
            self._rule_eval.append(RuleEvaluator.create_from_config(self._world,
                                                           self._world.vehicle_by_id(self._vehicle_id),
                                                           rule))
        self.rob_rule, self.rob_predicate, self.rob_abstraction, self.abstraction_names, \
            self.other_ids = self.evaluate_initially()
        # obtain the time-to-violation
        self._violated_rule_idx, self._tv, self._other_id = self._cal_tv_initial()
        self._prop_nodes = self._initialize_prop_rob()
        print("# =========== Traffic Rule Monitor ========== #")
        print("\tthe ego vehicle (id: {})'s initial\n\ttrajectory violates traffic rule {}".
              format(self._vehicle_id, self._rules[self._violated_rule_idx]))
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
        """
        For all propositions find the overlapping subsequences in the rule string
        and replace with the alphabet.
        """
        subformula_list = []
        prev_idx = 0
        for i, evaluator in enumerate(self._rule_eval):
            rule_node = evaluator._rule
            if len(rule_node.children) == 1:
                sat_formula = rule_node.children[0].rule_str
            else:
                sat_formula = rule_node.rule_str
                #for child in rule_node.children:
                #    if hasattr(child, 'quantified_vehicle'):
                #        sat_formula = sat_formula.replace(child.name,
                #                                          child.children[0].rule_str)
                #        print(sat_formula)
            sat_formula = sat_formula.replace('(', '').replace(')', '').replace('not', '!')
            clear_rob_abs = self.rob_abstraction[i][self.rob_abstraction[i]==self.rob_abstraction[i]]
            length = int(clear_rob_abs.shape[0] / self.rob_abstraction[i].shape[0])
            props_of_rule = self._prop_nodes[prev_idx:prev_idx+length]
            prev_idx += length
            for prop_node in props_of_rule:
                matches = SequenceMatcher(None, 
                                          sat_formula, 
                                          prop_node.name, 
                                          autojunk=True).get_matching_blocks()
                clean_matches = [match for match in matches if match.size>1]
                first_index = clean_matches[0].a
                last_index = clean_matches[-1].a+clean_matches[-1].size
                to_repl = sat_formula[first_index:last_index]
                sat_formula = sat_formula.replace(to_repl, prop_node.alphabet)  
            if 'implies' in sat_formula:
                impl_at = sat_formula.find('implies')
                sat_formula = '(' + sat_formula[:impl_at] + ') ' + sat_formula[impl_at:]
            subformula_list.append('(' + sat_formula + ')')
        for i, substr in enumerate(subformula_list[:-1]):
            subformula_list[i] = substr + ' and '
        return ''.join(subformula_list)

    @property
    @functools.lru_cache(128)
    def prop_robust_all(self):
        return self.rob_abstraction#[self._violated_rule_idx]

    @property
    @functools.lru_cache(128)
    def prop_robust_ttv(self):
        return self.rob_abstraction[self._violated_rule_idx][self._tv]


    def _initialize_prop_rob(self):
        """
        Construct 'nodes' for propositions for better backward compatibility.

        Returns:
        prop_nodes (List[PropositionNode]): List of proposition nodes
        """
        # Currently does not support multiple quantifiers at the same level
        # TODO: Incorporate support for multiple quantifiers
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
        all_prop_robs = self.rob_abstraction[:, self._tv]
        all_prop_names = self.abstraction_names[:, self._tv]
        prop_nodes = []
        for idx in np.transpose(np.isfinite(all_prop_robs).nonzero()):
            proposition = PropositionNode(all_prop_names[tuple(idx)],
                                          alphabet[len(prop_nodes)],
                                          all_prop_robs[tuple(idx)])
            pred_nodes = []
            retrieve_preds(self._rule_eval[idx[0]]._rule, pred_nodes)
            for pred in pred_nodes:
                if 'g0' not in all_prop_names[tuple(idx)]:
                    if pred.name in all_prop_names[tuple(idx)]:
                        proposition.children.append(pred)
                else:
                    other_props = np.delete(all_prop_names[idx[0]], idx[-1], 0)
                    if not any([pred.name in p_name for p_name in other_props[other_props==other_props]]):
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
        df_rule (np.ndarray): DF constructed of the rule robustness at each timestep
        df_pred (np.ndarray): DF constructed of each predicate robustness at each timestep for given other_id
        df_prop (np.ndarray): DF constructed of each proposition robustness at each timestep for given other_id
        other_ids List(Tuple): Vehicle ids w.r.t which the rule robustness was calculated
        """
        rule_rob_all = []
        prop_rob_all = []
        prop_names_all = []
        pred_rob_all = []
        other_ids_all =[]
        for evaluator in self._rule_eval:
            #TODO: Incorporate support for multiple rules.
            rule_rob = [] #init as list, convert to ndarray 2x faster, but 2x more memory
            prop_rob = []
            prop_names = []
            pred_rob = []
            other_ids = []
            # update until start time is reached
            # while self._rule_eval.current_time < self._rule_eval.ego_vehicle.start_time:
            #     self._rule_eval.update()
            for _ in range(
                    evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1
            ):
                rule_rob.append(evaluator.update())
                other_ids.append(evaluator.other_ids)
                prop, _, _ = evaluator.get_propositions()
                if prop:
                    prop_names.append([prop_name for prop_name in prop.keys()])
                    prop_rob.append([prop[prop_name] for prop_name in prop.keys()])
                else:
                    prop_names.append([])
                    prop_rob.append([])
                pred = evaluator.get_predicates()
                if pred:
                    pred_rob.append([pred[pred_name] for pred_name in pred.keys()])
                else:
                    pred_rob.append([])
            rule_rob_all.append(np.array(rule_rob, dtype=np.float64))
            prop_rob_all.append(np.array(prop_rob, dtype=np.float64))
            prop_names_all.append(np.array(prop_names, dtype=object))
            pred_rob_all.append(np.array(pred_rob, dtype=np.float64))
            other_ids_all.append(other_ids)
        assert len(rule_rob_all) == len(self._rule_eval)
        max_n_props = max([p.shape[1] for p in prop_rob_all])
        for idx, prop_array in enumerate(prop_rob_all):
            if prop_array.shape[1] < max_n_props:
                prop_rob_all[idx] = np.pad(prop_array, ((0,0),(0, max_n_props-prop_array.shape[1])), 
                                           'constant', constant_values=np.nan)
                prop_names_all[idx] = np.pad(prop_names_all[idx], ((0,0), (0, max_n_props-prop_array.shape[1])), 
                                             'constant', constant_values=np.nan)
        return np.array(rule_rob_all), np.array(pred_rob_all), np.array(prop_rob_all), np.array(prop_names_all), other_ids_all          

    def evaluate_consecutively(self, world, reset_time):
        """
        Evaluate the updated vehicle states (boolean assignments) in order to speed up the evaluation progress
        """
        world_state = copy.copy(world)
        rule_rob_all = []
        other_ids_all = []
        for evaluator in self._rule_eval:
            evaluator.reset(world_state.vehicle_by_id(self._vehicle_id), world_state, reset_time)
            self.switch_to_boolean(evaluator)
            rule_rob = []
            other_ids = []
            while evaluator.current_time < evaluator.ego_vehicle.end_time:
                rule_rob.append(evaluator.update())
                other_ids.append(evaluator.other_ids)
            rule_rob_all.append(np.array(rule_rob, dtype=np.float64))
            other_ids_all.append(other_ids)
        return np.array(rule_rob_all), other_ids_all

    def query_rule_rob_all(self):
        """
        Queries the robustness value and the other vehicle id with the minimum robustness
        """
        if self.rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return self.rob_rule, self.other_ids

    def _cal_tv_initial(self) -> Tuple[Any, Union[int, float], Any]:
        # calculate the time-to-violation: detect violation time using STL monitor
        # evaluated_robustness, evaluated_ids = self.query_rule_rob_all()
        if np.any(self.rob_rule[:, 0] < 0):
            rule_idx = np.where(self.rob_rule[:, 0] < 0)[0][0]
            if self.other_ids[rule_idx][0] is ():
                return None, -math.inf, None
            return None, -math.inf, self.other_ids[rule_idx][0][0]  # all violated
        tv_per_rule = np.argmax(self.rob_rule < 0, axis=-1)
        if np.all(tv_per_rule == 0):
            return None, math.inf, None  # no violation
        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if self.other_ids[rule_idx][min_tv] is ():# or self._rules[rule_idx] == 'R_G2':
            # R_G2: we focus on the ego vehicle
            return rule_idx, int(min_tv), self._vehicle_id
        return rule_idx, int(min_tv), self.other_ids[rule_idx][min_tv][0]

    def switch_to_boolean(self, evaluator):
        if not evaluator._eval_visitor.use_boolean:
            evaluator._eval_visitor.use_boolean = True

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

