import functools
import math
from typing import Iterable, Union, Tuple, Any, List
from collections import defaultdict
import numpy as np
import dataclasses
from dataclasses import dataclass
import copy
from difflib import SequenceMatcher
from multiprocessing import Process, Queue

import re
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.evaluation.evaluation import (
    get_evaluation_config,
    create_ego_vehicle_param,
)
from crmonitor.common.world import World, get_world_config
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.rule_node import PredicateNode

from crrepairer.utils.configuration import RepairerConfiguration, ScenarioType, MonitorType

from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg


@dataclass
class PropositionNode:
    name: str
    alphabet: str
    ttv_value: float
    children: List[PredicateNode] = dataclasses.field(default_factory=list)


class STLRuleMonitor:
    def __init__(
        self,
        config: RepairerConfiguration,
    ):
        # update the world configuration for repairing purposes
        world_config = get_world_config()
        traffic_rules_config = get_traffic_rule_config()

        world_config["scenario"] = self.scenario_type =\
            traffic_rules_config["traffic_rules_param"]["mpr_scenario"] = config.repair.scenario_type
        Cfg["common"]["scenario"] = self.scenario_type
        if self.scenario_type == ScenarioType.INTERSECTION:
            world_config["intersection_road_network_param"]["map_type"] = config.repair.intersection_type
        traffic_rules_config["traffic_rules_param"]["use_mpr"] = config.repair.use_mpr

        self._world: World = World.create_from_scenario(config.scenario, config=world_config)
        self._vehicle_id = config.repair.ego_id
        self.multiproc = config.repair.multiproc
        self._rules = config.repair.rules
        self._rule_eval = []
        self._start_time_step = self._world.vehicle_by_id(self._vehicle_id).start_time
        self._world.vehicle_by_id(
            self._vehicle_id
        ).vehicle_param = create_ego_vehicle_param(
            get_evaluation_config().get("ego_vehicle_param"), self._world.dt
        )
        for rule in self._rules:
            prop_rule_eval = PropositionRuleEvaluator.create_from_config(
                self._world, self._vehicle_id, rule, traffic_rules_config=traffic_rules_config
            )
            self._rule_eval.append(prop_rule_eval)
        if len(self._rule_eval) == 1:
            self.multiproc = False
        (
            self.rob_rule,
            self.rob_predicate,
            self.rob_abstraction,
            self.abstraction_names,
            self.other_ids,
        ) = self.evaluate_initially()
        # obtain the time-to-violation
        self._violated_rule_idx, self._tv, self._other_id = self._cal_tv_initial()
        self._prop_nodes = self._initialize_prop_rob()
        self._future_time_step = self.search_future_time_step()[self._violated_rule_idx]
        print("# =========== Traffic Rule Monitor ========== #")
        print(
            "\tthe ego vehicle (id: {})'s initial\n\ttrajectory violates traffic rule {}".format(
                self._vehicle_id, self._rules[self._violated_rule_idx]
            )
        )
        print(
            "\tw.r.t vehicle {} at time step {}.".format(
                self.other_id, self.tv_time_step
            )
        )
        print("# =========================================== #")

    @property
    def tv_time_step(self) -> Union[int, float]:
        return self._tv - self._future_time_step

    @property
    def future_time_step(self) -> Union[int, float]:
        return self._future_time_step

    @property
    def start_time_step(self):
        return self._start_time_step

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
                # for child in rule_node.children:
                #    if hasattr(child, 'quantified_vehicle'):
                #        sat_formula = sat_formula.replace(child.name,
                #                                          child.children[0].rule_str)
                #        print(sat_formula)
            # do not delete brackets in the rule
            # 'eventually' is replaced by 'once' because of the same replacement in rtamt
            sat_formula = (
                sat_formula.replace("not", "!")
                .replace("eventually", "once")
            )
            clear_rob_abs = self.rob_abstraction[i][
                self.rob_abstraction[i] == self.rob_abstraction[i]
            ]
            length = int(clear_rob_abs.shape[0] / self.rob_abstraction[i].shape[0])
            props_of_rule = self._prop_nodes[prev_idx : prev_idx + length]
            prev_idx += length
            for prop_node in props_of_rule:
                prop_node_name = prop_node.name
                # if proposition name starts with "once[x,x]", it will be considered as predicate
                if(
                    prop_node_name[0:4] == "once"
                    and prop_node_name[5:6] == prop_node_name[7:8]
                    ):
                    prop_node_name = prop_node_name.replace(
                        prop_node_name[0:9], ""
                    )
                else:
                    pattern = r"once\[(.*?)\]"
                    matches = re.findall(pattern, prop_node_name)
                    if len(matches) == 2:
                        delete_once_str = "once[" + matches[1] + "]("
                        prop_node_name = prop_node_name.replace(delete_once_str, "")
                        prop_node_name = prop_node_name[:-1]
                if prop_node_name.startswith('(') and prop_node_name.endswith(')'):
                    prop_node_name = prop_node_name[1:-1]
                matches = SequenceMatcher(
                    None, sat_formula, prop_node_name, autojunk=True
                ).get_matching_blocks()
                # TODO: match.size>1, further check necessary
                clean_matches = [match for match in matches if match.size > 1]
                first_index = clean_matches[0].a
                last_index = clean_matches[-1].a + clean_matches[-1].size
                to_repl = sat_formula[first_index:last_index]
                to_repl = re.escape(to_repl)
                # avoid issue of replacing wrong proposition
                pattern = rf"(?<!\]\(){to_repl}"
                sat_formula = re.sub(pattern, prop_node.alphabet, sat_formula)
            subformula_list.append("(" + sat_formula + ")")
        for i, substr in enumerate(subformula_list[:-1]):
            subformula_list[i] = substr + " and "
        return "".join(subformula_list)

    @property
    @functools.lru_cache(128)
    def prop_robust_all(self):
        return self.rob_abstraction  # [self._violated_rule_idx]

    @property
    @functools.lru_cache(128)
    def prop_robust_ttv(self):
        return self.rob_abstraction[self._violated_rule_idx][self._tv - self._start_time_step]

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
                if hasattr(child, "latest_value"):
                    liste.append(child)
                else:
                    retrieve_preds(child, liste)

        if self._tv in (math.inf, -math.inf):
            return None
        all_prop_robs = self.rob_abstraction[:, self._tv - self._start_time_step]
        all_prop_names = self.abstraction_names[:, self._tv - self._start_time_step]
        all_pre_rob_grad = self.rob_predicate[:, self._tv - self._start_time_step]

        prop_nodes = []
        for idx in np.transpose(np.isfinite(all_prop_robs).nonzero()):
            proposition = PropositionNode(
                all_prop_names[tuple(idx)],
                alphabet[len(prop_nodes)],
                all_prop_robs[tuple(idx)],
            )
            pred_nodes = []
            retrieve_preds(self._rule_eval[idx[0]]._rule, pred_nodes)
            for pred in pred_nodes:
                if "g0" not in all_prop_names[tuple(idx)]:
                    if pred.name in all_prop_names[tuple(idx)]:
                        # add missing values
                        pred.latest_value, pred.mpr_gradient = all_pre_rob_grad[tuple(idx)[0]][pred.name]
                        proposition.children.append(pred)
                else:
                    other_props = np.delete(all_prop_names[idx[0]], idx[-1], 0)
                    if not any(
                        [
                            pred.name in p_name
                            for p_name in other_props[other_props == other_props]
                        ]
                    ):
                        pred.latest_value, pred.mpr_gradient = all_pre_rob_grad[tuple(idx)[0]][pred.name]
                        proposition.children.append(pred)
            prop_nodes.append(proposition)
        return prop_nodes

    def search_future_time_step(self):
        future_time_step = np.zeros(len(self.rule_eval), dtype=int)
        for i in range(len(future_time_step)):
            prop_robust_rule = self.prop_robust_all[i, :, :]
            for j in range(prop_robust_rule.shape[0]):
                if np.any(np.isinf(prop_robust_rule[j, :])):
                    future_time_step[i] += 1
                else:
                    break
        return future_time_step

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
        other_ids_all = []

        if self.multiproc:
            rule_ids = []
            queue = Queue()
            processes = [
                Process(target=self.multiproc_evaluate, args=[i, queue])
                for i in range(len(self._rule_eval))
            ]
            for p in processes:
                p.start()
            for _ in range(len(self._rule_eval)):
                res = queue.get()
                rule_rob_all.append(res["rule"])
                prop_rob_all.append(res["prop"])
                prop_names_all.append(res["prop_name"])
                pred_rob_all.append(res["pred"])
                other_ids_all.append(res["other"])
                rule_ids.append(res["index"])
            for p in processes:
                p.join()
            self._rule_eval = [self._rule_eval[i] for i in rule_ids]
            self._rules = [self._rules[i] for i in rule_ids]

        else:
            for evaluator in self._rule_eval:
                rule_rob = []
                prop_rob = []
                prop_names = []
                pred_rob = []
                other_ids = []
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
                        try:
                            prop_names.append(prop_names[0])
                            prop_rob.append([-np.inf] * len(prop_rob[0]))
                        except:
                            prop_names.append([])
                            prop_rob.append([])
                    pred = evaluator.get_predicates()
                    if pred:
                        mpr_grad = evaluator.get_mpr_gradient()
                        pred_rob.append({key: [pred[key], mpr_grad[key]] for key in pred})
                    else:
                        pred_rob.append([])

                rule_rob_all.append(np.array(rule_rob, dtype=np.float64))
                prop_rob_all.append(np.array(prop_rob, dtype=np.float64))
                prop_names_all.append(np.array(prop_names, dtype=object))
                pred_rob_all.append(pred_rob)
                other_ids_all.append(other_ids)

        assert len(rule_rob_all) == len(self._rule_eval)
        max_n_props = max([p.shape[1] for p in prop_rob_all])
        for idx, prop_array in enumerate(prop_rob_all):
            if prop_array.shape[1] < max_n_props:
                prop_rob_all[idx] = np.pad(
                    prop_array,
                    ((0, 0), (0, max_n_props - prop_array.shape[1])),
                    "constant",
                    constant_values=np.nan,
                )
                prop_names_all[idx] = np.pad(
                    prop_names_all[idx],
                    ((0, 0), (0, max_n_props - prop_array.shape[1])),
                    "constant",
                    constant_values=np.nan,
                )
        return (
            np.array(rule_rob_all),
            np.array(pred_rob_all),
            np.array(prop_rob_all),
            np.array(prop_names_all),
            other_ids_all,
        )

    def multiproc_evaluate(self, index, q):
        evaluator = self._rule_eval[index]
        rule_rob = []
        prop_rob = []
        prop_names = []
        pred_rob = []
        other_ids = []
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
                mpr_grad = evaluator.get_mpr_gradient()
                pred_rob.append({key: [pred[key], mpr_grad[key]] for key in pred})
            else:
                pred_rob.append({})
        return_dict = {
            "rule": np.array(rule_rob, dtype=np.float64),
            "other": other_ids,
            "prop": np.array(prop_rob, dtype=np.float64),
            "prop_name": np.array(prop_names, dtype=object),
            "pred": np.array(pred_rob, dtype=object),
            "index": index,
        }
        q.put(return_dict)

    def evaluate_consecutively(self, world, reset_time):
        """
        Evaluate the updated vehicle states (boolean assignments) in order to speed up the evaluation progress
        """
        world_state = copy.copy(world)
        rule_rob_all = []
        other_ids_all = []
        for evaluator in self._rule_eval:
            evaluator.reset(
                world_state.vehicle_by_id(self._vehicle_id), world_state, reset_time
            )
            self.switch_to_boolean(evaluator)
            rule_rob = []
            other_ids = []
            while evaluator.current_time < evaluator.ego_vehicle.end_time:
                rule_rob.append(evaluator.update())
                other_ids.append(evaluator.other_ids)
                if rule_rob[-1] < 0:
                    break
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
            if self.other_ids[rule_idx][0] == ():
                return None, -math.inf, None
            return None, -math.inf, self.other_ids[rule_idx][0][0]  # all violated
        tv_per_rule = np.argmax(self.rob_rule < 0, axis=-1) + self._start_time_step
        if np.all(tv_per_rule == self._start_time_step):
            return None, math.inf, None  # no violation
        min_tv = np.min(tv_per_rule[tv_per_rule != self._start_time_step])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if (
            self.other_ids[rule_idx][min_tv - self._start_time_step] == ()
        ):  # or self._rules[rule_idx] == 'R_G2':
            # R_G2: we focus on the ego vehicle
            return rule_idx, int(min_tv), self._vehicle_id
        return rule_idx, int(min_tv), self.other_ids[rule_idx][min_tv - self._start_time_step][0]

    def switch_to_boolean(self, evaluator):
        if not evaluator._eval_visitor.use_boolean:
            evaluator._eval_visitor.use_boolean = True

    def switch_to_robustness(self, evaluator):
        if evaluator._eval_visitor.use_boolean:
            evaluator._eval_visitor.use_boolean = False


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
