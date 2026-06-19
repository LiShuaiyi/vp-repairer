from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)
from crrepairer.smt.sat_solver.dpll import DPLL
from crrepairer.smt.sat_solver.dpll_domain import DomainDPLL
from crmonitor.common.world import World
from shapely import affinity
import numpy as np
import time

import math


def StopLineEval(monitor: STLRuleMonitor, t_start: int, t_end: int) -> dict:
    dict = {}
    for t in range(t_start, t_end):
        if t >= len(monitor.rob_predicate[0]):
            continue
        for key, value in monitor.rob_predicate[0][t].items():
            if dict.get(key) is None:
                dict[key] = set()
            if value[0] > 0:
                dict[key].add(1)
            else:
                dict[key].add(0)
    return dict


if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    # scenario_id = "DEU_AachenBendplatz-1_152460_T-2479"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10161
    # config.repair.N_r = 20
    # config.update()
    # config.repair.use_mpr = False
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 1
    # config.debug.plot_limits = [40, 69, -45, -17]

    # scenario_id = "DEU_AachenBendplatz-1_151520_T-1539"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10108
    # config.repair.N_r = 20
    # config.update()
    # config.debug.show_plots = True
    # config.repair.planner = 2
    # config.repair.constraint_mode = 1
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False
    # config.debug.plot_limits = [37, 54, -30, -12]

    scenario_id = "DEU_AachenBendplatz-1_75140_T-5159"
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.rules = ["R_IN1"]
    config.repair.ego_id = 10203
    config.repair.N_r = 20
    config.update()
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = True
    config.repair.planner = 2
    config.repair.constraint_mode = 1
    config.debug.plot_limits = [53, 62, -30, -20]


    ego_initial = retrieve_ego_vehicle(config)
    t_0 = config.repair.t_0

    # ========== Velocity Planning Feasibility Estimation Test Demo =========
    # ========== Some Pre-processing =========
    traffic_rule_monitor = STLRuleMonitor(config)
    repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
    cart_trajectory_before_repair = ego_initial.prediction.trajectory.state_list    # prediction probably does not include the initial state

    # ========== Estimate Predicate Range =========
    time0 = time.time()
    
    pred_dict = StopLineEval(traffic_rule_monitor, 0, len(cart_trajectory_before_repair))
    if pred_dict is None:
        print("No predicate evaluation result for stop line rule.")
    time1 = time.time()
    # print(f"Average time to evaluate predicates: {((time1 - time0)/len(cart_trajectory_before_repair)):.6f}s")
    print(f"Time for predicate evaluation: {time1 - time0:.6f}s")
    
    # print(traffic_rule_monitor.rob_rule)
    # print(traffic_rule_monitor.rob_predicate)
    # print(traffic_rule_monitor.other_ids)

    # ========== SAT Solve with Domain Constraints =========
    prop_nodes = repairer.sat_solver._prop_nodes
    formula = repairer.sat_solver._formula
    
    domain_dict = {}
    for prop_node in prop_nodes:
        prop_name = prop_node.name
        for key, value in pred_dict.items():
            if key in prop_name and len(value) == 1:
                domain_dict[prop_node.alphabet] = value
                break
    
    # print('prop_nodes: ', prop_nodes)
    # print('pred_dict: ', pred_dict)
    print('domain_dict: ', domain_dict)

    time0 = time.time()
    domain_solver = DomainDPLL(formula, prop_nodes)
    time_set_domain_start = time.time()
    domain_solver.set_domains(domain_dict)
    time_set_domain_end = time.time()
    domain_sat_result = domain_solver.solve()
    time1 = time.time()
    print(f"DomainDPLL init time: {time_set_domain_start - time0:.6f}s")
    print(f"DomainDPLL set_domains time: {time_set_domain_end - time_set_domain_start:.6f}s")
    print(f"DomainDPLL solve time: {time1 - time_set_domain_end:.6f}s")
    print(f"Time to solve SAT with domain constraints for all prop tuples: {time1 - time0:.6f}s")
        
