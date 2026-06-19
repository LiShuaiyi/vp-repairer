from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import csv
import math
import logging
import warnings
from multiprocessing import Process, Queue
import time

file_path = "/home/liny/ind_scenarios_2024_repaired/"

def evaluate_single_case(result, file_path, queue):
    try:
        scenario_id, ego_id, rule = result[0], int(result[1]), result[2]
        scenario_id_ext = scenario_id.replace('-1_', f'-{ego_id}_')
        print(">>", rule, scenario_id, ego_id, flush=True)
        config = RepairerConfiguration()
        config.general.path_scenarios = file_path
        config.general.set_path_scenario(scenario_id_ext)
        config.update()

        config.repair.scenario_type = "intersection"
        config.repair.intersection_type = "dataset"
        config.repair.rules = [rule]
        config.repair.ego_id = ego_id
        config.repair.N_r = config.scenario.obstacle_by_id(ego_id).prediction.trajectory.final_state.time_step
        config.repair.use_mpr = False
        config.repair.use_mpr_derivative = False
        config.repair.use_dummy_tc = False
        config.debug.show_plots = False
        config.repair.planner = 3
        config.repair.constraint_mode = 2

        ego_initial = retrieve_ego_vehicle(config)
        traffic_rule_monitor = STLRuleMonitor(config)

        if traffic_rule_monitor.tv_time_step in (-math.inf, math.inf):
            queue.put([scenario_id, ego_id, rule, "initial feasibility"])
            return

        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()

        if repaired_traj is not None:
            queue.put([scenario_id, ego_id, rule, "bingo", repairer.model, repairer.tv, repairer.tc,
                       repairer.t_solver.compliant_maneuvers,
                       repairer.sat_reasoning_time,
                       getattr(repairer.t_solver, 'tc_search_time', 'N/A'),
                       getattr(repairer.t_solver, 'reach_set_time', 'N/A'),
                       getattr(repairer.t_solver, 'total_runtime', 0) + repairer.sat_reasoning_time,
                       len(config.scenario.obstacles),
                       repairer.nr_iter])
        else:
            queue.put([scenario_id, ego_id, rule, "not repairable", repairer.model, repairer.tv, repairer.tc,
                       repairer.t_solver.compliant_maneuvers,
                       repairer.sat_reasoning_time,
                       getattr(repairer.t_solver, 'tc_search_time', 'N/A'),
                       getattr(repairer.t_solver, 'reach_set_time', 'N/A'),
                       getattr(repairer.t_solver, 'total_runtime', 0) + repairer.sat_reasoning_time,
                       len(config.scenario.obstacles),
                       repairer.nr_iter])
    except Exception as e:
        queue.put([result[0], result[1], result[2], f"error/failed with {e}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

if __name__ == "__main__":
    nr_infeasible = 0
    nr_repairable = 0
    nr_not_repairable = 0

    with open("inD_evaluation_rin1_4_filtered.csv", "r") as f_r:
        reader = csv.reader(f_r)
        next(reader)
        result_inD = [row for row in reader]

    with open("inD_evaluation_rin_clrrt.csv", "a", newline="") as f_w:
        writer = csv.writer(f_w)
        headers = ["scenario_id", "ego_id", "rule", "repairability", "model", "TV", "TC",
                   "SAT_time", "TC_time", "reach_time", "total_time", "number_of_obstacles", "iterations"]
        writer.writerow(headers)

        for result in result_inD:
            q = Queue()
            p = Process(target=evaluate_single_case, args=(result, file_path, q))
            p.start()
            p.join(timeout=10)

            if p.is_alive():
                p.terminate()
                p.join()
                writer.writerow([result[0], result[1], result[2], "timeout", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue

            output = q.get()
            writer.writerow(output)
            if output[3] == "initial feasibility":
                nr_infeasible += 1
            elif output[3] == "bingo":
                nr_repairable += 1
            elif output[3] == "not repairable":
                nr_not_repairable += 1

        total = nr_infeasible + nr_repairable + nr_not_repairable
        if total > 0:
            writer.writerow(["Nr of scenarios", str(total), "Nr of infeasible trajectories", str(nr_infeasible),
                             str(round(nr_infeasible / total, 4)), "Nr of repairable trajectories",
                             str(nr_repairable),
                             str(round(nr_repairable / (nr_repairable + nr_not_repairable + 1e-9), 4))])
        print("Evaluation finished")