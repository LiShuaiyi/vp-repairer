import numpy as np

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import csv
import math
# https://syncandshare.lrz.de/dl/fi6kDs4fC1UnUu5HBm3cot/highd_scenarios_2024_repaired.zip

file_path = "/home/liny/highd_scenarios_2024_repaired/"

if __name__ == "__main__":
    nr_infeasible = 0
    nr_repairable = 0
    nr_not_repairable = 0

    # Read the scenario violation information from a CSV file
    with open("highD_evaluation_rg1_3_filtered.csv", "r") as f_r:
        reader = csv.reader(f_r)
        header = next(reader)  # Read header if needed
        result_inD = [row for row in reader]  # Read all rows from the file

    # Prepare to write to result CSV files
    with open("highD_evaluation_rg1_3_tc_jerk.csv", "w", newline="") as f_w:
        writer = csv.writer(f_w)

        # Write headers to the result file
        headers = ["scenario_id", "ego_id", "rule", "repairability", "model", "TV", "TC",
                   "SAT_time", "TC_time", "reach_time", "total_time", "number_of_obstacles", "iterations", "jerk"]
        writer.writerow(headers)

        for result in result_inD:
            scenario_id = result[0]
            ego_id = int(result[1])
            rule = result[2]
            print(">>", rule, scenario_id, ego_id)
            scenario_id_with_extension = scenario_id.replace('T-1', 'T-' + str(ego_id))

            # Load the scenario and configuration
            config = RepairerConfiguration()
            config.general.path_scenarios = file_path

            config.general.set_path_scenario(scenario_id_with_extension)
            config.update()

            config.repair.scenario_type = "interstate"

            config.repair.rules = ["R_G1", "R_G3"]
            config.repair.ego_id = ego_id
            config.repair.N_r = config.scenario.obstacle_by_id(ego_id).prediction.trajectory.final_state.time_step

            config.repair.use_mpr = False
            config.repair.use_mpr_derivative = False
            config.repair.use_dummy_tc = False
            config.debug.show_plots = False
            config.repair.planner = 2
            config.repair.constraint_mode = 2

            # Retrieve the ego vehicle
            ego_initial = retrieve_ego_vehicle(config)

            # ========== Traffic Rule Monitor =========
            try:
                traffic_rule_monitor = STLRuleMonitor(config)
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([scenario_id, ego_id, rule, f"initialization fails with {e}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue

            if traffic_rule_monitor.tv_time_step in (-math.inf, math.inf):
                writer.writerow([scenario_id, ego_id, rule, "initial feasibility"])
                nr_infeasible += 1
                continue

            # ========== Trajectory Repairing =========
            try:
                repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([scenario_id, ego_id, rule, f"initialization fails with {e}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue
            try:
                repaired_traj = repairer.repair()
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([scenario_id, ego_id, rule, f"error/failed with {e}", repairer.model, repairer.tv, repairer.tc,
                                 repairer.sat_reasoning_time,
                                 getattr(repairer.t_solver, 'tc_search_time', 'N/A'),
                                 getattr(repairer.t_solver, 'reach_set_time', 'N/A'),
                                 getattr(repairer.t_solver, 'total_runtime', 0) + repairer.sat_reasoning_time,
                                 len(config.scenario.obstacles), repairer.nr_iter])
                continue

            # Writing result based on repairability
            if repaired_traj is not None:
                acc_list = [state.acceleration for state in repaired_traj.state_list]
                dt = config.scenario.dt
                jerk_list = [(acc_list[i + 1] - acc_list[i]) / dt for i in range(len(acc_list) - 1)]
                jerk_magnitudes = [np.linalg.norm(jerk) for jerk in jerk_list]
                jerk_cost = sum(jerk_magnitudes)
                nr_repairable += 1
                writer.writerow([scenario_id, ego_id, rule, "bingo", repairer.model, repairer.tv, repairer.tc,
                                 repairer.sat_reasoning_time,
                                 getattr(repairer.t_solver, 'tc_search_time', 'N/A'),
                                 getattr(repairer.t_solver, 'reach_set_time', 'N/A'),
                                 getattr(repairer.t_solver, 'total_runtime', 0) + repairer.sat_reasoning_time,
                                 len(config.scenario.obstacles), repairer.nr_iter, jerk_cost])
            else:
                nr_not_repairable += 1
                writer.writerow([scenario_id, ego_id, rule, "not repairable", repairer.model, repairer.tv, repairer.tc,
                                 repairer.sat_reasoning_time,
                                 getattr(repairer.t_solver, 'tc_search_time', 'N/A'),
                                 getattr(repairer.t_solver, 'reach_set_time', 'N/A'),
                                 getattr(repairer.t_solver, 'total_runtime', 0) + repairer.sat_reasoning_time,
                                 len(config.scenario.obstacles), repairer.nr_iter])

        # Write the summary to the result file
        nr_total = nr_infeasible + nr_repairable + nr_not_repairable
        if nr_total > 0:
            writer.writerow(["Nr of scenarios", str(nr_total), "Nr of infeasible trajectories", str(nr_infeasible),
                             str(round(nr_infeasible / nr_total, 4)), "Nr of repairable trajectories",
                             str(nr_repairable),
                             str(round(nr_repairable / (nr_repairable + nr_not_repairable), 4))])

        print("Evaluation finished")
