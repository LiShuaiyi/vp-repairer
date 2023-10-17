from crrepairer.smt.monitor_wrapper import STLRuleMonitor, ScenarioType, IntersectionType
from crrepairer.repairer.smt_repairer_miqp import SMTTrajectoryRepairer

from commonroad.scenario.obstacle import ObstacleType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.prediction.prediction import Trajectory

from crrepairer.repairer.visualization import (
    visualize_repairing_result,
    visualize_repairing_result_thesis,
    visualize_a_profile,
    visualize_v_profile,
)

# other packages
import csv
import math
import glob
import pandas as pd
import os

scenario_path = "/home/ge23lac/scenarios/ind_rin5/"
csv_file_path = "./config"
csv_name = "data_violation_rin5.csv"
rule = ["R_IN5"]

if __name__ == "__main__":
    nr_infeasible = 0
    nr_repairable = 0
    nr_not_repairable = 0
    f_w = open("result_rin5.csv", "w")
    writer = csv.writer(f_w)
    writer.writerow(
        ["scenario_id", "ego_id", "rule", "repairability", "model", "TV", "TC"]
    )

    entry_id_names = ["scenario_id", "ego_id"]
    data = pd.read_csv(os.path.join(csv_file_path, csv_name), header=[0, 1], index_col=list(range(len(entry_id_names))))
    scenario_ids = []
    ego_ids = []
    for index, row in data.iterrows():
        test = index
        scenario_ids.append(index[0])
        ego_ids.append(index[1])
    print(len(ego_ids))
    for i in range(len(ego_ids)):
        scenario_id = scenario_ids[i]
        ego_id = ego_ids[i]
        scenario, planning_problem_set = CommonRoadFileReader(
            scenario_path + scenario_id + ".xml"
        ).open(lanelet_assignment=True)
        planning_problem = list(
            planning_problem_set.planning_problem_dict.values()
        )[0]
        print(rule, scenario_id, ego_id)
        N = 149
        ego_initial = scenario.obstacle_by_id(ego_id)
        ego_initial.prediction.trajectory = Trajectory(
            ego_initial.prediction.initial_time_step, ego_initial.prediction.trajectory.state_list[:N]
        )
        ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]
        if ego_initial.obstacle_type != ObstacleType.CAR:
            continue
        try:
            traffic_rule_monitor = STLRuleMonitor(
                    scenario, ego_id, rule[0], ScenarioType.INTERSECTION, IntersectionType.DATASET, use_mpr=False, mpr_scenario="intersection"
            )
            print(traffic_rule_monitor.tv_time_step, "+", traffic_rule_monitor.other_id)
            if traffic_rule_monitor.tv_time_step is not math.inf:
                writer.writerow([scenario.scenario_id, ego_id, rule, "initial feasibility"])
                nr_infeasible += 1
                repairer = SMTTrajectoryRepairer(
                    traffic_rule_monitor, planning_problem, ego_initial
                )
                repaired_traj = repairer.repair()
        except:
            repaired_traj = None
        if repaired_traj is not None:
            nr_repairable += 1
            writer.writerow(
                [
                    scenario.scenario_id,
                    ego_id,
                    rule,
                    "bingo",
                    repairer.model,
                    repairer.tv,
                    repairer.tc,
                ]
            )
        else:
            nr_not_repairable += 1
            writer.writerow(
                [
                    scenario.scenario_id,
                    ego_id,
                    rule,
                    "not repairable",
                    repairer.model,
                    repairer.tv,
                    repairer.tc,
                ]
            )

    nr_total = nr_not_repairable + nr_repairable + nr_infeasible
    writer.writerow(
        [
            "Nr of scenario",
            str(nr_total),
            "Nr of infeasible trajectory",
            str(nr_infeasible),
            str(round(nr_infeasible / nr_total, 4)),
            "Nr of reparable trajectory",
            str(nr_repairable),
            str(round(nr_repairable / (nr_repairable + nr_not_repairable), 4)),
        ]
    )
    print("evaluation finished")
    f_w.close()
