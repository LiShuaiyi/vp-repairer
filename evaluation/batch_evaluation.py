from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver
from commonroad_repair.crrepairer.t_solver.qp_planner_repair import QPPlannerRepair
from commonroad_repair.crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from commonroad_repair.crrepairer.t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_profile

from commonroad.scenario.obstacle import ObstacleType
from commonroad.common.file_reader import CommonRoadFileReader

# other packages
import csv
import os
import math
import glob

file_path = "../../highD-dataset/highD-cr-scenarios/"

if __name__ == '__main__':
    nr_infeasible = 0
    nr_repairable = 0
    nr_not_repairable = 0
    f_w = open("result_rg1+.csv", 'r+')
    f_w = open("result_rg1_not_rep.csv", 'r+')
    writer = csv.writer(f_w)
    writer.writerow(["scenario_id", "ego_id", 'rule', "repairability", "model", "TV", "TC"])
    for csv_file in list(glob.glob("config/*.csv", recursive=True)):
        if csv_file.split("/")[-1] != "violation_not_repair_R_G1.csv":
            continue
        f_r = open(csv_file, 'r+')
        reader = csv.reader(f_r)
        rule = "R_G" + csv_file[-5]
        for row in reader:
            scenario_id = list(row)[0]
            try:
                scenario, planning_problem_set = CommonRoadFileReader(file_path + scenario_id + ".xml"). \
                    open(lanelet_assignment=True)
            except:
                continue
            planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
            ego_id = int(list(row)[1])
            print(rule, scenario_id, ego_id)
            ego_initial = scenario.obstacle_by_id(ego_id)
            if ego_initial.obstacle_type != ObstacleType.CAR:
                continue
            try:
                rule_abstracter = RuleAbstracter(scenario,
                                                 planning_problem,
                                                 ego_id, rule)
                if rule_abstracter.rule_monitor.tv_time_step in (-math.inf, math.inf):
                    writer.writerow([scenario.scenario_id, ego_id, rule, "initial feasibility"])
                    nr_infeasible += 1
                    continue
                repairer = SMTTrajectoryRepairer(rule_abstracter,
                                             ego_initial)

                repaired_traj = repairer.repair()
            except:
                repaired_traj = None
                continue
            if repaired_traj is not None:
                nr_repairable += 1
                writer.writerow([scenario.scenario_id, ego_id, rule, "bingo", repairer.model, repairer.tv, repairer.tc])
            else:
                nr_not_repairable += 1
                writer.writerow(
                    [scenario.scenario_id, ego_id, rule, "not repairable", repairer.model, repairer.tv, repairer.tc])
        f_r.close()

    nr_total = nr_not_repairable + nr_repairable + nr_infeasible
    writer.writerow(["Nr of scenario", str(nr_total),
                     "Nr of infeasible trajectory", str(nr_infeasible), str(round(nr_infeasible / nr_total, 4)),
                     "Nr of reparable trajectory", str(nr_repairable), str(round(nr_repairable / (nr_repairable +
                                                                             nr_not_repairable), 4))])
    print('evaluation finished')
    f_w.close()