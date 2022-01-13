from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver
from commonroad_repair.crrepairer.t_solver.qp_planner import QPPlannerRepair
from commonroad_repair.crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from commonroad_repair.crrepairer.t_solver.utils import convert_traj_to_ego_vehicle
from commonroad_repair.crrepairer.repairer.visualization import visualize_repairing_result, visualize_profile
from commonroad_repair.crrepairer.t_solver.utils import calculate_safe_distance

from commonroad.common.file_reader import CommonRoadFileReader

# other packages
import csv
import os
import math

rule = "R_G1"
file_path = "/home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/"

if __name__ == '__main__':
    f_r = open(os.path.dirname(__file__) + "/highD_rule_evaluation_full.csv", 'r+')
    f_w = open(os.path.dirname(__file__) + "/highD_evaluation_result.csv", 'r+')
    reader = csv.reader(f_r)
    writer = csv.writer(f_w)
    writer.writerow(["scenario_id", "ego_id","repairability", "model", "TV", "TC"])
    nr_infeasible = 0
    nr_repairable = 0
    nr_not_repairable = 0
    for row in reader:
        if list(row)[2] == 'R_G1':
            scenario_id = list(row)[0]
            scenario, planning_problem_set = CommonRoadFileReader(file_path + scenario_id + ".xml").\
                open(lanelet_assignment=True)
            planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
            ego_id = int(list(row)[1])
            print(scenario_id, ego_id)
            ego_initial = scenario.obstacle_by_id(ego_id)
            rule_abstracter = RuleAbstracter(scenario,
                                             planning_problem,
                                             ego_id, rule)
            if rule_abstracter.rule_monitor.tv_time_step == -math.inf:
                writer.writerow([scenario.scenario_id, ego_id, "initial feasibility"])
                nr_infeasible += 1
                continue
            repairer = SMTTrajectoryRepairer(rule_abstracter,
                                             ego_initial)
            repaired_traj = repairer.repair()
            if repaired_traj is not None:
                nr_repairable += 1
                writer.writerow([scenario.scenario_id, ego_id, "bingo", repairer.model, repairer.tv, repairer.tc])
            else:
                nr_not_repairable +=1
                writer.writerow([scenario.scenario_id, ego_id, "not repairable", repairer.model, repairer.tv, repairer.tc])
    nr_total = nr_not_repairable+nr_repairable+nr_infeasible
    writer.writerow(["Nr of scenario", str(nr_total),
                     "Nr of infeasible trajectory", str(nr_infeasible), str(nr_infeasible/nr_total*100)+"%",
                     "Nr of reparable trajectory", str(nr_repairable), str(nr_repairable/(nr_repairable+nr_not_repairable))*100+"%"])
    f_r.close()
    f_w.close()