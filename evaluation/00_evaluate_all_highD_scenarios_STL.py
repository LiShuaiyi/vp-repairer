"""
Evaluate HighD scenarios using commonroad monitor:

the cr-highD scenarios are converted by the following commands:
crconvert highd /home/yuanfei/commonroad/highD-dataset/ /home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/
 --num-time-steps 100 --num-processes 4 --downsample 5 --keep-ego --obstacle-start-at-zero
(time step is 0.2s)

converter: version 2023.2
crconvert  --num-time-steps 20 --num-processes 4 --downsample 5 --keep-ego --obstacles-start-at-zero
/home/liny/Documents/commonroad/highD-dataset-v1.0 /home/liny/Documents/commonroad/highD-repair/ highd

version 2022.1
crconvert highd /home/liny/Documents/commonroad/highD-dataset-v1.0 /home/liny/Documents/commonroad/highD-repair/ --num-time-steps 20 --num-processes 4 --downsample 5 --keep-ego --obstacle-start-at-zero

(time step is 0.2s)

version 2023.2
crconvert  --num-time-steps 20 --num-processes 4 --downsample 5 --keep-ego --obstacles-start-at-zero /home/liny/Documents/commonroad/13_inD/ /home/liny/Documents/commonroad/inD-repair/ ind



use the STL monitor
"""
import os
import glob
import csv
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

if __name__ == "__main__":
    # the highD-cr scenario directory
    # file_path = "../../highD-dataset/highD-cr-scenarios/"
    file_path = "/home/liny/Documents/commonroad/highD-repair/"

    # file_path = "../../commonroad-scenarios-master-scenarios/scenarios/cooperative"
    # highD_scenario_dir = "/home/yuanfei/commonroad/highD-dataset/sebastian_evaluation/"
    filename = "highD_evaluation_result.csv"
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('')  # Create the file and write an empty string if needed
    f_w = open(filename, "r+")
    writer = csv.writer(f_w)
    # _ = f_r.readlines().pop(0)  # pop first line
    writer.writerow(["scenario_id", "ego_id", "rule_STL"])

    rules = ["R_G1", "R_G3"]
    for s in list(glob.glob(os.path.join(file_path, "*.xml"), recursive=True)):

        scenario, planning_problem_set = CommonRoadFileReader(s).open(
            lanelet_assignment=True
        )

        world = World.create_from_scenario(
            scenario
        )
        for veh in scenario.dynamic_obstacles:
            ego_id = veh.obstacle_id
            print(scenario.scenario_id, ego_id)
            row = [scenario.scenario_id, ego_id]
            violation = False
            for rule in rules:
                evaluator = RuleEvaluator.create_from_config(world,
                                                             ego_id,
                                                             rule=rule,
                                                             use_boolean=True)
                rule_robustness = evaluator.evaluate()
                # check whether there is an element of the rule_robustness is smaller than 0
                print(rule_robustness)
                if np.any(rule_robustness < 0) and rule_robustness[0] > 0 :
                    row.append(rule)
                    violation = True
                    print("violation", rule)
                else:
                    violation = False
            if violation:
                writer.writerow(row)
    f_w.close()
