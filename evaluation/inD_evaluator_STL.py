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
crconvert  --num-time-steps 100 --num-processes 4 --downsample 5 --keep-ego --obstacles-start-at-zero /home/liny/Documents/commonroad/13_inD/ /home/liny/Documents/commonroad/inD-repair/ ind

DEU_AachenBendplatz-1_151360_T-1379 10097
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.
  1.  1.]

  DEU_AachenBendplatz-1_152460_T-2479 10161
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.
  1.  1.]

DEU_AachenBendplatz-1_162840_T-2859 10217
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1.
  1.  1.]
DEU_AachenBendplatz-1_161140_T-1159 10098
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1.  1.  1.  1.
  1.  1.]
violation R_IN1

DEU_AachenBendplatz-1_164900_T-4919 10371
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1.  1.  1.
  1.  1.]
violation R_IN1

use the STL monitor
"""
import os
import glob
import csv
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.common.world import World, get_world_config

if __name__ == "__main__":
    # the highD-cr scenario directory
    # file_path = "../../highD-dataset/highD-cr-scenarios/"
    file_path = "/home/liny/Documents/commonroad/inD-repair/"

    # file_path = "../../commonroad-scenarios-master-scenarios/scenarios/cooperative"
    # highD_scenario_dir = "/home/yuanfei/commonroad/highD-dataset/sebastian_evaluation/"
    filename = "inD_evaluation_result.csv"
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('')  # Create the file and write an empty string if needed
    f_w = open(filename, "r+")
    writer = csv.writer(f_w)
    # _ = f_r.readlines().pop(0)  # pop first line
    writer.writerow(["scenario_id", "ego_id", "rule_STL"])

    rules = ["R_IN1"]
    for s in list(glob.glob(os.path.join(file_path, "*.xml"), recursive=True))[0:500]:

        scenario, planning_problem_set = CommonRoadFileReader(s).open(
            lanelet_assignment=True
        )
        world_config = get_world_config()
        world_config["scenario"] = "intersection"
        try:
            world = World.create_from_scenario(
                scenario, config=world_config
            )
        except:
            continue
        for veh in world.vehicles:
            ego_id = veh.id
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
            if violation:
                writer.writerow(row)
    f_w.close()
