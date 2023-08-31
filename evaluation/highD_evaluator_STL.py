"""
Evaluate HighD scenarios using commonroad monitor:

the cr-highD scenarios are converted by the following commands:
crconvert highd /home/yuanfei/commonroad/highD-dataset/ /home/yuanfei/commonroad/highD-dataset/highD-cr-scenarios/
 --num-time-steps 100 --num-processes 4 --downsample 5 --keep-ego --obstacle-start-at-zero
(time step is 0.2s)

use the STL monitor
"""
import os
import glob
import csv

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState
from crmonitor.evaluation.evaluation import RuleSetEvaluator

if __name__ == "__main__":

    # the highD-cr scenario directory
    file_path = "../../highD-dataset/highD-cr-scenarios/"
    # file_path = "../../commonroad-scenarios-master-scenarios/scenarios/cooperative"
    # highD_scenario_dir = "/home/yuanfei/commonroad/highD-dataset/sebastian_evaluation/"
    f_w = open("highD_evaluation_result.csv", "r+")
    writer = csv.writer(f_w)
    # _ = f_r.readlines().pop(0)  # pop first line
    writer.writerow(["scenario_id", "ego_id", "rule_STL"])
    for s in list(glob.glob(os.path.join(file_path, "*.xml"), recursive=True)):
        scenario, planning_problem_set = CommonRoadFileReader(s).open(
            lanelet_assignment=True
        )
        for veh in scenario.dynamic_obstacles:
            ego_id = veh.obstacle_id
            print(scenario.scenario_id, ego_id)
            try:
                world = WorldState.create_from_scenario(
                    scenario,
                    ego_obs_id=ego_id,
                )
                evaluator = RuleSetEvaluator.create_from_config(dt=scenario.dt)
                evaluator.switch_to_boolean()  # faster: boolean
                rule_robustness, _, _ = evaluator.evaluate_incremental(world)
                if not rule_robustness.query("robustness<0")["rule_name"].empty:
                    violate = []
                    for rule in list(
                        set(rule_robustness.query("robustness<0")["rule_name"])
                    ):
                        violate.append(rule)
                    row = [scenario.scenario_id, ego_id]
                    for v in violate:
                        row.append(str(v))
                    writer.writerow(row)
            except:
                continue
    f_w.close()
