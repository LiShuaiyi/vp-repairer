# Compare the SUMO result

import os
import glob
import csv

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState
from crmonitor.evaluation.evaluation import RuleSetEvaluator

if __name__ == '__main__':

    # the highD-cr scenario directory
    file_path = "/home/yuanfei/commonroad/commonroad_repair/scenarios/interactive/"

    for s in list(glob.glob(os.path.join(file_path, "*.xml"), recursive=True)):
        scenario, planning_problem_set = \
            CommonRoadFileReader(s).open(lanelet_assignment=True)
        for vehicle in scenario.dynamic_obstacles:
            ego_id = vehicle.obstacle_id
            world = WorldState.create_from_scenario(scenario, ego_obs_id=ego_id)
            evaluator = RuleSetEvaluator.create_from_config(dt=scenario.dt)
            rule_robustness, _, _ = evaluator.evaluate_incremental(world)
            violate = ""
            if not rule_robustness.query('robustness<0')["rule_name"].empty:
                for rule in list(rule_robustness.query('robustness<0')["rule_name"]):
                    violate += rule
                    print([str(scenario.scenario_id), ego_id, str(violate)])
