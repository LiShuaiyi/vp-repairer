# Compare the SUMO result

import os
import glob
import csv

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world_state import WorldState
from crmonitor.evaluation.evaluation import RuleSetEvaluator

if __name__ == '__main__':

    # the highD-cr scenario directory
    file_path = "/commonroad_repairer/scenarios/interactive/archive/"

    # for s in list(glob.glob(os.path.join(file_path, "*.xml"), recursive=True)):
    s = file_path + 'DEU_LocationELower-24_18_I-1.xml'
    scenario, planning_problem_set = \
            CommonRoadFileReader(s).open(lanelet_assignment=True)
    violating_result = []
    for vehicle in scenario.dynamic_obstacles:
        ego_id = vehicle.obstacle_id
        world = WorldState.create_from_scenario(scenario, ego_obs_id=ego_id)
        evaluator = RuleSetEvaluator.create_from_config(dt=scenario.dt)
        rule_robustness, _, _ = evaluator.evaluate_incremental(world)
        violate = ""
        if not rule_robustness.query('robustness<0')["rule_name"].empty:
            for rule in list(set(rule_robustness.query('robustness<0')["rule_name"])):
                violate += rule
            violating_result.append([str(scenario.scenario_id), ego_id, str(violate)])
        print(violating_result)