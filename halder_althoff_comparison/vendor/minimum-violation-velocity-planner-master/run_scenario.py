from commonroad.common.file_reader import CommonRoadFileReader

from minimum_violation_velocity_planner.common.utils import loadParameterFile, readRulebookFile
from minimum_violation_velocity_planner.run_velocity_planner import runVelocityPlanner


def main():
    # Select Scenario
    scenarios = {0: "DEU_Flensburg-6_2_T-1",
                 1: "ZAM_Zip-2_1_T-1",
                 2: "USA_Peach-5_1_T-1"}
    selected_scenario = 0

    # Load scenario and planning problem
    scenario_location = "examples/scenarios/" + scenarios[selected_scenario] + ".xml"
    scenario, planning_problem_set = CommonRoadFileReader(scenario_location).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Load rulebook
    rulebook = readRulebookFile("examples/rulebooks/example_rulebook.txt")

    # Load scenario parameters
    scenario_params = loadParameterFile("examples/scenarios/" + str(scenario.scenario_id) + ".yaml")

    # Select planner type
    planners = {0: 'BFS',
                1: 'DIJKSTRA',
                2: 'A_STAR',
                3: 'A_STAR_MIN_RULE_EVAL'}
    use_planner = 2

    # Execute the minimum violation velocity planner
    runVelocityPlanner(scenario, planning_problem,
                       total_order=rulebook,
                       scenario_params=scenario_params,
                       planner=planners[use_planner],
                       check_feasibility=False,
                       show_rule_evaluation_times=False,
                       make_visualizations=False,
                       verbose=True)


if __name__ == "__main__":
    main()
