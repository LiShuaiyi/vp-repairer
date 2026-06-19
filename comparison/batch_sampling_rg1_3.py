import numpy as np
import csv
import time
import logging

# cr monitor
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.evaluation import (
    get_evaluation_config,
    create_ego_vehicle_param,
)
from crmonitor.common.world import World, get_world_config

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.general import update_goal_state

from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.logger import initialize_logger
from commonroad_route_planner.route_planner import RoutePlanner

logger = logging.getLogger("RP_LOGGER")
rp_config = ReactivePlannerConfiguration()
initialize_logger(rp_config)

file_path = "/home/liny/Documents/commonroad/highd_scenarios_2024_repaired/"


# Read the scenario violation information from a CSV file
with open("./../evaluation/highD_evaluation_rg1_3_repair.csv", "r") as f_r:
    reader = csv.reader(f_r)
    header = next(reader)  # Read header if needed
    result_inD = [row for row in reader]  # Read all rows from the file

with open("highD_evaluation_rg1_3_sampling.csv", "w", newline="") as f_w:
    writer = csv.writer(f_w)

    # Write headers to the result file
    headers = ["scenario_id", "ego_id", "rule", "replanability", "total_time"]
    writer.writerow(headers)
    writer.writerow(headers)

    for result in result_inD:
        scenario_id = result[0]
        ego_id = int(result[1])

        print(">>", scenario_id, ego_id)

        scenario_id_with_extension = scenario_id.replace('T-1', 'T-' + str(ego_id))

        scenario_path = file_path + scenario_id_with_extension + ".xml"

        # build the config for sampling-based planner
        rp_config = ReactivePlannerConfiguration()
        rp_config.general.path_scenarios = file_path
        rp_config.general.set_path_scenario(scenario_id_with_extension + ".xml")
        rp_config.update()
        rp_config.planning.ego_id = ego_id
        rp_config.planning.rules = ["R_G1", "R_G3"]
        # set up the stl monitor world
        world_config = get_world_config()
        world_config["scenario"] = "interstate"
        world = World.create_from_scenario(rp_config.scenario, config=world_config)

        # set up the reactive planner
        rp_config.planning.dt = rp_config.scenario.dt

        if monitor_ego := world.vehicle_by_id(rp_config.planning.ego_id):
            monitor_ego.vehicle_param = create_ego_vehicle_param(
                get_evaluation_config().get("ego_vehicle_param"), world.dt
            )
            ego_initial = rp_config.scenario.obstacle_by_id(
                rp_config.planning.ego_id
            )
            rp_config.scenario.remove_obstacle(ego_initial)
            rp_config.planning_problem.initial_state = ego_initial.initial_state
            rp_config.planning_problem.goal = update_goal_state(
                ego_initial.prediction.trajectory
            )
            rp_config.vehicle.length = monitor_ego.shape.length
            rp_config.vehicle.width = monitor_ego.shape.width

            rp_config.planning.time_steps_computation = ego_initial.prediction.final_time_step - ego_initial.prediction.initial_time_step + 1
        else:
            writer.writerow([scenario_id, ego_id, rule, f"ego vehicle not found in the scenario", "N/A"])

        rule_evaluators = []
        for rule in rp_config.planning.rules:
            rule_evaluators.append(RuleEvaluator.create_from_config(world,
                                                                    rp_config.planning.ego_id,
                                                                    rule=rule,
                                                                    use_boolean=True))

        # initialize reactive planner
        planner = ReactivePlanner(rp_config)

        planner.rule_evaluators = rule_evaluators
        planner.world = world

        # run route planner and add reference path to config
        route_planner = RoutePlanner(rp_config.scenario.lanelet_network, rp_config.planning_problem)
        route = route_planner.plan_routes().retrieve_first_route()

        # set reference path for curvilinear coordinate system
        planner.set_reference_path(route.reference_path)

        planner.record_state_and_input(planner.x_0)

        planner.set_desired_velocity(current_speed=planner.x_0.velocity)

        time_start = time.time()
        try:
            optimal = planner.plan()
        except Exception as e:
            print(f"Error: {e}")
            writer.writerow([scenario_id, ego_id, rule, f"error/failed with {e}", "N/A"])
            continue
        # i = 1
        # while optimal is None and i <= planner.sampling_level:
        #     optimal = planner.plan(i)
        runtime = time.time() - time_start
        if not optimal:
            writer.writerow([scenario_id, ego_id, rule, "no", runtime])
        else:
            writer.writerow([scenario_id, ego_id, rule, "yes", runtime])

    print("Evaluation finished")

