import numpy as np
import matplotlib.pyplot as plt
import csv
from crmonitor.common.helper import load_yaml

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from micp.traffic_rule_4d import RIN4, RIN1

from stlpy.solvers import *

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.state import CustomState
from crmonitor.common.world import World

file_path = "/home/liny/Documents/commonroad/ind_scenarios_2024_repaired/"

# Read the scenario violation information from a CSV file
with open("./../evaluation/inD_evaluation_rin1_4_filtered.csv", "r") as f_r:
    reader = csv.reader(f_r)
    header = next(reader)  # Read header if needed
    result_inD = [row for row in reader]  # Read all rows from the file

with open("inD_evaluation_rin1_4.csv", "w", newline="") as f_w:
    writer = csv.writer(f_w)

    # Write headers to the result file
    headers = ["scenario_id", "ego_id", "rule", "replanability", "total_time"]
    writer.writerow(headers)
    writer.writerow(headers)

    for result in result_inD:
        scenario_id = result[0]
        ego_id = int(result[1])
        rule = result[2]

        print(">>", rule, scenario_id, ego_id)

        scenario_id_with_extension = scenario_id.replace('-1_', '-' + str(ego_id) + '_')

        scenario_path = file_path + scenario_id_with_extension + ".xml"

        # Open the scenario
        crscenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
        rule_config_path = (
            "/home/liny/repairverse/commonroad-stl-monitor/crmonitor/config.yaml"
        )
        config = load_yaml(rule_config_path)

        config["scenario"] = "intersection"
        config["intersection_road_network_param"]["map_type"] = "dataset"
        world = World.create_from_scenario(crscenario, config)

        T = crscenario.obstacle_by_id(ego_id).prediction.trajectory.final_state.time_step
        print(f"Time horizon: {T}")
        ego_vehicle = world.vehicle_by_id(ego_id)

        if rule == "R_IN4":
            repair_config = RepairerConfiguration()
            repair_config.general.path_scenarios = file_path

            repair_config.general.set_path_scenario(scenario_id_with_extension)
            repair_config.update()
            repair_config.repair.scenario_type = "intersection"
            repair_config.repair.intersection_type = "dataset"
            repair_config.repair.rules = ["R_IN4"]
            repair_config.repair.ego_id = ego_id
            repair_config.scenario = crscenario
            traffic_rule_monitor = STLRuleMonitor(repair_config)

            other_id = traffic_rule_monitor.other_id

            other_vehicle = world.vehicle_by_id(other_id)

            try:
                # Define the system and specification
                scenario = RIN4(T=T,
                                world=world,
                                ego_vehicle=ego_vehicle,
                                other_vehicle=other_vehicle,
                                lanelet_network=world.road_network.lanelet_network)
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([scenario_id, ego_id, rule, f"initialization fails with {e}", "N/A"])
                continue
        else:
            scenario = RIN1(T=T,
                            world=world,
                            ego_vehicle=ego_vehicle,
                            lanelet_network=world.road_network.lanelet_network)
        try:
            spec = scenario.GetSpecification()
        except Exception as e:
            print(f"Error: {e}")
            writer.writerow([scenario_id, ego_id, rule, f"initialization fails with {e}", "N/A"])
            continue
        sys = scenario.GetSystem()
        Q = np.diag([0.1, 0.1, 0.5, 1, 0.1, 0.1, 0.5, 1])
        R = 1 * np.eye(2)

        initial_state_lon = ego_vehicle.get_lon_state(0, ego_vehicle.ref_path_lane)
        initial_state_lat = ego_vehicle.get_lat_state(0, ego_vehicle.ref_path_lane)
        x0 = np.array(
            [
                initial_state_lon.s,
                initial_state_lat.d,
                initial_state_lon.v,
                0,
                initial_state_lon.a,
                0,
                0,
                0,
            ]
        )

        import time

        time_start = time.time()

        try:
            solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
        except Exception as e:
            print(f"Error: {e}")
            writer.writerow([scenario_id, ego_id, rule, f"initialization fails with {e}", "N/A"])
            continue
        # solver = ScipyGradientSolver(spec, sys, x0, T, verbose=True)

        u_min = np.array([-2000, -2000])
        u_max = np.array([2000, 2000, ])

        solver.AddQuadraticCost(Q, R)
        solver.AddControlBounds(u_min, u_max)
        try:
            x, u, _, _ = solver.Solve()

            runtime = time.time() - time_start
            print(f"Time used: {runtime:.2f}s")
            print(f"Optimal robustness: {solver.rho.X[0]}")
            writer.writerow([scenario_id, ego_id, rule, "yes", runtime])

        except Exception as e:
            runtime = time.time() - time_start
            print(f"Solver failed to find a solution: {str(e)}")
            writer.writerow([scenario_id, ego_id, rule, "no", runtime])

        print("Evaluation finished")

