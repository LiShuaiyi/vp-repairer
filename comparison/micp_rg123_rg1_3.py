import numpy as np
import matplotlib.pyplot as plt

from micp.traffic_rule_4d import RG123

from stlpy.solvers import *

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from micp.constraints import InSameLaneConstraint, InFrontOfConstraint, KeepsSafeDistanceConstraint
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/DEU_LocationDLower-8_154_T-1.xml"

# Open the scenario
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

world = World.create_from_scenario(scenario)

T = 20
ego_id = 11
other_id = 9

ego_vehicle = world.vehicle_by_id(ego_id)
other_vehicle = world.vehicle_by_id(other_id)


# Define the system and specification
scenario = RG123(T=T,
                 ego_vehicle=ego_vehicle,
                 other_vehicle=other_vehicle,
                 lanelet_network=world.road_network.lanelet_network)

spec = scenario.GetSpecification()
sys = scenario.GetSystem()
Q = 1e-1 * np.diag([0, 0, 10, 10, 0, 0, 1, 1])   # just penalize high velocities
R = 10 * np.eye(2)

initial_state_lon = ego_vehicle.get_lon_state(0, ego_vehicle.get_lane(0))
initial_state_lat = ego_vehicle.get_lat_state(0, ego_vehicle.get_lane(0))
x0 = np.array(
    [
        initial_state_lon.s,
        initial_state_lat.d,
        initial_state_lon.v,
        0,
        0,
        0,
        0,
        0,
    ]
)

import time
solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
# solver = ScipyGradientSolver(spec, sys, x0, T, verbose=True)

u_min = np.array([-10, -10])
u_max = np.array([12, 12,])

solver.AddQuadraticCost(Q, R)
solver.AddControlBounds(u_min, u_max)
time_start = time.time()
x, u, _, _ = solver.Solve()
print(x)
print(u)
print("", time.time() - time_start)
if x is not None:
    # Plot the solution

    ax = plt.gca()
    scenario.add_to_plot(ax)

    plt.scatter(*x[:2,:])
    plt.show()
