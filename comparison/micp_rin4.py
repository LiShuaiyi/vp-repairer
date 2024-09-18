import numpy as np
import matplotlib.pyplot as plt

from micp.traffic_rule_4d import RIN4

from stlpy.solvers import *

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.state import CustomState
from crmonitor.common.world import World
from crmonitor.common.helper import load_yaml

from crrepairer.utils.visualization import TUMColor
from micp.constraints import InSameLaneConstraint, InFrontOfConstraint, KeepsSafeDistanceConstraint
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/DEU_AachenBendplatz-1_162280_T-2299.xml"

# Open the scenario
crscenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
rule_config_path = (
    "/home/liny/repairverse/commonroad-stl-monitor/crmonitor/config.yaml"
)
config = load_yaml(rule_config_path)

config["scenario"] = "intersection"
config["intersection_road_network_param"]["map_type"] = "dataset"
world = World.create_from_scenario(crscenario, config)

T = 20
ego_id = 10179
other_id = 10180

ego_vehicle = world.vehicle_by_id(ego_id)
other_vehicle = world.vehicle_by_id(other_id)


# Define the system and specification
scenario = RIN4(T=T,
                 world=world,
                 ego_vehicle=ego_vehicle,
                 other_vehicle=other_vehicle,
                 lanelet_network=world.road_network.lanelet_network)

spec = scenario.GetSpecification()
sys = scenario.GetSystem()
Q = 1e-1 * np.diag([1, 0, 2, 1, 1, 1, 1, 1])
R = 100 * np.eye(2)

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

solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
# solver = ScipyGradientSolver(spec, sys, x0, T, verbose=True)

u_min = np.array([-2000, -2000])
u_max = np.array([2000, 2000,])

solver.AddQuadraticCost(Q, R)
solver.AddControlBounds(u_min, u_max)
x, u, _, _ = solver.Solve()

print(f"Time used: {time.time() - time_start:.2f}s")
print(f"Optimal robustness: {solver.rho.X[0]}")
traj_cr = list()
print()
# transform every trajectory point
for i in range(T + 1):
    clcs_pos = [x[0, i], x[1, i]]
    cart_pos = ego_vehicle.ref_path_lane.clcs.convert_to_cartesian_coords(clcs_pos[0], clcs_pos[1])
    state_values = {
        'position': np.array([cart_pos[0], cart_pos[1]]),
        'velocity': np.sqrt(x[2, i] **2 + x[3, i] **2),
        'acceleration': np.sqrt(x[4, i] **2 + x[5, i] **2),
        'time_step': i
        }
    state = CustomState(**state_values)
    traj_cr.append(state)


# plot velocity and acc
plt.figure(figsize=(6, 2.4))
plt.plot([state.velocity for state in traj_cr], linewidth=3, marker='D',
        markersize=4, color=TUMColor.TUMyellow.value)
plt.xticks(range(0, 20, 10))
plt.xlim(0, 20)

plt.ylim(19.363476994916265, 45.246605857384935)
plt.plot([state.acceleration for state in traj_cr])
plt.legend(['velocity', 'acceleration'])
plt.show()
# Plot the results
plot_limits = [158, 390, -32, -18.4]
from commonroad.visualization.mp_renderer import MPRenderer

def plot_scenario(crscenario, traj_cr, plot_limits, time_step):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    rnd = MPRenderer(ax=ax, plot_limits=plot_limits)

    # visualize scenario
    rnd.draw_params.time_begin = time_step
    rnd.draw_params.trajectory.draw_trajectory = False
    rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
    rnd.draw_params.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = (
        False
    )
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMgray.value
    )
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = (
        TUMColor.TUMblack.value
    )
    rnd.draw_params.dynamic_obstacle.draw_shape = True
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
    rnd.draw_params.dynamic_obstacle.trajectory.line_width = 0.3
    rnd.draw_params.dynamic_obstacle.draw_signals = False
    rnd.draw_params.dynamic_obstacle.draw_icon = True
    # rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    # rnd.draw_params.traffic_sign.draw_traffic_signs = True
    rnd.draw_params.lanelet_network.lanelet.stop_line_color = (
        TUMColor.TUMblack.value
    )
    rnd.draw_params.lanelet_network.lanelet.draw_stop_line = True
    crscenario.draw(rnd)

    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.draw_shape = True
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMblue.value
    )

    # render scenario and ego vehicle
    rnd.render()
    pos_x_replanned = []
    pos_y_replanned = []
    for state in traj_cr:
        pos_x_replanned.append(state.position[0])
        pos_y_replanned.append(state.position[1])

    rnd.ax.plot(
        pos_x_replanned[time_step:],
        pos_y_replanned[time_step:],
        color=TUMColor.TUMyellow.value,
        marker='D',
        markersize=4,
        zorder=22,
        linewidth=3,
        label="replanned trajectory",
    )
    plt.show()


for i in range(T + 1):
    plot_scenario(crscenario, traj_cr, plot_limits, i)
