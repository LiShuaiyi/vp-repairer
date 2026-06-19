from commonroad_crime.utility.simulation import SimulationLat, Maneuver, SimulationLong
import matplotlib.pyplot as plt

# import functions to read xml file and visualize commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
# import necessary classes from different modules
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from crrepairer.utils.visualization import TUMColor

veh_id = 10031
scenario_id = "DEU_AachenBendplatz-1_16220_T-239"
scenario, planning_problem_set = CommonRoadFileReader(scenario_id + ".xml").open()

config = CriMeConfiguration()
config.general.path_scenarios = "./"
config.general.set_scenario_name(scenario_id)

config.update()


def add_obs(state_list, sim_vehicle):
    state_list_updated = []
    for state in state_list:
        state_list_updated.append(
            CustomState(position=state.position,
                        velocity=state.velocity,
                        velocity_y = state.velocity_y,
                        orientation=state.orientation,
                        time_step=state.time_step)
        )
    dynamic_obstacle_trajectory = Trajectory(1, state_list_updated[1:])
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, sim_vehicle.obstacle_shape)
    dynamic_obstacle = DynamicObstacle(scenario.generate_object_id(),
                                       sim_vehicle.obstacle_type,
                                       sim_vehicle.obstacle_shape,
                                       sim_vehicle.initial_state,
                                       dynamic_obstacle_prediction)
    scenario.add_objects(dynamic_obstacle)

sim_vehicle = scenario.obstacle_by_id(veh_id)

for obs in scenario.dynamic_obstacles:
    scenario.remove_obstacle(obs)


# maneuver = Maneuver.TURNRIGHT
# cut_off_time = 9
# simulator = SimulationLat(maneuver,
#                           sim_vehicle,
#                           config)
# state_list = simulator.simulate_state_list(cut_off_time, 20)
# add_obs(state_list, sim_vehicle)

# maneuver = Maneuver.STEERLEFT
# simulator = SimulationLat(maneuver,
#                           sim_vehicle,
#                           config)
# config.time.steer_width = 1
# cut_off_time = 3
# state_list = simulator.simulate_state_list(cut_off_time, 20)
# add_obs(state_list, sim_vehicle)
#
#
cut_off_time = 15

maneuver = Maneuver.BRAKE
simulator = SimulationLong(maneuver,
                          sim_vehicle,
                          config)
state_list = simulator.simulate_state_list(cut_off_time, 20)
add_obs(state_list, sim_vehicle)
#
# maneuver = Maneuver.KICKDOWN
# simulator = SimulationLong(maneuver,
#                           sim_vehicle,
#                           config)
# state_list = simulator.simulate_state_list(10, 20)
# add_obs(state_list, sim_vehicle)



rnd = MPRenderer(plot_limits=[25, 65, -40, -12])
rnd.draw_params.time_begin = cut_off_time
rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = TUMColor.TUMgray.value
rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = TUMColor.TUMblack.value

rnd.draw_params.dynamic_obstacle.draw_icon = True
rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = True
scenario.draw(rnd)
rnd.render()
plt.show()