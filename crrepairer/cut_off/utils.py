from crmonitor.common.world_state import WorldState
from crmonitor.common.helper import (_compute_jerk,
                                     _compute_acceleration,
                                     update_curvilinear_states_long,
                                     create_curvilinear_states
                                     )
from typing import List, Union
from vehiclemodels.parameters_vehicle1 import VehicleParameters
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
from commonroad.common.solution import VehicleModel, VehicleType
import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer


def visualize_state_list(state_list: Union[State], scenario, parameters):
    rnd = MPRenderer()
    # scenario.draw(rnd)
    scenario.draw(rnd, draw_params={'time_begin': 0, 'scenario':{'dynamic_obstacle':{'show_label': True}}})
    trajectory = transfer_state_list_to_trajectory(scenario, state_list, parameters)
    trajectory.draw(rnd, draw_params={'time_begin': 0, 'trajectory': {'draw_trajectory': True}})
    rnd.render()
    plt.show()

def check_velocity_feasibility(state: State, parameters: VehicleParameters):
    if state.velocity < 0 or \
            state.velocity > parameters.longitudinal.v_max:
        return False
    return True


def check_steering_angle_feasibility(state: State, parameters: VehicleParameters):
    # if not hasattr(state, "steering_angle")
    if state.steering_angle < parameters.steering.min or \
            state.steering_angle > parameters.steering.max:
        return False
    return True


def transfer_state_list_to_trajectory(scenario, state_list, parameters):
    """
    Transfers given state list into a dummy vehicle.
    :param scenario: given scenario
    :param state_list: given state list
    :return:
    """
    for k in range(len(state_list) - 1):
        if not hasattr(state_list[k], "yaw_rate"):
            state_list[k].yaw_rate = (state_list[k + 1].orientation - state_list[k].orientation) / scenario.dt
        if not hasattr(state_list[k], "slip_angle"):
            state_list[k].slip_angle = 0
        if not hasattr(state_list[k], "steering_angle"):
            state_list[k].steering_angle = 0
        if not hasattr(state_list[k], "acceleration"):
            state_list[k].acceleration = (state_list[k + 1].velocity - state_list[k].velocity) / scenario.dt
        if not hasattr(state_list[k], "velocity_y"):
            state_list[k].velocity_y = 0
    state_list[-1].yaw_rate = 0
    state_list[-1].slip_angle = 0
    state_list[-1].steering_angle = 0
    state_list[-1].acceleration = 0
    dynamic_obstacle_trajectory = Trajectory(state_list[0].time_step, state_list)
    dynamic_obstacle_shape = Rectangle(width=parameters.w, length=parameters.l)
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = ObstacleType.CAR
    dynamic_obstacle_new = DynamicObstacle(dynamic_obstacle_id,
                                           dynamic_obstacle_type,
                                           dynamic_obstacle_shape,
                                           state_list[0],
                                           dynamic_obstacle_prediction)
    return dynamic_obstacle_new


def update_ego_vehicle(world_state: WorldState,
                       updated_ego_states: List[State],
                       cut_off_time: int = 1):
    """
    Update the ego vehicle based on the new given trajectory
    """
    ego_vehicle = world_state.ego_vehicle
    ego_initial_state = ego_vehicle.states_cr[0]
    lane_network = world_state.road_network
    dt = world_state.dt
    if cut_off_time == 0:
        acceleration = 0.0
        jerk = 0.0
    else:
        cut_off_state = updated_ego_states[cut_off_time - 1]
        if cut_off_time == 1:
            pre_cut_off_state = ego_initial_state
        else:
            pre_cut_off_state = updated_ego_states[cut_off_time - 2]
        acceleration = _compute_acceleration(pre_cut_off_state.velocity,
                                             cut_off_state.velocity, dt, )
        if not hasattr(pre_cut_off_state, "acceleration"):
            pre_cut_off_state.acceleration = 0
        jerk = _compute_jerk(acceleration, pre_cut_off_state.acceleration, dt)
    # cut-off state changes it's input values, but the states stay unchanged
    ego_vehicle.states_lon[cut_off_time] = update_curvilinear_states_long(ego_vehicle.states_lon[cut_off_time],
                                                                              acceleration, jerk)
    state_lon = ego_vehicle.states_lon[cut_off_time]
    state_lat = ego_vehicle.states_lat[cut_off_time]
    reference_lane = ego_vehicle.lane
    # print(ego_vehicle.lanelet_assignment)

    for state in updated_ego_states[cut_off_time:]:
        acceleration = _compute_acceleration(state_lon.v, state.velocity, dt)
        if state.time_step - 1 in ego_vehicle.states_lon:
            previous_acceleration = ego_vehicle.states_lon[state.time_step - 1].a
        else:  # previous state out of projection domain
            previous_acceleration = 0.0
        jerk = _compute_jerk(acceleration, previous_acceleration,
                dt)
        state_lon, state_lat = create_curvilinear_states(state.position,
                state.velocity, acceleration, jerk, state.orientation,
                reference_lane, )
        if state_lon is None or state_lat is None:
            break
        ego_vehicle.states_lon[state.time_step] = state_lon
        ego_vehicle.states_lat[state.time_step] = state_lat
        ego_vehicle.states_cr[state.time_step] = state
        # ego_vehicle.signal_series[state.time_step] = obstacle.signal_state_at_time_step(
        #     state.time_step) # todo: the signal state?

        ego_shape = ego_vehicle.shape.rotate_translate_local(state.position,
                                                             state.orientation)
        # use the shape lanelet assignment
        ego_vehicle.lanelet_assignment[state.time_step] = \
        set(lane_network.lanelet_network.find_lanelet_by_shape(ego_shape))
        # obstacle.prediction.shape_lanelet_assignment[state.time_step]
        # vehicle_classifications[state.time_step] = vehicle_classification
    if ego_vehicle.end_time > len(updated_ego_states):
        for time_step in range(len(updated_ego_states)+1, ego_vehicle.end_time+1):
            del ego_vehicle.states_lon[time_step]
            del ego_vehicle.states_lat[time_step]
            del ego_vehicle.states_cr[time_step]
            del ego_vehicle.lanelet_assignment[time_step]