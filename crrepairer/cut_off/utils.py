from crmonitor.common.world_state import WorldState
from crmonitor.common.helper import (_compute_jerk,
                                     _compute_acceleration,
                                     update_curvilinear_states_long,
                                     create_curvilinear_states
                                     )
from typing import List
from commonroad.scenario.obstacle import State

def update_ego_vehicle(world_state: WorldState,
                       updated_ego_states: List[State],
                       cut_off_time: int = 1):
    """
    Update the ego vehicle based on the new given trajectory
    """
    ego_vehicle = world_state.ego_vehicle
    ego_initial_state = ego_vehicle.initial_state
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