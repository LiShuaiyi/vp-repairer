from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import CustomState, ExtendedPMState

from crrepairer.utils.configuration import RepairerConfiguration


def retrieve_ego_vehicle(config: RepairerConfiguration):
    """Retrieves the ego vehicle based on the given time frame."""
    ego_initial = config.scenario.obstacle_by_id(config.repair.ego_id)
    new_state_list = []
    new_occupancy_list = []
    # todo: final time step? +1 ?
    for time_step in range(config.repair.t_0, config.repair.t_f + 1):
        if ego_initial.state_at_time(time_step):
            if isinstance(ego_initial.state_at_time(time_step), ExtendedPMState):
                new_state = CustomState(time_step=ego_initial.state_at_time(time_step).time_step,
                                        position=ego_initial.state_at_time(time_step).position,
                                        velocity=ego_initial.state_at_time(time_step).velocity,
                                        orientation=ego_initial.state_at_time(time_step).orientation,
                                        acceleration=ego_initial.state_at_time(time_step).acceleration)
            else:
                new_state = ego_initial.state_at_time(time_step)
            if time_step != 0:
                # skip the initial state with different type
                new_state_list.append(new_state)
            new_occupancy_list.append(ego_initial.occupancy_at_time(time_step))
        else:
            print(f"ego vehicle does not have state at time step {time_step}")
    ego_initial.prediction.trajectory = Trajectory(
        new_state_list[0].time_step,
        new_state_list
    )
    ego_initial.prediction.occupancy_set = new_occupancy_list
    # not always being equal to tf
    ego_initial.prediction.final_time_step = new_state_list[-1].time_step
    ego_initial.prediction.initial_time_step = config.repair.t_0
    return ego_initial
