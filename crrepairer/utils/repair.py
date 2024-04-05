from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import CustomState, ExtendedPMState

from crrepairer.utils.configuration import RepairerConfiguration


def retrieve_ego_vehicle(config: RepairerConfiguration):
    """Retrieves the ego vehicle based on the given time frame."""
    ego_initial = config.scenario.obstacle_by_id(config.repair.ego_id)
    new_state_list = []
    new_occupancy_list = []
    for time_step in range(config.repair.t_0, config.repair.t_f):
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
    return ego_initial
