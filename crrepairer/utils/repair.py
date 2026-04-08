from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import CustomState, ExtendedPMState, InitialState

from crrepairer.utils.configuration import RepairerConfiguration


def _normalize_ego_state(state):
    if state is None:
        return None

    if isinstance(state, InitialState):
        return state

    acceleration = getattr(state, "acceleration", 0.0)
    orientation = getattr(state, "orientation", 0.0)
    return CustomState(
        time_step=state.time_step,
        position=state.position,
        velocity=state.velocity,
        orientation=orientation,
        acceleration=acceleration,
    )


def retrieve_ego_vehicle(config: RepairerConfiguration):
    """Retrieves the ego vehicle based on the given time frame."""
    ego_initial = config.scenario.obstacle_by_id(config.repair.ego_id)
    new_state_list = []
    for time_step in range(config.repair.t_0, config.repair.t_f):
        state = ego_initial.state_at_time(time_step)
        if state:
            if isinstance(state, ExtendedPMState):
                new_state = CustomState(
                    time_step=state.time_step,
                    position=state.position,
                    velocity=state.velocity,
                    orientation=getattr(state, "orientation", 0.0),
                    acceleration=getattr(state, "acceleration", 0.0),
                )
            else:
                new_state = _normalize_ego_state(state)
            if not isinstance(new_state, InitialState): 
                # skip the initial state with different type
                new_state_list.append(new_state)
        else:
            print(f"ego vehicle does not have state at time step {time_step}")
    ego_initial.prediction.trajectory = Trajectory(
        new_state_list[0].time_step,
        new_state_list
    )
    return ego_initial
