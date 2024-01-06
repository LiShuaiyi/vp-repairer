from commonroad.prediction.prediction import Trajectory

from crrepairer.utils.configuration import RepairerConfiguration


def retrieve_ego_vehicle(config: RepairerConfiguration):
    """Retrieves the ego vehicle based on the given time frame."""
    ego_initial = config.scenario.obstacle_by_id(config.repair.ego_id)
    ego_initial.prediction.trajectory = Trajectory(
        ego_initial.prediction.trajectory.state_list[config.repair.t_0].time_step,
        ego_initial.prediction.trajectory.state_list[
            config.repair.t_0: config.repair.t_f
        ],
    )
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[
        config.repair.t_0: config.repair.t_f
    ]
    ego_initial.prediction.final_time_step = config.repair.t_f - 1
    ego_initial.prediction.initial_time_step = config.repair.t_0
    return ego_initial
