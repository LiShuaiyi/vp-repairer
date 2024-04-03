from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import InitialState

import math


if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "ZAM_Zip-1_56_T-1"

    # Build configuration object
    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()

    initial_time_step = 50
    final_time_step = initial_time_step + 20

    # # change the time horizon
    for veh in config.scenario.obstacles:
        updated_initial_state = veh.prediction.trajectory.state_list[initial_time_step - 1]
        for state in veh.prediction.trajectory.state_list:
            state.time_step -= initial_time_step
        veh.initial_state = InitialState(time_step=0,
                                         yaw_rate=0,
                                         slip_angle=0,
                                         position=updated_initial_state.position,
                                         velocity=updated_initial_state.velocity,
                                         orientation=updated_initial_state.orientation)
        veh.prediction.trajectory = Trajectory(
            1, veh.prediction.trajectory.state_list[initial_time_step: final_time_step]
        )

    ego_initial = retrieve_ego_vehicle(config)

    for i in range(ego_initial.prediction.trajectory.final_state.time_step):
        ego_initial.state_at_time(i).acceleration = (
            ego_initial.state_at_time(i + 1).velocity
            - ego_initial.state_at_time(i).velocity
        ) / config.scenario.dt

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )
            if config.debug.show_plots:
                # ============= Visualization =============
                visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
