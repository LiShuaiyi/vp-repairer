from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.repairer.visualization import (
    visualize_repairing_result,
    visualize_a_profile,
    visualize_v_profile,
)

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.prediction.prediction import Trajectory
import math

scenario_id = "DEU_test_consider_entering_vehicles_for_lane_change"
file_path = "../../scenarios/test_interstate/" + scenario_id + ".xml"
figure_path = "./figures"

flag_visualization = True

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(
        lanelet_assignment=True
    )
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 1000
    rule = ["R_I5"]
    N = 51
    ego_initial = scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory = Trajectory(
        1, ego_initial.prediction.trajectory.state_list[:N]
    )
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]
    import matplotlib.pyplot as plt
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.visualization.mp_renderer import MPRenderer
    rnd = MPRenderer()
    # set time step in draw_params
    # rnd.draw_params.time_begin = 100
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
    rnd.draw_params.trajectory.draw_trajectory = True
    rnd.draw_params.dynamic_obstacle.show_label = True
    scenario.draw(rnd)
    rnd.render()
    plt.show()
    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(scenario, ego_id, rule[0])
    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(
            traffic_rule_monitor, planning_problem, ego_initial
        )
        repaired_traj = repairer.repair()
        if repaired_traj is not None and flag_visualization:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )

            # ============= Visualization =============
            plot_limits = [-5, 50, -4.5, 3]
            target_veh = scenario.obstacle_by_id(traffic_rule_monitor.other_id)
            visualize_v_profile(
                ego_initial,
                ego_repaired,
                time_start=0,
                time_end=N,
                tc=repairer.tc,
                tv=repairer.tv,
            )
            visualize_a_profile(
                scenario.dt,
                ego_initial,
                ego_repaired,
                time_start=0,
                time_end=N,
                tc=repairer.tc,
                tv=repairer.tv,
            )
            for time_step in range(49, 50):
                visualize_repairing_result(
                    scenario,
                    ego_initial,
                    ego_repaired,
                    time_step,
                    tc=repairer.tc,
                    tv=repairer.tv,
                    plot_limits=plot_limits,
                    target_veh=target_veh,
                    world=traffic_rule_monitor.world,
                )  # , save_path=figure_path)
