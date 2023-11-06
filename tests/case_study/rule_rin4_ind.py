from crrepairer.smt.monitor_wrapper import STLRuleMonitor, ScenarioType, IntersectionType
from crrepairer.repairer.smt_repairer_miqp import SMTTrajectoryRepairer
from crrepairer.repairer.visualization import (
    visualize_repairing_result,
    visualize_a_profile,
    visualize_v_profile,
)

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.prediction.prediction import Trajectory
import math

scenario_id = "DEU_AAH1-2_81650_T-1799"
file_path = "../../scenarios/" + scenario_id + ".xml"
figure_path = "./figures"

flag_visualization = True

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(
        lanelet_assignment=True
    )
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_id = 10030
    rule = ["R_IN4"]
    N = 149
    ego_initial = scenario.obstacle_by_id(ego_id)
    ego_initial.prediction.trajectory = Trajectory(
        ego_initial.prediction.initial_time_step, ego_initial.prediction.trajectory.state_list[:N]
    )
    ego_initial.prediction.occupancy_set = ego_initial.prediction.occupancy_set[:N]

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(
        scenario, ego_id, rule[0], ScenarioType.INTERSECTION, IntersectionType.DATASET
    )
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
            plot_limits = [40, 70, -45, -15]
            target_veh = scenario.obstacle_by_id(traffic_rule_monitor.other_id)
            visualize_v_profile(
                ego_initial,
                ego_repaired,
                time_start=ego_initial.prediction.initial_time_step - 1,
                time_end=min(repairer.tv + traffic_rule_monitor.future_time_step + 25, ego_initial.prediction.final_time_step),
                tc=repairer.tc,
                tv=repairer.tv,
            )
            visualize_a_profile(
                scenario.dt,
                ego_initial,
                ego_repaired,
                time_start=ego_initial.prediction.initial_time_step - 1,
                time_end=min(repairer.tv + traffic_rule_monitor.future_time_step + 25, ego_initial.prediction.final_time_step),
                tc=repairer.tc,
                tv=repairer.tv,
            )
            for time_step in range(ego_initial.prediction.initial_time_step, min(repairer.tv + traffic_rule_monitor.future_time_step + 25, ego_initial.prediction.final_time_step)):
                visualize_repairing_result(
                    scenario,
                    ego_initial,
                    ego_repaired,
                    time_step=time_step,
                    tc=repairer.tc,
                    tv=repairer.tv,
                    plot_limits=plot_limits,
                    target_veh=target_veh,
                    world=traffic_rule_monitor.world,
                    end_time=min(repairer.tv + traffic_rule_monitor.future_time_step + 25, ego_initial.prediction.final_time_step),
                    # save_path=figure_path,
                )
