from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import math

if __name__ == "__main__":
    # THIS IS INVALID, ORIGINAL REPAIRER FAILED
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_AachenBendplatz-1_151860_T-1879"

    # Build configuration object
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.intersection_type = "dataset"
    config.repair.rules = ["R_IN4"]
    config.repair.ego_id = 10123

    config.repair.N_r = 20

    config.debug.show_plots = True
    config.repair.planner = 2
    config.repair.constraint_mode = 1

    from commonroad.visualization.mp_renderer import MPRenderer
    rnd = MPRenderer()
    rnd.draw_params.dynamic_obstacle.show_label = True
    config.scenario.draw(rnd)
    config.planning_problem.draw(rnd)
    rnd.render()
    from matplotlib.pyplot import show
    show()
    # Retrieve the ego vehicle
    ego_initial = retrieve_ego_vehicle(config)

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()
        # if repaired_traj is not None and config.debug.show_plots:
        #     ego_repaired = repairer.convert_traj_to_ego_vehicle(
        #         ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
        #     )
        #     if config.debug.show_plots:
        #         # ============= Visualization =============
        #         visualize_repaired_result(config, ego_initial, ego_repaired, repairer)

        print(repairer.t_solver._planner._constraints.longitudinal_constraints)
        print(f"repairer.t_solver._sel_prop:{repairer.t_solver._sel_prop}")
        # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._prop_full}")
        # for proposition in repairer.t_solver._prop_full:
        #     predicate = proposition.children[0]
        #     print(f"predicate: {len(proposition.children)}")
        #     print(f"predicate: {predicate.name}, {vars(predicate)}")
