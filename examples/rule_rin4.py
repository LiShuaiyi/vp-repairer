from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import math

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_AAH1-2_81650_T-1799"

    # Build configuration object
    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()
    config.repair.planner = 2
    config.repair.constraint_mode = 2

    # Retrieve the ego vehicle
    ego_initial = retrieve_ego_vehicle(config)

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

        lon_constr = repairer.t_solver._planner._constraints.longitudinal_constraints
        rule_constr_dict = lon_constr.rule_constraints
        collision_constr = lon_constr.collision_free_constraints
        for key,val in rule_constr_dict.items():
            print(f"rule: {key}, constr: {vars(val)}")
        print(f"collision_constr: {vars(collision_constr)}")
        # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._sel_prop}")
        # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._prop_full}")
        for proposition in repairer.t_solver._prop_full:
            predicate = proposition.children[0]
            # print(f"predicate: {len(proposition.children)}")
            # print(f"predicate: {predicate.name}, {vars(predicate)}")
