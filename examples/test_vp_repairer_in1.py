import math
import time

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle
from crrepairer.utils.visualization import visualize_repaired_result


def build_config():
    scenario_id = "DEU_AachenBendplatz-1_152460_T-2479"
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.rules = ["R_IN1"]
    config.repair.ego_id = 10161
    config.repair.N_r = 20
    config.update()
    config.repair.use_mpr = False
    config.debug.show_plots = True
    config.repair.planner = 2
    config.repair.constraint_mode = 1
    config.repair.sat_solver_mode = "domain_dpll"
    config.debug.plot_limits = [40, 69, -45, -17]

    # scenario_id = "DEU_AachenBendplatz-1_151520_T-1539"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10108
    # config.repair.N_r = 20
    # config.update()
    # config.debug.show_plots = True
    # config.repair.planner = 2
    # config.repair.constraint_mode = 1
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False
    # config.debug.plot_limits = [37, 54, -30, -12]

    # scenario_id = "DEU_AachenBendplatz-1_75140_T-5159"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10203
    # config.repair.N_r = 20
    # config.update()
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False
    # config.debug.show_plots = True
    # config.repair.planner = 2
    # config.repair.constraint_mode = 1
    # config.repair.sat_solver_mode = "domain_dpll"
    # config.debug.plot_limits = [53, 62, -30, -20]

    return scenario_id, config


def main():
    scenario_id, config = build_config()
    ego_initial = retrieve_ego_vehicle(config)
    traffic_rule_monitor = STLRuleMonitor(config)

    print(f"Scenario: {scenario_id}")
    print(f"Rules: {config.repair.rules}")
    print(f"Initial tv time step: {traffic_rule_monitor.tv_time_step}")
    print(f"SAT solver mode: {config.repair.sat_solver_mode}")

    if traffic_rule_monitor.tv_time_step in (math.inf, -math.inf):
        print("No repair needed or trajectory is not repairable at initialization.")
        return

    repairer = VPTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
    if repairer.sat_solver.solver_mode == "domain_dpll":
        domain_dict = repairer.ensure_domain_dict_initialized()
        print(f"DomainDPLL domain_dict construction time: {repairer.domain_dict_time:.6f}s")
        print(f"DomainDPLL domain_dict: {domain_dict}")

    start_time = time.time()
    repaired_traj = repairer.repair()
    total_time = time.time() - start_time

    if repaired_traj is None:
        print(f"VP repair failed after {total_time:.3f}s")
        return

    tv_updated, other_id = repairer.calc_tv_updated(
        repaired_traj.state_list,
        repairer.tc,
    )
    print(f"VP repair finished in {total_time:.3f}s")
    print(f"Repair started from tv={repairer.tv}, internal tc={repairer.tc}")
    print(f"Updated tv after VP repair: {tv_updated}, related other id: {other_id}")
    if repairer.sat_solver.solver_mode == "domain_dpll":
        print(f"Stored domain_dict construction time: {repairer.domain_dict_time:.6f}s")
        print(f"Stored domain_dict size: {len(repairer.domain_dict)}")
    if repairer.runtime_breakdown:
        print(f"Core runtime breakdown: {repairer.runtime_breakdown}")

    if config.debug.show_plots:
        ego_repaired = repairer.convert_traj_to_ego_vehicle(
            ego_initial.obstacle_shape,
            ego_initial.initial_state,
            repaired_traj,
        )
        visualize_repaired_result(config, ego_initial, ego_repaired, repairer)


if __name__ == "__main__":
    main()
