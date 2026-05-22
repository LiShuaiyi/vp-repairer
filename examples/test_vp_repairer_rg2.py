import math
import time
from pathlib import Path

from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import InitialState

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle
from crrepairer.utils.visualization import visualize_repaired_result


REPO_ROOT = Path(__file__).resolve().parents[1]


def trim_scenario_horizon(config, initial_time_step: int, horizon: int):
    """Align the scenario to the short RG2 horizon used by the SMT example."""
    final_time_step = initial_time_step + horizon
    for obstacle in config.scenario.obstacles:
        original_states = obstacle.prediction.trajectory.state_list
        updated_initial_state = original_states[initial_time_step - 1]

        shifted_states = original_states[initial_time_step:final_time_step]
        for state in shifted_states:
            state.time_step -= initial_time_step

        obstacle.initial_state = InitialState(
            time_step=0,
            yaw_rate=0,
            slip_angle=0,
            position=updated_initial_state.position,
            velocity=updated_initial_state.velocity,
            orientation=updated_initial_state.orientation,
        )
        obstacle.prediction.trajectory = Trajectory(1, shifted_states)


def populate_ego_acceleration(ego_vehicle, dt: float):
    for i in range(ego_vehicle.prediction.trajectory.final_state.time_step):
        ego_vehicle.state_at_time(i).acceleration = (
            ego_vehicle.state_at_time(i + 1).velocity
            - ego_vehicle.state_at_time(i).velocity
        ) / dt


def build_config():
    scenario_id = "ZAM_Zip-1_56_T-1"
    config = RepairerConfiguration.load(REPO_ROOT / "config" / f"{scenario_id}.yaml", scenario_id)
    config.update()

    initial_time_step = 50
    repair_horizon = 20
    trim_scenario_horizon(config, initial_time_step, repair_horizon)

    config.repair.rules = ["R_G2"]
    config.repair.planner = 2
    config.repair.constraint_mode = 1
    config.repair.sat_solver_mode = "domain_dpll"
    config.repair.use_mpr = False
    config.repair.N_r = repair_horizon

    return scenario_id, config


def main():
    scenario_id, config = build_config()
    ego_initial = retrieve_ego_vehicle(config)
    populate_ego_acceleration(ego_initial, config.scenario.dt)
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
