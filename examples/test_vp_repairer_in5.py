import math
import re
import time
from pathlib import Path

import crrepairer.smt.monitor_wrapper as monitor_wrapper
from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle
from crrepairer.utils.visualization import visualize_repaired_result


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMED_OPERATOR_PATTERN = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)


def align_bound_to_sampling_period(bound, dt):
    if dt <= 0:
        return bound
    ratio = bound / dt
    if math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
        return round(ratio) * dt
    return max(0, math.ceil(ratio)) * dt


def format_time_bound(value):
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.12g}"


def align_rtamt_bounds_in_rule(rule_str, dt):
    changes = []

    def replace(match):
        low = float(match.group("low"))
        high = float(match.group("high"))
        low_aligned = align_bound_to_sampling_period(low, dt)
        high_aligned = align_bound_to_sampling_period(high, dt)
        if not (
            math.isclose(low, low_aligned, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(high, high_aligned, rel_tol=1e-9, abs_tol=1e-9)
        ):
            changes.append((match.group(0), low, high, low_aligned, high_aligned))
        return (
            f"{match.group('op')}["
            f"{format_time_bound(low_aligned)},"
            f"{format_time_bound(high_aligned)}{match.group('unit')}]"
        )

    return TIMED_OPERATOR_PATTERN.sub(replace, rule_str), changes


def patch_rtamt_bound_alignment(config):
    original_get_config = monitor_wrapper.get_traffic_rule_config
    dt = float(config.scenario.dt)
    rules = set(config.repair.rules)

    def aligned_get_traffic_rule_config(*args, **kwargs):
        traffic_rules_config = original_get_config(*args, **kwargs)
        traffic_rules = traffic_rules_config.get("traffic_rules", {})
        for rule in rules:
            if rule not in traffic_rules:
                continue
            aligned_rule, changes = align_rtamt_bounds_in_rule(traffic_rules[rule], dt)
            traffic_rules[rule] = aligned_rule
            for original, low, high, low_aligned, high_aligned in changes:
                print(
                    f"Aligned {rule} RTAMT bound {original}: "
                    f"[{low},{high}] -> [{low_aligned},{high_aligned}] for dt={dt}"
                )
        return traffic_rules_config

    monitor_wrapper.get_traffic_rule_config = aligned_get_traffic_rule_config


def build_config():
    # Original hand-picked IN5 demo configuration:
    # scenario_id = "DEU_AAH1-2_7900_T-1049"
    # config = RepairerConfiguration.load(
    #     REPO_ROOT / "config" / f"{scenario_id}.yaml",
    #     scenario_id,
    # )
    # config.update()
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False
    # config.repair.constraint_mode = 1
    # config.repair.sat_solver_mode = "domain_dpll"
    # config.debug.show_plots = True
    # return scenario_id, config

    scenario_id = "DEU_AachenBendplatz-1_12960_T-979"
    config = RepairerConfiguration()
    config.general.path_scenarios = str(REPO_ROOT / "scenarios") + "/"
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.intersection_type = "dataset"
    config.repair.rules = ["R_IN5"]
    config.repair.ego_id = 10047
    config.repair.N_r = 20
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.repair.planner = 2
    config.repair.constraint_mode = 1
    config.repair.sat_solver_mode = "domain_dpll"
    config.debug.show_plots = True

    return scenario_id, config


def main():
    scenario_id, config = build_config()
    patch_rtamt_bound_alignment(config)
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
    # crconvert --routability-check nocheck --num-processes 16 --keep-ego --num-time-steps 2500 --output-type xml --obstacles-start-at-zero raw scenarios ind
    # crconvert --routability-check nocheck --num-processes 16 --keep-ego --num-time-steps 30 --output-type xml --obstacles-start-at-zero inD-dataset-v1.0 scenarios0 ind 
    # maximum step is num-time-steps/downsample --downsample 5 downsample only works for highd scenarios
