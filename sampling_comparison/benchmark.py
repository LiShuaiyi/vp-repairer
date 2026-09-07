#!/usr/bin/env python3
"""Batch runner for the isolated free lateral/longitudinal sampling baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Batch experiments must never open interactive visualization windows.
os.environ.setdefault("MPLBACKEND", "Agg")

# Direct script execution puts sampling_comparison/, rather than the current
# repository root, first on sys.path. Force this checkout ahead of an older
# editable crrepairer installation elsewhere on the machine.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The project has MICP/monitor and Reactive Planner in separate environments.
# Load the active environment's compiled scientific stack first, then append
# (never prepend) the existing Python-3.8 RP package directory.  This avoids
# accidentally importing its ABI-incompatible NumPy under Python 3.10.
import numpy as np
import scipy  # noqa: F401
import matplotlib  # noqa: F401

RP_SITE = Path("/data_linux/conda-envs/repair-rrt/lib/python3.8/site-packages")
PAPER_RP = REPO_ROOT / "sampling_comparison/vendor/reactive_planner_ab96a839"
if PAPER_RP.is_dir() and str(PAPER_RP) not in sys.path:
    # Exact rule-aware Reactive Planner revision used by Lin et al. (2025).
    # Keep it ahead of the old environment's unpatched commonroad_rp package.
    sys.path.insert(1, str(PAPER_RP))
if str(RP_SITE) not in sys.path:
    # Supplies pure-Python transitive dependencies (methodtools/wirerope) that
    # are absent from repairverse310_gpu. Numerical packages were imported
    # above, so the Python-3.8 NumPy build at this path cannot shadow them.
    sys.path.append(str(RP_SITE))

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from crmonitor.common.helper import load_yaml
from crmonitor.common.vehicle import CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import (
    RuleEvaluator,
    create_ego_vehicle_param,
    get_evaluation_config,
)
from crmonitor.evaluation.visitor import EvaluationMonitorTreeVisitor
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectoryBundle
from commonroad_rp.utility.config import ReactivePlannerConfiguration
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.trajectory_queries.trajectory_queries import (
    trajectory_preprocess_obb_sum,
)

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
import crrepairer.smt.monitor_wrapper as monitor_wrapper


RULE_GROUPS = {
    "R_G1": ("R_G1",), "R_G2": ("R_G2",), "R_G3": ("R_G3",),
    "R_G1_R_G3": ("R_G1", "R_G3"), "R_IN1": ("R_IN1",),
    "R_IN3": ("R_IN3",), "R_IN3_hand_draft": ("R_IN3_hand_draft",),
    "R_IN4": ("R_IN4",), "R_IN5": ("R_IN5",),
}

TIMED = re.compile(
    r"(?P<op>eventually|always|historically|once)\["
    r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*(?P<high>-?\d+(?:\.\d+)?)(?P<unit>s?)\]"
)


def align_rule_bounds(text, dt):
    def replace(match):
        values = []
        for name in ("low", "high"):
            value = float(match.group(name))
            steps = round(value / dt)
            if not math.isclose(value / dt, steps, abs_tol=1e-9):
                steps = math.ceil(value / dt)
            aligned = steps * dt
            values.append(str(int(aligned)) if math.isclose(aligned, round(aligned)) else f"{aligned:.12g}")
        return f"{match.group('op')}[{values[0]},{values[1]}{match.group('unit')}]"
    return TIMED.sub(replace, text)

FIELDS = [
    "scenario_id", "scenario_path", "ego_id", "rule", "repeat", "num_steps",
    "mode", "lin2025_replanable", "planner_returned", "monitor_compliant", "success", "setup_time",
    "planning_time", "core_total_time", "validation_time", "fixed_other_id", "updated_tv", "error",
    "rule_candidates_checked", "rule_candidate_limit", "in1_strategy",
]


class FixedVehicleEvaluationVisitor(EvaluationMonitorTreeVisitor):
    """Restrict rule quantifiers to the vehicle fixed before replanning."""

    def __init__(self, fixed_other_ids, **kwargs):
        super().__init__(**kwargs)
        self._fixed_other_ids = tuple(int(value) for value in fixed_other_ids)

    def _visit_quant_node(self, node, *ctx):
        world, mpr_world, time_step, bound_ids = ctx[:4]
        depth = len(bound_ids) - 1
        if depth >= len(self._fixed_other_ids):
            raise ValueError("Missing fixed vehicle ID for nested rule quantifier")
        vehicle_id = self._fixed_other_ids[depth]
        if vehicle_id in bound_ids or vehicle_id not in world.vehicle_ids_for_time_step(time_step):
            return [], []
        ids = bound_ids + (vehicle_id,)
        value = node.monitors[vehicle_id].visit(
            self, world, mpr_world, time_step, ids, *ctx[2:]
        )
        return [value], [ids]


def update_goal_state(trajectory):
    """Create the finite-horizon goal used by the repository's old runner."""
    final = trajectory.state_list[-1]
    goal = CustomState(
        position=Rectangle(1.0, 1.0, final.position),
        velocity=Interval(final.velocity, final.velocity + 5.0),
        orientation=AngleInterval(final.orientation - 0.2, final.orientation + 0.2),
        time_step=Interval(0, len(trajectory.state_list) + 5),
    )
    return GoalRegion([goal])


def stop_line_target_s(ego, lanelet_network, coordinate_system):
    """Stop target in the planner CLCS with vehicle-footprint clearance."""
    values = []
    for lanelet_id in ego.ref_path_lane.contained_lanelets:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet.stop_line is None:
            continue
        for point in (lanelet.stop_line.start, lanelet.stop_line.end):
            try:
                values.append(float(coordinate_system.convert_to_curvilinear_coords(*point)[0]))
            except ValueError:
                continue
    if not values:
        raise ValueError("No relevant stop line projects onto the planner CLCS")
    return min(values) - ego.shape.length / 2 - 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("highd", "ind"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario-dir", type=Path, action="append",
        help="Fallback scenario directory; may be repeated.",
    )
    parser.add_argument("--rule", choices=tuple(RULE_GROUPS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("lin2025", "rule_filtered"), default="rule_filtered",
        help="Paper-compatible native planner or strict STL-filtered sampling.",
    )
    parser.add_argument(
        "--max-rule-candidates", type=int, default=0,
        help="Strict-mode candidate cap; 0 means no artificial cap.",
    )
    parser.add_argument(
        "--in1-strategy",
        choices=("paper_example", "paper_batch", "stop_position"),
        default="paper_batch",
        help=(
            "Longitudinal sampling setup for R_IN1. paper_example uses the "
            "zero-speed interval from comparison/sampling_rin1.py; paper_batch "
            "keeps the current speed; stop_position is the earlier experimental "
            "stop-line-position setup."
        ),
    )
    parser.add_argument(
        "--monitor-config", type=Path,
        default=Path("/data_linux/Lab/commonroad-stl-monitor/crmonitor/config.yaml"),
    )
    return parser.parse_args()


def read_cases(path, forced_rule, limit, offset=0):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        seen, count = set(), 0
        for row in reader:
            if row.get("repairer_type") not in (None, "", "vp"):
                continue
            if row.get("sat_solver_mode") not in (None, "", "domain_dpll"):
                continue
            scenario_id = row.get("scenario_id") or row.get("scenario")
            ego_id = row.get("ego_id")
            rule = forced_rule or row.get("rule") or row.get("rule_STL")
            if not scenario_id or not ego_id or not rule:
                continue
            scenario_id = scenario_id.removesuffix(".xml")
            key = scenario_id, int(ego_id), rule
            if key in seen:
                continue
            seen.add(key)
            if len(seen) <= offset:
                continue
            yield dict(
                scenario_id=scenario_id, scenario_path=row.get("scenario_path", ""),
                ego_id=int(ego_id), rule=rule,
            )
            count += 1
            if limit is not None and count >= limit:
                return


def resolve_path(args, case):
    if case["scenario_path"]:
        path = Path(case["scenario_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_file():
            return path
    if not args.scenario_dir:
        raise FileNotFoundError("No valid scenario_path and no --scenario-dir")
    candidates = [case["scenario_id"]]
    if args.dataset == "highd":
        candidates.insert(0, case["scenario_id"].replace("T-1", f"T-{case['ego_id']}"))
    else:
        candidates.insert(0, case["scenario_id"].replace("-1_", f"-{case['ego_id']}_"))
    for directory in args.scenario_dir:
        for stem in dict.fromkeys(candidates):
            path = directory / f"{stem}.xml"
            if path.is_file():
                return path
    raise FileNotFoundError(f"Scenario not found for {case['scenario_id']}")


def make_monitor(args, path, scenario, case):
    config = RepairerConfiguration()
    config.general.path_scenarios = str(path.parent)
    config.general.set_path_scenario(path.stem)
    config.update()
    config.repair.scenario_type = "intersection" if args.dataset == "ind" else "interstate"
    if args.dataset == "ind":
        config.repair.intersection_type = "dataset"
    config.repair.rules = list(RULE_GROUPS[case["rule"]])
    config.repair.ego_id = case["ego_id"]
    config.scenario = scenario
    if case["rule"] not in {"R_IN3", "R_IN5"}:
        return STLRuleMonitor(config)
    original = monitor_wrapper.get_traffic_rule_config
    def aligned_config(*values, **kwargs):
        result = original(*values, **kwargs)
        for name in RULE_GROUPS[case["rule"]]:
            result["traffic_rules"][name] = align_rule_bounds(
                result["traffic_rules"][name], float(scenario.dt)
            )
        return result
    monitor_wrapper.get_traffic_rule_config = aligned_config
    try:
        return STLRuleMonitor(config)
    finally:
        monitor_wrapper.get_traffic_rule_config = original


def select_fixed_other_id(rule_monitor, rule, ego_id):
    """Select once the non-ego vehicle from the original violation."""
    if rule in {"R_G3", "R_IN1"}:
        return None
    quantified_rule = "R_G1" if rule == "R_G1_R_G3" else rule
    vehicle_id = rule_monitor.rule_to_other_id.get(quantified_rule)
    if vehicle_id is not None and int(vehicle_id) != int(ego_id):
        return int(vehicle_id)
    if rule != "R_G2":
        raise ValueError(f"{rule} has no related vehicle in the original violation")

    world = rule_monitor.world
    ego = world.vehicle_by_id(ego_id)
    trigger = rule_monitor.rule_to_tv.get("R_G2", rule_monitor.tv_time_step)
    step = 0 if not math.isfinite(trigger) else max(0, int(trigger))
    step = min(step, ego.end_time)
    lane = ego.get_lane(step)
    candidates = []
    for candidate_id in world.vehicle_ids_for_time_step(step):
        if int(candidate_id) == int(ego_id):
            continue
        target = world.vehicle_by_id(candidate_id)
        try:
            rear = target.rear_s(step, lane)
            front = ego.front_s(step, lane)
            if (
                rear is not None and front is not None and rear >= front
                and ego.lanes_at_state(step).intersection(target.lanes_at_state(step))
            ):
                candidates.append((rear - front, int(candidate_id)))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    # R_G2 can be violated solely by abrupt braking, with no original
    # preceding-vehicle witness. In that case the complete evaluator should
    # retain its normal quantification instead of inventing/failing a target.
    return min(candidates)[1] if candidates else None


def validate(monitor, ego_id, states, fixed_other_id=None, reusable_world=None):
    checked = copy.copy(monitor)
    world = reusable_world if reusable_world is not None else copy.deepcopy(monitor.world)
    checked._world = world
    if fixed_other_id is not None:
        for evaluator in checked._rule_eval:
            previous = evaluator._eval_visitor
            evaluator._eval_visitor = FixedVehicleEvaluationVisitor(
                (fixed_other_id,),
                use_boolean=previous.use_boolean,
                output_type=previous.output_type,
            )
    ego = world.vehicle_by_id(ego_id)
    ego.ccosy_cache = CurvilinearStateManager(world.road_network)
    for state in states:
        ego.states_cr[state.time_step] = state
        shape = ego.shape.rotate_translate_local(state.position, state.orientation)
        ego.lanelet_assignment[state.time_step] = set(
            world.road_network.lanelet_network.find_lanelet_by_shape(shape)
        )
        # Quantified predicates can cache on either participating vehicle.
        # Clear this time slice for every vehicle before reusing the world for
        # the next sampled candidate.
        for vehicle_id in world.vehicle_ids_for_time_step(state.time_step):
            world.vehicle_by_id(vehicle_id).predicate_cache.cache[state.time_step] = defaultdict()
    robustness, _ = checked.evaluate_consecutively(world, checked.start_time_step)
    lengths = [len(values) for values in robustness]
    if not lengths or any(length != lengths[0] for length in lengths):
        return -math.inf
    values = np.asarray(robustness)
    if np.any(values[:, 0] < 0):
        return -math.inf
    tv = np.argmax(values < 0, axis=-1)
    if np.all(tv == 0):
        return math.inf
    return float(np.min(tv[tv != 0]) * world.dt)


class RuleAwareReactivePlanner(ReactivePlanner):
    """Reactive Planner that actually rejects rule-violating samples.

    The historical comparison merely assigned a ``rule_evaluators`` attribute,
    but upstream Reactive Planner never consumes that attribute.  This override
    retains its native ordering and collision test and adds exact-monitor
    filtering before accepting a candidate.
    """

    def __init__(self, config, rule_monitor, ego_id, fixed_other_id, max_rule_candidates):
        self._comparison_monitor = rule_monitor
        self._comparison_ego_id = ego_id
        self._comparison_fixed_other_id = fixed_other_id
        self._comparison_world = copy.deepcopy(rule_monitor.world)
        self.rule_rejected = 0
        self.rule_candidates_checked = 0
        self.max_rule_candidates = max_rule_candidates
        super().__init__(config)

    def _collision_free(self, trajectory):
        """Paper-revision collision check, separated from its rule monitor."""
        half_length = 0.5 * self.vehicle_params.length
        half_width = 0.5 * self.vehicle_params.width
        pos1 = (
            trajectory.cartesian.x
            + self.vehicle_params.wb_rear_axle
            * np.cos(trajectory.cartesian.theta)
        )
        pos2 = (
            trajectory.cartesian.y
            + self.vehicle_params.wb_rear_axle
            * np.sin(trajectory.cartesian.theta)
        )
        theta = trajectory.cartesian.theta
        for i in range(len(pos1)):
            ego = pycrcc.TimeVariantCollisionObject(
                self.x_0.time_step + i * self.config.planning.factor
            )
            ego.append_obstacle(
                pycrcc.RectOBB(
                    half_length, half_width, theta[i], pos1[i], pos2[i]
                )
            )
            if self._cc.collide(ego):
                self._infeasible_count_collision += 1
                return False
        if self.config.planning.continuous_collision_check:
            ego_tvo = pycrcc.TimeVariantCollisionObject(self.x_0.time_step)
            for i in range(len(pos1)):
                ego_tvo.append_obstacle(
                    pycrcc.RectOBB(
                        half_length, half_width, theta[i], pos1[i], pos2[i]
                    )
                )
            ego_tvo, _ = trajectory_preprocess_obb_sum(ego_tvo)
            if self._cc.collide(ego_tvo):
                self._infeasible_count_collision += 1
                return False
        return True

    def _check_collisions(self, trajectory_bundle):
        for trajectory in trajectory_bundle.get_sorted_list():
            if (
                self.max_rule_candidates > 0
                and self.rule_candidates_checked >= self.max_rule_candidates
            ):
                return None
            if not self._collision_free(trajectory):
                continue
            states = self._compute_trajectory_pair(trajectory)[0].state_list
            self.rule_candidates_checked += 1
            updated_tv = validate(
                self._comparison_monitor, self._comparison_ego_id, states,
                fixed_other_id=self._comparison_fixed_other_id,
                reusable_world=self._comparison_world,
            )
            if math.isinf(updated_tv) and updated_tv > 0:
                return trajectory
            self.rule_rejected += 1
        return None

    # The paper's patched RP revision routes candidate selection through this
    # method name. Delegate it to the stricter complete-monitor implementation
    # above, while retaining the historical sampling and cost ordering.
    def _check_collisions_rule_compliance(self, trajectory_bundle):
        return self._check_collisions(trajectory_bundle)


def evaluate(args, case, repeat, base_world_config):
    row = {field: "" for field in FIELDS}
    row.update(case, repeat=repeat, planner_returned=False, monitor_compliant=False, success=False)
    row["mode"] = args.mode
    row["in1_strategy"] = args.in1_strategy
    row["lin2025_replanable"] = False
    row["rule_candidate_limit"] = args.max_rule_candidates
    try:
        path = resolve_path(args, case)
        row["scenario_path"] = str(path)
        scenario, planning_problem_set = CommonRoadFileReader(str(path)).open(
            lanelet_assignment=True
        )
        monitor = make_monitor(args, path, scenario, case)
        fixed_other_id = select_fixed_other_id(
            monitor, case["rule"], case["ego_id"]
        )
        row["fixed_other_id"] = (
            "" if fixed_other_id is None else fixed_other_id
        )
        started = time.perf_counter()
        config = ReactivePlannerConfiguration()
        config.general.path_scenarios = str(path.parent)
        config.general.set_path_scenario(path.name)
        planning_problem = next(
            iter(planning_problem_set.planning_problem_dict.values()), None
        )
        config.update(scenario=scenario, planning_problem=planning_problem)
        config.planning.ego_id = case["ego_id"]
        config.planning.rules = list(RULE_GROUPS[case["rule"]])
        config.planning.dt = config.scenario.dt
        world = World.create_from_scenario(config.scenario, copy.deepcopy(base_world_config))
        ego = world.vehicle_by_id(case["ego_id"])
        ego.vehicle_param = create_ego_vehicle_param(
            get_evaluation_config().get("ego_vehicle_param"), world.dt
        )
        obstacle = config.scenario.obstacle_by_id(case["ego_id"])
        config.scenario.remove_obstacle(obstacle)
        config.planning_problem.initial_state = obstacle.initial_state
        config.planning_problem.goal = update_goal_state(obstacle.prediction.trajectory)
        config.vehicle.length, config.vehicle.width = ego.shape.length, ego.shape.width
        config.planning.time_steps_computation = (
            obstacle.prediction.final_time_step - obstacle.prediction.initial_time_step + 1
        )
        # Reactive Planner 2023.1 indexes this lookahead unconditionally when
        # the initial speed is near zero.  Short repair horizons can contain
        # fewer than the default 11 samples, so clamp it to a valid index.
        config.planning.standstill_lookahead = min(
            config.planning.standstill_lookahead,
            max(0, config.planning.time_steps_computation - 1),
        )
        row["num_steps"] = config.planning.time_steps_computation
        evaluators = [
            RuleEvaluator.create_from_config(
                world, case["ego_id"], rule=rule, use_boolean=True
            )
            for rule in RULE_GROUPS[case["rule"]]
        ]
        if args.mode == "lin2025":
            planner = ReactivePlanner(config)
        else:
            planner = RuleAwareReactivePlanner(
                config, rule_monitor=monitor, ego_id=case["ego_id"],
                fixed_other_id=fixed_other_id,
                max_rule_candidates=args.max_rule_candidates,
            )
        planner.rule_evaluators = evaluators
        planner.world = world
        # This path defines the CLCS only. Reactive Planner still samples d,
        # d_dot and d_ddot, so returned trajectories are not fixed to it.
        reference_lane = ego.ref_path_lane or ego.get_lane(0)
        if reference_lane is None:
            raise ValueError("Ego vehicle has no usable CLCS reference lane")
        planner.set_reference_path(
            np.asarray(reference_lane.clcs.reference_path())
        )
        planner.record_state_and_input(planner.x_0)
        if (
            args.mode == "rule_filtered"
            and case["rule"] == "R_IN1"
            and args.in1_strategy == "stop_position"
        ):
            target_s = stop_line_target_s(
                ego, world.road_network.lanelet_network, planner.coordinate_system
            )
            # If the vehicle is already safely waiting closer to the line than
            # the nominal clearance target, do not ask the polynomial sampler
            # to drive backwards. Holding the current safe position keeps the
            # crossing antecedent false.
            current_s = float(
                planner.coordinate_system.convert_to_curvilinear_coords(
                    *planner.x_0.position
                )[0]
            )
            target_s = max(target_s, current_s)
            planner.set_desired_lon_position(
                target_s, delta_s_min=-1.0, delta_s_max=0.0
            )
        elif (
            args.mode == "rule_filtered"
            and case["rule"] == "R_IN1"
            and args.in1_strategy == "paper_example"
        ):
            # Match the rule-specific setup used by the paper's R_IN1 example:
            # sample zero/near-zero terminal speeds while retaining the native
            # free lateral quintic sampling.
            planner.set_desired_velocity(
                desired_velocity=0.0, current_speed=planner.x_0.velocity
            )
            planner.set_v_sampling_parameters(0.01, 15.0)
        elif args.mode == "rule_filtered" and case["rule"] in {"R_G3", "R_G1_R_G3"}:
            planner.set_desired_velocity(
                desired_velocity=min(42.0, planner.x_0.velocity),
                current_speed=planner.x_0.velocity,
            )
        else:
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
        row["setup_time"] = time.perf_counter() - started

        started = time.perf_counter()
        optimal = planner.plan()
        row["planning_time"] = time.perf_counter() - started
        row["core_total_time"] = row["planning_time"]
        if args.mode == "rule_filtered":
            row["rule_candidates_checked"] = planner.rule_candidates_checked
            row["lin2025_replanable"] = planner.rule_candidates_checked > 0
        if optimal:
            row["planner_returned"] = True
            row["lin2025_replanable"] = True
            started = time.perf_counter()
            row["updated_tv"] = validate(
                monitor, case["ego_id"], optimal[0].state_list,
                fixed_other_id=fixed_other_id,
            )
            row["validation_time"] = time.perf_counter() - started
            row["monitor_compliant"] = math.isinf(row["updated_tv"]) and row["updated_tv"] > 0
            row["success"] = row["monitor_compliant"]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return row


def main():
    args = parse_args()
    rp_logger = logging.getLogger("RP_LOGGER")
    rp_logger.handlers.clear()
    rp_logger.setLevel(logging.WARNING)
    config = load_yaml(str(args.monitor_config))
    config["scenario"] = "intersection" if args.dataset == "ind" else "interstate"
    if args.dataset == "ind":
        config["intersection_road_network_param"]["map_type"] = "dataset"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = list(read_cases(args.input, args.rule, args.limit, args.offset))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for repeat in range(args.repeat):
                print(f"sampling {case['rule']} {case['scenario_id']} repeat={repeat}", flush=True)
                row = evaluate(args, case, repeat, config)
                writer.writerow(row)
                stream.flush()
                print(f"  success={row['success']} core={row['core_total_time'] or 'N/A'} error={row['error'] or '-'}", flush=True)


if __name__ == "__main__":
    main()
