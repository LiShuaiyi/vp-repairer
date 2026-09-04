"""Free lateral/longitudinal MICP encodings for all VP experiment groups.

The dynamic state is ``[s,d,v_s,v_d,a_s,a_d,j_s,j_d]``.  Predicates see the
ten-dimensional system output.  The exact monitor still decides whether a
returned trajectory is counted as successful.
"""

from __future__ import annotations

import math
from functools import reduce
import numpy as np
import shapely
from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.predicates.predicate_factory import PredicateFactory
from stlpy.STL import LinearPredicate
from stlpy.STL.formula import STLTree
from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.benchmarks.common import outside_rectangle_formula

from comparison.micp.constraints import (
    CollisionFreeConstraint,
    compute_lane_bounds,
)
from comparison.micp.formula import (
    collision_free_formula,
    in_front_of_formula,
    in_same_lane_formula,
    inside_interval_formula,
    linearized_keeps_safe_distance_formula,
    no_backwards_driving,
    not_braking_formula,
    not_braking_abruptly_formula,
    not_in_front_of_formula,
    not_in_same_lane_formula,
    outside_interval_formula,
    phantom_false,
    relative_braking_abruptly_formula,
)
from comparison.micp.vehicle_models_dt import VehicleModel


DIM = 10


def polygonal_speed_limit_formula(
    speed_limit, v_s_index, v_d_index, d, name=None, sides=16,
):
    """Inscribed polygon approximation of the monitor's Euclidean speed cap."""
    predicates = []
    bound = float(speed_limit) * math.cos(math.pi / int(sides))
    angles = (
        np.linspace(0.0, 2.0 * math.pi, int(sides), endpoint=False)
        + math.pi / int(sides)
    )
    for angle in angles:
        a = np.zeros((1, d))
        a[:, v_s_index] = -math.cos(angle)
        a[:, v_d_index] = -math.sin(angle)
        predicates.append(LinearPredicate(a, -bound, name=name))
    return reduce(lambda x, y: x & y, predicates)


# Backward-compatible name used by the solver encoding tests.  The former
# implementation was a four-sided L1 diamond; the 16-gon is still an inner
# approximation but removes most of that unintended conservatism.
l1_speed_limit_formula = polygonal_speed_limit_formula


def at_each_step(formulas):
    formulas = list(formulas)
    if not formulas:
        raise ValueError("Empty planning horizon")
    return STLTree(formulas, "and", list(range(len(formulas))))


def collision_formula(bounds, ego):
    return collision_free_formula(bounds, 0, DIM, ego.shape.length, 2.578)


def false_formula():
    """A state-independent false predicate for empty logical branches."""
    return LinearPredicate(np.zeros((1, DIM)), 1.0)


def true_formula():
    """A state-independent predicate with unit positive robustness."""
    return LinearPredicate(np.zeros((1, DIM)), -1.0)


def outside_convex_polygon_formula(
    vertices, x_index=0, y_index=1, s_padding=0.0, d_padding=0.0,
):
    """Linear disjunction for lying outside a convex polygon in CLCS."""
    vertices = np.asarray(vertices, dtype=float)
    if s_padding or d_padding:
        offsets = np.asarray([
            [-s_padding, -d_padding], [-s_padding, d_padding],
            [s_padding, -d_padding], [s_padding, d_padding],
        ])
        vertices = np.vstack([vertices + offset for offset in offsets])
    polygon = shapely.Polygon(vertices).convex_hull
    points = np.asarray(polygon.exterior.coords[:-1], dtype=float)
    if not polygon.exterior.is_ccw:
        points = points[::-1]
    sides = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        dx, dy = end - start
        a = np.zeros((1, DIM))
        # For CCW vertices the polygon interior is left of every edge;
        # this predicate selects the complementary right half-plane.
        a[:, x_index] = dy
        a[:, y_index] = -dx
        b = dy * start[0] - dx * start[1]
        sides.append(LinearPredicate(a, b))
    return reduce(lambda left, right: left | right, sides)


def negate_pnf(formula):
    """Push negation through an and/or tree into its linear predicates."""
    if isinstance(formula, LinearPredicate):
        return formula.negation()
    kind = "or" if formula.combination_type == "and" else "and"
    return STLTree(
        [negate_pnf(child) for child in formula.subformula_list],
        kind,
        list(formula.timesteps),
    )


def safe_collision_bounds(world, ego, final_step):
    """Build the legacy longitudinal corridor, skipping invalid projections."""
    result = {}
    # Some extracted inD windows have no lanelet assignment at sample zero,
    # while their reference route and curvilinear states are still valid.
    # Collision projection only needs that reference lane.
    lane = ego.ref_path_lane
    if lane is None:
        first_assigned = min(ego.lanelet_assignment)
        lane = ego.get_lane(first_assigned)
    for k in range(final_step + 1):
        lower, upper = -np.inf, np.inf
        for vehicle_id in world.vehicle_ids_for_time_step(k):
            vehicle = world.vehicle_by_id(vehicle_id)
            if vehicle.id == ego.id:
                continue
            try:
                if not ego.lanes_at_state(k).intersection(vehicle.lanes_at_state(k)):
                    continue
                rear, front = vehicle.rear_s(k, lane), vehicle.front_s(k, lane)
                ego_rear, ego_front = ego.rear_s(k, lane), ego.front_s(k, lane)
                if any(value is None for value in (rear, front, ego_rear, ego_front)):
                    continue
                if rear > ego_front:
                    upper = min(upper, rear)
                elif front < ego_rear:
                    lower = max(lower, front)
            except (KeyError, TypeError, ValueError):
                continue
        result[k] = (lower, upper)
    return result


def free_2d_collision_formula(world, ego, time_step, reference_lane):
    """Axis-aligned CLCS obstacle avoidance without fixed front/rear order."""
    formulas = []
    for vehicle_id in world.vehicle_ids_for_time_step(time_step):
        if int(vehicle_id) == int(ego.id):
            continue
        other = world.vehicle_by_id(vehicle_id)
        try:
            center = reference_lane.clcs.convert_to_curvilinear_coords(
                *other.states_cr[time_step].position
            )
            half_s = (ego.shape.length + other.shape.length) / 2.0 + 0.05
            half_d = (ego.shape.width + other.shape.width) / 2.0 + 0.05
            formulas.append(outside_rectangle_formula(
                (
                    float(center[0] - half_s), float(center[0] + half_s),
                    float(center[1] - half_d), float(center[1] + half_d),
                ),
                0, 1, DIM,
            ))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    if not formulas:
        return true_formula()
    return reduce(lambda left, right: left & right, formulas)


class Rule(BenchmarkScenario):
    def __init__(
        self, num_steps, world, ego, lanelet_network, dt, other=None,
        rule_name=None, trigger_step=None, rule_semantics="lin2025",
    ):
        self.num_steps = int(num_steps)
        self.T = self.num_steps - 1
        self.world = world
        self.ego = ego
        self.other = other
        self.lanelet_network = lanelet_network
        self.dt = float(dt)
        self.rule_name = rule_name
        self.trigger_step = trigger_step
        self.rule_semantics = rule_semantics

    def reference_lane(self):
        """Return the CLCS used by the optimizer for every dataset variant."""
        lane = getattr(self.ego, "ref_path_lane", None)
        return lane if lane is not None else self.ego.get_lane(0)

    def GetSystem(self):
        # Lin2025 uses the full 8-state position/velocity/acceleration/jerk
        # chain in both longitudinal and lateral directions.  Keeping the
        # smaller direct-acceleration model here would materially change the
        # benchmark size and timing.
        return VehicleModel(self.dt)

    def control_bounds(self):
        return np.full(2, -2000.0), np.full(2, 2000.0)

    def state_bounds(self):
        return np.full(8, -np.inf), np.full(8, np.inf)

    def physical_speed_limit(self):
        dynamics = self.ego.vehicle_param.get("dynamics_param")
        longitudinal = getattr(dynamics, "longitudinal", None)
        value = getattr(longitudinal, "v_max", None)
        return float(value) if value is not None and math.isfinite(value) else 60.0

    def speed_envelope(self, limit=None):
        return polygonal_speed_limit_formula(
            self.physical_speed_limit() if limit is None else limit,
            2, 3, DIM, "two-dimensional speed limit",
        )

    def add_to_plot(self, ax):
        """BenchmarkScenario compatibility; plotting is handled separately."""
        return None


class RG1(Rule):
    def GetSpecification(self):
        if self.other is None:
            raise ValueError("R_G1 requires the monitor-selected preceding vehicle")
        collision = CollisionFreeConstraint()
        collision.compute(self.world, self.ego, self.other, 0, self.T)
        result = []
        reference_lane = self.ego.get_lane(0)
        for k in range(self.num_steps):
            try:
                target_lane = self.other.get_lane(k)
                rear_l = self.other.rear_s(k, reference_lane)
                lead_velocity = self.other.get_lon_state(k, reference_lane).v
                same_bounds = compute_lane_bounds(target_lane, reference_lane)
                safe_distance = linearized_keeps_safe_distance_formula(
                    rear_l, lead_velocity, 0, 2, DIM,
                    self.ego.shape.length, 2.578,
                )
                clause = (
                    not_in_front_of_formula(
                        (-np.inf, rear_l), 0, DIM,
                        self.ego.shape.length, 2.578,
                    )
                    | not_in_same_lane_formula(same_bounds, 0, 1, DIM)
                    | safe_distance
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                clause = true_formula()
            collision_free = collision_formula(collision.constraint_dict[k], self.ego)
            if collision_free is not None:
                clause = clause & collision_free
            result.append(clause & no_backwards_driving(2, DIM))
        return at_each_step(result)


class RG2(Rule):
    def GetSpecification(self):
        if self.rule_semantics == "vp_witness":
            result = []
            reference_lane = self.ego.get_lane(0)
            for k in range(self.num_steps):
                fixed_preceding = []
                ego_front = self.ego.front_s(k, reference_lane)
                for vehicle_id in self.world.vehicle_ids_for_time_step(k):
                    if int(vehicle_id) == int(self.ego.id):
                        continue
                    target = self.world.vehicle_by_id(vehicle_id)
                    try:
                        rear_l = target.rear_s(k, reference_lane)
                        if (
                            rear_l is not None and ego_front is not None
                            and rear_l >= ego_front
                            and self.ego.lanes_at_state(k).intersection(
                                target.lanes_at_state(k)
                            )
                        ):
                            fixed_preceding.append((rear_l - ego_front, target))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        continue

                clause = not_braking_abruptly_formula(4, DIM)
                if fixed_preceding:
                    target = min(fixed_preceding, key=lambda item: item[0])[1]
                    try:
                        rear_l = target.rear_s(k, reference_lane)
                        same_bounds = compute_lane_bounds(
                            target.get_lane(k), reference_lane
                        )
                        precedes = (
                            in_front_of_formula(
                                (-np.inf, rear_l), 0, DIM,
                                self.ego.shape.length, 2.578,
                            )
                            & in_same_lane_formula(same_bounds, 0, 1, DIM)
                        )
                        safe = linearized_keeps_safe_distance_formula(
                            rear_l,
                            target.get_lon_state(k, reference_lane).v,
                            0, 2, DIM, self.ego.shape.length, 2.578,
                        )
                        rel_abrupt = relative_braking_abruptly_formula(
                            target.get_lon_state(k, reference_lane).a, 4, DIM,
                        )
                        justified = (
                            negate_pnf(safe) | rel_abrupt.negation()
                        )
                        clause = clause | (precedes & justified)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                result.append(clause & no_backwards_driving(2, DIM))
            return at_each_step(result)
        if self.rule_semantics == "vp_quantified":
            result = []
            reference_lane = self.ego.get_lane(0)
            for k in range(self.num_steps):
                candidates = []
                for vehicle_id in self.world.vehicle_ids_for_time_step(k):
                    if int(vehicle_id) == int(self.ego.id):
                        continue
                    target = self.world.vehicle_by_id(vehicle_id)
                    try:
                        target_lane = target.get_lane(k)
                        rear_l = target.rear_s(k, reference_lane)
                        if target_lane is None or rear_l is None:
                            continue
                        same_bounds = compute_lane_bounds(target_lane, reference_lane)
                        front = in_front_of_formula(
                            (-np.inf, rear_l), 0, DIM,
                            self.ego.shape.length, 2.578,
                        )
                        not_front = not_in_front_of_formula(
                            (-np.inf, rear_l), 0, DIM,
                            self.ego.shape.length, 2.578,
                        )
                        same = in_same_lane_formula(same_bounds, 0, 1, DIM)
                        not_same = not_in_same_lane_formula(same_bounds, 0, 1, DIM)
                        safe = linearized_keeps_safe_distance_formula(
                            rear_l,
                            target.get_lon_state(k, reference_lane).v,
                            0, 2, DIM, self.ego.shape.length, 2.578,
                        )
                        rel_abrupt = relative_braking_abruptly_formula(
                            target.get_lon_state(k, reference_lane).a, 4, DIM,
                        )
                        candidates.append({
                            "rear": float(rear_l), "front": front,
                            "not_front": not_front, "same": same,
                            "not_same": not_same, "safe": safe,
                            "rel_abrupt": rel_abrupt,
                        })
                    except (AttributeError, KeyError, TypeError, ValueError):
                        continue

                witnesses = []
                for target in candidates:
                    # PredPreceding selects the nearest vehicle ahead in the
                    # same lane.  For this target to be the witness, every
                    # geometrically closer fixed vehicle must either be behind
                    # the optimized ego or outside its selected lane.
                    precedes = target["front"] & target["same"]
                    for blocker in candidates:
                        if blocker is target or blocker["rear"] >= target["rear"]:
                            continue
                        precedes = precedes & (
                            blocker["not_front"] | blocker["not_same"]
                        )
                    justified = negate_pnf(target["safe"]) | target["rel_abrupt"].negation()
                    witnesses.append(precedes & justified)

                clause = not_braking_abruptly_formula(4, DIM)
                if witnesses:
                    clause = clause | reduce(lambda x, y: x | y, witnesses)
                result.append(clause & no_backwards_driving(2, DIM))
            return at_each_step(result)
        if self.rule_semantics == "vp_compatible":
            # A sufficient, monitor-exact repair of R_G2 is to make the
            # implication antecedent false: the ego must not brake abruptly.
            # This avoids Lin's non-equivalent fixed-witness formula.  It is
            # deliberately kept as a separate semantics mode because it is
            # more restrictive than the complete existential monitor rule.
            formulas = [
                not_braking_abruptly_formula(4, DIM)
                & no_backwards_driving(2, DIM)
                for _ in range(1, self.num_steps)
            ]
            # x[0] is fixed to the recorded initial state.  All batch inputs
            # have a strictly later violation time, so the monitor has already
            # established that this state satisfies R_G2 (possibly because an
            # abrupt brake is justified by its preceding vehicle).
            a0 = float(self.ego.states_cr[0].acceleration)
            formulas.insert(
                0,
                true_formula() if a0 < -2.0 else not_braking_abruptly_formula(4, DIM),
            )
            return at_each_step(formulas)
        result = []
        reference_lane = self.ego.get_lane(0)
        for k in range(self.num_steps):
            not_braking = not_braking_formula(4, DIM)
            not_abrupt = not_braking_abruptly_formula(4, DIM)
            # Apply the selected-witness implication over the complete
            # horizon.  Restricting it to the monitor trigger neighbourhood is
            # a case-guided simplification and is not part of Lin2025.
            if self.other is None:
                clause = not_abrupt
            else:
                try:
                    target_lane = self.other.get_lane(k)
                    rear_l = self.other.rear_s(k, reference_lane)
                    lead_velocity = self.other.get_lon_state(k, reference_lane).v
                    target_acceleration = self.other.get_lon_state(k, reference_lane).a
                    same_bounds = compute_lane_bounds(target_lane, reference_lane)
                    front = in_front_of_formula(
                        (-np.inf, rear_l), 0, DIM,
                        self.ego.shape.length, 2.578,
                    )
                    same = in_same_lane_formula(same_bounds, 0, 1, DIM)
                    safe = linearized_keeps_safe_distance_formula(
                        rear_l, lead_velocity, 0, 2, DIM,
                        self.ego.shape.length, 2.578,
                    )
                    relative = relative_braking_abruptly_formula(
                        target_acceleration, 4, DIM,
                    )
                    not_front = not_in_front_of_formula(
                        (-np.inf, rear_l), 0, DIM,
                        self.ego.shape.length, 2.578,
                    )
                    not_same = not_in_same_lane_formula(
                        same_bounds, 0, 1, DIM
                    )
                    # Preserve the formula used in Lin2025's released MICP
                    # implementation.  It is intentionally not replaced by a
                    # trigger-local or algebraically stronger implication.
                    clause = (
                        not_braking | not_front | not_same | not_abrupt
                        | (safe & front & same & relative)
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    clause = not_abrupt
            result.append(clause & no_backwards_driving(2, DIM))
        return at_each_step(result)


class RG3(Rule):
    def monitor_speed_limit(self, time_step):
        """Cap used by the monitor on the ego's lane assignment at this step."""
        limits = []
        interpreter = TrafficSignInterpreter(
            SupportedTrafficSignCountry.GERMANY, self.lanelet_network
        )
        lanelet_ids = self.ego.lanelet_assignment.get(time_step, set())
        value = interpreter.speed_limit(frozenset(lanelet_ids))
        if value is not None and math.isfinite(value):
            limits.append(float(value))
        for key in ("fov_speed_limit", "braking_speed_limit"):
            value = self.ego.vehicle_param.get(key)
            if value is not None and math.isfinite(value):
                limits.append(float(value))
        if self.ego.obstacle_type is ObstacleType.TRUCK:
            limits.append(22.22)
        return min(limits) if limits else self.physical_speed_limit()

    def GetSpecification(self):
        return at_each_step([
            self.speed_envelope(self.monitor_speed_limit(k))
            & no_backwards_driving(2, DIM)
            for k in range(self.num_steps)
        ])


class RG1RG3(RG1):
    def GetSpecification(self):
        speed = RG3(self.num_steps, self.world, self.ego, self.lanelet_network, self.dt)
        return super().GetSpecification() & speed.GetSpecification()


class RIN1(Rule):
    def stop_lines(self):
        with_stop = {x.lanelet_id for x in self.lanelet_network.lanelets if x.stop_line is not None}
        relevant = self.ego.ref_path_lane.contained_lanelets & with_stop
        values = []
        for lanelet_id in relevant:
            stop = self.lanelet_network.find_lanelet_by_id(lanelet_id).stop_line
            projected = []
            for point in (stop.start, stop.end):
                try:
                    projected.append(self.ego.ref_path_lane.clcs.convert_to_curvilinear_coords(*point)[0])
                except ValueError:
                    pass
            if projected:
                values.append(min(projected) - self.ego.shape.length / 2.0)
        if not values:
            raise ValueError("No relevant stop line projects onto the ego CLCS")
        return np.asarray(values)

    def GetSpecification(self):
        stop_s = float(np.min(self.stop_lines()))
        if self.rule_semantics in {
            "vp_no_crossing_temporal", "vp_no_crossing_rule_only",
        }:
            # A no-crossing repair is sufficient for R_IN1 without encoding
            # the optional stop-for-three-seconds-and-continue branch.  The
            # projected stop position returned by ``stop_lines`` already
            # subtracts half the vehicle length.  Do not subtract the half
            # diagonal as well: that assumes a freely rotating body although
            # the CLCS state and reconstructed heading follow the route
            # tangent.  A small clearance handles strict monitor inequalities
            # and numerical projection error without making near-line cases
            # physically impossible.
            stop_clearance = 0.02
            before_by_step = [true_formula()]
            for k in range(1, self.num_steps):
                a = np.zeros((1, DIM))
                a[:, 0] = -1.0
                before_by_step.append(
                    LinearPredicate(a, -(stop_s - stop_clearance))
                )
            specification = at_each_step(before_by_step)

            # Do not impose a heading cone here: under the no-crossing repair
            # a fixed rear vehicle may otherwise make stopping infeasible.
            # Lateral motion remains a genuine decision variable; collision
            # constraints below prevent using it as a rule-bypass shortcut.
            heading_cone = true_formula()
            reference_lane = self.reference_lane()
            per_step = []
            for k in range(self.num_steps):
                collision_free = (
                    true_formula()
                    if k == 0 or self.rule_semantics == "vp_no_crossing_rule_only"
                    else free_2d_collision_formula(
                        self.world, self.ego, k, reference_lane
                    )
                )
                per_step.append(
                    no_backwards_driving(2, DIM)
                    & heading_cone
                    & collision_free
                )
            return specification & at_each_step(per_step)
        if self.rule_semantics == "vp_compatible":
            # Prevent crossing the stop line during this finite repair
            # horizon.  This makes the antecedent of the monitor's crossing
            # implication false and is therefore a sound repair.  Unlike the
            # released Lin tree, it cannot be bypassed by choosing d <= 0.
            # ``stop_s`` is based on half the vehicle length.  The monitor
            # projects the rotated footprint, whose longitudinal extent can
            # be as large as the half diagonal under free lateral motion.
            footprint_extra = (
                math.hypot(self.ego.shape.length, self.ego.shape.width)
                - self.ego.shape.length
            ) / 2.0
            # Extra allowance covers CLCS curvature/projection tolerance in
            # addition to the worst-case rotated rectangular footprint.
            conservative_stop_s = stop_s - footprint_extra - 0.10
            a = np.zeros((1, DIM))
            a[:, 0] = -1.0
            stays_before_stop_line = LinearPredicate(a, -conservative_stop_s)
            result = []
            for k in range(self.num_steps):
                # The recorded x[0] is immutable and the input manifests only
                # contain violations after t=0.  Preserve that already valid
                # initial state; apply the repair from the first successor.
                stop_clause = true_formula() if k == 0 else stays_before_stop_line
                clause = stop_clause & no_backwards_driving(2, DIM)
                result.append(clause)
            return at_each_step(result)
        stop_zone = (stop_s - 1.0, stop_s)
        inside = inside_interval_formula(stop_zone, 0, DIM)
        outside = outside_interval_formula(stop_zone, 0, DIM)
        # Preserve the three auxiliary branches in Lin2025 verbatim.  They
        # are redundant/problematic as logical constants when lateral motion
        # is free, but removing them changes the MICP tree and timing.
        no_crossing = (
            outside | inside.eventually(1, 1)
            | phantom_false(1, DIM)
            | phantom_false(1, DIM)
            | phantom_false(1, DIM)
        )
        specification = no_crossing.always(0, self.T - 1)
        collision = safe_collision_bounds(self.world, self.ego, self.T)
        result = []
        for k in range(self.num_steps):
            clause = no_backwards_driving(2, DIM)
            collision_free = collision_formula(collision[k], self.ego)
            if collision_free is not None:
                clause = clause & collision_free
            result.append(clause)
        return specification & at_each_step(result)


class IntersectionPriority(Rule):
    """Free-2D encoding of the shared R_IN3/R_IN4/R_IN5 consequent."""

    SAME_PRIORITY = (
        "turning_right_ego_turning_right_target_same_priority",
        "turning_right_ego_turning_left_target_same_priority",
        "turning_right_ego_going_straight_target_same_priority",
        "turning_left_ego_turning_right_target_same_priority",
        "turning_left_ego_turning_left_target_same_priority",
        "turning_left_ego_going_straight_target_same_priority",
        "going_straight_ego_turning_right_target_same_priority",
        "going_straight_ego_turning_left_target_same_priority",
        "going_straight_ego_going_straight_target_same_priority",
    )
    HAS_PRIORITY = (
        "turning_right_target_turning_right_ego_target_has_priority",
        "turning_right_target_turning_left_ego_target_has_priority_not_oncoming",
        "turning_right_target_going_straight_ego_target_has_priority",
        "going_straight_target_turning_right_ego_target_has_priority",
        "going_straight_target_turning_left_ego_target_has_priority_not_oncoming",
        "going_straight_target_going_straight_ego_target_has_priority",
        "turning_left_target_turning_left_ego_target_has_priority",
        "turning_left_target_turning_right_ego_target_has_priority",
        "turning_left_target_going_straight_ego_target_has_priority",
    )
    ONCOMING_PRIORITY = (
        "turning_right_target_turning_left_ego_target_has_priority_oncoming",
        "going_straight_target_turning_left_ego_target_has_priority_oncoming",
    )

    def _outside_lanelets(self, lanelets, label):
        """Return outside of each projected lanelet, including ego footprint."""
        outside_all = []
        for lanelet in lanelets:
            projected = []
            for x, y in lanelet.polygon.shapely_object.exterior.coords:
                try:
                    sd = self.ego.ref_path_lane.clcs.convert_to_curvilinear_coords(x, y)
                except ValueError:
                    continue
                if np.all(np.isfinite(sd)):
                    projected.append(sd)
            if len(projected) >= 3:
                outside_all.append(outside_convex_polygon_formula(
                    projected,
                    s_padding=self.ego.shape.length / 2.0 + 0.01,
                    d_padding=self.ego.shape.width / 2.0 + 0.01,
                ))
        if not outside_all:
            raise ValueError(f"{label} does not project onto the ego CLCS")
        return reduce(lambda left, right: left & right, outside_all)

    @staticmethod
    def _intersection_lanelets(vehicle, network):
        return [
            network.find_lanelet_by_id(lanelet_id)
            for lanelet_id in vehicle.lanelets_dir
            if LaneletType.INTERSECTION
            in network.find_lanelet_by_id(lanelet_id).lanelet_type
        ]

    def conflict_avoidance_formula(self):
        """Lin2025 geometric overlap, enlarged and reduced to a CLCS box."""
        ego_lanelets = self._intersection_lanelets(self.ego, self.lanelet_network)
        other_lanelets = self._intersection_lanelets(self.other, self.lanelet_network)
        if not ego_lanelets or not other_lanelets:
            return true_formula()
        ego_region = shapely.unary_union([
            lanelet.polygon.shapely_object for lanelet in ego_lanelets
        ])
        other_region = shapely.unary_union([
            lanelet.polygon.shapely_object for lanelet in other_lanelets
        ])
        overlap = ego_region.intersection(other_region)
        if overlap.is_empty:
            return true_formula()
        try:
            enlarged = shapely.Polygon(
                shapely.offset_curve(overlap, self.ego.shape.length / 2.0)
            )
            if enlarged.is_empty or not enlarged.is_valid:
                enlarged = overlap.buffer(self.ego.shape.length / 2.0)
        except (TypeError, ValueError):
            enlarged = overlap.buffer(self.ego.shape.length / 2.0)
        projected = []
        for x, y in enlarged.convex_hull.exterior.coords:
            try:
                projected.append(
                    self.ego.ref_path_lane.clcs.convert_to_curvilinear_coords(x, y)
                )
            except ValueError:
                continue
        if len(projected) < 3:
            return true_formula()
        xmin, ymin, xmax, ymax = shapely.Polygon(projected).bounds
        return outside_rectangle_formula(
            (xmin, xmax, ymin, ymax), 0, 1, DIM
        )

    def conflict_lanelets(self):
        ego_lanelets = self._intersection_lanelets(self.ego, self.lanelet_network)
        other_lanelets = self._intersection_lanelets(self.other, self.lanelet_network)
        if not ego_lanelets or not other_lanelets:
            raise ValueError("No intersection lanelets available for conflict area")
        # Mirror PredInIntersectionConflictArea: for the ego, conflict
        # lanelets are intersection lanelets on the target reference route,
        # excluding lanelets belonging to the ego's own directed route.  The
        # former road-polygon intersection was a smaller, non-equivalent set.
        ego_directed = set(self.ego.lanelets_dir)
        conflict_lanelets = [
            lanelet for lanelet in other_lanelets
            if lanelet.lanelet_id not in ego_directed
        ]
        if not conflict_lanelets:
            raise ValueError("No monitor-equivalent conflict lanelets")
        return conflict_lanelets

    def yield_region_formula(self):
        """Lin-style branch allowing exit through any conflict-area boundary."""
        return self.conflict_avoidance_formula()

    def intersection_avoidance_formula(self):
        """Negation of on_lanelet_with_type_intersection(ego)."""
        lanelets = [
            lanelet for lanelet in self.lanelet_network.lanelets
            if LaneletType.INTERSECTION in lanelet.lanelet_type
        ]
        return self._outside_lanelets(lanelets, "Intersection area")

    def antecedent_holds(self, time_step):
        """Evaluate the rule's fixed route/priority antecedent exactly."""
        if not hasattr(self, "_predicate_factory"):
            params = get_traffic_rule_config()["traffic_rules_param"]
            self._predicate_factory = PredicateFactory(params)
        factory = self._predicate_factory
        vehicle_ids = [self.ego.id, self.other.id]
        if self.rule_name in {"R_IN3", "R_IN3_hand_draft"}:
            incoming = factory.get_predicate("on_incoming_left_of").evaluate_boolean(
                self.world, time_step, vehicle_ids
            )
            traffic_light = factory.get_predicate("relevant_traffic_light").evaluate_boolean(
                self.world, time_step, [self.ego.id]
            )
            names = self.SAME_PRIORITY
            return incoming and not traffic_light and any(
                factory.get_predicate(name).evaluate_boolean(
                    self.world, time_step, vehicle_ids
                ) for name in names
            )
        names = self.ONCOMING_PRIORITY if self.rule_name == "R_IN5" else self.HAS_PRIORITY
        return any(
            factory.get_predicate(name).evaluate_boolean(
                self.world, time_step, vehicle_ids
            ) for name in names
        )

    def causes_braking_forbidden_bounds(self, time_step, d_br=15.0):
        """Conservative CLCS box for monitor predicate causes_braking.

        The predicate can only hold when the ego rear lies 0..d_br ahead of
        the target front along the target path and the target brakes.  This
        box excludes that spatial corridor for those fixed target states.
        """
        target_lane = self.other.ref_path_lane
        target_acceleration = self.other.get_lon_state(time_step, target_lane).a
        if target_acceleration > -1.0:
            return None
        target_front = self.other.front_s(time_step, target_lane)
        if target_front is None:
            return None
        radius = math.hypot(
            self.ego.shape.length / 2.0,
            self.ego.shape.width / 2.0,
        )
        projected = []
        for s_target in (target_front + radius, target_front + d_br + radius):
            for d_target in (-radius - 2.0, radius + 2.0):
                try:
                    xy = target_lane.clcs.convert_to_cartesian_coords(
                        s_target, d_target
                    )
                    projected.append(
                        self.ego.ref_path_lane.clcs.convert_to_curvilinear_coords(*xy)
                    )
                except ValueError:
                    continue
        if len(projected) < 2:
            return None
        values = np.asarray(projected)
        return (
            float(values[:, 0].min()), float(values[:, 0].max()),
            float(values[:, 1].min()), float(values[:, 1].max()),
        )

    def GetSpecification(self):
        if self.other is None:
            raise ValueError(f"{self.rule_name} requires the monitor-selected vehicle")
        yield_region = self.yield_region_formula()
        # Preserve Lin2025's 27 auxiliary predicates (nine groups of three)
        # verbatim.  Collapsing them is an algebraic/model-size simplification
        # and invalidates a timing reproduction even though all leaves have
        # the same expression in the released code.
        auxiliary = [phantom_false(1, DIM) for _ in range(27)]
        groups = [
            auxiliary[i] | auxiliary[i + 1] | auxiliary[i + 2]
            for i in range(0, 27, 3)
        ]
        not_entire_subformula = reduce(lambda left, right: left & right, groups)
        rule_formula = (not_entire_subformula | yield_region).always(0, self.T)

        collision = safe_collision_bounds(self.world, self.ego, self.T)
        result = []
        for k in range(self.num_steps):
            clause = no_backwards_driving(2, DIM)
            collision_free = collision_formula(collision[k], self.ego)
            if collision_free is not None:
                clause = clause & collision_free
            result.append(clause)
        return rule_formula & at_each_step(result)


RULE_CLASSES = {
    "R_G1": RG1,
    "R_G2": RG2,
    "R_G3": RG3,
    "R_G1_R_G3": RG1RG3,
    "R_IN1": RIN1,
    "R_IN3": IntersectionPriority,
    "R_IN3_hand_draft": IntersectionPriority,
    "R_IN4": IntersectionPriority,
    "R_IN5": IntersectionPriority,
}
