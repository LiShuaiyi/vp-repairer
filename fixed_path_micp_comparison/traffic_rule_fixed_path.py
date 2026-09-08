"""Dimension-reduced MICP traffic-rule scenarios on a fixed reference path.

The legacy MICP comparison optimizes both longitudinal and lateral Frenet
motion.  The classes in this module project its predicates onto ``d = 0`` and
optimize only ``[s, v, a, j]`` plus longitudinal snap.  ``num_steps`` always
means the number of signal samples, matching the public stlpy 0.3.0 solver
API; valid STL time indices are therefore ``0 .. num_steps - 1``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import shapely
from stlpy.STL.formula import STLTree

from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter

from comparison.micp.formula import (
    collision_free_formula,
    inside_interval_formula,
    linearized_keeps_safe_distance_formula,
    no_backwards_driving,
    not_in_front_of_formula,
    outside_interval_formula,
)
from comparison.micp.traffic_rule_4d import RG123, RIN1, RIN4
from fixed_path_micp_comparison.vehicle_models_fixed_path import FixedPathVehicleModel


S_INDEX = FixedPathVehicleModel.S
V_INDEX = FixedPathVehicleModel.V
SIGNAL_DIM = FixedPathVehicleModel.OUTPUT_DIM


def _and_at_each_step(formulas: Sequence, num_steps: int):
    if len(formulas) != num_steps:
        raise ValueError(
            f"Expected one formula per step ({num_steps}), got {len(formulas)}"
        )
    return STLTree(list(formulas), "and", list(range(num_steps)))


def _outside_rectangle_on_fixed_path(
    bounds, lateral_offset: float = 0.0
) -> Optional[object]:
    """Project ``outside_rectangle(bounds)`` onto a fixed lateral offset.

    ``None`` denotes the constant-true result: if the fixed path is already
    laterally outside the rectangle, no longitudinal constraint remains.
    Otherwise, being outside the rectangle is equivalent to being outside its
    longitudinal interval.
    """

    s_min, s_max, d_min, d_max = (float(value) for value in bounds)
    d = float(lateral_offset)
    if d <= d_min or d >= d_max:
        return None
    return outside_interval_formula((s_min, s_max), S_INDEX, SIGNAL_DIM)


def _effective_speed_limit(ego_vehicle, lanelet_network) -> float:
    country = SupportedTrafficSignCountry.GERMANY
    lanelet_ids = ego_vehicle.lanelet_assignment[0]
    interpreter = TrafficSignInterpreter(country, lanelet_network)
    lane_speed_limit = interpreter.speed_limit(frozenset(lanelet_ids))
    if lane_speed_limit is None:
        lane_speed_limit = 60.0

    # These are the four simultaneous upper bounds used by RG123.GetSpecification.
    # Replacing their conjunction by the minimum is logically exact and avoids
    # three redundant binary predicate variables in stlpy's MICP encoding.
    return min(float(lane_speed_limit), 60.0, 50.0, 43.0)


def _speed_bounds_formula(speed_limit: float):
    from comparison.micp.formula import keeps_speed_limit

    return keeps_speed_limit(speed_limit, V_INDEX, SIGNAL_DIM) & no_backwards_driving(
        V_INDEX, SIGNAL_DIM
    )


def _relative_vehicle_constraints(
    ego_vehicle, target_vehicle, reference_lane, num_steps: int
):
    target_lane = target_vehicle.get_lane(0)
    reference_path = getattr(reference_lane, "reference_path", None)
    if reference_path is None:
        reference_path = np.asarray(reference_lane.clcs.reference_path(), dtype=float)
    path_in_lane = shapely.LineString(reference_path).intersection(
        target_lane.lanelet.polygon.shapely_object
    )
    lane_coordinates = []

    def collect_coordinates(geometry):
        if hasattr(geometry, "geoms"):
            for sub_geometry in geometry.geoms:
                collect_coordinates(sub_geometry)
        elif hasattr(geometry, "coords"):
            lane_coordinates.extend(geometry.coords)

    collect_coordinates(path_in_lane)
    projected_lane_s = [
        reference_lane.clcs.convert_to_curvilinear_coords(x, y)[0]
        for x, y in lane_coordinates
    ]
    same_lane_interval = (
        None
        if not projected_lane_s
        else (min(projected_lane_s), max(projected_lane_s))
    )
    in_front = defaultdict(tuple)
    safe_distance = defaultdict(tuple)
    for time_step in range(num_steps):
        if time_step > target_vehicle.end_time:
            rear_s = np.inf
            velocity = 0.0
        else:
            try:
                rear_s = target_vehicle.rear_s(time_step, reference_lane)
                lon_state = target_vehicle.get_lon_state(time_step, reference_lane)
            except Exception as exc:
                raise ValueError(
                    f"Target vehicle cannot be projected at time step {time_step}"
                ) from exc
            if rear_s is None or lon_state is None:
                raise ValueError(
                    f"Target vehicle cannot be projected at time step {time_step}"
                )
            velocity = lon_state.v
        in_front[time_step] = (-np.inf, rear_s)
        safe_distance[time_step] = (rear_s, velocity)
    return same_lane_interval, in_front, safe_distance


def _collision_bounds(world, ego_vehicle, reference_lane, num_steps: int):
    constraints = defaultdict(tuple)
    for time_step in range(num_steps):
        s_min, s_max = -np.inf, np.inf
        ego_lanes = ego_vehicle.lanes_at_state(time_step)
        ego_front = ego_vehicle.front_s(time_step, reference_lane)
        ego_rear = ego_vehicle.rear_s(time_step, reference_lane)
        if ego_front is None or ego_rear is None:
            raise ValueError(
                f"Ego vehicle cannot be projected at time step {time_step}"
            )
        for vehicle_id in world.vehicle_ids_for_time_step(time_step):
            vehicle = world.vehicle_by_id(vehicle_id)
            if vehicle.id == ego_vehicle.id:
                continue
            if not ego_lanes.intersection(vehicle.lanes_at_state(time_step)):
                continue
            try:
                rear_s = vehicle.rear_s(time_step, reference_lane)
                front_s = vehicle.front_s(time_step, reference_lane)
            except Exception:
                continue
            if rear_s is None or front_s is None:
                continue
            if rear_s > ego_front:
                s_max = min(s_max, rear_s)
            elif front_s < ego_rear:
                s_min = max(s_min, front_s)
        constraints[time_step] = (s_min, s_max)
    return constraints


class _FixedPathMixin:
    """Shared horizon and dynamics behavior for fixed-path scenarios."""

    lateral_offset = 0.0

    def _validate_num_steps(self):
        if not isinstance(self.T, int) or self.T < 1:
            raise ValueError(f"num_steps must be a positive integer, got {self.T!r}")

    def GetSystem(self):
        return FixedPathVehicleModel(0.2)


class RG123FixedPath(_FixedPathMixin, RG123):
    """Fixed-path version of the legacy combined R_G1/R_G3 MICP scenario."""

    def __init__(
        self,
        num_steps: int,
        world,
        ego_vehicle,
        other_vehicle,
        lanelet_network,
        reference_lane=None,
    ):
        super().__init__(
            T=num_steps,
            world=world,
            ego_vehicle=ego_vehicle,
            other_vehicle=other_vehicle,
            lanelet_network=lanelet_network,
        )
        self.reference_lane = (
            reference_lane or ego_vehicle.ref_path_lane or ego_vehicle.get_lane(0)
        )
        self._validate_num_steps()

    def GetSpecification(self):
        final_step = self.T - 1

        (
            self.in_same_lane_interval,
            self.in_front_of_constraints,
            self.safe_distance_constraints,
        ) = _relative_vehicle_constraints(
            self.ego_vehicle, self.other_vehicle, self.reference_lane, self.T
        )
        self.collision_avoidance_constraints = _collision_bounds(
            self.world, self.ego_vehicle, self.reference_lane, self.T
        )

        per_step = []
        for time_step in range(self.T):
            not_in_front = not_in_front_of_formula(
                self.in_front_of_constraints[time_step],
                S_INDEX,
                SIGNAL_DIM,
                self.ego_vehicle.shape.length,
                2.578,
            )
            not_in_same_lane = (
                None
                if self.in_same_lane_interval is None
                else outside_interval_formula(
                    self.in_same_lane_interval, S_INDEX, SIGNAL_DIM
                )
            )
            safe_distance = linearized_keeps_safe_distance_formula(
                self.safe_distance_constraints[time_step][0],
                self.safe_distance_constraints[time_step][1],
                S_INDEX,
                V_INDEX,
                SIGNAL_DIM,
                self.ego_vehicle.shape.length,
                2.578,
            )
            collision_free = collision_free_formula(
                self.collision_avoidance_constraints[time_step],
                S_INDEX,
                SIGNAL_DIM,
                self.ego_vehicle.shape.length,
                2.578,
            )

            # If d=0 is outside the target lane, the RG1 implication is already
            # true and disappears.  Otherwise only its longitudinal part remains.
            if not_in_same_lane is None:
                step_formula = collision_free
            else:
                step_formula = not_in_front | not_in_same_lane | safe_distance
                if collision_free is not None:
                    step_formula = step_formula & collision_free

            if step_formula is None:
                step_formula = no_backwards_driving(V_INDEX, SIGNAL_DIM)
            per_step.append(step_formula)

        rule_and_collision = _and_at_each_step(per_step, self.T)
        speed_bounds = _speed_bounds_formula(
            _effective_speed_limit(self.ego_vehicle, self.lanelet_network)
        ).always(0, final_step)
        spec = speed_bounds & rule_and_collision
        spec.name = "RG123_fixed_path"
        return spec


class RIN1FixedPath(_FixedPathMixin, RIN1):
    """Fixed-path version of the legacy stop-line MICP scenario."""

    def __init__(
        self, num_steps: int, world, ego_vehicle, lanelet_network, reference_lane=None
    ):
        super().__init__(
            T=num_steps,
            world=world,
            ego_vehicle=ego_vehicle,
            lanelet_network=lanelet_network,
        )
        self.reference_lane = (
            reference_lane or ego_vehicle.ref_path_lane or ego_vehicle.get_lane(0)
        )
        self._validate_num_steps()

    def obtain_stop_line_s(self):
        lanelets_with_stop_line = {
            lanelet.lanelet_id
            for lanelet in self.lanelet_network.lanelets
            if lanelet.stop_line is not None
        }
        intersection_lanelets = self.reference_lane.contained_lanelets.intersection(
            lanelets_with_stop_line
        )
        stop_line_positions = []
        for lanelet_id in intersection_lanelets:
            stop_line = self.lanelet_network.find_lanelet_by_id(
                lanelet_id
            ).stop_line
            start = np.asarray(stop_line.start, dtype=float)
            end = np.asarray(stop_line.end, dtype=float)
            # A stop-line endpoint can be millimetres outside the narrow
            # trajectory-CLCS projection domain.  The segment midpoint lies on
            # the lane path in the normal case and is the relevant fixed-path
            # crossing position.
            candidates = ((start + end) / 2.0, start, end)
            projected = []
            for point in candidates:
                try:
                    projected.append(
                        self.reference_lane.clcs.convert_to_curvilinear_coords(
                            *point
                        )[0]
                    )
                except Exception:
                    continue
            if not projected:
                raise ValueError(
                    f"Stop line on lanelet {lanelet_id} does not intersect the "
                    "fixed reference-path projection domain"
                )
            stop_line_positions.append(projected[0])
        return np.asarray(stop_line_positions, dtype=float)

    def GetSpecification(self):
        stop_lines = np.asarray(self.obtain_stop_line_s(), dtype=float).reshape(-1)
        if stop_lines.size != 1:
            raise ValueError(
                "Fixed-path R_IN1 currently requires exactly one stop line on the "
                f"reference path, found {stop_lines.size}"
            )
        self.stop_line_s = float(stop_lines[0] - self.ego_vehicle.shape.length / 2.0)

        final_step = self.T - 1
        self.collision_avoidance_constraints = _collision_bounds(
            self.world, self.ego_vehicle, self.reference_lane, self.T
        )

        # This is the intended legacy formula after removing the p_false dummy
        # branches.  The legacy helper called "phantom_false" is actually d<=0
        # and would become true at fixed d=0, making the stop-line rule vacuous.
        spec = None
        if self.T >= 2:
            stop_zone = (self.stop_line_s - 1.0, self.stop_line_s)
            in_stop_zone = inside_interval_formula(
                stop_zone, S_INDEX, SIGNAL_DIM
            )
            outside_stop_zone = outside_interval_formula(
                stop_zone, S_INDEX, SIGNAL_DIM
            )
            not_crossing = outside_stop_zone | in_stop_zone.eventually(1, 1)
            spec = not_crossing.always(0, self.T - 2)

        per_step = []
        for time_step in range(self.T):
            collision_free = collision_free_formula(
                self.collision_avoidance_constraints[time_step],
                S_INDEX,
                SIGNAL_DIM,
                self.ego_vehicle.shape.length,
                2.578,
            )
            no_backwards = no_backwards_driving(V_INDEX, SIGNAL_DIM)
            per_step.append(
                no_backwards
                if collision_free is None
                else collision_free & no_backwards
            )

        physical_constraints = _and_at_each_step(per_step, self.T)
        spec = physical_constraints if spec is None else spec & physical_constraints
        spec.name = "RIN1_fixed_path"
        return spec


class RIN4FixedPath(_FixedPathMixin, RIN4):
    """Fixed-path version of the legacy intersection-priority MICP scenario."""

    def __init__(
        self,
        num_steps: int,
        world,
        ego_vehicle,
        other_vehicle,
        lanelet_network,
        reference_lane=None,
    ):
        super().__init__(
            T=num_steps,
            world=world,
            ego_vehicle=ego_vehicle,
            other_vehicle=other_vehicle,
            lanelet_network=lanelet_network,
        )
        self.reference_lane = (
            reference_lane or ego_vehicle.ref_path_lane or ego_vehicle.get_lane(0)
        )
        self._validate_num_steps()

    def obtain_conflict_interval(self):
        lanelets_ego = self.get_intersection_lanelets(
            self.lanelet_network, self.ego_vehicle.lanelets_dir
        )
        lanelets_other = self.get_intersection_lanelets(
            self.lanelet_network, self.other_vehicle.lanelets_dir
        )
        conflict_cartesian = self.get_lanelet_union(lanelets_ego).intersection(
            self.get_lanelet_union(lanelets_other)
        )
        enlarged_polygon = conflict_cartesian.buffer(
            self.ego_vehicle.shape.length / 2.0
        )
        reference_path = getattr(self.reference_lane, "reference_path", None)
        if reference_path is None:
            reference_path = np.asarray(
                self.reference_lane.clcs.reference_path(), dtype=float
            )
        path_in_conflict = shapely.LineString(reference_path).intersection(
            enlarged_polygon
        )
        if path_in_conflict.is_empty:
            return None

        coordinates = []

        def collect_coordinates(geometry):
            if hasattr(geometry, "geoms"):
                for sub_geometry in geometry.geoms:
                    collect_coordinates(sub_geometry)
            elif hasattr(geometry, "coords"):
                coordinates.extend(geometry.coords)

        collect_coordinates(path_in_conflict)
        projected_s = [
            self.reference_lane.clcs.convert_to_curvilinear_coords(x, y)[0]
            for x, y in coordinates
        ]
        if not projected_s:
            return None
        return min(projected_s), max(projected_s)

    def GetSpecification(self):
        # The legacy implementation repeats this static geometry calculation T
        # times.  It has no time-dependent input, so compute it once.
        self.conflict_interval = self.obtain_conflict_interval()
        outside_conflict = (
            None
            if self.conflict_interval is None
            else outside_interval_formula(
                self.conflict_interval, S_INDEX, SIGNAL_DIM
            )
        )
        final_step = self.T - 1
        spec = (
            None
            if outside_conflict is None
            else outside_conflict.always(0, final_step)
        )

        self.collision_avoidance_constraints = _collision_bounds(
            self.world, self.ego_vehicle, self.reference_lane, self.T
        )
        per_step = []
        for time_step in range(self.T):
            collision_free = collision_free_formula(
                self.collision_avoidance_constraints[time_step],
                S_INDEX,
                SIGNAL_DIM,
                self.ego_vehicle.shape.length,
                2.578,
            )
            no_backwards = no_backwards_driving(V_INDEX, SIGNAL_DIM)
            per_step.append(
                no_backwards
                if collision_free is None
                else collision_free & no_backwards
            )
        physical_constraints = _and_at_each_step(per_step, self.T)
        spec = physical_constraints if spec is None else spec & physical_constraints
        spec.name = "RIN4_fixed_path"
        return spec
