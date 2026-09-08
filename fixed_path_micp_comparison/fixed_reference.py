"""Reference-lane adapters for the standalone fixed-path MICP comparison."""

from __future__ import annotations

import numpy as np
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams
from commonroad_clcs.util import resample_polyline


class FixedReferenceLane:
    """Minimal lane-like wrapper around a trajectory-aligned CLCS.

    crmonitor's vehicle projection cache accepts any object exposing ``clcs``,
    ``clcs_large_step``, and ``orientation(s)``.  Keeping the route's contained
    lanelets also lets intersection stop lines be selected exactly as before.
    """

    def __init__(self, clcs, contained_lanelets=(), reference_path=None):
        self.clcs = clcs
        self.clcs_large_step = clcs
        self.contained_lanelets = frozenset(contained_lanelets)
        self.reference_path = (
            None
            if reference_path is None
            else np.asarray(reference_path, dtype=float)
        )
        # CurvilinearStateManager uses lane_id as part of its cache key.
        self.lane_id = ("fixed_reference", id(self))
        projection_domain = np.asarray(
            self.clcs.curvilinear_projection_domain(), dtype=float
        )
        self.s_min = float(np.min(projection_domain[:, 0]))
        self.s_max = float(np.max(projection_domain[:, 0]))

    def orientation(self, s: float) -> float:
        s = float(s)
        tangent = np.asarray(self.clcs.tangent(s), dtype=float).reshape(-1)
        if tangent.size < 2 or np.linalg.norm(tangent[:2]) <= 1e-9:
            raise ValueError(f"Cannot determine reference-path orientation at s={s}")
        return float(np.arctan2(tangent[1], tangent[0]))


def build_trajectory_reference_lane(
    ego_obstacle, world_ego, rule: str, dt: float = 0.2
):
    """Build VP's stable trajectory-aligned reference representation.

    This mirrors the stable branch in ``VPTrajectoryContext`` without importing
    the repairer package (whose package initializer also loads the much heavier
    reachability backend).  ``rule`` is accepted so the batch API records which
    rule the path belongs to; the stable geometry itself is rule-independent.
    """

    states = [ego_obstacle.initial_state] + list(
        ego_obstacle.prediction.trajectory.state_list
    )
    del rule
    source_path = []
    for state in states:
        position = np.asarray(state.position, dtype=float).reshape(-1)[:2]
        if position.size < 2 or not np.all(np.isfinite(position)):
            raise ValueError("Trajectory reference contains an invalid ego position")
        if source_path and np.linalg.norm(position - source_path[-1]) < 1e-2:
            continue
        if source_path:
            displacement = position - source_path[-1]
            heading = float(getattr(state, "orientation", 0.0))
            forward = np.array([np.cos(heading), np.sin(heading)])
            if float(np.dot(displacement, forward)) <= 0.0:
                continue
        source_path.append(position)

    if len(source_path) < 2:
        raise ValueError(
            "Cannot build a trajectory reference from fewer than two distinct "
            "forward positions"
        )
    source_path = np.asarray(source_path, dtype=float)

    # Extend the recorded path along its already-selected route.  A fixed path
    # still has to cover positions reachable by a repaired (especially faster)
    # trajectory and the future positions of relevant vehicles.  This follows
    # VP's acceleration-reference construction: preserve the recorded geometry,
    # then blend its last route-relative offset smoothly to the route center.
    route_lane = world_ego.ref_path_lane or world_ego.get_lane(0)
    if route_lane is None:
        raise ValueError("Ego vehicle has no route or lane reference CLCS")
    route_clcs = route_lane.clcs
    route_projection = np.asarray(
        [
            route_clcs.convert_to_curvilinear_coords(float(x), float(y))
            for x, y in source_path
        ],
        dtype=float,
    )
    progress = np.diff(route_projection[:, 0])
    progress = progress[np.abs(progress) > 1e-4]
    direction = 1.0 if not len(progress) or np.median(progress) >= 0.0 else -1.0
    last_s = float(route_projection[-1, 0])
    last_d = float(route_projection[-1, 1])
    route_domain = np.asarray(route_clcs.curvilinear_projection_domain(), dtype=float)
    route_s_min = float(np.min(route_domain[:, 0]))
    route_s_max = float(np.max(route_domain[:, 0]))
    route_end = route_s_max if direction > 0.0 else route_s_min
    route_remaining = direction * (route_end - last_s)
    horizon_seconds = max(
        0.0,
        float(states[-1].time_step - states[0].time_step) * float(dt),
    )
    observed_max_velocity = max(
        max(0.0, float(getattr(state, "velocity", 0.0))) for state in states
    )
    wanted_extension = max(30.0, 1.5 * observed_max_velocity * horizon_seconds + 15.0)
    extension_length = min(route_remaining, wanted_extension)
    if extension_length > 0.1:
        continuation = []
        for distance in np.arange(0.1, extension_length, 0.1):
            blend_u = min(1.0, float(distance) / 5.0)
            smoothstep = blend_u * blend_u * (3.0 - 2.0 * blend_u)
            lateral_offset = last_d * (1.0 - smoothstep)
            route_s = last_s + direction * float(distance)
            try:
                point = route_clcs.convert_to_cartesian_coords(
                    route_s, lateral_offset
                )
            except Exception:
                try:
                    point = route_clcs.convert_to_cartesian_coords(route_s, 0.0)
                except Exception:
                    break
            continuation.append(point)
        if continuation:
            source_path = np.vstack(
                (source_path, np.asarray(continuation, dtype=float))
            )

    start_endpoint = 2.0 * source_path[0] - source_path[1]
    end_endpoint = 2.0 * source_path[-1] - source_path[-2]
    extended_source_path = np.vstack(
        (start_endpoint, source_path, end_endpoint)
    )
    processed_reference = resample_polyline(extended_source_path, step=0.1)
    clcs = CurvilinearCoordinateSystem(
        reference_path=processed_reference,
        params=CLCSParams(),
        preprocess_path=False,
        validity_checks=False,
    )
    contained_lanelets = getattr(route_lane, "contained_lanelets", ())
    return FixedReferenceLane(clcs, contained_lanelets, processed_reference)
