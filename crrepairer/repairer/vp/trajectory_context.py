"""Trajectory and planner context helpers for velocity-planning repair."""

from typing import List, Tuple

import numpy as np

from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, TrajectoryPrediction
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams
from commonroad_clcs.util import resample_polyline


class VPTrajectoryContext:
    """Provides trajectory, CLCS, and mode-selection helpers shared by VP steps."""

    def _set_fixed_cutoff_time(self, time_step: int = 0):
        """Set the fixed VP cutoff time step when the default tc=0 should change."""
        self._tc = int(time_step)

    def _get_states_with_initial(self) -> List[CustomState]:
        """Return the ego initial state followed by predicted trajectory states."""
        states = [self.ego_vehicle.initial_state] + list(
            self.ego_vehicle.prediction.trajectory.state_list
        )
        self._validate_state_positions(states, context="ego trajectory")
        return states

    @staticmethod
    def _validate_state_positions(states: List[CustomState], context: str = "states"):
        for idx, state in enumerate(states):
            pos = np.asarray(getattr(state, "position", None), dtype=float).reshape(-1)
            if pos.size < 2:
                raise ValueError(
                    f"Invalid position shape in {context} at index {idx}, "
                    f"time_step={getattr(state, 'time_step', None)}: "
                    f"position={getattr(state, 'position', None)!r}"
                )

    def _get_lanelet_clcs_and_dt(self):
        return self._get_vp_lanelet_clcs(), self.config.scenario.dt

    def _get_vp_lanelet_clcs(self):
        world_ego = self.rule_monitor.world.vehicle_by_id(self.config.repair.ego_id)
        ref_path_lane = getattr(world_ego, "ref_path_lane", None)
        if ref_path_lane is not None and getattr(ref_path_lane, "clcs", None) is not None:
            return ref_path_lane.clcs

        candidate_times = [
            self._tc,
            getattr(world_ego, "start_time", None),
            getattr(world_ego, "end_time", None),
        ]
        for time_step in candidate_times:
            if time_step is None:
                continue
            try:
                lane = world_ego.get_lane(int(time_step))
            except Exception:
                lane = None
            if lane is not None and getattr(lane, "clcs", None) is not None:
                return lane.clcs
        raise RuntimeError("No CLCS available for VP lanelet constraints.")

    def _get_dt(self) -> float:
        return self._get_lanelet_clcs_and_dt()[1]

    @staticmethod
    def convert_traj_to_ego_vehicle(
        shape,
        initial_state,
        cr_trajectory: Trajectory,
        vehicle_id: int = 0,
    ) -> DynamicObstacle:
        """Wrap a repaired CommonRoad trajectory as a DynamicObstacle for visualization."""
        return DynamicObstacle(
            obstacle_id=vehicle_id,
            obstacle_type=ObstacleType.CAR,
            prediction=TrajectoryPrediction(cr_trajectory, shape),
            obstacle_shape=shape,
            initial_state=initial_state,
        )

    @staticmethod
    def _max_discrete_curvature(path: np.ndarray) -> float:
        """Estimate maximum three-point curvature on a sanitized path."""
        path = np.asarray(path, dtype=float)
        if len(path) < 3:
            return 0.0
        first = path[1:-1] - path[:-2]
        second = path[2:] - path[1:-1]
        chord = path[2:] - path[:-2]
        denominator = (
            np.linalg.norm(first, axis=1)
            * np.linalg.norm(second, axis=1)
            * np.linalg.norm(chord, axis=1)
        )
        cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        curvature = np.divide(
            2.0 * cross,
            denominator,
            out=np.full_like(cross, np.inf),
            where=denominator > 1e-12,
        )
        return float(np.max(curvature, initial=0.0))

    @staticmethod
    def _build_stable_reference_paths(
        source_path: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the source and its one-segment endpoint extensions at 0.1 m."""
        start_endpoint = 2.0 * source_path[0] - source_path[1]
        end_endpoint = 2.0 * source_path[-1] - source_path[-2]
        extended_source_path = np.vstack(
            (start_endpoint, source_path, end_endpoint)
        )
        return (
            resample_polyline(source_path, step=0.1),
            resample_polyline(extended_source_path, step=0.1),
        )

    def _build_trajectory_clcs(
        self,
        all_states: List[CustomState],
        resampling_factor: int = 10,
        num_extend_pts: int = 10,
        position_debounce: float = 1e-2,
    ) -> Tuple[CurvilinearCoordinateSystem, np.ndarray]:
        """Build a trajectory-aligned CLCS used by the VP longitudinal planner."""
        source_path = []
        for i, state in enumerate(all_states):
            pos = np.asarray(state.position, dtype=float).reshape(-1)
            if pos.size < 2:
                raise ValueError(
                    f"Invalid ref_path source position at state index {i}, "
                    f"time_step={state.time_step}: position={state.position!r}"
                )
            pos = pos[:2]
            if not np.all(np.isfinite(pos)):
                raise ValueError(
                    f"Non-finite ref_path source position at state index {i}, "
                    f"time_step={state.time_step}: position={state.position!r}"
                )
            # Recorded trajectories often jitter by fractions of a millimetre
            # after the vehicle has stopped.  Treat displacement relative to
            # the last accepted point as accumulated motion: genuine slow
            # travel eventually exceeds this threshold, while stop jitter
            # cannot introduce tiny backwards segments into the CLCS path.
            if (
                source_path
                and np.linalg.norm(pos - source_path[-1]) < position_debounce
            ):
                continue
            if source_path:
                displacement = pos - source_path[-1]
                heading = float(getattr(state, "orientation", 0.0))
                forward = np.array([np.cos(heading), np.sin(heading)])
                if float(np.dot(displacement, forward)) <= 0.0:
                    continue
            source_path.append(pos)

        if len(source_path) < 2:
            raise ValueError(
                "Cannot build VP trajectory CLCS from fewer than two distinct positions."
            )

        source_path = np.asarray(source_path, dtype=float)
        # Most recorded trajectories are already clean enough for a direct
        # CLCS.  Start with that inexpensive representation and fall back to
        # fixed-step preprocessing only when its curvature is unreliable or a
        # repair attempt later proves sensitive to the parameterization.
        discrete_curvature = self._max_discrete_curvature(source_path)
        trajectory_clcs = None
        repair_config = getattr(getattr(self, "config", None), "repair", None)
        repair_rules = set(getattr(repair_config, "rules", ()))
        intersection_rule = any(rule.startswith("R_IN") for rule in repair_rules)
        fast_path_allowed = (
            not intersection_rule
            and not getattr(self, "_force_trajectory_clcs_preprocess", False)
        )

        if fast_path_allowed:
            ref_path = []
            for i in range(len(source_path) - 1):
                pos = source_path[i]
                next_pos = source_path[i + 1]
                delta = (next_pos - pos) / resampling_factor
                for j in range(resampling_factor):
                    ref_path.append(pos + j * delta)
            ref_path.append(source_path[-1])
            ref_path = np.asarray(ref_path, dtype=float)
            start_dir = ref_path[0] - ref_path[1]
            end_dir = ref_path[-1] - ref_path[-2]
            start_extend = np.array(
                [ref_path[0] + start_dir * i for i in range(num_extend_pts, 0, -1)],
                dtype=float,
            )
            end_extend = np.array(
                [ref_path[-1] + end_dir * i for i in range(1, num_extend_pts + 1)],
                dtype=float,
            )
            processed_ref_path = np.vstack((start_extend, ref_path, end_extend))
        else:
            # The stable branch immediately needs a uniformly sampled 0.1 m
            # path.  Feeding it the former factor-10 interpolation only makes
            # the same piecewise-linear geometry denser before it is sampled
            # again.  Resample the sanitized source path directly instead.
            ref_path, processed_ref_path = self._build_stable_reference_paths(
                source_path
            )

        try:
            if not fast_path_allowed:
                raise ValueError(
                    "trajectory CLCS fast path is disabled for this rule family"
                )
            candidate = CurvilinearCoordinateSystem(
                reference_path=processed_ref_path,
                params=CLCSParams(),
                preprocess_path=False,
            )
            curvature_min, curvature_max = candidate.curvature_range(
                0.0, float(candidate.length())
            )
            candidate_curvature = max(
                abs(float(curvature_min)), abs(float(curvature_max))
            )
            curvature_is_reliable = (
                np.isfinite(candidate_curvature)
                and np.isfinite(discrete_curvature)
                and candidate_curvature <= 1.0
                and candidate_curvature + 1e-6
                >= 0.95 * discrete_curvature
            )
            if curvature_is_reliable:
                trajectory_clcs = candidate
                self._trajectory_clcs_preprocessed = False
        except Exception:
            trajectory_clcs = None

        if trajectory_clcs is None:
            # If the optimistic RG path fell back after it was constructed,
            # replace its factor-10 sampling with the stable 0.1 m sampling.
            if fast_path_allowed:
                ref_path, processed_ref_path = self._build_stable_reference_paths(
                    source_path
                )
            trajectory_clcs = CurvilinearCoordinateSystem(
                reference_path=processed_ref_path,
                params=CLCSParams(),
                preprocess_path=False,
                validity_checks=False,
            )
            self._trajectory_clcs_preprocessed = True
        return trajectory_clcs, ref_path

    def _retry_with_preprocessed_trajectory_clcs(self) -> bool:
        """Switch one repair call from the optimistic CLCS to the stable path."""
        if getattr(self, "_trajectory_clcs_preprocessed", True):
            return False
        if getattr(self, "_force_trajectory_clcs_preprocess", False):
            return False
        self._force_trajectory_clcs_preprocess = True
        self._shared_trajectory_clcs = None
        return True

    def _build_acceleration_trajectory_clcs(
        self,
        all_states: List[CustomState],
        position_debounce: float = 1e-2,
        continuation_step: float = 0.1,
        lateral_blend_distance: float = 5.0,
        stationary_span: float = 0.5,
    ) -> Tuple[CurvilinearCoordinateSystem, np.ndarray]:
        """Extend the recorded ego path smoothly along its remaining route.

        Acceleration repair can move beyond the finite recorded trajectory, but
        replacing that trajectory with the lane centerline also changes its
        already-observed geometry.  Preserve every reliable original point and
        append samples from the complete route CLCS.  The last recorded lateral
        offset decays smoothly to the route center over a short transition.
        """
        source_path = []
        for state in all_states:
            position = np.asarray(state.position, dtype=float).reshape(-1)[:2]
            if position.size < 2 or not np.all(np.isfinite(position)):
                raise ValueError(
                    "Acceleration reference contains an invalid ego position."
                )
            if (
                source_path
                and np.linalg.norm(position - source_path[-1])
                < position_debounce
            ):
                continue
            if source_path:
                displacement = position - source_path[-1]
                heading = float(getattr(state, "orientation", 0.0))
                forward = np.array([np.cos(heading), np.sin(heading)])
                if float(np.dot(displacement, forward)) <= 0.0:
                    continue
            source_path.append(position)

        source_path = np.asarray(source_path, dtype=float)
        if len(source_path) == 0:
            raise ValueError("Acceleration reference contains no ego positions.")
        if (
            len(source_path) < 2
            or float(np.max(np.linalg.norm(source_path - source_path[0], axis=1)))
            < stationary_span
        ):
            # A stopped vehicle often has centimetre-scale position jitter.
            # Treat that as a single observed position instead of turning the
            # noise into a high-curvature path tangent.
            current_idx = int(self._tc - all_states[0].time_step)
            current_position = np.asarray(
                all_states[current_idx].position, dtype=float
            ).reshape(-1)[:2]
            source_path = current_position[None, :]

        world_ego = self.rule_monitor.world.vehicle_by_id(
            self.config.repair.ego_id
        )
        route_clcs = world_ego.ref_path_lane.clcs
        projected = np.asarray(
            [
                route_clcs.convert_to_curvilinear_coords(
                    float(point[0]), float(point[1])
                )
                for point in source_path
            ],
            dtype=float,
        )
        nonzero_progress = np.diff(projected[:, 0])
        nonzero_progress = nonzero_progress[np.abs(nonzero_progress) > 1e-4]
        last_s = float(projected[-1, 0])
        last_d = float(projected[-1, 1])
        if len(nonzero_progress):
            direction = 1.0 if float(np.median(nonzero_progress)) >= 0.0 else -1.0
        else:
            ds = min(0.5, 0.25 * float(route_clcs.length()))
            s_before = max(0.0, last_s - ds)
            s_after = min(float(route_clcs.length()), last_s + ds)
            point_before = np.asarray(
                route_clcs.convert_to_cartesian_coords(s_before, last_d),
                dtype=float,
            )
            point_after = np.asarray(
                route_clcs.convert_to_cartesian_coords(s_after, last_d),
                dtype=float,
            )
            route_tangent = point_after - point_before
            current_idx = int(self._tc - all_states[0].time_step)
            heading = float(getattr(all_states[current_idx], "orientation", 0.0))
            forward = np.array([np.cos(heading), np.sin(heading)])
            direction = 1.0 if float(np.dot(route_tangent, forward)) >= 0.0 else -1.0
        route_end = float(route_clcs.length()) if direction > 0.0 else 0.0
        remaining_length = direction * (route_end - last_s)
        current_idx = int(self._tc - all_states[0].time_step)
        current_velocity = max(
            0.0, float(getattr(all_states[current_idx], "velocity", 0.0))
        )
        horizon_seconds = max(
            0.0,
            (all_states[-1].time_step - int(self._tc))
            * float(self.config.scenario.dt),
        )
        qp_config = self.config.vehicle.qp_veh_config
        maximum_travel = min(
            float(qp_config.v_lon_max) * horizon_seconds,
            current_velocity * horizon_seconds
            + 0.5 * float(qp_config.a_lon_max) * horizon_seconds**2,
        )
        remaining_length = min(
            remaining_length,
            max(30.0, maximum_travel + 15.0),
        )
        if remaining_length <= continuation_step:
            raise ValueError(
                "Acceleration reference route has no usable continuation "
                "after the recorded trajectory."
            )

        continuation_distances = np.arange(
            continuation_step,
            remaining_length,
            continuation_step,
            dtype=float,
        )
        continuation = []
        for distance in continuation_distances:
            blend_u = min(1.0, distance / lateral_blend_distance)
            smoothstep = blend_u * blend_u * (3.0 - 2.0 * blend_u)
            lateral_offset = last_d * (1.0 - smoothstep)
            route_s = last_s + direction * float(distance)
            continuation.append(
                route_clcs.convert_to_cartesian_coords(route_s, lateral_offset)
            )
        if not continuation:
            raise ValueError(
                "Acceleration reference route continuation contains no samples."
            )

        extended_source_path = np.vstack(
            (source_path, np.asarray(continuation, dtype=float))
        )
        ref_path, processed_ref_path = self._build_stable_reference_paths(
            extended_source_path
        )
        trajectory_clcs = CurvilinearCoordinateSystem(
            reference_path=processed_ref_path,
            params=CLCSParams(),
            preprocess_path=False,
            validity_checks=False,
        )
        # The extended path already passes through the original trajectory;
        # unlike the lane-centerline implementation it needs no fixed offset.
        self._trajectory_clcs_lateral_offset = 0.0
        self._trajectory_clcs_preprocessed = True
        return trajectory_clcs, ref_path

    def _get_shared_trajectory_clcs(
        self,
        all_states: List[CustomState],
    ) -> Tuple[CurvilinearCoordinateSystem, np.ndarray]:
        """Return the trajectory CLCS shared by estimation and VP candidates.

        The ego trajectory is immutable during one repair call.  Rebuilding
        and preprocessing the same CLCS for DomainDPLL and again for every SAT
        candidate therefore adds cost without changing the planning geometry.
        Use the planner-resolution path for both consumers and cache it on the
        repairer instance.
        """
        cached = getattr(self, "_shared_trajectory_clcs", None)
        if cached is None:
            if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
                if getattr(
                    self.config.repair,
                    "extend_acceleration_reference_path",
                    True,
                ):
                    cached = self._build_acceleration_trajectory_clcs(all_states)
                else:
                    # Ablation: use the complete route CLCS without building a
                    # trajectory-aligned extension of the recorded ego path.
                    world_ego = self.rule_monitor.world.vehicle_by_id(
                        self.config.repair.ego_id
                    )
                    lane = world_ego.ref_path_lane
                    ref_path = np.asarray(lane.center_vertices, dtype=float)
                    if ref_path.ndim != 2 or len(ref_path) < 2:
                        raise ValueError(
                            "Acceleration reference lane has fewer than two points."
                        )
                    current_idx = int(self._tc - all_states[0].time_step)
                    current_state = all_states[current_idx]
                    current_ct = lane.clcs.convert_to_curvilinear_coords(
                        float(current_state.position[0]),
                        float(current_state.position[1]),
                    )
                    self._trajectory_clcs_lateral_offset = float(current_ct[1])
                    self._trajectory_clcs_preprocessed = True
                    cached = lane.clcs, ref_path
            else:
                self._trajectory_clcs_lateral_offset = 0.0
                cached = self._build_trajectory_clcs(all_states)
            self._shared_trajectory_clcs = cached
        return cached

    def _convert_states_to_clcs(
        self,
        all_states: List[CustomState],
        clcs: CurvilinearCoordinateSystem,
    ) -> List[np.ndarray]:
        cl_states = []
        for state in all_states:
            cl_states.append(
                clcs.convert_to_curvilinear_coords(
                    float(state.position[0]),
                    float(state.position[1]),
                )
            )
        return cl_states

    def _extract_constraints_from_corridor(self):
        raise NotImplementedError(
            "VP planner no longer depends on QP/MIQP planner corridor extraction. "
            "Use constraint_mode == 1 or implement a VP-native corridor extractor."
        )
