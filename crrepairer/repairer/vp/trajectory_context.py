"""Trajectory and planner context helpers for velocity-planning repair."""

from typing import List, Tuple

import numpy as np

from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, TrajectoryPrediction
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams, ProcessingOption, ResamplingOption


class VPTrajectoryContext:
    def _uses_in_series_processing(self) -> bool:
        return any(rule.startswith("R_IN") for rule in self.config.repair.rules)

    def _uses_rg3_specific_second_lp(self) -> bool:
        return "R_G3" in self.config.repair.rules

    def _should_pin_velocity_planning_initial_conditions(self) -> bool:
        return "R_G3" in self.config.repair.rules

    def _prepare_fixed_cutoff_time(self):
        self._tc = self._vp_tc_time_step

    def _get_states_with_initial(self) -> List[CustomState]:
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
            self._vp_tc_time_step,
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
        return DynamicObstacle(
            obstacle_id=vehicle_id,
            obstacle_type=ObstacleType.CAR,
            prediction=TrajectoryPrediction(cr_trajectory, shape),
            obstacle_shape=shape,
            initial_state=initial_state,
        )

    def _build_trajectory_clcs(
        self,
        all_states: List[CustomState],
        resampling_factor: int = 10,
        num_extend_pts: int = 10,
    ) -> Tuple[CurvilinearCoordinateSystem, np.ndarray]:
        ref_path = []
        for i in range(len(all_states) - 1):
            pos = np.asarray(all_states[i].position, dtype=float).reshape(-1)
            next_pos = np.asarray(all_states[i + 1].position, dtype=float).reshape(-1)
            if pos.size < 2:
                raise ValueError(
                    f"Invalid ref_path source position at state index {i}, "
                    f"time_step={all_states[i].time_step}: position={all_states[i].position!r}"
                )
            if next_pos.size < 2:
                raise ValueError(
                    f"Invalid ref_path source position at state index {i + 1}, "
                    f"time_step={all_states[i + 1].time_step}: position={all_states[i + 1].position!r}"
                )
            pos = pos[:2]
            next_pos = next_pos[:2]
            delta = (next_pos - pos) / resampling_factor
            for j in range(resampling_factor):
                ref_path.append(pos + j * delta)
        last_pos = np.asarray(all_states[-1].position, dtype=float).reshape(-1)
        if last_pos.size < 2:
            raise ValueError(
                f"Invalid ref_path source position at final state index {len(all_states) - 1}, "
                f"time_step={all_states[-1].time_step}: position={all_states[-1].position!r}"
            )
        ref_path.append(last_pos[:2])
        ref_path = np.asarray(ref_path, dtype=float)

        params = CLCSParams()
        params.processing_option = ProcessingOption.SPLINE_SMOOTHING
        params.resampling.option = ResamplingOption.ADAPTIVE

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

        trajectory_clcs = CurvilinearCoordinateSystem(
            reference_path=processed_ref_path,
            params=params,
            preprocess_path=False,
        )
        return trajectory_clcs, ref_path

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
