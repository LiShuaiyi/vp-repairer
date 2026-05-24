"""LP solving and repaired-trajectory construction for velocity planning."""

import math
from typing import List

import numpy as np
from scipy.optimize import linprog

from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from commonroad_clcs.clcs import CurvilinearCoordinateSystem


class VPOptimization:
    """Solves the velocity-planning optimization problem and reconstructs trajectories."""

    def _build_reference_longitudinal_positions(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
    ) -> np.ndarray:
        """Project the original ego trajectory to longitudinal reference positions."""
        s_hat = np.zeros(all_states[-1].time_step - int(self._tc))
        for time_step in range(int(self._tc) + 1, all_states[-1].time_step + 1):
            state = all_states[time_step - all_states[0].time_step]
            ct_pos = trajectory_clcs.convert_to_curvilinear_coords(
                float(state.position[0]),
                float(state.position[1]),
            )
            s_hat[time_step - int(self._tc) - 1] = ct_pos[0]
        return s_hat

    def _get_velocity_planning_initial_conditions(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
    ):
        first_plan_idx = int(self._tc - all_states[0].time_step) + 1
        if first_plan_idx >= len(all_states):
            return None, None
        state = all_states[first_plan_idx]
        s0 = trajectory_clcs.convert_to_curvilinear_coords(
            float(state.position[0]),
            float(state.position[1]),
        )[0]
        v0 = state.velocity
        return s0, v0

    @staticmethod
    def _initial_conditions_within_bounds(s0, v0, smin, smax, vmin, vmax) -> bool:
        if s0 is None or v0 is None:
            return False
        if len(smin) == 0 or len(vmin) == 0:
            return False
        return (
            float(smin[0]) <= float(s0) <= float(smax[0])
            and float(vmin[0]) <= float(v0) <= float(vmax[0])
        )

    def _get_longitudinal_planning_limits(self):
        qp_veh_config = self.config.vehicle.qp_veh_config
        return (
            qp_veh_config.a_lon_min,
            qp_veh_config.a_lon_max,
            qp_veh_config.j_lon_min,
            qp_veh_config.j_lon_max,
        )

    def _get_longitudinal_reachability_limits(self):
        qp_veh_config = self.config.vehicle.qp_veh_config
        return (
            qp_veh_config.a_lon_max,
            qp_veh_config.a_lon_min,
            qp_veh_config.v_lon_max,
        )

    def _solve_velocity_planning_lp(
        self,
        dt,
        s_hat,
        vmin,
        vmax,
        smin,
        smax,
        amin,
        amax,
        jmin,
        jmax,
        s0=None,
        v0=None,
        time_offset=0,
    ):
        """Solve the longitudinal LP under position, velocity, acceleration, and jerk bounds."""
        s_hat = np.asarray(s_hat, dtype=float)
        vmin = np.asarray(vmin, dtype=float)
        vmax = np.asarray(vmax, dtype=float)
        smin = np.asarray(smin, dtype=float)
        smax = np.asarray(smax, dtype=float)

        T = len(s_hat)
        n_s = T
        n_v = T
        n_x = n_s + n_v

        def s_idx(t):
            return t

        def v_idx(t):
            return n_s + t

        c = np.zeros(n_x)
        for t in range(T):
            c[s_idx(t)] = -1.0

        A_eq = []
        b_eq = []
        for t in range(T - 1):
            row = np.zeros(n_x)
            row[s_idx(t + 1)] = 1.0
            row[s_idx(t)] = -1.0
            row[v_idx(t)] = -0.5 * dt
            row[v_idx(t + 1)] = -0.5 * dt
            A_eq.append(row)
            b_eq.append(0.0)

        if s0 is not None:
            row = np.zeros(n_x)
            row[s_idx(0)] = 1.0
            A_eq.append(row)
            b_eq.append(float(s0))

        if v0 is not None:
            row = np.zeros(n_x)
            row[v_idx(0)] = 1.0
            A_eq.append(row)
            b_eq.append(float(v0))

        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None

        A_ub = []
        b_ub = []
        for t in range(T - 1):
            row = np.zeros(n_x)
            row[v_idx(t + 1)] = 1.0
            row[v_idx(t)] = -1.0
            A_ub.append(row)
            b_ub.append(amax * dt)

            row = np.zeros(n_x)
            row[v_idx(t + 1)] = -1.0
            row[v_idx(t)] = 1.0
            A_ub.append(row)
            b_ub.append(-amin * dt)

        for t in range(T - 2):
            row = np.zeros(n_x)
            row[v_idx(t)] = 1.0
            row[v_idx(t + 1)] = -2.0
            row[v_idx(t + 2)] = 1.0
            A_ub.append(row)
            b_ub.append(jmax * dt * dt)

            row = np.zeros(n_x)
            row[v_idx(t)] = -1.0
            row[v_idx(t + 1)] = 2.0
            row[v_idx(t + 2)] = -1.0
            A_ub.append(row)
            b_ub.append(-jmin * dt * dt)

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        bounds = []
        for t in range(T):
            lb = smin[t]
            ub = min(smax[t], s_hat[t])
            if lb > ub:
                abs_time_step = time_offset + t
                lb = 0
                print(
                    "* \t<VPRepairer>: infeasible longitudinal position bound details: "
                    f"lp_index={t}, abs_time_step={abs_time_step}, "
                    f"smin={lb}, smax={ub}, s_hat={s_hat[t]}"
                )
                # raise ValueError(
                #     f"Infeasible velocity bounds at t={t}: vmin={lb}, vmax={ub}"
                # )
            bounds.append((lb, ub))

        for t in range(T):
            lb = vmin[t]
            ub = vmax[t]
            if lb > ub:
                abs_time_step = time_offset + t
                lb = 0
                print(
                    "* \t<VPRepairer>: infeasible velocity bound details: "
                    f"lp_index={t}, abs_time_step={abs_time_step}, "
                    f"vmin={lb}, vmax={ub}"
                )
                # raise ValueError(
                #     f"Infeasible velocity bounds at t={t}: vmin={lb}, vmax={ub}"
                # )
            bounds.append((lb, ub))

        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")

        x = result.x
        s = x[:T]
        v = x[T:]
        return {
            "s": s,
            "v": v,
            "objective_min_sum_s_hat_minus_s": np.sum(s_hat - s),
            "raw_result": result,
        }

    def _build_repaired_trajectory(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
        s: np.ndarray,
        v: np.ndarray,
        dt: float,
    ) -> Trajectory:
        repaired_state_list = []
        start_idx = int(self._tc - all_states[0].time_step)

        for i in range(start_idx + 1):
            state = all_states[i]
            repaired_state_list.append(self._copy_state(state))

        for i in range(len(s)):
            s_i = float(s[i])
            v_i = float(v[i])
            cart_pos = trajectory_clcs.convert_to_cartesian_coords(s_i, 0.0)
            prev_state = repaired_state_list[-1]
            orientation = self._orientation_from_trajectory_clcs(
                trajectory_clcs,
                s_i,
                fallback=getattr(prev_state, "orientation", 0.0),
            )

            if len(v) == 1:
                acceleration = 0.0
            elif i < len(v) - 1:
                acceleration = float((v[i + 1] - v[i]) / dt)
            else:
                acceleration = float((v[i] - v[i - 1]) / dt)

            repaired_state_list.append(
                CustomState(
                    time_step=int(self._tc) + i + 1,
                    position=np.asarray(cart_pos, dtype=float),
                    velocity=v_i,
                    orientation=orientation,
                    acceleration=acceleration,
                )
            )

        return Trajectory(all_states[0].time_step, repaired_state_list)

    @staticmethod
    def _orientation_from_trajectory_clcs(
        trajectory_clcs: CurvilinearCoordinateSystem,
        s: float,
        fallback: float = 0.0,
    ) -> float:
        ds_candidates = (0.25, 0.1, 0.05)
        for ds in ds_candidates:
            try:
                p_prev = np.asarray(
                    trajectory_clcs.convert_to_cartesian_coords(s - ds, 0.0),
                    dtype=float,
                )
                p_next = np.asarray(
                    trajectory_clcs.convert_to_cartesian_coords(s + ds, 0.0),
                    dtype=float,
                )
            except Exception:
                continue
            delta = p_next - p_prev
            if np.linalg.norm(delta) > 1e-6:
                return math.atan2(delta[1], delta[0])
        return fallback

    def _copy_state(self, state) -> CustomState:
        return CustomState(
            time_step=state.time_step,
            position=np.asarray(state.position, dtype=float),
            velocity=getattr(state, "velocity", 0.0),
            orientation=getattr(state, "orientation", 0.0),
            acceleration=getattr(state, "acceleration", 0.0),
        )
