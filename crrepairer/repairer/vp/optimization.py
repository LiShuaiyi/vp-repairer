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

    def _get_velocity_planning_current_conditions(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
    ):
        """Return the fixed state immediately before the VP decision horizon."""
        current_idx = int(self._tc - all_states[0].time_step)
        if current_idx < 0 or current_idx >= len(all_states):
            return None, None, None
        state = all_states[current_idx]
        current_s = trajectory_clcs.convert_to_curvilinear_coords(
            float(state.position[0]),
            float(state.position[1]),
        )[0]
        current_v = max(0.0, float(getattr(state, "velocity", 0.0)))
        current_a = float(getattr(state, "acceleration", 0.0))
        if not np.isfinite(current_a):
            current_a = 0.0
        return float(current_s), current_v, current_a

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

    def _build_constant_acceleration_reference(
        self,
        *,
        initial_s,
        initial_v,
        acceleration,
        horizon,
        dt,
        path_max,
    ):
        """Build the longitudinal reference used by one acceleration attempt.

        The earlier acceleration feasibility audit swept constant positive
        accelerations.  Keeping the same family here makes that audit an exact
        regression oracle, while the LP still enforces all position, speed,
        acceleration, jerk, and temporal constraints around the reference.
        """
        vehicle_v_max = float(self.config.vehicle.qp_veh_config.v_lon_max)
        s_values = []
        v_values = []
        s_prev = float(initial_s)
        v_prev = max(0.0, float(initial_v))
        path_exhausted = False
        for _ in range(int(horizon)):
            v_next = min(
                vehicle_v_max,
                v_prev + float(acceleration) * float(dt),
            )
            s_next = s_prev + 0.5 * (v_prev + v_next) * float(dt)
            if s_next > float(path_max) + 1e-9:
                path_exhausted = True
            s_next = min(s_next, float(path_max))
            s_values.append(float(s_next))
            v_values.append(float(v_next))
            s_prev, v_prev = s_next, v_next
        return {
            "acceleration": float(acceleration),
            "s": np.asarray(s_values, dtype=float),
            "v": np.asarray(v_values, dtype=float),
            "path_exhausted": bool(path_exhausted),
        }

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
        repair_mode="deceleration",
        initial_s=None,
        initial_v=None,
        initial_a=None,
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
        use_reference_deviation = repair_mode == "acceleration"
        n_delta_s = T if use_reference_deviation else 0
        n_delta_acceleration = T if use_reference_deviation else 0
        # One auxiliary covers the transition from the fixed acceleration at
        # tc to the first LP acceleration; the remaining auxiliaries cover
        # second velocity differences inside the planning horizon.
        n_delta_jerk = max(0, T - 1) if use_reference_deviation else 0
        n_x = (
            n_s
            + n_v
            + n_delta_s
            + n_delta_acceleration
            + n_delta_jerk
        )

        def s_idx(t):
            return t

        def v_idx(t):
            return n_s + t

        def delta_s_idx(t):
            return n_s + n_v + t

        def delta_jerk_idx(t):
            return n_s + n_v + n_delta_s + n_delta_acceleration + t

        def delta_acceleration_idx(t):
            return n_s + n_v + n_delta_s + t

        c = np.zeros(n_x)
        if use_reference_deviation:
            for t in range(T):
                # Acceleration uses the recorded trajectory projected onto its
                # route-extended CLCS.  Explicit exit bounds require the extra
                # progress, while this L1 objective keeps all other positions
                # as close to the original motion as possible.
                c[delta_s_idx(t)] = 1.0
        else:
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

        if initial_s is not None and initial_v is not None and T:
            # Connect the first modifiable state to the fixed state at tc.  In
            # particular, acceleration repair must not obtain an artificial
            # one-frame position or velocity jump before the LP horizon.
            row = np.zeros(n_x)
            row[s_idx(0)] = 1.0
            row[v_idx(0)] = -0.5 * dt
            A_eq.append(row)
            b_eq.append(float(initial_s) + 0.5 * float(initial_v) * dt)

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
        if initial_v is not None and T:
            row = np.zeros(n_x)
            row[v_idx(0)] = 1.0
            A_ub.append(row)
            b_ub.append(float(initial_v) + amax * dt)

            row = np.zeros(n_x)
            row[v_idx(0)] = -1.0
            A_ub.append(row)
            b_ub.append(-float(initial_v) - amin * dt)

            if initial_a is not None:
                row = np.zeros(n_x)
                row[v_idx(0)] = 1.0
                A_ub.append(row)
                b_ub.append(
                    float(initial_v) + float(initial_a) * dt + jmax * dt * dt
                )

                row = np.zeros(n_x)
                row[v_idx(0)] = -1.0
                A_ub.append(row)
                b_ub.append(
                    -float(initial_v) - float(initial_a) * dt - jmin * dt * dt
                )

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

        if use_reference_deviation:
            # Linearize |s - s_hat|.  Acceleration mode then advances only as
            # far as the exit constraint requires instead of maximizing speed
            # for the complete horizon.
            for t in range(T):
                row = np.zeros(n_x)
                row[s_idx(t)] = 1.0
                row[delta_s_idx(t)] = -1.0
                A_ub.append(row)
                b_ub.append(float(s_hat[t]))

                row = np.zeros(n_x)
                row[s_idx(t)] = -1.0
                row[delta_s_idx(t)] = -1.0
                A_ub.append(row)
                b_ub.append(-float(s_hat[t]))

            # Absolute velocity differences are proportional to absolute
            # longitudinal acceleration.  Include the transition from the
            # fixed state at tc so smoothing cannot hide a first-frame jump.
            for t in range(T):
                if t == 0:
                    if initial_v is None:
                        continue
                    previous_velocity = float(initial_v)
                    row = np.zeros(n_x)
                    row[v_idx(t)] = 1.0
                    row[delta_acceleration_idx(t)] = -1.0
                    A_ub.append(row)
                    b_ub.append(previous_velocity)

                    row = np.zeros(n_x)
                    row[v_idx(t)] = -1.0
                    row[delta_acceleration_idx(t)] = -1.0
                    A_ub.append(row)
                    b_ub.append(-previous_velocity)
                    continue

                row = np.zeros(n_x)
                row[v_idx(t)] = 1.0
                row[v_idx(t - 1)] = -1.0
                row[delta_acceleration_idx(t)] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

                row = np.zeros(n_x)
                row[v_idx(t)] = -1.0
                row[v_idx(t - 1)] = 1.0
                row[delta_acceleration_idx(t)] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

            # Auxiliary variables for the absolute second differences of
            # velocity.  A lexicographic second LP minimizes these values while
            # keeping the original-trajectory position objective near-optimal.
            if T >= 2 and initial_v is not None:
                initial_acceleration = float(initial_a or 0.0)
                expected_first_velocity = (
                    float(initial_v) + initial_acceleration * dt
                )
                row = np.zeros(n_x)
                row[v_idx(0)] = 1.0
                row[delta_jerk_idx(0)] = -1.0
                A_ub.append(row)
                b_ub.append(expected_first_velocity)

                row = np.zeros(n_x)
                row[v_idx(0)] = -1.0
                row[delta_jerk_idx(0)] = -1.0
                A_ub.append(row)
                b_ub.append(-expected_first_velocity)

            for t in range(T - 2):
                row = np.zeros(n_x)
                row[v_idx(t)] = 1.0
                row[v_idx(t + 1)] = -2.0
                row[v_idx(t + 2)] = 1.0
                row[delta_jerk_idx(t + 1)] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

                row = np.zeros(n_x)
                row[v_idx(t)] = -1.0
                row[v_idx(t + 1)] = 2.0
                row[v_idx(t + 2)] = -1.0
                row[delta_jerk_idx(t + 1)] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        bounds = []
        for t in range(T):
            lb = smin[t]
            if repair_mode == "deceleration":
                ub = min(smax[t], s_hat[t])
            elif repair_mode == "acceleration":
                ub = smax[t]
            else:
                raise ValueError(f"Unsupported VP repair mode: {repair_mode!r}")
            if lb > ub:
                abs_time_step = time_offset + t
                if repair_mode == "acceleration":
                    raise RuntimeError(
                        "Infeasible acceleration position bounds at "
                        f"time_step={abs_time_step}: smin={lb}, smax={ub}, "
                        f"s_hat={s_hat[t]}"
                    )
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
                if repair_mode == "acceleration":
                    raise RuntimeError(
                        "Infeasible acceleration velocity bounds at "
                        f"time_step={abs_time_step}: vmin={lb}, vmax={ub}"
                    )
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

        if use_reference_deviation:
            bounds.extend((0.0, None) for _ in range(T))
            bounds.extend((0.0, None) for _ in range(n_delta_acceleration))
            bounds.extend((0.0, None) for _ in range(n_delta_jerk))

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

        if use_reference_deviation and (n_delta_acceleration or n_delta_jerk):
            # Preserve closeness to the original trajectory as the primary
            # objective.  The small tolerance gives the secondary smoothness
            # solve room to remove bang-bang velocity changes without changing
            # which logical/exit constraints are satisfied.
            repair_config = self.config.repair
            primary_limit = float(result.fun) + max(
                float(
                    repair_config.acceleration_smoothing_position_absolute_tolerance
                ),
                float(
                    repair_config.acceleration_smoothing_position_relative_tolerance
                )
                * max(1.0, abs(float(result.fun))),
            )
            primary_row = np.zeros(n_x)
            for t in range(T):
                primary_row[delta_s_idx(t)] = 1.0
            secondary_A_ub = np.vstack((A_ub, primary_row))
            secondary_b_ub = np.append(b_ub, primary_limit)
            secondary_c = np.zeros(n_x)
            for t in range(n_delta_acceleration):
                secondary_c[delta_acceleration_idx(t)] = 1.0
            jerk_weight = float(
                repair_config.acceleration_smoothing_jerk_weight
            )
            for t in range(n_delta_jerk):
                secondary_c[delta_jerk_idx(t)] = jerk_weight
            secondary_result = linprog(
                c=secondary_c,
                A_ub=secondary_A_ub,
                b_ub=secondary_b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if secondary_result.success:
                result = secondary_result

        x = result.x
        s = x[:T]
        v = x[T : 2 * T]
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
            lateral_offset = (
                float(getattr(self, "_trajectory_clcs_lateral_offset", 0.0))
                if getattr(self, "_vp_repair_mode", "deceleration")
                == "acceleration"
                else 0.0
            )
            cart_pos = trajectory_clcs.convert_to_cartesian_coords(
                s_i, lateral_offset
            )
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
