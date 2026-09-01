"""LP solving and repaired-trajectory construction for velocity planning."""

import math
from typing import List

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack as sparse_vstack

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
        cache = getattr(self, "_reference_longitudinal_positions_cache", None)
        if cache is None:
            cache = {}
            self._reference_longitudinal_positions_cache = cache
        key = (
            id(trajectory_clcs),
            int(self._tc),
            int(all_states[0].time_step),
            int(all_states[-1].time_step),
            len(all_states),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached.copy()

        s_hat = np.zeros(all_states[-1].time_step - int(self._tc))
        for time_step in range(int(self._tc) + 1, all_states[-1].time_step + 1):
            state = all_states[time_step - all_states[0].time_step]
            ct_pos = trajectory_clcs.convert_to_curvilinear_coords(
                float(state.position[0]),
                float(state.position[1]),
            )
            s_hat[time_step - int(self._tc) - 1] = ct_pos[0]
        cache[key] = s_hat.copy()
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

    def _build_acceleration_lp_template(
        self,
        dt,
        s_hat,
        amin,
        amax,
        jmin,
        jmax,
        s0,
        v0,
        initial_s,
        initial_v,
        initial_a,
    ):
        """Build and cache the acceleration LP matrices shared by SAT candidates.

        Candidate-specific longitudinal and velocity requirements are variable
        bounds, not matrix coefficients.  The dynamics, absolute-deviation,
        acceleration, and jerk rows therefore remain identical throughout one
        acceleration phase and only need to be assembled once.
        """
        s_hat = np.asarray(s_hat, dtype=float)
        T = len(s_hat)
        n_s = n_v = T
        n_delta_s = T
        n_delta_acceleration = T
        n_delta_jerk = max(0, T - 1)
        n_x = n_s + n_v + n_delta_s + n_delta_acceleration + n_delta_jerk

        def s_idx(t):
            return t

        def v_idx(t):
            return n_s + t

        def delta_s_idx(t):
            return n_s + n_v + t

        def delta_acceleration_idx(t):
            return n_s + n_v + n_delta_s + t

        def delta_jerk_idx(t):
            return n_s + n_v + n_delta_s + n_delta_acceleration + t

        key = (
            T,
            float(dt),
            float(amin),
            float(amax),
            float(jmin),
            float(jmax),
            None if s0 is None else float(s0),
            None if v0 is None else float(v0),
            None if initial_s is None else float(initial_s),
            None if initial_v is None else float(initial_v),
            None if initial_a is None else float(initial_a),
            s_hat.tobytes(),
        )
        cache = getattr(self, "_acceleration_lp_template_cache", None)
        if cache is None:
            cache = {}
            self._acceleration_lp_template_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        def build_sparse(rows):
            row_indices = []
            column_indices = []
            values = []
            rhs = []
            for row_index, (coefficients, bound) in enumerate(rows):
                for column, value in coefficients:
                    if value:
                        row_indices.append(row_index)
                        column_indices.append(column)
                        values.append(float(value))
                rhs.append(float(bound))
            matrix = csr_matrix(
                (values, (row_indices, column_indices)),
                shape=(len(rows), n_x),
                dtype=float,
            )
            return matrix, np.asarray(rhs, dtype=float)

        eq_rows = []
        for t in range(T - 1):
            eq_rows.append(
                (
                    (
                        (s_idx(t + 1), 1.0),
                        (s_idx(t), -1.0),
                        (v_idx(t), -0.5 * dt),
                        (v_idx(t + 1), -0.5 * dt),
                    ),
                    0.0,
                )
            )
        if initial_s is not None and initial_v is not None and T:
            eq_rows.append(
                (
                    ((s_idx(0), 1.0), (v_idx(0), -0.5 * dt)),
                    float(initial_s) + 0.5 * float(initial_v) * dt,
                )
            )
        if s0 is not None:
            eq_rows.append((((s_idx(0), 1.0),), float(s0)))
        if v0 is not None:
            eq_rows.append((((v_idx(0), 1.0),), float(v0)))
        A_eq, b_eq = build_sparse(eq_rows)

        ub_rows = []
        if initial_v is not None and T:
            ub_rows.extend(
                (
                    (((v_idx(0), 1.0),), float(initial_v) + amax * dt),
                    (((v_idx(0), -1.0),), -float(initial_v) - amin * dt),
                )
            )
            if initial_a is not None:
                ub_rows.extend(
                    (
                        (
                            ((v_idx(0), 1.0),),
                            float(initial_v) + float(initial_a) * dt + jmax * dt * dt,
                        ),
                        (
                            ((v_idx(0), -1.0),),
                            -float(initial_v) - float(initial_a) * dt - jmin * dt * dt,
                        ),
                    )
                )
        for t in range(T - 1):
            ub_rows.extend(
                (
                    (
                        ((v_idx(t + 1), 1.0), (v_idx(t), -1.0)),
                        amax * dt,
                    ),
                    (
                        ((v_idx(t + 1), -1.0), (v_idx(t), 1.0)),
                        -amin * dt,
                    ),
                )
            )
        for t in range(T - 2):
            ub_rows.extend(
                (
                    (
                        (
                            (v_idx(t), 1.0),
                            (v_idx(t + 1), -2.0),
                            (v_idx(t + 2), 1.0),
                        ),
                        jmax * dt * dt,
                    ),
                    (
                        (
                            (v_idx(t), -1.0),
                            (v_idx(t + 1), 2.0),
                            (v_idx(t + 2), -1.0),
                        ),
                        -jmin * dt * dt,
                    ),
                )
            )
        for t in range(T):
            ub_rows.extend(
                (
                    (
                        ((s_idx(t), 1.0), (delta_s_idx(t), -1.0)),
                        float(s_hat[t]),
                    ),
                    (
                        ((s_idx(t), -1.0), (delta_s_idx(t), -1.0)),
                        -float(s_hat[t]),
                    ),
                )
            )
        for t in range(T):
            if t == 0:
                if initial_v is None:
                    continue
                coefficients = ((v_idx(t), 1.0), (delta_acceleration_idx(t), -1.0))
                opposite = ((v_idx(t), -1.0), (delta_acceleration_idx(t), -1.0))
                rhs = float(initial_v)
            else:
                coefficients = (
                    (v_idx(t), 1.0),
                    (v_idx(t - 1), -1.0),
                    (delta_acceleration_idx(t), -1.0),
                )
                opposite = (
                    (v_idx(t), -1.0),
                    (v_idx(t - 1), 1.0),
                    (delta_acceleration_idx(t), -1.0),
                )
                rhs = 0.0
            ub_rows.extend(((coefficients, rhs), (opposite, -rhs)))
        if T >= 2 and initial_v is not None:
            expected_first_velocity = float(initial_v) + float(initial_a or 0.0) * dt
            ub_rows.extend(
                (
                    (
                        ((v_idx(0), 1.0), (delta_jerk_idx(0), -1.0)),
                        expected_first_velocity,
                    ),
                    (
                        ((v_idx(0), -1.0), (delta_jerk_idx(0), -1.0)),
                        -expected_first_velocity,
                    ),
                )
            )
        for t in range(T - 2):
            ub_rows.extend(
                (
                    (
                        (
                            (v_idx(t), 1.0),
                            (v_idx(t + 1), -2.0),
                            (v_idx(t + 2), 1.0),
                            (delta_jerk_idx(t + 1), -1.0),
                        ),
                        0.0,
                    ),
                    (
                        (
                            (v_idx(t), -1.0),
                            (v_idx(t + 1), 2.0),
                            (v_idx(t + 2), -1.0),
                            (delta_jerk_idx(t + 1), -1.0),
                        ),
                        0.0,
                    ),
                )
            )
        A_ub, b_ub = build_sparse(ub_rows)

        primary_c = np.zeros(n_x)
        primary_c[n_s + n_v : n_s + n_v + n_delta_s] = 1.0
        secondary_c = np.zeros(n_x)
        secondary_c[
            n_s + n_v + n_delta_s : n_s + n_v + n_delta_s + n_delta_acceleration
        ] = 1.0
        secondary_c[n_s + n_v + n_delta_s + n_delta_acceleration :] = float(
            self.config.repair.acceleration_smoothing_jerk_weight
        )
        primary_row = csr_matrix(
            (
                np.ones(T, dtype=float),
                (
                    np.zeros(T, dtype=int),
                    np.arange(n_s + n_v, n_s + n_v + T, dtype=int),
                ),
            ),
            shape=(1, n_x),
        )
        template = {
            "A_eq": A_eq,
            "b_eq": b_eq,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "secondary_A_ub": sparse_vstack((A_ub, primary_row), format="csr"),
            "primary_c": primary_c,
            "secondary_c": secondary_c,
            "n_x": n_x,
            "T": T,
            "n_delta_acceleration": n_delta_acceleration,
            "n_delta_jerk": n_delta_jerk,
        }
        cache.clear()
        cache[key] = template
        return template

    def _solve_acceleration_velocity_planning_lp_cached(
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
        initial_s=None,
        initial_v=None,
        initial_a=None,
    ):
        """Solve the unchanged two-stage acceleration LP using cached sparse matrices."""
        arrays = tuple(
            np.asarray(values, dtype=float)
            for values in (s_hat, vmin, vmax, smin, smax)
        )
        s_hat, vmin, vmax, smin, smax = arrays
        template = self._build_acceleration_lp_template(
            dt,
            s_hat,
            amin,
            amax,
            jmin,
            jmax,
            s0,
            v0,
            initial_s,
            initial_v,
            initial_a,
        )
        T = template["T"]
        bounds = []
        for t in range(T):
            lb, ub = float(smin[t]), float(smax[t])
            if lb > ub:
                raise RuntimeError(
                    "Infeasible acceleration position bounds at "
                    f"time_step={time_offset + t}: smin={lb}, smax={ub}, "
                    f"s_hat={s_hat[t]}"
                )
            bounds.append((lb, ub))
        for t in range(T):
            lb, ub = float(vmin[t]), float(vmax[t])
            if lb > ub:
                raise RuntimeError(
                    "Infeasible acceleration velocity bounds at "
                    f"time_step={time_offset + t}: vmin={lb}, vmax={ub}"
                )
            bounds.append((lb, ub))
        bounds.extend((0.0, None) for _ in range(3 * T - 1))

        result = linprog(
            c=template["primary_c"],
            A_ub=template["A_ub"],
            b_ub=template["b_ub"],
            A_eq=template["A_eq"],
            b_eq=template["b_eq"],
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")

        primary_limit = float(result.fun) + max(
            float(self.config.repair.acceleration_smoothing_position_absolute_tolerance),
            float(self.config.repair.acceleration_smoothing_position_relative_tolerance)
            * max(1.0, abs(float(result.fun))),
        )
        secondary_result = linprog(
            c=template["secondary_c"],
            A_ub=template["secondary_A_ub"],
            b_ub=np.append(template["b_ub"], primary_limit),
            A_eq=template["A_eq"],
            b_eq=template["b_eq"],
            bounds=bounds,
            method="highs",
        )
        if secondary_result.success:
            result = secondary_result
        s = result.x[:T]
        v = result.x[T : 2 * T]
        return {
            "s": s,
            "v": v,
            "objective_min_sum_s_hat_minus_s": np.sum(s_hat - s),
            "raw_result": result,
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
        if repair_mode == "acceleration":
            return self._solve_acceleration_velocity_planning_lp_cached(
                dt=dt,
                s_hat=s_hat,
                vmin=vmin,
                vmax=vmax,
                smin=smin,
                smax=smax,
                amin=amin,
                amax=amax,
                jmin=jmin,
                jmax=jmax,
                s0=s0,
                v0=v0,
                time_offset=time_offset,
                initial_s=initial_s,
                initial_v=initial_v,
                initial_a=initial_a,
            )
        return self._solve_velocity_planning_lp_uncached(
            dt=dt,
            s_hat=s_hat,
            vmin=vmin,
            vmax=vmax,
            smin=smin,
            smax=smax,
            amin=amin,
            amax=amax,
            jmin=jmin,
            jmax=jmax,
            s0=s0,
            v0=v0,
            time_offset=time_offset,
            repair_mode=repair_mode,
            initial_s=initial_s,
            initial_v=initial_v,
            initial_a=initial_a,
        )

    def _solve_velocity_planning_lp_uncached(
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
