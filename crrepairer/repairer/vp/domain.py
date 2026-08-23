"""DomainDPLL domain estimation helpers for velocity-planning repair."""

import math
import os
import time

import numpy as np
from shapely import affinity



class VPPredicateEstimation:
    """Builds predicate-value domains used to prune SAT models in DomainDPLL mode."""

    def ensure_domain_dict_initialized(self):
        """Build and cache SAT domain information when DomainDPLL is enabled."""
        if self.sat_solver.solver_mode != "domain_dpll":
            return self.domain_dict
        if self._domain_dict_initialized:
            return self.domain_dict

        domain_start_time = time.time()
        try:
            domain_dict = self._build_domain_dict_for_sat()
        except Exception as exc:
            print(
                "* \t<VPRepairer>: domain_dict construction failed, "
                f"falling back to empty domains: {exc}"
            )
            domain_dict = {}
        phase_domain_time = time.time() - domain_start_time
        self.domain_dict_time += phase_domain_time
        self.runtime_breakdown["domain_dict"] = self.domain_dict_time
        self.domain_dict = dict(domain_dict)
        self.sat_solver.set_domain_dict(
            domain_dict,
            hard_domain_vars=self._hard_domain_vars,
            repair_literals=self._repair_literals,
        )
        self._domain_dict_initialized = True
        repair_mode = getattr(self, "_vp_repair_mode", "deceleration")
        breakdown_by_mode = getattr(self, "domain_dict_breakdown_by_mode", None)
        if breakdown_by_mode is not None:
            breakdown_by_mode[repair_mode] = dict(self.domain_dict_breakdown)
        print(
            "* \t<VPRepairer>: domain_dict construction time "
            f"({repair_mode}) = {phase_domain_time:.6f}s"
        )
        if self.domain_dict_breakdown:
            print(f"* \t<VPRepairer>: domain_dict breakdown = {self.domain_dict_breakdown}")
        print(f"* \t<VPRepairer>: domain_dict for DomainDPLL = {domain_dict}")
        return self.domain_dict

    def _build_domain_dict_for_sat(self):
        """Estimate possible truth values for SAT proposition nodes from VP reachability."""
        if any(rule in self.config.repair.rules for rule in ("R_IN1", "R_IN4", "R_G2", "R_IN3_hand_draft", "R_IN5")):
            return self._build_domain_dict_for_sat_direct()
        if self._tv in (-math.inf, math.inf):
            self.domain_dict_breakdown = {}
            return {}

        breakdown = {}
        all_states = self._get_states_with_initial()
        start = time.time()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        breakdown["get_clcs_dt"] = time.time() - start

        start = time.time()
        trajectory_clcs, ref_path = self._get_shared_trajectory_clcs(all_states)
        breakdown["build_trajectory_clcs"] = time.time() - start

        start = time.time()
        theta_bounds = self._estimate_theta_bounds(ref_path, lanelet_clcs)
        breakdown["estimate_theta_bounds"] = time.time() - start

        start = time.time()
        ct_reach = self._estimate_trajectory_reachable_set(all_states, trajectory_clcs, dt, ref_path)
        breakdown["estimate_trajectory_reachable_set"] = time.time() - start

        start = time.time()
        cart_reach, cl_reach = self._convert_reachset_back_to_lanelet(
            ct_reach,
            trajectory_clcs,
            lanelet_clcs,
            theta_bounds,
        )
        breakdown["convert_reachset_back_to_lanelet"] = time.time() - start

        start = time.time()
        predicate_values = self._estimate_predicate_ranges(cart_reach, cl_reach, theta_bounds[1])
        breakdown["estimate_predicate_ranges"] = time.time() - start
        if self._domain_predicate_timing:
            breakdown["estimate_predicate_ranges_detail"] = dict(
                self._domain_predicate_timing
            )

        start = time.time()
        self._apply_once_operator(predicate_values["cut_in"])
        breakdown["apply_once_operator"] = time.time() - start

        start = time.time()
        domain_dict = self._domain_dict_construct_general(
            predicate_values, self.sat_solver._prop_nodes
        )
        fixed_rg_count = self._fix_uncontrollable_rg_predicates(
            domain_dict,
            self.sat_solver._prop_nodes,
        )
        self._repair_literals = self._rg_controllable_repair_literals(
            self.sat_solver._prop_nodes
        )
        breakdown["infer_domain_dict"] = time.time() - start
        breakdown["fixed_rg_predicate_count"] = fixed_rg_count
        breakdown["hard_rg_front_speed_domain_count"] = getattr(
            self, "_rg_front_speed_hard_domain_count", 0
        )
        breakdown["repair_literal_count"] = len(self._repair_literals)
        self.domain_dict_breakdown = breakdown
        return domain_dict

    @staticmethod
    def _rg_controllable_repair_literals(prop_nodes):
        """Prioritize RG literals backed by an explicit VP constraint.

        Fixed facts such as the other vehicle's cut-in state may satisfy a SAT
        clause.  These literals guide DomainDPLL's branch order toward a real
        ego action, but are not added to the working CNF and therefore do not
        change the Boolean formula's semantics.
        """
        literals = []
        for prop_node in prop_nodes:
            name = prop_node.name
            alphabet = prop_node.alphabet[-1]
            current_positive = float(prop_node.ttv_value) > 0.0
            if "distance" in name and not current_positive:
                literals.append(alphabet)
            elif "lane" in name and "same" in name and not current_positive:
                literals.append(alphabet)
            elif "speed" in name and not current_positive:
                literals.append(alphabet)
            elif "brakes_abruptly" in name and current_positive:
                literals.append(f"~{alphabet}")
        return list(dict.fromkeys(literals))

    def _fix_uncontrollable_rg_predicates(self, domain_dict, prop_nodes):
        """Initialize RG facts that velocity planning cannot change.

        A cut-in event is determined by the other vehicle's lateral motion and
        is not an action available to the ego-only velocity planner.  Its SAT
        search domain initially agrees with the monitored trajectory.  It is
        intentionally not part of the IN-only hard-priority exception.
        """
        fixed_count = 0
        for prop_node in prop_nodes:
            if "cut_in" not in prop_node.name:
                continue
            alphabet = prop_node.alphabet[-1]
            current_value = int(float(prop_node.ttv_value) > 0.0)
            domain_dict[alphabet] = {current_value}
            fixed_count += 1
        return fixed_count

    def _build_domain_dict_for_sat_direct(self):
        """Build initial domains and VP-action literals for directly handled rules.

        Propositions with a dedicated VP constraint remain searchable.  Every
        other proposition is fixed to its current value because the velocity-
        only repairer has no semantics-preserving way to change it.
        """
        self.domain_dict_breakdown = {}
        if any(
            rule in self.config.repair.rules
            for rule in ("R_IN1", "R_IN4", "R_IN3_hand_draft", "R_IN5")
        ):
            return self._build_constraint_guided_intersection_domains()
        if "R_IN4" in self.config.repair.rules:
            pred_dict = self._eval_turning_priority(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._domain_dict_construct_turning_priority(
                pred_dict, self.sat_solver._prop_nodes
            )
        if "R_IN1" in self.config.repair.rules:
            pred_dict = self._eval_stop_line(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._domain_dict_construct_stop_line(
                pred_dict, self.sat_solver._prop_nodes
            )
        if "R_G2" in self.config.repair.rules:
            pred_dict = self._eval_abrupt_brake(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._domain_dict_construct_abrupt_brake(
                pred_dict, self.sat_solver._prop_nodes
            )
        if "R_IN3_hand_draft" in self.config.repair.rules:
            pred_dict = self._eval_same_priority(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._domain_dict_construct_same_priority(
                pred_dict, self.sat_solver._prop_nodes
            )
        if "R_IN5" in self.config.repair.rules:
            pred_dict = self._eval_has_priority_oncoming(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._domain_dict_construct_has_priority_oncoming(
                pred_dict, self.sat_solver._prop_nodes
            )
        raise NotImplementedError(
            "Direct extraction of domain dict for DomainDPLL support currently supports R_IN1, R_IN4 and R_G2 only."
        )

    def _build_constraint_guided_intersection_domains(self):
        """Keep only propositions backed by explicit VP constraints searchable."""
        is_in1 = "R_IN1" in self.config.repair.rules
        initial_domains = {}
        hard_priority_vars = set()
        repair_literals = []
        repair_mode = getattr(self, "_vp_repair_mode", "deceleration")
        acceleration_reachability = {}
        acceleration_diagnostics = {}
        acceleration_reachability_time = 0.0
        deceleration_reachability = {}
        deceleration_diagnostics = {}
        deceleration_reachability_time = 0.0
        if repair_mode == "acceleration":
            self._acceleration_exit_step_by_variable = {}
            self._acceleration_exit_last_step_by_variable = {}
        constraint_props = [
            prop_node
            for prop_node in self.sat_solver._prop_nodes
            if (
                "stop_line" in prop_node.name
                if is_in1
                else "in_intersection_conflict_area__0_1" in prop_node.name
            )
        ]
        if repair_mode == "deceleration":
            deceleration_reachability_start = time.time()
            try:
                (
                    deceleration_reachability,
                    deceleration_diagnostics,
                ) = self._estimate_deceleration_constraint_reachability(
                    constraint_props,
                    is_in1=is_in1,
                )
            except Exception as exc:
                # Reachability domains only guide SAT.  Leave a literal
                # searchable on geometry/CLCS failure so that exact extraction
                # and the LP remain the final feasibility proof.
                deceleration_diagnostics = {
                    "estimate_error": f"{type(exc).__name__}: {exc}"
                }
            deceleration_reachability_time = (
                time.time() - deceleration_reachability_start
            )
        if not is_in1 and repair_mode == "acceleration":
            acceleration_reachability_start = time.time()
            try:
                (
                    acceleration_reachability,
                    acceleration_diagnostics,
                ) = self._estimate_acceleration_conflict_reachability(
                    constraint_props
                )
            except Exception as exc:
                # Reachability domains are search guidance.  Failure to build
                # the estimate must not create a false hard rejection; exact
                # constraint extraction and the LP remain the final guards.
                acceleration_diagnostics = {
                    "estimate_error": f"{type(exc).__name__}: {exc}"
                }
            acceleration_reachability_time = (
                time.time() - acceleration_reachability_start
            )

        for prop_node in self.sat_solver._prop_nodes:
            alphabet = prop_node.alphabet[-1]
            name = prop_node.name
            current_value = float(prop_node.ttv_value) > 0.0
            if is_in1:
                extractable = "stop_line" in name
            else:
                # The current IN-series VP constraint can only make the ego
                # stay out of its conflict area.  It cannot change the target
                # trajectory (__1_0), nor can it force a currently-false
                # conflict proposition to become true.
                extractable = (
                    "in_intersection_conflict_area__0_1" in name
                    and current_value
                )
            if (
                extractable
                and repair_mode == "deceleration"
                and deceleration_reachability.get(alphabet) is False
            ):
                extractable = False
            if not is_in1:
                if (
                    extractable
                    and repair_mode == "acceleration"
                    and acceleration_reachability.get(alphabet) is False
                ):
                    extractable = False
            if extractable:
                repair_literals.append(f"~{alphabet}")
            else:
                initial_domains[alphabet] = {int(current_value)}
                # These IN antecedents encode route/right-of-way relations.
                # Ego-only velocity planning cannot change them, so unlike
                # ordinary reachability domains they must survive candidate
                # failure and domain relaxation.
                if "same_priority" in name or "target_has_priority" in name:
                    hard_priority_vars.add(alphabet)

        self._hard_domain_vars = hard_priority_vars
        self._repair_literals = repair_literals
        self.domain_dict_breakdown = {
            "repair_mode": repair_mode,
            "initial_domain_count": len(initial_domains),
            "hard_priority_domain_count": len(hard_priority_vars),
            "repair_literal_count": len(repair_literals),
            "deceleration_reachability": deceleration_diagnostics,
            "acceleration_reachability": acceleration_diagnostics,
            # CLCS construction belongs to VP planning in the paper timing.
            # The remaining time is the actual optimistic predicate/reachable
            # computation added by the active reachability phase.
            "build_trajectory_clcs": float(
                (
                    deceleration_diagnostics
                    if repair_mode == "deceleration"
                    else acceleration_diagnostics
                ).get("build_trajectory_clcs", 0.0)
            ),
            "estimate_deceleration_reachability": max(
                0.0,
                deceleration_reachability_time
                - float(
                    deceleration_diagnostics.get(
                        "build_trajectory_clcs", 0.0
                    )
                ),
            ),
            "estimate_acceleration_reachability": max(
                0.0,
                acceleration_reachability_time
                - float(
                    acceleration_diagnostics.get(
                        "build_trajectory_clcs", 0.0
                    )
                ),
            ),
        }
        return initial_domains

    def _estimate_deceleration_constraint_reachability(
        self,
        constraint_props,
        *,
        is_in1,
    ):
        """Check constraint-backed IN literals over a braking interval.

        The recorded trajectory is the upper endpoint of deceleration repair.
        Only predicates with an explicit stop-line/conflict-area VP constraint
        are considered.  The lower endpoint matches the current IN LP
        semantics; a physically continuous maximum-braking rollout is retained
        separately in diagnostics.  A literal is rejected only if even the
        optimistic LP endpoint violates its upper-bound constraint on a
        required temporal frame.
        """
        if not constraint_props:
            return {}, {"constraint_proposition_count": 0}

        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        trajectory_clcs_start = time.time()
        trajectory_clcs, ref_path = self._get_shared_trajectory_clcs(all_states)
        trajectory_clcs_time = time.time() - trajectory_clcs_start

        current_s, current_v, current_a = (
            self._get_velocity_planning_current_conditions(
                all_states,
                trajectory_clcs,
            )
        )
        if current_s is None:
            return {}, {
                "constraint_proposition_count": len(constraint_props),
                "estimate_error": "current VP state is unavailable",
                "build_trajectory_clcs": trajectory_clcs_time,
            }

        original_s = self._build_reference_longitudinal_positions(
            all_states,
            trajectory_clcs,
        )
        amin, amax, jmin, _ = self._get_longitudinal_planning_limits()
        maximum_braking_s = []
        maximum_braking_v = []
        s_prev = float(current_s)
        v_prev = max(0.0, float(current_v))
        a_prev = float(np.clip(current_a, amin, amax))
        for original_endpoint in original_s:
            a_next = float(np.clip(a_prev + jmin * dt, amin, amax))
            v_next = max(0.0, v_prev + a_next * dt)
            s_next = s_prev + 0.5 * (v_prev + v_next) * dt
            # Projection noise or an unusually abrupt recorded stop must not
            # invert the interval endpoints.
            s_next = min(float(s_next), float(original_endpoint))
            maximum_braking_s.append(float(s_next))
            maximum_braking_v.append(float(v_next))
            actual_acceleration = (v_next - v_prev) / dt
            s_prev, v_prev, a_prev = (
                float(s_next),
                float(v_next),
                float(np.clip(actual_acceleration, amin, amax)),
            )

        maximum_braking_s = np.asarray(maximum_braking_s, dtype=float)
        maximum_braking_v = np.asarray(maximum_braking_v, dtype=float)

        # The current IN velocity-planning LP intentionally leaves the
        # current-to-first-planning-frame transition free (s0/v0 are not
        # supplied).  Its optimistic reachable lower endpoint is therefore a
        # stationary trajectory at the current longitudinal coordinate.  Keep
        # the physically continuous maximum-braking rollout above as a useful
        # diagnostic, but do not use that stricter model to reject a SAT
        # literal which the actual LP is still allowed to realize.
        reachable_lower_s = np.full(
            len(original_s), float(current_s), dtype=float
        )
        temporal_steps = self._temporal_constraint_steps(
            all_states,
            propositions=constraint_props,
        )

        interval_mode = "stop_line"
        if is_in1:
            constraint_upper = self._constraint_stop_line_on_trajectory(
                self.rule_monitor.world,
                self.rule_monitor.world.vehicle_by_id(
                    self.config.repair.ego_id
                ),
                ref_path,
                trajectory_clcs,
            )
            if constraint_upper is None:
                return {}, {
                    "constraint_proposition_count": len(constraint_props),
                    "geometry_available": False,
                    "kind": "stop_line",
                    "build_trajectory_clcs": trajectory_clcs_time,
                }
        else:
            conflict_interval = self._get_intersection_conflict_trajectory_interval(
                lanelet_clcs=lanelet_clcs,
                trajectory_clcs=trajectory_clcs,
                ref_path=ref_path,
            )
            if conflict_interval is None:
                return {}, {
                    "constraint_proposition_count": len(constraint_props),
                    "geometry_available": False,
                    "kind": "conflict",
                    "build_trajectory_clcs": trajectory_clcs_time,
                }
            constraint_upper, _, interval_mode = conflict_interval

        raw_constraint_upper = float(constraint_upper)
        effective_constraint_upper = raw_constraint_upper
        if not is_in1 and interval_mode.startswith("legacy"):
            ct_s_min = float(
                trajectory_clcs.convert_to_curvilinear_coords(
                    float(ref_path[0][0]), float(ref_path[0][1])
                )[0]
            )
            first_plan_s = (
                float(original_s[0])
                if len(original_s)
                else float(current_s)
            )
            if (
                not os.environ.get("CRREPAIR_VP_DISABLE_CONFLICT_CAP_CLAMP")
                and effective_constraint_upper < ct_s_min
            ):
                effective_constraint_upper = max(
                    first_plan_s, effective_constraint_upper
                )

        reachable = {}
        per_literal = {}
        planning_start = int(self._tc) + 1
        for prop in constraint_props:
            alphabet = prop.alphabet[-1]
            interval = temporal_steps[id(prop)]
            first_index = max(0, interval.start - planning_start)
            last_index = min(
                len(reachable_lower_s) - 1,
                interval.end - planning_start,
            )
            if interval.count == 0 or last_index < first_index:
                value = False
                min_margin = None
                active_minimum = []
                active_original = []
            else:
                active_minimum = reachable_lower_s[
                    first_index : last_index + 1
                ]
                active_braking = maximum_braking_s[
                    first_index : last_index + 1
                ]
                active_original = original_s[first_index : last_index + 1]
                margins = effective_constraint_upper - active_minimum
                braking_margins = (
                    effective_constraint_upper - active_braking
                )
                value = bool(np.all(margins >= -1e-6))
                min_margin = float(np.min(margins))
                braking_reachable = bool(
                    np.all(braking_margins >= -1e-6)
                )
                braking_min_margin = float(np.min(braking_margins))
            # Legacy intersection bounds are deliberately conservative
            # projections and the exact constraint converter may recover from
            # a local bound mismatch.  A negative interval overlap is thus
            # ``unknown`` rather than a proof of infeasibility.  Only exact
            # stop-line/monitor geometry may turn a negative estimate into a
            # hard SAT-domain rejection.
            conclusive = bool(
                is_in1 or not interval_mode.startswith("legacy")
            )
            candidate_searchable = bool(value or not conclusive)
            reachable[alphabet] = candidate_searchable
            per_literal[alphabet] = {
                "active_start": int(interval.start),
                "active_end": int(interval.end),
                "active_count": int(interval.count),
                "constraint_upper": effective_constraint_upper,
                "raw_constraint_upper": raw_constraint_upper,
                "minimum_margin": min_margin,
                "reachable": value,
                "conclusive": conclusive,
                "candidate_searchable": candidate_searchable,
                "maximum_braking_reachable": (
                    braking_reachable
                    if interval.count and last_index >= first_index
                    else None
                ),
                "maximum_braking_minimum_margin": (
                    braking_min_margin
                    if interval.count and last_index >= first_index
                    else None
                ),
                "minimum_s_start": (
                    float(active_minimum[0]) if len(active_minimum) else None
                ),
                "minimum_s_end": (
                    float(active_minimum[-1]) if len(active_minimum) else None
                ),
                "original_s_start": (
                    float(active_original[0]) if len(active_original) else None
                ),
                "original_s_end": (
                    float(active_original[-1]) if len(active_original) else None
                ),
            }

        return reachable, {
            "constraint_proposition_count": len(constraint_props),
            "geometry_available": True,
            "kind": "stop_line" if is_in1 else "conflict",
            "interval_mode": interval_mode,
            "current_s": float(current_s),
            "current_v": float(current_v),
            "reachable_interval_semantics": "current_in_lp",
            "reachable_lower_terminal_s": float(reachable_lower_s[-1]),
            "maximum_braking_terminal_s": float(maximum_braking_s[-1]),
            "maximum_braking_terminal_v": float(maximum_braking_v[-1]),
            "original_terminal_s": float(original_s[-1]),
            "literals": per_literal,
            "build_trajectory_clcs": trajectory_clcs_time,
        }

    def _estimate_acceleration_conflict_reachability(self, conflict_props):
        """Estimate whether maximum-progress dynamics can clear the exit.

        This is deliberately an optimistic predicate-domain estimate.  It uses
        the longitudinal acceleration, jerk, and vehicle-speed limits, while
        the exact LP remains responsible for enforcing local curvature limits
        and proving every selected proposition.
        """
        if not conflict_props:
            return {}, {"conflict_proposition_count": 0}

        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        trajectory_clcs_start = time.time()
        trajectory_clcs, ref_path = self._get_shared_trajectory_clcs(all_states)
        trajectory_clcs_time = time.time() - trajectory_clcs_start
        conflict_interval = self._get_intersection_conflict_trajectory_interval(
            lanelet_clcs=lanelet_clcs,
            trajectory_clcs=trajectory_clcs,
            ref_path=ref_path,
        )
        if conflict_interval is None:
            return {}, {
                "conflict_proposition_count": len(conflict_props),
                "geometry_available": False,
                "build_trajectory_clcs": trajectory_clcs_time,
            }

        _, after_lower, interval_mode = conflict_interval
        temporal_steps = self._temporal_constraint_steps(
            all_states,
            propositions=conflict_props,
        )
        current_s, current_v, current_a = (
            self._get_velocity_planning_current_conditions(
                all_states,
                trajectory_clcs,
            )
        )
        if current_s is None:
            return {}, {
                "geometry_available": True,
                "estimate_error": "current VP state is unavailable",
                "build_trajectory_clcs": trajectory_clcs_time,
            }

        amin, amax, _, jmax = self._get_longitudinal_planning_limits()
        qp_veh_config = self.config.vehicle.qp_veh_config
        v_lon_max = float(qp_veh_config.v_lon_max)
        path_progress = sorted(
            (
                float(
                    trajectory_clcs.convert_to_curvilinear_coords(
                        float(ref_path[0][0]), float(ref_path[0][1])
                    )[0]
                ),
                float(
                    trajectory_clcs.convert_to_curvilinear_coords(
                        float(ref_path[-1][0]), float(ref_path[-1][1])
                    )[0]
                ),
            )
        )
        _, path_max = path_progress
        start_idx = int(self._tc - all_states[0].time_step)
        horizon = max(0, len(all_states) - start_idx - 1)
        maximum_progress = []
        s_prev = float(current_s)
        v_prev = float(current_v)
        a_prev = float(np.clip(current_a, amin, amax))
        for _ in range(horizon):
            a_next = min(float(amax), a_prev + float(jmax) * dt)
            v_next = min(v_lon_max, max(0.0, v_prev + a_next * dt))
            s_next = min(
                path_max,
                s_prev + 0.5 * (v_prev + v_next) * dt,
            )
            maximum_progress.append(float(s_next))
            actual_acceleration = (v_next - v_prev) / dt
            s_prev, v_prev, a_prev = (
                s_next,
                v_next,
                float(np.clip(actual_acceleration, amin, amax)),
            )

        reachable = {}
        per_literal = {}
        for prop in conflict_props:
            alphabet = prop.alphabet[-1]
            interval = temporal_steps[id(prop)]
            if interval.count == 0:
                value = False
                first_required_step = None
                exit_step = None
                reachable_s = None
            else:
                first_required_step = max(int(self._tc) + 1, interval.start)
                exit_step = None
                reachable_s = None
                if current_s >= after_lower:
                    exit_step = first_required_step
                    reachable_s = current_s
                else:
                    for time_step in range(first_required_step, interval.end + 1):
                        index = time_step - int(self._tc) - 1
                        if not (0 <= index < len(maximum_progress)):
                            continue
                        candidate_s = maximum_progress[index]
                        if candidate_s >= after_lower - 1e-6:
                            exit_step = time_step
                            reachable_s = candidate_s
                            break
                value = exit_step is not None
            reachable[alphabet] = bool(value)
            per_literal[alphabet] = {
                "first_required_step": first_required_step,
                "earliest_exit_step": exit_step,
                "maximum_reachable_s": reachable_s,
                "after_lower": float(after_lower),
                "reachable": bool(value),
            }

        return reachable, {
            "conflict_proposition_count": len(conflict_props),
            "geometry_available": True,
            "interval_mode": interval_mode,
            "current_s": float(current_s),
            "path_max": float(path_max),
            "literals": per_literal,
            "build_trajectory_clcs": trajectory_clcs_time,
        }

    def _eval_abrupt_brake(self, t_start: int, t_end: int) -> dict:
        pred_dict = {}
        for t in range(t_start, t_end):
            if t >= len(self.rule_monitor.rob_abstraction[0]):
                continue
            for idx, key in enumerate(self.rule_monitor.abstraction_names[0][t]):
                if not key or "brakes_abruptly" in key:
                    continue
                value = self.rule_monitor.rob_abstraction[0][t][idx]
                pred_dict.setdefault(key, set())
                if value > 0:
                    pred_dict[key].add(1)
                else:
                    pred_dict[key].add(0)
        return pred_dict
    
    def _domain_dict_construct_abrupt_brake(self, pred_dict, prop_nodes):
        domain_dict = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            if prop_name == "g1":
                domain_dict[prop_node.alphabet[-1]] = {0}
                continue
            if "brakes_abruptly" in prop_name:
                continue
            for key, value in pred_dict.items():
                if key in prop_name and len(value) == 1:
                    domain_dict[prop_node.alphabet[-1]] = set(value)
                    break
        return domain_dict

    def _eval_stop_line(self, t_start: int, t_end: int) -> dict:
        pred_dict = {}
        for t in range(t_start, t_end):
            if t >= len(self.rule_monitor.rob_predicate[0]):
                continue
            for key, value in self.rule_monitor.rob_predicate[0][t].items():
                pred_dict.setdefault(key, set())
                if value[0] > 0:
                    pred_dict[key].add(1)
                else:
                    pred_dict[key].add(0)
        return pred_dict

    def _domain_dict_construct_stop_line(self, pred_dict, prop_nodes):
        domain_dict = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            for key, value in pred_dict.items():
                if key in prop_name and len(value) == 1:
                    domain_dict[prop_node.alphabet[-1]] = set(value)
                    break
        return domain_dict

    def _eval_turning_priority(self, t_start: int, t_end: int) -> dict:
        pred_dict = {}
        for t in range(t_start, t_end):
            if t >= len(self.rule_monitor.rob_predicate[0]):
                continue
            for key, value in self.rule_monitor.rob_predicate[0][t].items():
                if "has_priority" not in key:
                    continue
                pred_dict.setdefault(key, set())
                if value[0] > 0:
                    pred_dict[key].add(1)
                else:
                    pred_dict[key].add(0)
        return pred_dict

    def _extract_in4_priority_atom(self, name: str):
        atom_prefixes = (
            "turning_right_target_",
            "going_straight_target_",
            "turning_left_target_",
        )
        start = -1
        for prefix in atom_prefixes:
            start = name.find(prefix)
            if start != -1:
                break
        if start == -1:
            return None

        suffix = "__0_1"
        end = name.find(suffix, start)
        if end == -1:
            return None
        return name[start : end + len(suffix)]

    def _domain_dict_construct_turning_priority(self, pred_dict, prop_nodes):
        prop_name_to_alphabet = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            if "has_priority" not in prop_name:
                continue
            atomic_name = self._extract_in4_priority_atom(prop_name)
            if atomic_name is not None:
                prop_name_to_alphabet[atomic_name] = prop_node.alphabet[-1]

        domain_dict = {}
        for atomic_name, value in pred_dict.items():
            if len(value) != 1:
                continue
            alphabet = prop_name_to_alphabet.get(atomic_name)
            if alphabet is not None:
                domain_dict[alphabet] = set(value)
        return domain_dict

    def _eval_same_priority(self, t_start: int, t_end: int) -> dict:
        pred_dict = {}
        for t in range(t_start, t_end):
            if t >= len(self.rule_monitor.rob_predicate[0]):
                continue
            for key, value in self.rule_monitor.rob_predicate[0][t].items():
                if "same_priority" not in key:
                    continue
                pred_dict.setdefault(key, set())
                if value[0] > 0:
                    pred_dict[key].add(1)
                else:
                    pred_dict[key].add(0)
        return pred_dict

    def _extract_in3_same_priority_atom(self, name: str):
        atom_prefixes = (
            "turning_right_ego_",
            "turning_left_ego_",
            "going_straight_ego_",
        )
        start = -1
        for prefix in atom_prefixes:
            start = name.find(prefix)
            if start != -1:
                break
        if start == -1:
            return None

        suffix = "__0_1"
        end = name.find(suffix, start)
        if end == -1:
            return None
        return name[start : end + len(suffix)]

    def _domain_dict_construct_same_priority(self, pred_dict, prop_nodes):
        prop_name_to_alphabet = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            if "same_priority" not in prop_name:
                continue
            atomic_name = self._extract_in3_same_priority_atom(prop_name)
            if atomic_name is not None:
                prop_name_to_alphabet[atomic_name] = prop_node.alphabet[-1]

        domain_dict = {}
        for atomic_name, value in pred_dict.items():
            if len(value) != 1:
                continue
            alphabet = prop_name_to_alphabet.get(atomic_name)
            if alphabet is not None:
                domain_dict[alphabet] = set(value)
        return domain_dict

    def _eval_has_priority_oncoming(self, t_start: int, t_end: int) -> dict:
        pred_dict = {}
        for t in range(t_start, t_end):
            if t >= len(self.rule_monitor.rob_predicate[0]):
                continue
            for key, value in self.rule_monitor.rob_predicate[0][t].items():
                if "has_priority_oncoming" not in key:
                    continue
                pred_dict.setdefault(key, set())
                if value[0] > 0:
                    pred_dict[key].add(1)
                else:
                    pred_dict[key].add(0)
        return pred_dict

    def _extract_in5_priority_oncoming_atom(self, name: str):
        atom_prefixes = (
            "turning_right_target_",
            "going_straight_target_",
            "turning_left_target_",
        )
        start = -1
        for prefix in atom_prefixes:
            start = name.find(prefix)
            if start != -1:
                break
        if start == -1:
            return None

        suffix = "__0_1"
        end = name.find(suffix, start)
        if end == -1:
            return None
        return name[start : end + len(suffix)]

    def _domain_dict_construct_has_priority_oncoming(self, pred_dict, prop_nodes):
        prop_name_to_alphabet = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            if "has_priority_oncoming" not in prop_name:
                continue
            atomic_name = self._extract_in5_priority_oncoming_atom(prop_name)
            if atomic_name is not None:
                prop_name_to_alphabet[atomic_name] = prop_node.alphabet[-1]

        domain_dict = {}
        for atomic_name, value in pred_dict.items():
            if len(value) != 1:
                continue
            alphabet = prop_name_to_alphabet.get(atomic_name)
            if alphabet is not None:
                domain_dict[alphabet] = set(value)
        return domain_dict

    def _estimate_theta_bounds(self, ref_path: np.ndarray, lanelet_clcs):
        lanelet_pts = np.empty((len(ref_path), 2), dtype=float)
        for k, point in enumerate(ref_path):
            point_arr = np.asarray(point, dtype=float).reshape(-1)
            if point_arr.size < 2:
                raise ValueError(
                    f"Invalid reference-path point at index {k}: {point!r}"
                )
            x = float(point_arr[0])
            y = float(point_arr[1])
            lanelet_pts[k] = lanelet_clcs.convert_to_curvilinear_coords(x, y)
        ds = np.gradient(lanelet_pts[:, 0])
        dd = np.gradient(lanelet_pts[:, 1])
        eps = 1e-6
        ds_safe = np.where(np.abs(ds) < eps, eps, ds)
        ratio_cos = np.sqrt(ds ** 2 / (ds_safe ** 2 + dd ** 2))
        min_ratio_cos = np.min(ratio_cos)
        max_ratio_cos = np.max(ratio_cos)
        min_theta = float(np.arccos(max_ratio_cos))
        max_theta = float(np.arccos(min_ratio_cos))
        return min_theta, max_theta

    def _estimate_trajectory_reachable_set(self, all_states, trajectory_clcs, dt, ref_path):
        ct_initial_pos = trajectory_clcs.convert_to_curvilinear_coords(
            float(self.ego_vehicle.initial_state.position[0]),
            float(self.ego_vehicle.initial_state.position[1]),
        )
        v0 = self.ego_vehicle.initial_state.velocity
        s0 = ct_initial_pos[0]
        a_lon_max, a_lon_min, v_lon_max = self._get_longitudinal_reachability_limits()
        v_lon_min = 0.0
        maximum = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[-1][0]),
            float(ref_path[-1][1]),
        )[0]

        ct_reach = []
        for t in range(len(all_states) - 1):
            if t == 0:
                v_prev_max = v0
                v_prev_min = v0
                s_prev_max = s0
                s_prev_min = s0
            v_max = min(v_prev_max + a_lon_max * dt, v_lon_max)
            v_min = max(v_prev_min + a_lon_min * dt, v_lon_min)
            s_max = s_prev_max + (v_prev_max + v_max) / 2 * dt
            s_min = s_prev_min + (v_prev_min + v_min) / 2 * dt
            if v_prev_max + a_lon_max * dt > v_lon_max:
                s_max = s_prev_max + v_prev_max * dt + 0.5 * a_lon_max * dt ** 2
            if v_prev_min + a_lon_min * dt < v_lon_min:
                s_min = s_prev_min + v_prev_min * dt + 0.5 * a_lon_min * dt ** 2
            s_max = min(s_max, maximum)
            s_min = max(s_min, s0)
            ct_reach.append((t + 1 + self.config.repair.t_0, s_min, s_max, v_min, v_max))
            v_prev_max = v_max
            v_prev_min = v_min
            s_prev_max = s_max
            s_prev_min = s_min
        return ct_reach

    def _convert_reachset_back_to_lanelet(self, ct_reach, trajectory_clcs, lanelet_clcs, theta_bounds):
        min_theta, max_theta = theta_bounds
        a_lon_max, a_lon_min, _ = self._get_longitudinal_reachability_limits()
        max_ratio_cos = np.cos(min_theta)
        min_ratio_cos = np.cos(max_theta)

        cart_reach = []
        cl_reach = []
        for t in range(len(ct_reach)):
            s_min = ct_reach[t][1]
            s_max = ct_reach[t][2]
            pos_min = trajectory_clcs.convert_to_cartesian_coords(s_min, 0.0)
            pos_max = trajectory_clcs.convert_to_cartesian_coords(s_max, 0.0)
            cart_reach.append((t, pos_min, pos_max))
            pos_min_cl = lanelet_clcs.convert_to_curvilinear_coords(float(pos_min[0]), float(pos_min[1]))
            pos_max_cl = lanelet_clcs.convert_to_curvilinear_coords(float(pos_max[0]), float(pos_max[1]))
            v_lon_max_cl = ct_reach[t][4] * max_ratio_cos
            v_lon_min_cl = ct_reach[t][3] * min_ratio_cos
            a_lon_max_cl = a_lon_max * max_ratio_cos
            a_lon_min_cl = a_lon_min * min_ratio_cos
            cl_reach.append(
                (t + 1 + self.config.repair.t_0, pos_min_cl, pos_max_cl, v_lon_min_cl, v_lon_max_cl, a_lon_min_cl, a_lon_max_cl)
            )
        return cart_reach, cl_reach

    def _estimate_predicate_ranges(self, cart_reach, cl_reach, max_theta):
        other_id = self.rule_monitor.other_id
        world = self.rule_monitor.world
        road_network = world.road_network
        follow_vehicle = world.vehicle_by_id(self.ego_vehicle.obstacle_id)
        other_vehicle = world.vehicle_by_id(other_id)
        safe_dist = []
        in_same_lane = []
        cut_in = []
        in_front_of = []
        lane_speed = []
        type_speed = []
        fov_speed = []
        brake_speed = []
        speed_limits = self._extract_speed_limit_values()
        follow_width = follow_vehicle.shape.width
        follow_length = follow_vehicle.shape.length
        follow_cos_theta = np.cos(max_theta)
        follow_sin_theta = np.sin(max_theta)
        a_min_follow = follow_vehicle.vehicle_param.get("a_min")
        t_react_follow = follow_vehicle.vehicle_param.get("t_react")
        predicate_timing = {
            "safe_dist": 0.0,
            "in_same_lane": 0.0,
            "cut_in": 0.0,
            "in_front_of": 0.0,
            "speed_limits": 0.0,
        }

        for t in range(len(cart_reach)):
            time_step = t + 1
            abs_time_step = time_step + self.config.repair.t_0
            other_context = self._build_other_vehicle_predicate_context(
                road_network=road_network,
                other_vehicle=other_vehicle,
                time_step=time_step,
            )

            start = time.time()
            safe_dist_min, in_front_of_min = self._evaluate_longitudinal_predicates(
                follow_vehicle=follow_vehicle,
                other_context=other_context,
                time_step=time_step,
                follow_cart_pos=cart_reach[t][1],
                follow_velocity=cl_reach[t][3],
                follow_theta=max_theta,
                follow_width=follow_width,
                follow_length=follow_length,
                follow_cos_theta=follow_cos_theta,
                follow_sin_theta=follow_sin_theta,
                a_min_follow=a_min_follow,
                t_react_follow=t_react_follow,
            )
            safe_dist_max, in_front_of_max = self._evaluate_longitudinal_predicates(
                follow_vehicle=follow_vehicle,
                other_context=other_context,
                time_step=time_step,
                follow_cart_pos=cart_reach[t][2],
                follow_velocity=cl_reach[t][4],
                follow_theta=max_theta,
                follow_width=follow_width,
                follow_length=follow_length,
                follow_cos_theta=follow_cos_theta,
                follow_sin_theta=follow_sin_theta,
                a_min_follow=a_min_follow,
                t_react_follow=t_react_follow,
            )
            safe_dist.append((abs_time_step, 2 if safe_dist_min != safe_dist_max else int(safe_dist_min)))
            predicate_timing["safe_dist"] += time.time() - start

            start = time.time()
            in_same_lane_min = self._in_same_lane_eval(
                road_network=road_network,
                other_context=other_context,
                ego_vehicle=follow_vehicle,
                ego_cart_pos=cart_reach[t][1],
                ego_theta=max_theta,
            )
            in_same_lane_max = self._in_same_lane_eval(
                road_network=road_network,
                other_context=other_context,
                ego_vehicle=follow_vehicle,
                ego_cart_pos=cart_reach[t][2],
                ego_theta=max_theta,
            )
            in_same_lane.append((abs_time_step, 2 if in_same_lane_min != in_same_lane_max else int(in_same_lane_min)))
            predicate_timing["in_same_lane"] += time.time() - start

            start = time.time()
            cut_in_min = self._cut_in_eval(
                road_network=road_network,
                other_context=other_context,
                ego_vehicle=follow_vehicle,
                ego_cart_pos=cart_reach[t][1],
                ego_theta=max_theta,
                in_same_lane_value=in_same_lane_min,
            )
            cut_in_max = self._cut_in_eval(
                road_network=road_network,
                other_context=other_context,
                ego_vehicle=follow_vehicle,
                ego_cart_pos=cart_reach[t][2],
                ego_theta=max_theta,
                in_same_lane_value=in_same_lane_max,
            )
            cut_in.append((abs_time_step, 2 if cut_in_min != cut_in_max else int(cut_in_min)))
            predicate_timing["cut_in"] += time.time() - start

            start = time.time()
            in_front_of.append((abs_time_step, 2 if in_front_of_min != in_front_of_max else int(in_front_of_min)))
            predicate_timing["in_front_of"] += time.time() - start

            v_min = cl_reach[t][3]
            v_max = cl_reach[t][4]
            start = time.time()
            if speed_limits["lane"] is not None:
                lane_speed_min = self._keep_speed_limit_eval(v_min, speed_limits["lane"])
                lane_speed_max = self._keep_speed_limit_eval(v_max, speed_limits["lane"])
                lane_speed.append((abs_time_step, 2 if lane_speed_min != lane_speed_max else int(lane_speed_min)))
            if speed_limits["type"] is not None:
                type_speed_min = self._keep_speed_limit_eval(v_min, speed_limits["type"])
                type_speed_max = self._keep_speed_limit_eval(v_max, speed_limits["type"])
                type_speed.append((abs_time_step, 2 if type_speed_min != type_speed_max else int(type_speed_min)))
            if speed_limits["fov"] is not None:
                fov_speed_min = self._keep_speed_limit_eval(v_min, speed_limits["fov"])
                fov_speed_max = self._keep_speed_limit_eval(v_max, speed_limits["fov"])
                fov_speed.append((abs_time_step, 2 if fov_speed_min != fov_speed_max else int(fov_speed_min)))
            if speed_limits["brake"] is not None:
                brake_speed_min = self._keep_speed_limit_eval(v_min, speed_limits["brake"])
                brake_speed_max = self._keep_speed_limit_eval(v_max, speed_limits["brake"])
                brake_speed.append((abs_time_step, 2 if brake_speed_min != brake_speed_max else int(brake_speed_min)))
            predicate_timing["speed_limits"] += time.time() - start

        predicate_values = {
            "safe_dist": safe_dist,
            "in_same_lane": in_same_lane,
            "cut_in": cut_in,
            "in_front_of": in_front_of,
        }
        self._domain_predicate_timing = predicate_timing
        if lane_speed:
            predicate_values["lane_speed"] = lane_speed
        if type_speed:
            predicate_values["type_speed"] = type_speed
        if fov_speed:
            predicate_values["fov_speed"] = fov_speed
        if brake_speed:
            predicate_values["brake_speed"] = brake_speed
        return predicate_values

    def _extract_speed_limit_values(self):
        speed_limits = {"lane": None, "type": None, "fov": None, "brake": None}
        stlmonitor_world = self.rule_monitor.world
        t_0 = self.config.repair.t_0
        for proposition in self.rule_monitor.proposition_nodes:
            for predicate in proposition.children:
                if "speed" not in getattr(predicate, "base_name", ""):
                    continue
                speed_limit = predicate.evaluator.get_speed_limit(
                    stlmonitor_world, t_0, [self.ego_vehicle.obstacle_id]
                )
                if "lane" in predicate.base_name:
                    speed_limits["lane"] = speed_limit 
                elif "type" in predicate.base_name:
                    speed_limits["type"] = speed_limit 
                elif "fov" in predicate.base_name:
                    speed_limits["fov"] = speed_limit
                elif "brake" in predicate.base_name:
                    speed_limits["brake"] = speed_limit 
        return speed_limits

    def _domain_dict_construct_general(self, predicate_values, prop_nodes):
        """Build singleton RG domains and preserve proven one-way facts.

        ``in_front_of`` and the four speed-limit predicates are hard only when
        reachability has already proved their domain to be a singleton.  In
        particular, a violated speed predicate which braking can repair keeps
        the domain ``{0, 1}`` and is never fixed here.
        """
        domain_dict = {}
        hard_domain_vars = set()
        for prop_node in prop_nodes:
            predicate_values_key = self._prop_node_name_to_predicate_values_key(
                prop_node.name, predicate_values
            )
            if predicate_values_key == {0} or predicate_values_key == {1}:
                variable = prop_node.alphabet[-1]
                domain_dict[variable] = set(predicate_values_key)
                if self._is_rg_front_or_speed_predicate(prop_node.name):
                    hard_domain_vars.add(variable)

        self._hard_domain_vars.update(hard_domain_vars)
        self._rg_front_speed_hard_domain_count = len(hard_domain_vars)
        return domain_dict

    @staticmethod
    def _is_rg_front_or_speed_predicate(prop_name):
        # Temporal wrappers may be retained in a proposition name, as for the
        # existing once(cut_in) form, so match the atomic predicate anywhere.
        return any(
            predicate_name in prop_name
            for predicate_name in (
                "in_front_of__",
                "keeps_lane_speed_limit__",
                "keeps_type_speed_limit__",
                "keeps_fov_speed_limit__",
                "keeps_brake_speed_limit__",
            )
        )

    def _prop_node_name_to_predicate_values_key(self, prop_name, predicate_values):
        key = None
        if "distance" in prop_name:
            key = "safe_dist"
        elif "lane" in prop_name and "same" in prop_name:
            key = "in_same_lane"
        elif "front" in prop_name:
            key = "in_front_of"
        elif "cut_in" in prop_name:
            key = "cut_in"
        elif "lane" in prop_name and "speed" in prop_name:
            key = "lane_speed"
        elif "type" in prop_name and "speed" in prop_name:
            key = "type_speed"
        elif "fov" in prop_name and "speed" in prop_name:
            key = "fov_speed"
        elif "brake" in prop_name and "speed" in prop_name:
            key = "brake_speed"

        if key is None or key not in predicate_values:
            return {0, 1}

        possible_values = set()
        for _, value in predicate_values[key]:
            if value == 2:
                possible_values.update({0, 1})
            else:
                possible_values.add(int(value))
        return possible_values

    def _apply_once_operator(self, cut_in):
        for prop_node in self.sat_solver._prop_nodes:
            prop_name = prop_node.name
            if "once" not in prop_name:
                continue
            time_horizon = [int(prop_name[5]), int(prop_name[7])]
            cut_seq = np.array([value for _, value in cut_in], dtype=int)
            not_cut_seq = self._logic_not(cut_seq)
            prev_not_cut_seq = self._logic_previous(not_cut_seq)
            cut_and_prev_not_cut_seq = self._logic_and(cut_seq, prev_not_cut_seq)
            once_seq = self._logic_once(time_horizon[0], time_horizon[1], cut_and_prev_not_cut_seq)
            for i in range(len(cut_in)):
                cut_in[i] = (cut_in[i][0], int(once_seq[i]))

    def _logic_not(self, value_seq: np.ndarray):
        lut = np.array([1, 0, 2])
        return lut[value_seq]

    def _logic_previous(self, value_seq: np.ndarray, t=0):
        if t < -1:
            raise ValueError("t must be >= -1")
        if t == -1:
            return value_seq
        prev_value_seq = np.empty_like(value_seq)
        prev_value_seq[: t + 1] = 2
        prev_value_seq[t + 1 :] = value_seq[: -1 - t]
        return prev_value_seq

    def _logic_and(self, value_seq1: np.ndarray, value_seq2: np.ndarray):
        lut = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 2]])
        return lut[value_seq1, value_seq2]

    def _logic_or(self, value_seq1: np.ndarray, value_seq2: np.ndarray):
        lut = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 2]])
        return lut[value_seq1, value_seq2]

    def _logic_once(self, t1, t2, value_seq: np.ndarray):
        for i in range(t1, t2 + 1):
            if i == t1:
                once_value_seq = self._logic_previous(value_seq, t=i - 1)
            else:
                tmp = self._logic_previous(value_seq, t=i - 1)
                once_value_seq = self._logic_or(once_value_seq, tmp)
        return once_value_seq

    def _calculate_safe_distance(self, v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow):
        return (
            (v_lead ** 2) / (-2 * np.abs(a_min_lead))
            - (v_follow ** 2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )

    def _build_other_vehicle_predicate_context(self, road_network, other_vehicle, time_step):
        context = {
            "time_step": time_step,
            "vehicle": other_vehicle,
            "valid": self._vehicle_has_valid_time_step(other_vehicle, time_step),
            "lanelets": set(),
            "lanes": set(),
            "lane": None,
            "rear_s": None,
            "velocity": None,
            "lat_theta": None,
            "cart_pos": None,
        }
        if not context["valid"]:
            return context

        lanelets = other_vehicle.lanelet_assignment.get(time_step)
        if not lanelets:
            context["valid"] = False
            return context

        context["lanelets"] = set(lanelets)
        context["lanes"] = road_network.find_lanes_by_lanelets(context["lanelets"])
        context["lane"] = other_vehicle.get_lane(time_step)
        context["cart_pos"] = other_vehicle.states_cr[time_step].position
        context["velocity"] = other_vehicle.states_cr[time_step].velocity
        context["lat_theta"] = other_vehicle.get_lat_state(time_step).theta
        if context["lane"] is not None:
            context["rear_s"] = other_vehicle.rear_s(time_step)
        return context

    def _compute_follow_front_s(self, follow_vehicle, time_step, follow_cart_pos, follow_theta):
        return self._compute_follow_front_s_fast(
            follow_vehicle=follow_vehicle,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_theta=follow_theta,
        )

    def _compute_follow_front_s_fast(
        self,
        follow_vehicle,
        time_step,
        follow_cart_pos,
        follow_theta,
        follow_width=None,
        follow_length=None,
        follow_cos_theta=None,
        follow_sin_theta=None,
    ):
        if not self._vehicle_has_valid_time_step(follow_vehicle, time_step):
            return None
        lane_follow = follow_vehicle.get_lane(time_step)
        if lane_follow is None:
            return None
        if follow_width is None:
            follow_width = follow_vehicle.shape.width
        if follow_length is None:
            follow_length = follow_vehicle.shape.length
        if follow_cos_theta is None:
            follow_cos_theta = np.cos(follow_theta)
        if follow_sin_theta is None:
            follow_sin_theta = np.sin(follow_theta)
        follow_curvi_pos = lane_follow.clcs.convert_to_curvilinear_coords(
            follow_cart_pos[0], follow_cart_pos[1]
        )

        long_offset = follow_length * 0.5 * follow_cos_theta
        lat_offset = follow_width * 0.5 * follow_sin_theta
        return follow_curvi_pos[0] + max(
            long_offset - lat_offset,
            long_offset + lat_offset,
            -long_offset - lat_offset,
            -long_offset + lat_offset,
        )

    def _ego_lanelets_for_predicate_eval(
        self,
        road_network,
        ego_vehicle,
        ego_cart_pos,
        ego_theta,
    ):
        lanelet_network = road_network.lanelet_network
        try:
            lanelet_matches = lanelet_network.find_lanelet_by_position([ego_cart_pos])
            if lanelet_matches and lanelet_matches[0]:
                return set(lanelet_matches[0])
        except Exception:
            pass

        ego_cos = np.cos(ego_theta)
        ego_sin = np.sin(ego_theta)
        ego_mat = [ego_cos, -ego_sin, ego_sin, ego_cos, ego_cart_pos[0], ego_cart_pos[1]]
        ego_vehicle_shapely_object = affinity.affine_transform(
            ego_vehicle.shape.shapely_object, ego_mat
        )
        ego_lanelets = set()
        for idx in lanelet_network._strtee.query(ego_vehicle_shapely_object):
            lanelet_shapely_polygon = lanelet_network._strtee.geometries[idx]
            if lanelet_shapely_polygon.intersects(ego_vehicle_shapely_object):
                ego_lanelets.add(
                    lanelet_network._get_lanelet_id_by_shapely_polygon(
                        lanelet_shapely_polygon
                    )
                )
        return ego_lanelets

    def _evaluate_longitudinal_predicates(
        self,
        follow_vehicle,
        other_context,
        time_step,
        follow_cart_pos,
        follow_velocity,
        follow_theta,
        max_lon_dist=200.0,
        follow_width=None,
        follow_length=None,
        follow_cos_theta=None,
        follow_sin_theta=None,
        a_min_follow=None,
        t_react_follow=None,
    ):
        front_s = self._compute_follow_front_s_fast(
            follow_vehicle=follow_vehicle,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_theta=follow_theta,
            follow_width=follow_width,
            follow_length=follow_length,
            follow_cos_theta=follow_cos_theta,
            follow_sin_theta=follow_sin_theta,
        )
        if front_s is None:
            return True, True
        if not other_context["valid"] or other_context["rear_s"] is None:
            return True, True

        delta_s = other_context["rear_s"] - front_s
        in_front_of = np.clip(delta_s / max_lon_dist, -1.0, 1.0) > 0.0

        safe_distance = self._calculate_safe_distance(
            follow_velocity,
            other_context["velocity"],
            other_context["vehicle"].vehicle_param.get("a_min"),
            follow_vehicle.vehicle_param.get("a_min") if a_min_follow is None else a_min_follow,
            follow_vehicle.vehicle_param.get("t_react") if t_react_follow is None else t_react_follow,
        )
        keep_safe_distance = (
            np.clip((delta_s - safe_distance) / max_lon_dist, -1.0, 1.0) > 0.0
        )
        return keep_safe_distance, in_front_of
    
    def _in_front_of_eval(
        self,
        follow_vehicle,
        other_context,
        time_step,
        follow_cart_pos,
        follow_theta,
        max_lon_dist=200.0,
    ):
        front_s = self._compute_follow_front_s(
            follow_vehicle=follow_vehicle,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_theta=follow_theta,
        )
        if front_s is None:
            return True
        if not other_context["valid"] or other_context["rear_s"] is None:
            return True

        delta_s = other_context["rear_s"] - front_s
        in_front_of = np.clip(delta_s / max_lon_dist, -1.0, 1.0) > 0.0

        return in_front_of
    
    def _keep_safe_distance_eval(
        self,
        follow_vehicle,
        other_context,
        time_step,
        follow_cart_pos,
        follow_velocity,
        follow_theta,
        max_lon_dist=200.0,
    ):
        front_s = self._compute_follow_front_s(
            follow_vehicle=follow_vehicle,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_theta=follow_theta,
        )
        if front_s is None:
            return True
        if not other_context["valid"] or other_context["rear_s"] is None:
            return True

        delta_s = other_context["rear_s"] - front_s
        
        a_min_follow = follow_vehicle.vehicle_param.get("a_min")
        a_min_lead = other_context["vehicle"].vehicle_param.get("a_min")
        t_react_follow = follow_vehicle.vehicle_param.get("t_react")
        safe_distance = self._calculate_safe_distance(
            follow_velocity,
            other_context["velocity"],
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )
        keep_safe_distance = (
            np.clip((delta_s - safe_distance) / max_lon_dist, -1.0, 1.0) > 0.0
        )
        return keep_safe_distance

    # def _keep_safe_distance_eval(
    #     self,
    #     world,
    #     time_step,
    #     lead_id,
    #     follow_id,
    #     follow_cart_pos,
    #     follow_velocity,
    #     follow_theta,
    #     max_lon_dist=200.0,
    # ):
    #     follow_vehicle = world.vehicle_by_id(follow_id)
    #     other_context = self._build_other_vehicle_predicate_context(
    #         road_network=world.road_network,
    #         other_vehicle=world.vehicle_by_id(lead_id),
    #         time_step=time_step,
    #     )
    #     keep_safe_distance, _ = self._evaluate_longitudinal_predicates(
    #         follow_vehicle=follow_vehicle,
    #         other_context=other_context,
    #         time_step=time_step,
    #         follow_cart_pos=follow_cart_pos,
    #         follow_velocity=follow_velocity,
    #         follow_theta=follow_theta,
    #         max_lon_dist=max_lon_dist,
    #     )
    #     return keep_safe_distance

    def _keep_speed_limit_eval(self, ego_velocity, speed_limit, max_speed=250.0 / 3.6, eps=1e-5):
        if speed_limit is None:
            robustness = math.inf
        else:
            robustness = speed_limit + eps - ego_velocity
        robustness = np.clip(robustness / max_speed, -1.0, 1.0)
        return robustness > 0.0

    def _in_same_lane_eval(self, road_network, other_context, ego_vehicle, ego_cart_pos, ego_theta):
        if not other_context["valid"] or not other_context["lanelets"]:
            return False
        ego_lanelets = self._ego_lanelets_for_predicate_eval(
            road_network=road_network,
            ego_vehicle=ego_vehicle,
            ego_cart_pos=ego_cart_pos,
            ego_theta=ego_theta,
        )
        if not ego_lanelets:
            return False
        ego_lane = road_network.find_lanes_by_lanelets(ego_lanelets)
        return bool(other_context["lanes"] & ego_lane)

    def _cut_in_eval(self, road_network, other_context, ego_vehicle, ego_cart_pos, ego_theta, in_same_lane_value=None, eps=1e-5):
        if not other_context["valid"]:
            return False
        if len(other_context["lanes"]) == 1:
            return False
        in_same_lane = (
            bool(in_same_lane_value)
            if in_same_lane_value is not None
            else self._in_same_lane_eval(
                road_network=road_network,
                other_context=other_context,
                ego_vehicle=ego_vehicle,
                ego_cart_pos=ego_cart_pos,
                ego_theta=ego_theta,
            )
        )
        if not in_same_lane:
            return False
        lane = other_context["lane"]
        if lane is None:
            return False
        other_cart = other_context["cart_pos"]
        other_d = lane.clcs.convert_to_curvilinear_coords(other_cart[0], other_cart[1])[1]
        ego_d = lane.clcs.convert_to_curvilinear_coords(ego_cart_pos[0], ego_cart_pos[1])[1]
        other_orient = other_context["lat_theta"]
        return (other_d < ego_d and other_orient > eps) or (other_d > ego_d and other_orient < -eps)

    # def _in_front_of_eval(self, world, time_step, lead_id, follow_id, follow_cart_pos, follow_theta, max_lon_dist=200.0):
    #     follow_vehicle = world.vehicle_by_id(follow_id)
    #     other_context = self._build_other_vehicle_predicate_context(
    #         road_network=world.road_network,
    #         other_vehicle=world.vehicle_by_id(lead_id),
    #         time_step=time_step,
    #     )
    #     _, in_front_of = self._evaluate_longitudinal_predicates(
    #         follow_vehicle=follow_vehicle,
    #         other_context=other_context,
    #         time_step=time_step,
    #         follow_cart_pos=follow_cart_pos,
    #         follow_velocity=getattr(follow_vehicle.states_cr.get(time_step), "velocity", 0.0),
    #         follow_theta=follow_theta,
    #         max_lon_dist=max_lon_dist,
    #     )
    #     return in_front_of
