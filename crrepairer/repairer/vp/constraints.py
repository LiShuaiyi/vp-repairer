"""Constraint extraction helpers for velocity-planning repair."""

import math
import os
import time
from typing import List, Union

import numpy as np
import shapely
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Polygon
from sympy.logic.boolalg import simplify_logic

from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.state import CustomState

from commonroad_clcs.clcs import CurvilinearCoordinateSystem

from crmonitor.common.world import World

from crrepairer.repairer.vp.semantic_predicate_regions import (
    _project_points_to_s,
)
from crrepairer.repairer.vp.temporal import (
    constraint_steps_for_anchors,
    constraint_time_interval,
)


class UnsupportedVPCandidateError(RuntimeError):
    """A SAT assignment that cannot be represented by VP constraints."""


class AccelerationExitStepInfeasibleError(RuntimeError):
    """The selected acceleration exit deadline is dynamically infeasible."""


class VPConstraintExtraction:
    """Extracts longitudinal position and velocity constraints for VP repair."""

    def _reject_unsupported_vp_candidates(self):
        """Return whether missing VP semantics must reject the SAT model.

        Plain DPLL intentionally preserves the original permissive baseline:
        unsupported literals add no LP constraint. DomainDPLL uses the
        explicit controllable/fixed/unsupported classification.
        """
        sat_solver = getattr(self, "sat_solver", None)
        return getattr(sat_solver, "solver_mode", "domain_dpll") != "dpll"

    def _proposition_sequence(self, proposition):
        """Return the monitor sequence associated with one proposition node."""
        try:
            rule_index = self.rule_monitor._rules.index(proposition.source_rule)
            by_vehicle = self.rule_monitor.all_props_all_ids_all[rule_index].get(
                proposition.name,
                {},
            )
            return self.rule_monitor._safe_prop_sequence(
                by_vehicle,
                self.rule_monitor.other_id,
                self.rule_monitor.vehicle_id,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return ()

    def _implication_active_source_anchors(
        self,
        proposition,
        trajectory_start,
        trajectory_end,
    ):
        """Find source anchors where ``proposition`` is critical to the rule.

        A temporal leaf constraint is unnecessary at an anchor where changing
        the selected literal cannot change the Boolean rule value--most
        importantly, while the antecedent of an implication is false.  The
        Per-frame monitor values are substituted only for hard/fixed
        propositions; VP-controllable consequent variables remain symbolic.
        Missing evaluations are retained conservatively.

        ``None`` asks the caller to use the legacy all-anchor expansion.  This
        is used before a SAT model exists (during domain estimation) and if the
        monitor formula cannot be evaluated reliably.
        """
        if getattr(self, "_model", None) is None:
            return None
        if not (
            getattr(self, "_vp_repair_mode", "deceleration") == "acceleration"
            and "in_intersection_conflict_area" in proposition.name
        ):
            return None

        try:
            formula = self.rule_monitor.sat_formula_sep[proposition.source_rule]
            formula_symbols = {str(symbol): symbol for symbol in formula.free_symbols}
        except (AttributeError, KeyError, TypeError):
            return None

        target_variable = proposition.alphabet[-1]
        target_symbol = formula_symbols.get(target_variable)
        if target_symbol is None:
            return None

        rule_propositions = {}
        for node in self.rule_monitor.proposition_nodes:
            if node.source_rule != proposition.source_rule:
                continue
            variable = node.alphabet[-1]
            if variable in formula_symbols:
                rule_propositions[variable] = node
        if set(formula_symbols) - set(rule_propositions):
            return None

        hard_variables = set(getattr(self, "_hard_domain_vars", set()))
        fixed_variables = hard_variables & set(formula_symbols)
        if not fixed_variables:
            return None
        constrained_variables = {
            node.alphabet[-1]
            for node in (getattr(self, "_sel_prop", None) or ())
        }
        sequences = {
            variable: self._proposition_sequence(rule_propositions[variable])
            for variable in fixed_variables
        }

        start_time_step = int(self.rule_monitor.start_time_step)
        active_anchors = []
        filtered_anchors = []
        unknown_anchors = []
        anchor_evaluations = []
        for source_anchor in range(int(trajectory_start), int(trajectory_end) + 1):
            # ``all_props_all_ids_all`` is indexed by the monitor's evaluation
            # frame.  For a pastified future rule this is already the delayed
            # anchor.  The temporal leaf offsets are applied below, but the
            # global ``future_time_step`` must not be added a second time.
            evaluation_index = source_anchor - start_time_step
            assignment = {}
            complete = True
            for variable in sorted(fixed_variables):
                sequence = sequences[variable]
                if not (0 <= evaluation_index < len(sequence)):
                    complete = False
                    break
                value = sequence[evaluation_index]
                try:
                    numeric_value = float(value)
                    if math.isnan(numeric_value):
                        complete = False
                        break
                    assignment[formula_symbols[variable]] = bool(numeric_value >= 0.0)
                except (TypeError, ValueError):
                    complete = False
                    break

            if not complete:
                # Unknown anchors must remain constrained: filtering them would
                # be an unsound under-approximation.
                active_anchors.append(source_anchor)
                unknown_anchors.append(source_anchor)
                continue

            try:
                substituted_formula = formula.subs(assignment)
                # Symbol membership alone is not a semantic relevance test.
                # For example, ``~j | (~j & ~n)`` still contains ``n``
                # syntactically but is equivalent to ``~j``.  Expanding n's
                # temporal window at such an anchor imposes an artificial
                # early exit deadline.  Hard/fixed substitution leaves only a
                # few VP-controllable symbols here, so simplify before testing
                # whether this particular target can affect the rule value.
                reduced_formula = simplify_logic(
                    substituted_formula, force=True
                )
                target_relevant = target_symbol in reduced_formula.free_symbols
                if (
                    not target_relevant
                    and reduced_formula not in (True, False)
                ):
                    # A different VP-controllable literal may absorb this
                    # target (e.g. ``~l | (~l & ~p) == ~l``).  Dropping p is
                    # sound only if l is itself being enforced by the current
                    # VP candidate.  PropositionNode.ttv_value is evaluated at
                    # one reference frame and cannot prove l's polarity at all
                    # implication anchors.  If the absorbing symbols are not
                    # constrained, retain the target conservatively.
                    remaining_variables = {
                        str(symbol) for symbol in reduced_formula.free_symbols
                    }
                    if not remaining_variables.issubset(constrained_variables):
                        target_relevant = (
                            target_symbol in substituted_formula.free_symbols
                        )
            except (TypeError, ValueError):
                return None
            anchor_evaluations.append(
                {
                    "source_anchor": source_anchor,
                    "target_relevant": target_relevant,
                    "reduced_formula": str(reduced_formula),
                    "hard_values": {
                        variable: assignment[formula_symbols[variable]]
                        for variable in sorted(hard_variables & set(formula_symbols))
                    },
                }
            )

            # Substituting only hard/fixed facts leaves the VP-controllable
            # consequent symbolic.  The target disappears exactly when the
            # fixed implication guard makes it irrelevant at this anchor.
            if target_relevant:
                active_anchors.append(source_anchor)
            else:
                filtered_anchors.append(source_anchor)

        self._last_implication_anchor_debug = {
            "proposition": proposition.name,
            "active_source_anchors": active_anchors,
            "filtered_source_anchors": filtered_anchors,
            "unknown_source_anchors": unknown_anchors,
            "anchor_evaluations": anchor_evaluations,
            "formula": str(formula),
        }
        return active_anchors

    def _temporal_constraint_steps(self, all_states, propositions=None):
        """Map each selected proposition to its VP leaf-predicate frames."""
        start = time.perf_counter()
        trajectory_start = int(all_states[0].time_step)
        trajectory_end = int(all_states[-1].time_step)
        planning_start = int(self._tc) + 1
        dt = float(self.config.scenario.dt)
        future_time_step = int(self.rule_monitor.future_time_step)
        propositions = self._sel_prop if propositions is None else propositions
        propositions = tuple(propositions or ())
        model_key = tuple(sorted(str(item) for item in (self._model or ())))
        cache_key = (
            getattr(self, "_vp_repair_mode", "deceleration"),
            model_key,
            tuple(id(prop) for prop in propositions),
            trajectory_start,
            trajectory_end,
            planning_start,
            dt,
            future_time_step,
        )
        cache = getattr(self, "_temporal_constraint_steps_cache", None)
        if cache is None:
            cache = {}
            self._temporal_constraint_steps_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            active_steps, diagnostics = cached
            self._last_temporal_expansion_time = time.perf_counter() - start
            self._last_temporal_expansion_debug = diagnostics
            return active_steps

        active_steps = {}
        diagnostics = []
        for prop in propositions:
            source_anchors = self._implication_active_source_anchors(
                proposition=prop,
                trajectory_start=trajectory_start,
                trajectory_end=trajectory_end,
            )
            if source_anchors is None:
                interval, expansion, pair_count = constraint_time_interval(
                    expression=prop.name,
                    dt=dt,
                    trajectory_start=trajectory_start,
                    planning_start=planning_start,
                    trajectory_end=trajectory_end,
                    future_time_step=future_time_step,
                )
            else:
                interval, expansion, pair_count = constraint_steps_for_anchors(
                    expression=prop.name,
                    dt=dt,
                    source_anchors=source_anchors,
                    planning_start=planning_start,
                    trajectory_end=trajectory_end,
                    # These anchors come from the already-pastified monitor
                    # sequences, hence they are delayed evaluation anchors.
                    # Applying ``future_time_step`` again would shift the leaf
                    # window twice (e.g. evaluation anchor 9 for once[0,1]
                    # must constrain frames 4..9, not 9..14 for dt=0.2).
                    future_time_step=0,
                )
            active_steps[id(prop)] = interval
            diagnostics.append(
                {
                    "proposition": prop.name,
                    "operators": expansion.operators,
                    "leaf_expression": expansion.leaf_expression,
                    "offsets": expansion.offsets,
                    "active_start": interval.start,
                    "active_end": interval.end,
                    "active_count": interval.count,
                    "represented_anchor_offset_pairs": pair_count,
                    "implication_aware": source_anchors is not None,
                    "source_anchor_count": (
                        len(source_anchors) if source_anchors is not None else None
                    ),
                    "universalized": any(
                        operator in {"once", "eventually"}
                        for operator in expansion.operators
                    ),
                }
            )
        self._last_temporal_expansion_time = time.perf_counter() - start
        self._last_temporal_expansion_debug = diagnostics
        cache[cache_key] = (active_steps, diagnostics)
        return active_steps

    def _extract_interstate_constraints_manually(
        self,
        all_states: List[CustomState],
        lanelet_clcs: CurvilinearCoordinateSystem,
    ):
        """Build VP bounds directly from selected traffic-rule propositions."""
        start_idx = int(self._tc - all_states[0].time_step)
        horizon = len(all_states) - start_idx - 1
        if horizon <= 0:
            raise RuntimeError("No horizon available after tv for VP repair.")

        lane_start = lanelet_clcs.convert_to_curvilinear_coords(
            float(all_states[0].position[0]),
            float(all_states[0].position[1]),
        )[0]
        lane_end = lanelet_clcs.convert_to_curvilinear_coords(
            float(all_states[-1].position[0]),
            float(all_states[-1].position[1]),
        )[0]

        s_min = np.ones(horizon) * min(lane_start, lane_end)
        s_max = np.ones(horizon) * max(lane_start, lane_end)
        v_min = np.zeros(horizon)
        v_max = np.ones(horizon) * math.inf
        speed_limits = self._extract_speed_limit_values() 
        temporal_steps = self._temporal_constraint_steps(all_states)

        final_time_step = all_states[-1].time_step
        for time_step in range(int(self._tc) + 1, final_time_step + 1):
            idx = time_step - int(self._tc) - 1
            follow_velocity = all_states[time_step - all_states[0].time_step].velocity
            v_max_list = []
            s_max_list = []
            s_min_list = []

            for prop in self._sel_prop:
                if not temporal_steps[id(prop)].contains(time_step):
                    continue
                if "distance" in prop.name:
                    s_up, v_up = self._constraint_keep_safe_distance(
                        world=self.rule_monitor.world,
                        lanelet_clcs=lanelet_clcs,
                        time_step=time_step,
                        lead_id=self.rule_monitor.other_id,
                        follow_id=self.ego_vehicle.obstacle_id,
                        follow_velocity=follow_velocity,
                    )
                    s_max_list.append(s_up)
                    v_max_list.append(v_up)
                elif "lane" in prop.name and "same" in prop.name:
                    if prop.alphabet.startswith("~"):
                        if self._reject_unsupported_vp_candidates():
                            raise UnsupportedVPCandidateError(
                                "Negative in-same-lane RG literal is not representable "
                                f"by the positive VP lane constraint: {prop.name} "
                                f"({prop.alphabet})."
                            )
                    s_low, s_up = self._constraint_in_same_lane(
                        world=self.rule_monitor.world,
                        lanelet_clcs=lanelet_clcs,
                        time_step=time_step,
                        other_id=self.rule_monitor.other_id,
                        ego_id=self.ego_vehicle.obstacle_id,
                        t_c=int(self._tc),
                        t_f=final_time_step,
                    )
                    if s_low is None or s_up is None:
                        raise RuntimeError(
                            f"Infeasible lane constraint at time step {time_step} with prop {prop.name}."
                        )
                    s_up_safe, v_up_safe = self._constraint_keep_safe_distance(
                        world=self.rule_monitor.world,
                        lanelet_clcs=lanelet_clcs,
                        time_step=time_step,
                        lead_id=self.rule_monitor.other_id,
                        follow_id=self.ego_vehicle.obstacle_id,
                        follow_velocity=follow_velocity,
                    )
                    s_min_list.append(s_low)
                    s_max_list.append(min(s_up, s_up_safe))
                    v_max_list.append(v_up_safe)
                elif "lane" in prop.name and "speed" in prop.name and speed_limits["lane"] is not None:
                    v_max_list.append(speed_limits["lane"] - 0.01)
                elif "type" in prop.name and "speed" in prop.name and speed_limits["type"] is not None:
                    v_max_list.append(speed_limits["type"] - 0.01)
                elif "fov" in prop.name and "speed" in prop.name and speed_limits["fov"] is not None:
                    v_max_list.append(speed_limits["fov"] - 0.01)
                elif "brake" in prop.name and "speed" in prop.name and speed_limits["brake"] is not None:
                    v_max_list.append(speed_limits["brake"] - 0.01)
                elif "brakes_abruptly" in prop.name:
                    self._constraint_not_break_abruptly(prop)
                else:
                    if self._reject_unsupported_vp_candidates():
                        raise UnsupportedVPCandidateError(
                            "Unsupported RG predicate has no VP constraint: "
                            f"{prop.name} ({prop.alphabet})."
                        )
                    raise RuntimeError(
                        "Unsupported RG predicate has no VP constraint: "
                        f"{prop.name} ({prop.alphabet})."
                    )
                v_max_list.append(follow_velocity)
                # print('v_max candidates at time step {}: {}'.format(time_step, v_max_list))

            v_max[idx] = min(v_max_list) if v_max_list else math.inf
            s_max[idx] = min(s_max_list) if s_max_list else max(lane_start, lane_end)
            s_min[idx] = max(s_min_list) if s_min_list else min(lane_start, lane_end)

        return s_min, s_max, v_min, v_max

    def _constraint_not_break_abruptly(self, prop):
        for predicate in prop.children:
            if "abrupt" not in getattr(predicate, "base_name", ""):
                continue
            a_abrupt = predicate.evaluator.config.get("a_abrupt")
            if a_abrupt is None:
                continue
            qp_veh_config = self.config.vehicle.qp_veh_config
            qp_veh_config.a_lon_min = max(qp_veh_config.a_lon_min, a_abrupt)
            return

    def _extract_intersection_constraints_manually(
        self,
        all_states: List[CustomState],
        lanelet_clcs: CurvilinearCoordinateSystem,
        trajectory_clcs: CurvilinearCoordinateSystem,
        cl_trajectory_before: List[np.ndarray],
        ref_path: np.ndarray,
    ):
        if not any(
            rule in self.config.repair.rules
            for rule in ("R_IN1", "R_IN3", "R_IN3_hand_draft", "R_IN4", "R_IN5")
        ):
            raise NotImplementedError(
                "Intersection VP constraints currently support R_IN1, R_IN3, "
                "R_IN3_hand_draft, R_IN4 and R_IN5 only."
            )

        start_idx = int(self._tc - all_states[0].time_step)
        horizon = len(all_states) - start_idx - 1
        if horizon <= 0:
            raise RuntimeError("No horizon available after tc for IN-series VP repair.")

        lane_start = cl_trajectory_before[0][0]
        lane_end = cl_trajectory_before[-1][0]
        s_min = np.ones(horizon) * min(lane_start, lane_end)
        s_max = np.ones(horizon) * max(lane_start, lane_end)
        if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
            # The recorded violating trajectory may stop inside the conflict
            # area.  Its final lane progress is not a physical upper bound for
            # the route-extended acceleration reference path.
            s_max[:] = math.inf
        v_min = np.zeros(horizon)
        v_max = np.ones(horizon) * math.inf

        ct_s_min = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[0][0]),
            float(ref_path[0][1]),
        )[0]
        ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[-1][0]),
            float(ref_path[-1][1]),
        )[0]
        if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
            ct_s_max = float(trajectory_clcs.length())
        trajectory_s_min_cap = np.ones(horizon) * ct_s_min
        trajectory_s_max_cap = np.ones(horizon) * ct_s_max
        first_plan_state = all_states[min(start_idx + 1, len(all_states) - 1)]
        first_plan_s_trajectory = trajectory_clcs.convert_to_curvilinear_coords(
            float(first_plan_state.position[0]),
            float(first_plan_state.position[1]),
        )[0]

        wheelbase = self._get_planner_wheelbase()
        final_time_step = all_states[-1].time_step
        temporal_steps = self._temporal_constraint_steps(all_states)
        self._last_extraction_debug = []
        in1_trajectory_stop_cap = None
        if "R_IN1" in self.config.repair.rules:
            in1_trajectory_stop_cap = self._constraint_stop_line_on_trajectory(
                self.rule_monitor.world,
                self.rule_monitor.world.vehicle_by_id(self.config.repair.ego_id),
                ref_path,
                trajectory_clcs,
            )
        conflict_trajectory_interval = None
        if any(
            rule in self.config.repair.rules
            for rule in ("R_IN3", "R_IN3_hand_draft", "R_IN4", "R_IN5")
        ):
            conflict_trajectory_interval = (
                self._get_intersection_conflict_trajectory_interval(
                    lanelet_clcs=lanelet_clcs,
                    trajectory_clcs=trajectory_clcs,
                    ref_path=ref_path,
                    wheelbase=wheelbase,
                )
            )

        self._validate_intersection_candidate_support(
            temporal_steps=temporal_steps,
            conflict_trajectory_interval=conflict_trajectory_interval,
        )

        for prop in self._sel_prop:
            prop_debug_recorded = False
            for time_step in range(int(self._tc) + 1, final_time_step + 1):
                if not temporal_steps[id(prop)].contains(time_step):
                    continue
                idx = time_step - int(self._tc) - 1
                if "R_IN1" in self.config.repair.rules:
                    if "stop_line" in prop.name:
                        upper_bound = self._constraint_stop_line(
                            self.rule_monitor.world,
                            self.rule_monitor.world.vehicle_by_id(self.config.repair.ego_id),
                            wheelbase,
                            lanelet_clcs,
                        )
                        s_max[idx] = min(s_max[idx], upper_bound)
                        if in1_trajectory_stop_cap is not None:
                            trajectory_s_max_cap[idx] = min(
                                trajectory_s_max_cap[idx],
                                in1_trajectory_stop_cap,
                            )
                        if not prop_debug_recorded:
                            self._last_extraction_debug.append(
                                {
                                    "proposition": prop.name,
                                    "kind": "stop_line_upper",
                                    "upper_bound_lane_clcs": float(upper_bound),
                                    "upper_bound_trajectory_clcs": (
                                        None
                                        if in1_trajectory_stop_cap is None
                                        else float(in1_trajectory_stop_cap)
                                    ),
                                }
                            )
                            prop_debug_recorded = True
                    else:
                        if self._reject_unsupported_vp_candidates():
                            raise UnsupportedVPCandidateError(
                                "IN1 SAT candidate requires a non-stop-line predicate "
                                f"that VP cannot constrain: {prop.name} ({prop.alphabet})."
                            )
                    continue

                if "in_intersection_conflict_area" in prop.name:
                    prop_assignment = -1 if prop.alphabet.startswith("~") else 1
                    if prop_assignment > 0:
                        continue
                    if conflict_trajectory_interval is None:
                        if self._reject_unsupported_vp_candidates():
                            raise UnsupportedVPCandidateError(
                                "Conflict geometry is unavailable for VP SAT candidate: "
                                f"{prop.name} ({prop.alphabet})."
                            )
                        continue

                    before_upper, after_lower, interval_mode = (
                        conflict_trajectory_interval
                    )
                    start_s = float(first_plan_s_trajectory)
                    repair_mode = getattr(
                        self, "_vp_repair_mode", "deceleration"
                    )
                    if repair_mode == "acceleration":
                        semantic_builder = getattr(
                            self, "_semantic_in_region_builder", None
                        )
                        if semantic_builder is not None:
                            semantic_region = semantic_builder.ego_conflict_region()
                            if semantic_region.complete and semantic_region.outer_true:
                                # ``outer_true`` covers every path position at
                                # which the monitor's Boolean conflict predicate
                                # may still hold.  Its final endpoint is thus a
                                # certified acceleration-side exit boundary on
                                # the processed trajectory path itself.
                                semantic_after_lower = max(
                                    interval.upper
                                    for interval in semantic_region.outer_true
                                )
                                if semantic_after_lower > after_lower:
                                    after_lower = float(semantic_after_lower)
                                    interval_mode = (
                                        f"{interval_mode}_semantic_outer"
                                    )
                        # The temporal operator has already selected the exact
                        # frames at which this negative conflict predicate must
                        # hold.  The acceleration branch chooses the spatial
                        # disjunct after the conflict region at every one of
                        # those frames; it must not postpone the requirement to
                        # an exit step guessed from an acceleration profile.
                        if start_s >= after_lower:
                            branch = "acceleration_already_after_conflict"
                        else:
                            branch = "acceleration_after_conflict_lower"
                            trajectory_s_min_cap[idx] = max(
                                trajectory_s_min_cap[idx], after_lower
                            )
                    elif interval_mode == "legacy_in4":
                        branch = "legacy_conflict_rear_upper"
                        if (
                            not os.environ.get(
                                "CRREPAIR_VP_DISABLE_CONFLICT_CAP_CLAMP"
                            )
                            and before_upper < ct_s_min
                        ):
                            before_upper = max(start_s, before_upper)
                        trajectory_s_max_cap[idx] = min(
                            trajectory_s_max_cap[idx], before_upper
                        )
                    elif interval_mode == "legacy_interval":
                        if (
                            not os.environ.get(
                                "CRREPAIR_VP_DISABLE_CONFLICT_CAP_CLAMP"
                            )
                            and before_upper < ct_s_min
                        ):
                            before_upper = max(start_s, before_upper)
                        if start_s <= before_upper:
                            branch = "legacy_before_conflict_upper"
                            trajectory_s_max_cap[idx] = min(
                                trajectory_s_max_cap[idx], before_upper
                            )
                        elif start_s >= after_lower:
                            branch = "legacy_after_conflict_lower"
                            trajectory_s_min_cap[idx] = max(
                                trajectory_s_min_cap[idx], after_lower
                            )
                        elif start_s - before_upper <= after_lower - start_s:
                            branch = "legacy_inside_choose_before_upper"
                            trajectory_s_max_cap[idx] = min(
                                trajectory_s_max_cap[idx], before_upper
                            )
                        else:
                            branch = "legacy_inside_choose_after_lower"
                            trajectory_s_min_cap[idx] = max(
                                trajectory_s_min_cap[idx], after_lower
                            )
                    elif start_s <= before_upper:
                        branch = "before_conflict_upper"
                        trajectory_s_max_cap[idx] = min(
                            trajectory_s_max_cap[idx], before_upper
                        )
                    elif start_s >= after_lower:
                        # The first plannable state is already beyond the
                        # monitored interval; braking cannot make it re-enter.
                        branch = "already_after_conflict"
                    else:
                        # A deceleration-only repair cannot move an ego that is
                        # already inside the monitored conflict interval back
                        # to its entrance.  Preserve the contradiction so this
                        # SAT assignment is rejected instead of producing an
                        # unrelated trajectory.
                        branch = "inside_conflict_unreachable_before"
                        trajectory_s_max_cap[idx] = min(
                            trajectory_s_max_cap[idx], before_upper
                        )
                    if not prop_debug_recorded:
                        self._last_extraction_debug.append(
                            {
                                "proposition": prop.name,
                                "kind": "monitor_conflict_interval",
                                "start_s": start_s,
                                "before_upper": float(before_upper),
                                "after_lower": float(after_lower),
                                "interval_mode": interval_mode,
                                "repair_mode": repair_mode,
                                "branch": branch,
                            }
                        )
                        prop_debug_recorded = True
                else:
                    if self._reject_unsupported_vp_candidates():
                        raise UnsupportedVPCandidateError(
                            "Unsupported IN predicate has no VP constraint: "
                            f"{prop.name} ({prop.alphabet})."
                        )
                    raise RuntimeError(
                        "Unsupported IN predicate has no VP constraint: "
                        f"{prop.name} ({prop.alphabet})."
                    )
        
        return s_min, s_max, v_min, v_max, trajectory_s_min_cap, trajectory_s_max_cap

    def _get_intersection_conflict_trajectory_interval(
        self,
        lanelet_clcs,
        trajectory_clcs,
        ref_path,
        wheelbase=None,
    ):
        """Return the shared entry/exit interval used by estimation and LP."""
        geometry_mode = bool(
            getattr(self, "_use_monitor_conflict_geometry", False)
        )
        cache = getattr(self, "_conflict_trajectory_interval_cache", None)
        if cache is None:
            cache = {}
            self._conflict_trajectory_interval_cache = cache
        if geometry_mode in cache:
            return cache[geometry_mode]

        world = self.rule_monitor.world
        ego = world.vehicle_by_id(self.config.repair.ego_id)
        target = world.vehicle_by_id(self.rule_monitor.other_id)
        if wheelbase is None:
            wheelbase = self._get_planner_wheelbase()

        conflict_interval = None
        if geometry_mode:
            monitor_interval = self._monitor_conflict_interval_on_trajectory(
                world=world,
                ego_vehicle=ego,
                target_vehicle=target,
                ref_path=ref_path,
                lanelet_clcs=lanelet_clcs,
                trajectory_clcs=trajectory_clcs,
            )
            if monitor_interval is not None:
                conflict_interval = (
                    monitor_interval[0],
                    monitor_interval[1],
                    "monitor",
                )
        else:
            conflict_interval = self._legacy_conflict_interval_on_trajectory(
                world=world,
                ego_vehicle=ego,
                target_vehicle=target,
                trajectory_clcs=trajectory_clcs,
                wheelbase=wheelbase,
            )
            # Geometry representation is not part of the repair action.  Use
            # the monitor-aligned construction if the legacy interval cannot
            # represent this otherwise controllable predicate.
            if conflict_interval is None:
                monitor_interval = self._monitor_conflict_interval_fallback(
                    world=world,
                    ego_vehicle=ego,
                    target_vehicle=target,
                    ref_path=ref_path,
                    lanelet_clcs=lanelet_clcs,
                    trajectory_clcs=trajectory_clcs,
                )
                if monitor_interval is not None:
                    conflict_interval = (
                        monitor_interval[0],
                        monitor_interval[1],
                        "monitor",
                    )

        if (
            conflict_interval is not None
            and getattr(self, "_vp_repair_mode", "deceleration")
            == "acceleration"
        ):
            # The reference path is sampled at 0.1 m and monitor overlap uses
            # a strict polygon boundary.  Requiring one additional sample past
            # the mapped rear-clearance boundary prevents a numerically
            # boundary-touching LP solution from remaining non-compliant.
            conflict_interval = (
                conflict_interval[0],
                conflict_interval[1] + 0.1,
                conflict_interval[2],
            )

        cache[geometry_mode] = conflict_interval
        return conflict_interval

    def _monitor_conflict_interval_fallback(self, **kwargs):
        """Evaluate monitor geometry under its required geometry-mode flag."""
        previous_geometry_mode = getattr(
            self, "_use_monitor_conflict_geometry", False
        )
        self._use_monitor_conflict_geometry = True
        try:
            return self._monitor_conflict_interval_on_trajectory(**kwargs)
        finally:
            self._use_monitor_conflict_geometry = previous_geometry_mode

    def _validate_intersection_candidate_support(
        self,
        temporal_steps,
        conflict_trajectory_interval,
    ):
        """Reject selected SAT literals that cannot produce a VP constraint.

        Fixed predicates are filtered by SAT hard domains.  This validation is
        the final guard for genuinely unsupported selected literals, so the LP
        is never solved with a silently missing constraint.
        """
        if not self._reject_unsupported_vp_candidates():
            return
        is_in1 = "R_IN1" in self.config.repair.rules
        for prop in self._sel_prop:
            interval = temporal_steps[id(prop)]
            if interval.count == 0:
                continue
            if is_in1 and "stop_line" not in prop.name:
                self._last_extraction_debug.append(
                    {
                        "proposition": prop.name,
                        "kind": "unsupported_in1_non_stop_line",
                    }
                )
                raise UnsupportedVPCandidateError(
                    "IN1 SAT candidate requires a non-stop-line predicate "
                    f"that VP cannot constrain: {prop.name} ({prop.alphabet})."
                )
            if (
                not is_in1
                and "in_intersection_conflict_area" in prop.name
                and prop.alphabet.startswith("~")
                and conflict_trajectory_interval is None
            ):
                self._last_extraction_debug.append(
                    {
                        "proposition": prop.name,
                        "kind": "conflict_geometry_unavailable",
                    }
                )
                raise UnsupportedVPCandidateError(
                    "Conflict geometry is unavailable for VP SAT candidate: "
                    f"{prop.name} ({prop.alphabet})."
                )

    def _constraint_in_same_lane(
        self,
        world: World,
        lanelet_clcs: CurvilinearCoordinateSystem,
        time_step,
        other_id,
        ego_id,
        t_c,
        t_f,
    ):
        other_vehicle = world.vehicle_by_id(other_id)
        ego_vehicle = world.vehicle_by_id(ego_id)
        if not self._vehicle_has_valid_time_step(other_vehicle, time_step):
            return None, None

        steps = []
        ego_lane = set()
        other_lane_at_time = other_vehicle.get_lane(time_step)
        if other_lane_at_time is None:
            return None, None
        other_lane = {other_lane_at_time}
        for t in range(t_c, t_f + 1):
            try:
                ego_lane.add(ego_vehicle.get_lane(t))
            except Exception:
                continue
            common_lanelets = other_lane & ego_lane
            if common_lanelets:
                steps.append(t)

        if not steps:
            return None, None

        start = steps[0]
        end = steps[-1]
        start_cart_pos = ego_vehicle.states_cr[start].position
        end_cart_pos = ego_vehicle.states_cr[end].position
        start_cl_pos = lanelet_clcs.convert_to_curvilinear_coords(start_cart_pos[0], start_cart_pos[1])
        end_cl_pos = lanelet_clcs.convert_to_curvilinear_coords(end_cart_pos[0], end_cart_pos[1])
        return start_cl_pos[0], end_cl_pos[0]

    def _constraint_stop_line(
        self,
        world: World,
        ego_vehicle,
        wheelbase: float,
        lanelet_clcs: CurvilinearCoordinateSystem,
    ) -> float:
        upper_bound = np.inf
        for lanelet_id in ego_vehicle.lanelets_dir:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is not None:
                stop_line_s = min(
                    lanelet_clcs.convert_to_curvilinear_coords(*lanelet.stop_line.start)[0],
                    lanelet_clcs.convert_to_curvilinear_coords(*lanelet.stop_line.end)[0],
                )
                upper_bound = min(
                    upper_bound,
                    stop_line_s
                    - ego_vehicle.circle_radius
                    - ego_vehicle.shape.length / 3
                    - wheelbase / 2,
                )
        return upper_bound

    def _constraint_stop_line_on_trajectory(
        self,
        world: World,
        ego_vehicle,
        ref_path: np.ndarray,
        trajectory_clcs: CurvilinearCoordinateSystem,
        clearance: float = 0.05,
    ):
        """Map the monitor stop line into the finite repair-path coordinates.

        The lane CLCS and the finite trajectory CLCS use different longitudinal
        origins and domains.  Converting a lane-CLCS stop coordinate through a
        Cartesian point can therefore fall outside the trajectory projection
        domain.  A monotone longitudinal mapping along the shared reference
        path also handles a stop line just beyond the recorded trajectory.
        """
        lanelet_clcs = ego_vehicle.ref_path_lane.clcs
        points = np.asarray(ref_path, dtype=float)
        lane_values = _project_points_to_s(lanelet_clcs, points)
        trajectory_values = _project_points_to_s(trajectory_clcs, points)
        lane_progress = []
        trajectory_progress = []
        for lane_s, trajectory_s in zip(lane_values, trajectory_values):
            if not (np.isfinite(lane_s) and np.isfinite(trajectory_s)):
                continue
            if lane_progress and lane_s <= lane_progress[-1] + 1e-9:
                continue
            lane_progress.append(float(lane_s))
            trajectory_progress.append(float(trajectory_s))
        if len(lane_progress) < 2:
            return None

        def map_progress(lane_s):
            if lane_s < lane_progress[0]:
                scale = (
                    (trajectory_progress[1] - trajectory_progress[0])
                    / (lane_progress[1] - lane_progress[0])
                )
                return trajectory_progress[0] + scale * (lane_s - lane_progress[0])
            if lane_s > lane_progress[-1]:
                scale = (
                    (trajectory_progress[-1] - trajectory_progress[-2])
                    / (lane_progress[-1] - lane_progress[-2])
                )
                return trajectory_progress[-1] + scale * (lane_s - lane_progress[-1])
            return float(np.interp(lane_s, lane_progress, trajectory_progress))

        candidate_caps = []
        lanelet_ids = set(ego_vehicle.ref_path_lane.contained_lanelets)
        for lanelet_id in lanelet_ids:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(
                lanelet_id
            )
            if lanelet.stop_line is None:
                continue
            stop_lane_s = min(
                lanelet_clcs.convert_to_curvilinear_coords(
                    *lanelet.stop_line.start
                )[0],
                lanelet_clcs.convert_to_curvilinear_coords(
                    *lanelet.stop_line.end
                )[0],
            )
            candidate_caps.append(float(map_progress(stop_lane_s)))

        if not candidate_caps:
            return None

        first_time = min(ego_vehicle.states_cr)
        state = ego_vehicle.states_cr[first_time]
        center_s = lanelet_clcs.convert_to_curvilinear_coords(
            float(state.position[0]), float(state.position[1])
        )[0]
        front_s = ego_vehicle.front_s(first_time, ego_vehicle.ref_path_lane)
        front_extent = (
            max(0.0, float(front_s) - float(center_s))
            if front_s is not None
            else ego_vehicle.shape.length / 2
        )
        stop_line_cap = min(candidate_caps) - front_extent - clearance

        # A stop-line cap alone still lets the LP choose a much later stop.
        # IN1 can be made vacuously compliant by never crossing, so also cap
        # progress at the position reached by braking from the initial frame.
        # This uses the same discrete integration as the dedicated feasibility
        # audit and remains within the configured longitudinal acceleration.
        initial_s = trajectory_clcs.convert_to_curvilinear_coords(
            float(state.position[0]), float(state.position[1])
        )[0]
        speed = max(0.0, float(getattr(state, "velocity", 0.0)))
        acceleration_min = min(
            0.0, float(self.config.vehicle.qp_veh_config.a_lon_min)
        )
        dt = float(world.dt)
        braking_distance = 0.0
        while speed > 1e-9:
            next_speed = max(0.0, speed + acceleration_min * dt)
            braking_distance += 0.5 * (speed + next_speed) * dt
            if next_speed >= speed:
                break
            speed = next_speed
        immediate_brake_cap = float(initial_s) + braking_distance
        result_cap = min(stop_line_cap, immediate_brake_cap)

        if os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
            print(
                "* \t<VPRepairer>: trajectory stop-line candidates: "
                f"{candidate_caps}, front_extent={front_extent}, "
                f"clearance={clearance}, stop_line_cap={stop_line_cap}, "
                f"immediate_brake_cap={immediate_brake_cap}"
            )
        return result_cap

    def _get_planner_wheelbase(self):
        qp_veh_config = self.config.vehicle.qp_veh_config
        return getattr(qp_veh_config, "wheelbase")

    @staticmethod
    def _vehicle_has_valid_time_step(vehicle, time_step: int) -> bool:
        if vehicle is None:
            return False

        start_time = getattr(vehicle, "start_time", None)
        if start_time is None:
            initial_state = getattr(vehicle, "initial_state", None)
            start_time = getattr(initial_state, "time_step", None)

        end_time = getattr(vehicle, "end_time", None)
        if end_time is None:
            prediction = getattr(vehicle, "prediction", None)
            end_time = getattr(prediction, "final_time_step", None)

        lanelet_assignment = getattr(vehicle, "lanelet_assignment", None)
        has_lanelet_assignment = lanelet_assignment is not None

        if start_time is None or end_time is None:
            return False

        if time_step < start_time or time_step > end_time:
            return False

        if hasattr(vehicle, "is_valid"):
            try:
                if not vehicle.is_valid(time_step):
                    return False
            except Exception:
                state_at_time = getattr(vehicle, "state_at_time", None)
                if state_at_time is None or state_at_time(time_step) is None:
                    return False
        else:
            state_at_time = getattr(vehicle, "state_at_time", None)
            if state_at_time is not None and state_at_time(time_step) is None:
                return False

        if has_lanelet_assignment and time_step not in lanelet_assignment:
            return False

        return (
            True
        )

    def _constraint_keep_safe_distance(
        self,
        world: World,
        lanelet_clcs: CurvilinearCoordinateSystem,
        time_step,
        lead_id,
        follow_id,
        follow_velocity,
        delta_s=0.5,
    ):
        def calculate_safe_distance(v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow):
            return (
                (v_lead ** 2) / (-2 * np.abs(a_min_lead))
                - (v_follow ** 2) / (-2 * np.abs(a_min_follow))
                + v_follow * t_react_follow
            )

        vehicle_follow = world.vehicle_by_id(follow_id)
        vehicle_lead = world.vehicle_by_id(lead_id)
        follow_length = vehicle_follow.shape.length
        if not self._vehicle_has_valid_time_step(vehicle_lead, time_step):
            return math.inf, math.inf
        if vehicle_lead.get_lane(time_step) is None:
            return math.inf, math.inf

        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = calculate_safe_distance(
            follow_velocity + 1.0,
            vehicle_lead.states_cr[time_step].velocity,
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )
        lead_rear_s = lanelet_clcs.convert_to_curvilinear_coords(
            vehicle_lead.states_cr[time_step].position[0],
            vehicle_lead.states_cr[time_step].position[1],
        )[0]
        lead_rear_s = lead_rear_s - vehicle_lead.shape.length / 2
        s = lead_rear_s - safe_distance - follow_length - delta_s
        return s, follow_velocity

    def _convert_lanelet_constraints_to_trajectory_constraints(
        self,
        s_min,
        s_max,
        v_min,
        v_max,
        all_states,
        ref_path,
        lanelet_clcs,
        trajectory_clcs,
        cl_trajectory_before,
        trajectory_s_min_cap=None,
        trajectory_s_max_cap=None,
        apply_curvature_limits=True,
    ):
        estimated_s_min = []
        estimated_s_max = []
        estimated_v_min = []
        estimated_v_max = []

        ds = np.gradient(ref_path[:, 0])
        dd = np.gradient(ref_path[:, 1])
        eps = 1e-6
        ds_safe = np.where(np.abs(ds) < eps, eps, ds)
        ratio_1_cos = np.sqrt(1.0 + (dd / ds_safe) ** 2)
        rmax = np.max(ratio_1_cos)
        rmin = np.min(ratio_1_cos)

        ct_s_min = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[0][0]),
            float(ref_path[0][1]),
        )[0]
        ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[-1][0]),
            float(ref_path[-1][1]),
        )[0]

        for i in range(len(s_min)):
            d = cl_trajectory_before[i][1]
            try:
                if not np.isfinite(s_min[i]):
                    raise ValueError("s_min outside projection domain")
                min_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(
                    float(s_min[i]), float(d)
                )
            except Exception:
                min_lane_to_cart = (float(ref_path[0][0]), float(ref_path[0][1]))
            try:
                if not np.isfinite(s_max[i]):
                    raise ValueError("s_max outside projection domain")
                max_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(
                    float(s_max[i]), float(d)
                )
            except Exception:
                max_lane_to_cart = (float(ref_path[-1][0]), float(ref_path[-1][1]))

            try:
                min_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(
                    float(min_lane_to_cart[0]),
                    float(min_lane_to_cart[1]),
                )[0]
            except Exception:
                min_cart_to_traj = ct_s_min
            try:
                max_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(
                    float(max_lane_to_cart[0]),
                    float(max_lane_to_cart[1]),
                )[0]
            except Exception:
                max_cart_to_traj = ct_s_max
            if (
                getattr(self, "_vp_repair_mode", "deceleration")
                == "acceleration"
                and not np.isfinite(s_max[i])
            ):
                max_cart_to_traj = ct_s_max

            s_min_traj = min_cart_to_traj
            if trajectory_s_min_cap is not None:
                s_min_traj = max(s_min_traj, trajectory_s_min_cap[i])
            estimated_s_min.append(s_min_traj)
            s_max_traj = max_cart_to_traj
            if trajectory_s_max_cap is not None:
                s_max_traj = min(s_max_traj, trajectory_s_max_cap[i])
            estimated_s_max.append(s_max_traj)
            estimated_v_max.append(v_max[i] * rmin)
            estimated_v_min.append(v_min[i] * rmax)

        if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
            # All temporal/proposition requirements for one frame have already
            # been consolidated into these arrays by repeated max(lower) and
            # min(upper) updates.  Detect an empty merged interval before doing
            # curvature work or constructing the LP; adding separate rows for
            # the original requirements would be redundant.
            merged_s_min = np.asarray(estimated_s_min, dtype=float)
            merged_s_max = np.asarray(estimated_s_max, dtype=float)
            infeasible_frames = np.flatnonzero(merged_s_min > merged_s_max)
            if len(infeasible_frames):
                first = int(infeasible_frames[0])
                raise AccelerationExitStepInfeasibleError(
                    "Merged acceleration position constraints are infeasible at "
                    f"time_step={int(self._tc) + first + 1}: "
                    f"smin={merged_s_min[first]}, smax={merged_s_max[first]}"
                )
            estimated_s_min = merged_s_min.tolist()
            estimated_s_max = merged_s_max.tolist()

        if apply_curvature_limits:
            curvature_s_min = estimated_s_min
            curvature_s_max = estimated_s_max
            if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
                reachable_s_min, reachable_s_max = (
                    self._acceleration_reachable_s_intervals(
                        all_states,
                        trajectory_clcs,
                        len(estimated_s_min),
                    )
                )
                curvature_s_min = np.maximum(
                    np.asarray(estimated_s_min, dtype=float),
                    reachable_s_min,
                )
                curvature_s_max = np.minimum(
                    np.asarray(estimated_s_max, dtype=float),
                    reachable_s_max,
                )
                # An empty intersection is already position-infeasible for the
                # LP.  Query curvature at its required lower boundary rather
                # than swapping endpoints and scanning an unrelated segment.
                curvature_s_max = np.maximum(curvature_s_min, curvature_s_max)
                self._last_curvature_reachable_intervals = [
                    (float(lower), float(upper))
                    for lower, upper in zip(curvature_s_min, curvature_s_max)
                ]
            if getattr(self, "_vp_repair_mode", "deceleration") == "acceleration":
                curvature_v_max = self._acceleration_curvature_velocity_limits(
                    trajectory_clcs,
                    curvature_s_min,
                    curvature_s_max,
                    self.config.vehicle.qp_veh_config.a_lat_max,
                )
            else:
                curvature_v_max = self._curvature_velocity_limits(
                    trajectory_clcs,
                    curvature_s_min,
                    curvature_s_max,
                    self.config.vehicle.qp_veh_config.a_lat_max,
                )
            estimated_v_max = np.minimum(
                np.asarray(estimated_v_max, dtype=float),
                curvature_v_max,
            ).tolist()

        return estimated_s_min, estimated_s_max, estimated_v_min, estimated_v_max

    def _acceleration_reachable_s_intervals(
        self,
        all_states,
        trajectory_clcs,
        horizon,
    ):
        """Return a sound forward reachable ``s`` envelope for acceleration VP."""
        current_s, current_v, current_a = (
            self._get_velocity_planning_current_conditions(
                all_states,
                trajectory_clcs,
            )
        )
        if current_s is None or horizon <= 0:
            return np.zeros(horizon), np.full(horizon, float(trajectory_clcs.length()))

        dt = float(self.config.scenario.dt)
        amin, amax, jmin, jmax = self._get_longitudinal_planning_limits()
        v_max = float(self.config.vehicle.qp_veh_config.v_lon_max)
        path_max = float(trajectory_clcs.length())

        lower_s = upper_s = float(current_s)
        lower_v = upper_v = max(0.0, float(current_v))
        lower_a = upper_a = float(np.clip(current_a, amin, amax))
        reachable_lower = []
        reachable_upper = []
        for _ in range(horizon):
            next_lower_a = max(float(amin), lower_a + float(jmin) * dt)
            next_upper_a = min(float(amax), upper_a + float(jmax) * dt)
            next_lower_v = max(0.0, lower_v + next_lower_a * dt)
            next_upper_v = min(v_max, upper_v + next_upper_a * dt)
            lower_s = max(
                float(current_s),
                lower_s + 0.5 * (lower_v + next_lower_v) * dt,
            )
            upper_s = min(
                path_max,
                upper_s + 0.5 * (upper_v + next_upper_v) * dt,
            )
            reachable_lower.append(lower_s)
            reachable_upper.append(upper_s)
            lower_v, upper_v = next_lower_v, next_upper_v
            lower_a, upper_a = next_lower_a, next_upper_a
        return (
            np.asarray(reachable_lower, dtype=float),
            np.asarray(reachable_upper, dtype=float),
        )

    @staticmethod
    def _smoothed_curvature_profile(
        trajectory_clcs,
        smoothing_distance=2.0,
    ):
        """Estimate path curvature from a de-jittered heading profile.

        The planning CLCS is deliberately left unchanged.  Its reference path
        is already sampled at roughly 0.1 m, so smoothing the unwrapped heading
        and differentiating it with respect to longitudinal progress is enough
        to suppress localization-induced vertex spikes without another CLCS or
        Cartesian/curvilinear remapping pass.
        """

        def values(attribute):
            value = getattr(trajectory_clcs, attribute)
            return np.asarray(value() if callable(value) else value, dtype=float).reshape(-1)

        positions = values("ref_pos")
        headings = values("ref_theta")
        if len(positions) != len(headings) or len(positions) < 5:
            raise ValueError("Trajectory CLCS has no usable heading profile.")
        finite = np.isfinite(positions) & np.isfinite(headings)
        positions = positions[finite]
        headings = headings[finite]
        order = np.argsort(positions)
        positions = positions[order]
        headings = headings[order]
        keep = np.r_[True, np.diff(positions) > 1e-9]
        positions = positions[keep]
        headings = headings[keep]
        if len(positions) < 5:
            raise ValueError("Trajectory CLCS heading samples are degenerate.")

        step = float(np.median(np.diff(positions)))
        window_length = max(5, int(round(float(smoothing_distance) / step)))
        if window_length % 2 == 0:
            window_length += 1
        maximum_window = len(positions) if len(positions) % 2 else len(positions) - 1
        window_length = min(window_length, maximum_window)
        smoothed_heading = savgol_filter(
            np.unwrap(headings),
            window_length=window_length,
            polyorder=min(3, window_length - 2),
            mode="interp",
        )
        curvature = np.abs(np.gradient(smoothed_heading, positions))
        if not np.all(np.isfinite(curvature)):
            raise ValueError("Smoothed trajectory curvature is non-finite.")
        return positions, curvature

    def _acceleration_curvature_velocity_limits(
        self,
        trajectory_clcs,
        s_min,
        s_max,
        a_lat_max,
        curvature_epsilon=1e-9,
    ):
        """Acceleration-only cached curvature range queries.

        The smoothed path profile is invariant across SAT candidates.  A
        sparse-table range maximum preserves the exact sample maximum used by
        :meth:`_curvature_velocity_limits` while replacing a full Boolean mask
        and scan for every planning frame with two ``searchsorted`` calls and
        an O(1) maximum query.
        """
        if len(s_min) != len(s_max):
            raise ValueError(
                f"s_min and s_max must have equal lengths: {len(s_min)} != {len(s_max)}"
            )
        if not np.isfinite(a_lat_max) or a_lat_max <= 0:
            raise ValueError(f"a_lat_max must be positive and finite, got {a_lat_max!r}")
        path_length = float(trajectory_clcs.length())
        if not np.isfinite(path_length) or path_length <= 0:
            raise ValueError(f"Invalid trajectory CLCS length: {path_length!r}")

        cache = getattr(self, "_acceleration_curvature_cache", None)
        if cache is None:
            cache = {}
            self._acceleration_curvature_cache = cache
        key = id(trajectory_clcs)
        profile_data = cache.get(key)
        if profile_data is None:
            try:
                positions, profile = self._smoothed_curvature_profile(
                    trajectory_clcs
                )
            except (AttributeError, TypeError, ValueError):
                # Preserve the CLCS-native fallback used by the original
                # implementation for unusual/mock coordinate systems.
                return self._curvature_velocity_limits(
                    trajectory_clcs,
                    s_min,
                    s_max,
                    a_lat_max,
                    curvature_epsilon=curvature_epsilon,
                )
            levels = [np.asarray(profile, dtype=float)]
            span = 2
            while span <= len(profile):
                half = span // 2
                previous = levels[-1]
                levels.append(
                    np.maximum(previous[:-half], previous[half:])
                )
                span *= 2
            profile_data = (positions, profile, levels)
            cache.clear()
            cache[key] = profile_data
        positions, profile, levels = profile_data

        def sampled_maximum(lower, upper):
            left = int(np.searchsorted(positions, lower, side="left"))
            right = int(np.searchsorted(positions, upper, side="right"))
            if left >= right:
                return -math.inf
            count = right - left
            level = count.bit_length() - 1
            width = 1 << level
            return max(
                float(levels[level][left]),
                float(levels[level][right - width]),
            )

        limits = np.full(len(s_min), math.inf, dtype=float)
        for t, (lower, upper) in enumerate(zip(s_min, s_max)):
            lower = float(lower)
            upper = float(upper)
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = 0.0, path_length
            else:
                lower, upper = min(lower, upper), max(lower, upper)
                lower = min(max(lower, 0.0), path_length)
                upper = min(max(upper, 0.0), path_length)
            boundary_values = np.interp(
                (lower, upper), positions, profile
            )
            max_curvature = max(
                float(np.max(boundary_values)),
                sampled_maximum(lower, upper),
            )
            if max_curvature > curvature_epsilon:
                limits[t] = math.sqrt(float(a_lat_max) / max_curvature)
        return limits

    @staticmethod
    def _curvature_velocity_limits(
        trajectory_clcs,
        s_min,
        s_max,
        a_lat_max,
        curvature_epsilon=1e-9,
    ):
        """Compute the conservative curvature speed limit for each time step.

        The maximum absolute curvature over each feasible longitudinal interval
        is used so that ``v <= sqrt(a_lat_max / kappa_max)`` remains linear.
        """
        if len(s_min) != len(s_max):
            raise ValueError(
                f"s_min and s_max must have equal lengths: {len(s_min)} != {len(s_max)}"
            )
        if not np.isfinite(a_lat_max) or a_lat_max <= 0:
            raise ValueError(f"a_lat_max must be positive and finite, got {a_lat_max!r}")

        path_length = float(trajectory_clcs.length())
        if not np.isfinite(path_length) or path_length <= 0:
            raise ValueError(f"Invalid trajectory CLCS length: {path_length!r}")
        try:
            curvature_positions, curvature_profile = (
                VPConstraintExtraction._smoothed_curvature_profile(
                    trajectory_clcs
                )
            )
        except (AttributeError, TypeError, ValueError):
            curvature_positions = curvature_profile = None
        limits = np.full(len(s_min), math.inf, dtype=float)
        for t, (lower, upper) in enumerate(zip(s_min, s_max)):
            lower = float(lower)
            upper = float(upper)
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = 0.0, path_length
            else:
                lower, upper = min(lower, upper), max(lower, upper)
                lower = min(max(lower, 0.0), path_length)
                upper = min(max(upper, 0.0), path_length)

            if curvature_positions is None:
                curvature_min, curvature_max = trajectory_clcs.curvature_range(
                    lower, upper
                )
                if not np.isfinite(curvature_min) or not np.isfinite(curvature_max):
                    raise ValueError(
                        "Trajectory CLCS returned a non-finite curvature range at "
                        f"t={t}: ({curvature_min}, {curvature_max})"
                    )
                max_curvature = max(
                    abs(float(curvature_min)), abs(float(curvature_max))
                )
            else:
                inside = (
                    (curvature_positions >= lower)
                    & (curvature_positions <= upper)
                )
                boundary_values = np.interp(
                    (lower, upper),
                    curvature_positions,
                    curvature_profile,
                )
                max_curvature = float(np.max(boundary_values))
                if np.any(inside):
                    max_curvature = max(
                        max_curvature,
                        float(np.max(curvature_profile[inside])),
                    )

            if max_curvature > curvature_epsilon:
                limits[t] = math.sqrt(float(a_lat_max) / max_curvature)
        return limits

    def _find_conflict_points(
        self,
        curved_line: LineString,
        conflict_polygon: Union[Polygon, LineString],
    ):
        conflict_line_points = []
        intersection = curved_line.intersection(conflict_polygon)
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif intersection.geom_type in ("LineString", "LinearRing"):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif intersection.geom_type in ("MultiPoint", "MultiLineString"):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            return None
        return [conflict_line_points[0], conflict_line_points[-1]]

    def _create_conflict_area_parameter(
        self,
        ego_vehicle,
        target_vehicle,
        world: World,
        clcs=None,
        cart: bool = False,
    ):
        road_network = world.road_network
        conflict_lanelets_shape = []
        for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                conflict_lanelets_shape.append(lanelet.polygon.shapely_object)
        if not conflict_lanelets_shape:
            return np.array([np.inf, -np.inf])

        if not getattr(self, "_use_monitor_conflict_geometry", False):
            conflict_area_shape = shapely.unary_union(conflict_lanelets_shape)
            conflict_linestring = shapely.offset_curve(
                conflict_area_shape, ego_vehicle.circle_radius
            )
            traj_xy = [
                (
                    ego_vehicle.states_cr[t].position[0],
                    ego_vehicle.states_cr[t].position[1],
                )
                for t in ego_vehicle.states_cr
            ]
            points = self._find_conflict_points(
                LineString(traj_xy), conflict_linestring
            )
            if points is None:
                return np.array([np.inf, -np.inf])
            if cart:
                return points
            coordinate_system = clcs or ego_vehicle.ref_path_lane.clcs
            progress = sorted(
                coordinate_system.convert_to_curvilinear_coords(
                    float(point[0]), float(point[1])
                )[0]
                for point in points
            )
            return progress[0], progress[-1]

        # Match PredInIntersectionConflictArea's fallback geometry: intersect
        # the complete ego lane-direction centerline with every intersection
        # polygon on the target reference path.  The previous implementation
        # intersected the finite recorded ego trajectory with
        # ``offset_curve(Polygon)``.  For these polygons Shapely produces an
        # open offset line, so the result was commonly zero or one point; that
        # single point was then incorrectly used as both conflict endpoints.
        line_center = LineString(ego_vehicle.lanelets_dir_center_vertices)
        candidates = []
        for conflict_shape in conflict_lanelets_shape:
            points = self._find_conflict_points(line_center, conflict_shape)
            if points is None:
                continue
            for point in points:
                try:
                    progress = ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        float(point[0]), float(point[1])
                    )[0]
                except Exception:
                    progress = line_center.project(
                        shapely.Point(float(point[0]), float(point[1]))
                    )
                candidates.append((float(progress), np.asarray(point, dtype=float)))
        if not candidates:
            return np.array([np.inf, -np.inf])
        candidates.sort(key=lambda item: item[0])
        conflict_circle_center_center = [candidates[0][1], candidates[-1][1]]
        if cart:
            return conflict_circle_center_center

        if clcs is not None:
            s_circle_center_center = [
                clcs.convert_to_curvilinear_coords(*conflict_circle_center_center[0])[0],
                clcs.convert_to_curvilinear_coords(*conflict_circle_center_center[1])[0],
            ]
        else:
            s_circle_center_center = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_center[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_center[1]
                )[0],
            ]
        s_circle_center_center = np.sort(s_circle_center_center)
        return s_circle_center_center[0], s_circle_center_center[1]

    def _monitor_conflict_interval_on_trajectory(
        self,
        world: World,
        ego_vehicle,
        target_vehicle,
        ref_path: np.ndarray,
        lanelet_clcs: CurvilinearCoordinateSystem,
        trajectory_clcs: CurvilinearCoordinateSystem,
        clearance: float = 0.05,
    ):
        """Return monitor-aligned ego-center bounds in trajectory coordinates.

        Conflict endpoints are defined on the ego reference-lane CLCS, while
        the LP uses a finite trajectory CLCS.  Direct Cartesian conversion of
        an endpoint before/after that finite path raises a projection-domain
        error.  A monotone longitudinal mapping through the shared repair path
        keeps those out-of-horizon endpoints meaningful.
        """
        center_start, center_end = self._create_conflict_area_parameter(
            ego_vehicle,
            target_vehicle,
            world,
            clcs=lanelet_clcs,
            cart=False,
        )
        if not np.isfinite([center_start, center_end]).all():
            return None

        points = np.asarray(ref_path, dtype=float)
        lane_values = _project_points_to_s(lanelet_clcs, points)
        trajectory_values = _project_points_to_s(trajectory_clcs, points)
        raw_progress = [
            (float(lane_s), float(trajectory_s))
            for lane_s, trajectory_s in zip(lane_values, trajectory_values)
            if np.isfinite(lane_s) and np.isfinite(trajectory_s)
        ]
        if len(raw_progress) < 2:
            return None

        # The route used to extend a stopped trajectory and the lane CLCS used
        # by the monitor can describe the same path in opposite orientations.
        # Infer that direction before discarding non-monotone samples.
        lane_deltas = np.diff([item[0] for item in raw_progress])
        lane_deltas = lane_deltas[np.abs(lane_deltas) > 1e-9]
        if len(lane_deltas) == 0:
            return None
        lane_direction = 1.0 if float(np.median(lane_deltas)) >= 0.0 else -1.0
        monotone_progress = []
        for lane_s, trajectory_s in raw_progress:
            if monotone_progress:
                progress_delta = lane_direction * (
                    lane_s - monotone_progress[-1][0]
                )
                if progress_delta <= 1e-9:
                    continue
            monotone_progress.append((lane_s, trajectory_s))
        if lane_direction < 0.0:
            monotone_progress.reverse()

        lane_progress = [item[0] for item in monotone_progress]
        trajectory_progress = [item[1] for item in monotone_progress]
        if len(lane_progress) < 2:
            return None

        def map_progress(lane_s):
            if lane_s < lane_progress[0]:
                scale = (
                    (trajectory_progress[1] - trajectory_progress[0])
                    / (lane_progress[1] - lane_progress[0])
                )
                return trajectory_progress[0] + scale * (lane_s - lane_progress[0])
            if lane_s > lane_progress[-1]:
                scale = (
                    (trajectory_progress[-1] - trajectory_progress[-2])
                    / (lane_progress[-1] - lane_progress[-2])
                )
                return trajectory_progress[-1] + scale * (lane_s - lane_progress[-1])
            return float(np.interp(lane_s, lane_progress, trajectory_progress))

        center_start_ct, center_end_ct = sorted(
            (map_progress(center_start), map_progress(center_end))
        )
        first_time = min(ego_vehicle.states_cr)
        center_s = lanelet_clcs.convert_to_curvilinear_coords(
            *ego_vehicle.states_cr[first_time].position
        )[0]
        front_s = ego_vehicle.front_s(first_time, ego_vehicle.ref_path_lane)
        rear_s = ego_vehicle.rear_s(first_time, ego_vehicle.ref_path_lane)
        front_extent = max(0.0, float(front_s) - float(center_s))
        rear_extent = max(0.0, float(center_s) - float(rear_s))
        initial_trajectory_s = trajectory_clcs.convert_to_curvilinear_coords(
            *ego_vehicle.states_cr[first_time].position
        )[0]
        speed = max(
            0.0, float(getattr(ego_vehicle.states_cr[first_time], "velocity", 0.0))
        )
        acceleration_min = min(
            0.0, float(self.config.vehicle.qp_veh_config.a_lon_min)
        )
        dt = float(world.dt)
        braking_distance = 0.0
        while speed > 1e-9:
            next_speed = max(0.0, speed + acceleration_min * dt)
            braking_distance += 0.5 * (speed + next_speed) * dt
            if next_speed >= speed:
                break
            speed = next_speed
        immediate_brake_cap = float(initial_trajectory_s) + braking_distance
        return (
            min(
                float(center_start_ct - front_extent - clearance),
                immediate_brake_cap,
            ),
            float(center_end_ct + rear_extent + clearance),
        )

    def _legacy_conflict_interval_on_trajectory(
        self,
        world: World,
        ego_vehicle,
        target_vehicle,
        trajectory_clcs: CurvilinearCoordinateSystem,
        wheelbase: float,
    ):
        """Reproduce the original VP interval for first-pass compatibility."""
        points = self._create_conflict_area_parameter(
            ego_vehicle, target_vehicle, world, cart=True
        )
        if (
            len(points) != 2
            or not np.isfinite(np.asarray(points, dtype=float)).all()
        ):
            return None
        try:
            progress = [
                float(trajectory_clcs.convert_to_curvilinear_coords(
                    float(point[0]), float(point[1])
                )[0])
                for point in points
            ]
        except Exception:
            return None
        extent = wheelbase / 2 + self.ego_vehicle.obstacle_shape.length / 3
        if "R_IN4" in self.config.repair.rules:
            return float(progress[1] - extent), float(progress[1]), "legacy_in4"
        lower, upper = sorted(progress)
        return float(lower - extent), float(upper), "legacy_interval"

    def _constraint_in_intersection_conflict_area(
        self,
        time_step: int,
        prop_assignment: float,
        lanelet_clcs: CurvilinearCoordinateSystem = None,
        cart: bool = False,
    ):
        if prop_assignment > 0:
            print(
                f"* \t<VPRepairer>: time step {time_step}: conflict area constraint is not added"
            )
            return math.inf, -math.inf

        world = self.rule_monitor.world
        ego_vehicle = world.vehicle_by_id(self.config.repair.ego_id)
        target_vehicle = world.vehicle_by_id(self.rule_monitor.other_id)
        wheelbase = self._get_planner_wheelbase()
        s_circle_center_front, s_circle_center_rear = self._create_conflict_area_parameter(
            ego_vehicle,
            target_vehicle,
            world,
            lanelet_clcs,
            cart,
        )
        if cart:
            return s_circle_center_front, s_circle_center_rear

        front_constr = (
            s_circle_center_front - ego_vehicle.shape.length / 3 - wheelbase / 2
        )
        rear_constr = s_circle_center_rear
        return front_constr, rear_constr

   
