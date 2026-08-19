import math
import os
import time
import traceback

import numpy as np
from z3 import sat

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory

from crrepairer.repairer.base import TrajectoryRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.repairer.vp.constraints import (
    UnsupportedVPCandidateError,
    VPConstraintExtraction,
)
from crrepairer.repairer.vp.domain import VPPredicateEstimation
from crrepairer.repairer.vp.optimization import VPOptimization
from crrepairer.repairer.vp.trajectory_context import VPTrajectoryContext
from crrepairer.repairer.vp.utils import VPUtils


class VPTrajectoryRepairer(
    VPPredicateEstimation,  # Estimates predicate domains for DomainDPLL SAT pruning.
    VPOptimization,  # Solves the VP linear program and builds repaired trajectories.
    VPConstraintExtraction,  # Converts selected propositions into VP bounds.
    VPTrajectoryContext,  # Provides ego trajectory, CLCS, and planner context helpers.
    VPUtils,  # Holds shared repair utilities such as proposition selection and tv re-checking.
    TrajectoryRepair,  # Common repair base class storing the current trajectory.
):
    """
    Trajectory repairer that uses SAT to select violated propositions and velocity planning
    to repair the trajectory from tv onward without searching tc.
    """

    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        ego_vehicle: DynamicObstacle,
        config: RepairerConfiguration,
    ):
        super().__init__(ego_vehicle.prediction.trajectory)
        self.rule_monitor = rule_monitor
        self._model = None
        # CommonRoad scenarios need not start at absolute time step zero (the
        # updated inD converter can produce ego trajectories beginning at 1 or
        # 2).  VP repairs from the first available ego state, so its fixed
        # cutoff must use that state rather than an out-of-range global zero.
        self._tc = int(getattr(ego_vehicle.initial_state, "time_step", 0))
        self._tv = -math.inf
        self.sat_solver = SATSolver(self.rule_monitor, config)
        self.config = config
        self.ego_vehicle = ego_vehicle
        self._sel_prop = None
        self._prop_full = None
        self.sat_reasoning_time = 0
        self.nr_iter = 0
        self.domain_dict_time = 0.0
        self.domain_dict = {}
        self._domain_dict_initialized = False
        self.runtime_breakdown = {}
        self.domain_dict_breakdown = {}
        self._domain_predicate_timing = {}
        self._hard_domain_vars = set()
        self._repair_literals = []
        self.candidate_tvs = []
        self.candidate_diagnostics = []
        self._use_monitor_conflict_geometry = False

    @property
    def tv(self):
        return self._tv

    @property
    def core_runtime(self):
        """Runtime of the proposed repair method, excluding monitor validation."""
        return sum(
            elapsed
            for stage, elapsed in self.runtime_breakdown.items()
            if stage != "compliance_check"
        )

    @property
    def tc(self):
        return self._tc

    @property
    def target_vehicle(self):
        return self.config.scenario.obstacle_by_id(self.rule_monitor.other_id)

    @property
    def model(self):
        """
        SAT models.
        """
        return self._model

    def repair(self, check_flag=True, *args, **kwargs):
        # A repairer can be reused.  Each call starts optimistically with the
        # inexpensive trajectory CLCS and switches to preprocessing only if
        # the current scenario demonstrates that it needs the stable path.
        self._force_trajectory_clcs_preprocess = False
        self._shared_trajectory_clcs = None
        self._tv = self.rule_monitor.tv_time_step
        if self._tv in (-math.inf, math.inf):
            return None
        if self.sat_solver.solver_mode == "domain_dpll":
            self.ensure_domain_dict_initialized()

        nr = 1
        start_time = time.time()
        self.runtime_breakdown = {
            "domain_dict": self.domain_dict_time if self._domain_dict_initialized else 0.0,
            "sat": 0.0,
            "clcs": 0.0,
            "constraint_extraction": 0.0,
            "constraint_conversion": 0.0,
            "lp": 0.0,
            "trajectory_build": 0.0,
            "compliance_check": 0.0,
        }
        self.candidate_tvs = []
        self.candidate_diagnostics = []
        print("******** Velocity-Planning Trajectory Repairing starts! ********")
        while True:
            sat_start_time = time.time()
            solve_result = self.sat_solver.solve()
            if solve_result != sat:
                self.runtime_breakdown["sat"] += time.time() - sat_start_time
                break

            self.nr_iter += 1
            print("* {}. iteration...".format(nr))
            if self.rule_monitor.proposition_nodes is None:
                self.runtime_breakdown["sat"] += time.time() - sat_start_time
                return None

            select_proposition, self._model = self.sat_solver.model()
            print(f"selected proposition: {select_proposition}")
            sat_elapsed = time.time() - sat_start_time
            self.sat_reasoning_time += sat_elapsed
            self.runtime_breakdown["sat"] += sat_elapsed
            print("* \t<SATSolver>: SAT reasoning time: {:.3f}s".format(self.sat_reasoning_time))

            self._assign_proposition(
                select_proposition,
                list(self._model),
            )
            try:
                repaired_traj = self._repair_with_velocity_planning()
            except UnsupportedVPCandidateError as exc:
                print(f"* \t<VPRepairer>: rejecting unsupported SAT model: {exc}")
                self.candidate_diagnostics.append(
                    {
                        "status": "unsupported_candidate",
                        "selected": [prop.name for prop in self._sel_prop],
                        "error": str(exc),
                    }
                )
                repaired_traj = None
            except Exception as exc:
                print(f"* \t<VPRepairer>: VP repair failed for current SAT model: {exc}")
                if os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
                    traceback.print_exc()
                self.candidate_diagnostics.append(
                    {
                        "status": "planning_failed",
                        "selected": [prop.name for prop in self._sel_prop],
                        "error": str(exc),
                    }
                )
                repaired_traj = None
                if self._retry_with_preprocessed_trajectory_clcs():
                    print(
                        "* \t<VPRepairer>: retrying the same SAT model with "
                        "preprocessed trajectory CLCS"
                    )
                    # The next loop reuses the same logical assignment; this
                    # is a geometry fallback, not another SAT iteration.
                    self.nr_iter -= 1
                    nr += 1
                    continue
            if repaired_traj is not None:
                candidate_tv = math.inf
                if check_flag:
                    compliance_start_time = time.time()
                    try:
                        candidate_tv, _ = self.calc_tv_updated(
                            repaired_traj.state_list,
                            self._tc,
                        )
                    except Exception as exc:
                        print(
                            "* \t<VPRepairer>: compliance check failed for current "
                            f"SAT model: {exc}"
                        )
                        candidate_tv = -math.inf
                    finally:
                        self.runtime_breakdown["compliance_check"] += (
                            time.time() - compliance_start_time
                        )
                    self.candidate_tvs.append(candidate_tv)
                    ego_vehicle = getattr(self, "ego_vehicle", None)
                    prediction = getattr(ego_vehicle, "prediction", None)
                    original_trajectory = getattr(prediction, "trajectory", None)
                    original_by_time = {
                        state.time_step: state
                        for state in getattr(original_trajectory, "state_list", [])
                    }
                    velocity_deltas = []
                    position_deltas = []
                    changed_time_steps = []
                    selected_props = getattr(self, "_sel_prop", None) or []
                    for state in repaired_traj.state_list:
                        original = original_by_time.get(state.time_step)
                        if original is None:
                            continue
                        velocity_delta = abs(
                            float(getattr(state, "velocity", 0.0))
                            - float(getattr(original, "velocity", 0.0))
                        )
                        position_delta = float(
                            np.linalg.norm(
                                np.asarray(state.position, dtype=float)
                                - np.asarray(original.position, dtype=float)
                            )
                        )
                        velocity_deltas.append(velocity_delta)
                        position_deltas.append(position_delta)
                        if velocity_delta > 1e-6 or position_delta > 1e-6:
                            changed_time_steps.append(int(state.time_step))
                    self.candidate_diagnostics.append(
                        {
                            "status": "checked",
                            "selected": [prop.name for prop in selected_props],
                            "selected_assignments": [
                                {
                                    "name": prop.name,
                                    "literal": prop.alphabet,
                                    "initial_ttv_value": prop.ttv_value,
                                    "desired_positive": not prop.alphabet.startswith("~"),
                                }
                                for prop in selected_props
                            ],
                            "updated_tv": candidate_tv,
                            "max_velocity_delta": max(velocity_deltas, default=0.0),
                            "max_position_delta": max(position_deltas, default=0.0),
                            "first_changed_time_step": (
                                min(changed_time_steps) if changed_time_steps else None
                            ),
                            **(
                                {
                                    "predicate_debug": self._last_candidate_predicate_debug
                                }
                                if getattr(self, "_last_candidate_predicate_debug", None)
                                else {}
                            ),
                            **(
                                {"constraint_debug": self._last_constraint_debug}
                                if getattr(self, "_last_constraint_debug", None)
                                else {}
                            ),
                        }
                    )

                if not check_flag or (
                    math.isinf(candidate_tv) and candidate_tv > 0
                ):
                    core_total_time = self.core_runtime
                    print(f"----- Computation Time: {time.time() - start_time:.3f}s -----")
                    print(f"*****  Successfully Repaired in {self.nr_iter} iteration(s)! •ᴗ•  *****")
                    print(
                        "----- Core Time details ----- "
                        f"\n***** Domain Dict: {self.runtime_breakdown['domain_dict']:.6f}s"
                        f"\n***** SAT: {self.runtime_breakdown['sat']:.6f}s"
                        f"\n***** Trajectory CLCS: {self.runtime_breakdown['clcs']:.6f}s"
                        f"\n***** Constraint Extraction: {self.runtime_breakdown['constraint_extraction']:.6f}s"
                        f"\n***** Constraint Conversion: {self.runtime_breakdown['constraint_conversion']:.6f}s"
                        f"\n***** LP: {self.runtime_breakdown['lp']:.6f}s"
                        f"\n***** Trajectory Build: {self.runtime_breakdown['trajectory_build']:.6f}s"
                        f"\n***** Monitor Validation (excluded): {self.runtime_breakdown['compliance_check']:.6f}s"
                        f"\n***** Repair Method Total: {core_total_time:.6f}s"
                    )
                    return repaired_traj

                print(
                    "* \t<VPRepairer>: candidate remains non-compliant "
                    f"(updated TV={candidate_tv}); trying another SAT model"
                )

                if self._retry_with_preprocessed_trajectory_clcs():
                    print(
                        "* \t<VPRepairer>: retrying the same SAT model with "
                        "preprocessed trajectory CLCS"
                    )
                    # The next loop reuses the same logical assignment; this
                    # is a geometry fallback, not another SAT iteration.
                    self.nr_iter -= 1
                    nr += 1
                    continue

                selected_negative_conflict = any(
                    "in_intersection_conflict_area" in prop.name
                    and prop.alphabet.startswith("~")
                    for prop in (getattr(self, "_sel_prop", None) or [])
                )
                if (
                    selected_negative_conflict
                    and not self._use_monitor_conflict_geometry
                ):
                    self._use_monitor_conflict_geometry = True
                    print(
                        "* \t<VPRepairer>: retrying the same SAT model with "
                        "monitor-aligned conflict geometry"
                    )
                    nr += 1
                    continue

            self._use_monitor_conflict_geometry = False
            self.sat_solver.update_formula()
            if self.sat_solver.solver_mode == "domain_dpll":
                domain_solver = getattr(self.sat_solver, "_dpll_solver", None)
                relaxed_domains = (
                    domain_solver.relax_domains_for_model(self.model)
                    if domain_solver is not None
                    else []
                )
                if relaxed_domains:
                    self.domain_dict = dict(domain_solver.domains)
                    self.sat_solver.set_domain_dict(
                        self.domain_dict,
                        hard_domain_vars=self._hard_domain_vars,
                        repair_literals=self._repair_literals,
                    )
                    print(
                        "* \t<VPRepairer>: relaxed domains selected by failed model: "
                        f"{relaxed_domains}"
                    )
            nr += 1

        print(f"*******   Repairing Failed ಠ_ಠ with {nr} iteration(s)  *******")
        return None

    def _repair_with_velocity_planning(self) -> Trajectory:
        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        clcs_start_time = time.time()
        trajectory_clcs, ref_path = self._get_shared_trajectory_clcs(all_states)
        self.runtime_breakdown["clcs"] += time.time() - clcs_start_time
        cl_trajectory_before = self._convert_states_to_clcs(all_states, lanelet_clcs)

        constraint_extraction_start_time = time.time()
        trajectory_s_min_cap = None
        trajectory_s_max_cap = None
        if self.config.repair.constraint_mode == 2:
            s_min, s_max, v_min, v_max = self._extract_constraints_from_corridor()
        elif self.config.repair.constraint_mode == 1:
            if any(rule in self.config.repair.rules for rule in ("R_IN1", "R_IN4", "R_IN3_hand_draft", "R_IN5")):
                (
                    s_min,
                    s_max,
                    v_min,
                    v_max,
                    trajectory_s_min_cap,
                    trajectory_s_max_cap,
                ) = (
                    self._extract_intersection_constraints_manually(
                        all_states,
                        lanelet_clcs,
                        trajectory_clcs,
                        cl_trajectory_before,
                        ref_path,
                    )
                )
            elif any(rule in self.config.repair.rules for rule in ("R_G1", "R_G3", "R_G2")):
                s_min, s_max, v_min, v_max = self._extract_interstate_constraints_manually(
                    all_states,
                    lanelet_clcs,
                )
            else:
                raise ValueError(
                    f"Unsupported rules for constraint extraction: {self.config.repair.rules}"
                )
        else:
            raise ValueError(
                f"Unsupported constraint_mode: {self.config.repair.constraint_mode}"
            )
        self.runtime_breakdown["constraint_extraction"] += time.time() - constraint_extraction_start_time

        constraint_conversion_start_time = time.time()
        est_s_min, est_s_max, est_v_min, est_v_max = (
            self._convert_lanelet_constraints_to_trajectory_constraints(
                s_min,
                s_max,
                v_min,
                v_max,
                all_states,
                ref_path,
                lanelet_clcs,
                trajectory_clcs,
                cl_trajectory_before,
                trajectory_s_min_cap=trajectory_s_min_cap,
                trajectory_s_max_cap=trajectory_s_max_cap,
            )
        )
        self.runtime_breakdown["constraint_conversion"] += time.time() - constraint_conversion_start_time

        s_hat = self._build_reference_longitudinal_positions(all_states, trajectory_clcs)
        amin, amax, jmin, jmax = self._get_longitudinal_planning_limits()
        s0, v0 = self._get_velocity_planning_initial_conditions(
            all_states,
            trajectory_clcs,
        )
        if not (
            self.config.repair.rules == ["R_G3"]
            and self._initial_conditions_within_bounds(
                s0,
                v0,
                est_s_min,
                est_s_max,
                est_v_min,
                est_v_max,
            )
        ):
            s0, v0 = None, None

        lp_start_time = time.time()

        sol = self._solve_velocity_planning_lp(
            dt=dt,
            s_hat=s_hat,
            vmin=np.asarray(est_v_min),
            vmax=np.asarray(est_v_max),
            smin=np.asarray(est_s_min),
            smax=np.asarray(est_s_max),
            amin=amin,
            amax=amax,
            jmin=jmin,
            jmax=jmax,
            time_offset=int(self._tc) + 1,
            s0=s0,
            v0=v0,
        )

        self.runtime_breakdown["lp"] += time.time() - lp_start_time
        if os.environ.get("CRREPAIR_VP_PREDICATE_DEBUG"):
            self._last_constraint_debug = {
                "extraction": getattr(self, "_last_extraction_debug", []),
                "s_hat": np.asarray(s_hat, dtype=float).round(9).tolist(),
                "s_min": np.asarray(est_s_min, dtype=float).round(9).tolist(),
                "s_max": np.asarray(est_s_max, dtype=float).round(9).tolist(),
                "solution_s": np.asarray(sol["s"], dtype=float).round(9).tolist(),
                "solution_v": np.asarray(sol["v"], dtype=float).round(9).tolist(),
            }

        trajectory_build_start_time = time.time()
        repaired_trajectory = self._build_repaired_trajectory(
            all_states,
            trajectory_clcs,
            sol["s"],
            sol["v"],
            dt,
        )
        self.runtime_breakdown["trajectory_build"] += time.time() - trajectory_build_start_time
        return repaired_trajectory
