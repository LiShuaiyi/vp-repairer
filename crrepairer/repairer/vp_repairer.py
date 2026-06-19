import math
import time

import numpy as np
from z3 import sat

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory

from crrepairer.repairer.base import TrajectoryRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.repairer.vp.constraints import VPConstraintExtraction
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
        self._tc = 0
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

    def reset(
        self,
        config: RepairerConfiguration = None,
        rule_monitor: STLRuleMonitor = None,
        ego_vehicle: DynamicObstacle = None,
    ):
        self._tc = 0
        self._tv = -math.inf
        if config is not None:
            self.config = config
        if rule_monitor is not None:
            self.rule_monitor = rule_monitor
        if ego_vehicle is not None:
            self.ego_vehicle = ego_vehicle
            self._initial_trajectory = ego_vehicle.prediction.trajectory
        if self.sat_solver is not None and self.rule_monitor is not None:
            self.sat_solver.reset(
                config=self.config,
                rule_monitor=self.rule_monitor,
            )
        self._model = None
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

    @property
    def tv(self):
        return self._tv

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
            "constraint_extraction": 0.0,
            "constraint_conversion": 0.0,
            "lp": 0.0,
            "trajectory_build": 0.0,
            "compliance_check": 0.0,
        }
        print("******** Velocity-Planning Trajectory Repairing starts! ********")
        while self.sat_solver.solve() == sat:
            self.nr_iter += 1
            print("* {}. iteration...".format(nr))
            if self.rule_monitor.proposition_nodes is None:
                return None

            sat_start_time = time.time()
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
            except Exception as exc:
                print(f"* \t<VPRepairer>: VP repair failed for current SAT model: {exc}")
                repaired_traj = None
            if repaired_traj is not None:
                core_total_time = sum(self.runtime_breakdown.values())
                print(f"----- Computation Time: {time.time() - start_time:.3f}s -----")
                print(f"*****  Successfully Repaired in {self.nr_iter} iteration(s)! •ᴗ•  *****")
                print(
                    "----- Core Time details ----- "
                    f"\n***** Domain Dict: {self.runtime_breakdown['domain_dict']:.6f}s"
                    f"\n***** SAT: {self.runtime_breakdown['sat']:.6f}s"
                    f"\n***** Constraint Extraction: {self.runtime_breakdown['constraint_extraction']:.6f}s"
                    f"\n***** Constraint Conversion: {self.runtime_breakdown['constraint_conversion']:.6f}s"
                    f"\n***** LP: {self.runtime_breakdown['lp']:.6f}s"
                    f"\n***** Trajectory Build: {self.runtime_breakdown['trajectory_build']:.6f}s"
                    f"\n***** Compliance Check: {self.runtime_breakdown['compliance_check']:.6f}s"
                    f"\n***** Total: {core_total_time:.6f}s"
                )
                return repaired_traj

            self.sat_solver.update_formula()
            if self.sat_solver.solver_mode == "domain_dpll":
                domain_solver = getattr(self.sat_solver, "_dpll_solver", None)
                relaxed_domains = domain_solver.relax_domains_for_model(
                    self.model
                ) if domain_solver is not None else []
                if relaxed_domains:
                    self.domain_dict = dict(domain_solver.domains)
                    self.sat_solver.set_domain_dict(self.domain_dict)
                    print(
                        "* \t<VPRepairer>: relaxed DomainDPLL domains after failed model: "
                        f"{relaxed_domains}"
                    )
            nr += 1

        print(f"*******   Repairing Failed ಠ_ಠ with {nr} iteration(s)  *******")
        return None

    def _repair_with_velocity_planning(self) -> Trajectory:
        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        trajectory_clcs, ref_path = self._build_trajectory_clcs(all_states)
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
