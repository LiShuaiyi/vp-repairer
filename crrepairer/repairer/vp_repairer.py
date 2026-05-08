import copy
import math
import time

import numpy as np
from z3 import sat

from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory

from crrepairer.cut_off.utils import update_ego_vehicle
from crrepairer.repairer.base import TrajectoryRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.repairer.vp.constraints import VPConstraintExtraction
from crrepairer.repairer.vp.domain import VPPredicateEstimation
from crrepairer.repairer.vp.optimization import VPOptimization
from crrepairer.repairer.vp.trajectory_context import VPTrajectoryContext


class VPTrajectoryRepairer(
    VPPredicateEstimation,
    VPOptimization,
    VPConstraintExtraction,
    VPTrajectoryContext,
    TrajectoryRepair,
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
        self._tc = -math.inf
        self._tv = -math.inf
        self.sat_solver = SATSolver(self.rule_monitor, config)
        self.config = config
        self.ego_vehicle = ego_vehicle
        self._sel_prop = None
        self._prop_full = None
        self._vp_tc_time_step = 0
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

    @staticmethod
    def convert_traj_to_ego_vehicle(
        shape,
        initial_state,
        cr_trajectory: Trajectory,
        vehicle_id: int = 0,
    ) -> DynamicObstacle:
        pred = TrajectoryPrediction(cr_trajectory, shape)
        return DynamicObstacle(
            obstacle_id=vehicle_id,
            obstacle_type=ObstacleType.CAR,
            prediction=pred,
            obstacle_shape=shape,
            initial_state=initial_state,
        )

    def _assign_proposition(self, propositions, model):
        self._prop_full = propositions
        self._sel_prop = []
        for prop in propositions:
            if prop is not None and prop.alphabet in model:
                if (prop.ttv_value < 0 and prop.alphabet[0] != "~") or (
                    prop.ttv_value > 0 and prop.alphabet[0] == "~"
                ):
                    self._sel_prop.append(prop)
                    print(
                        f"* \t<VPRepairer>: selected propositions: "
                        f"{prop.alphabet[-1]} {prop.name} = {prop.ttv_value}"
                    )

    def calc_tv_updated(self, updated_states, cut_off_time=None):
        monitor = copy.copy(self.rule_monitor)
        world = copy.deepcopy(self.rule_monitor.world)
        monitor._world = world
        world_ego = world.vehicle_by_id(self.ego_vehicle.obstacle_id)
        update_ego_vehicle(
            world.road_network,
            world_ego,
            updated_states,
            0,
            world.dt,
        )

        rule_rob, other_ids = monitor.evaluate_consecutively(
            world,
            monitor.start_time_step,
        )
        if not all(len(arr) == len(rule_rob[0]) for arr in rule_rob):
            return -math.inf, None

        rule_rob = np.array(rule_rob)
        if np.any(rule_rob[:, 0] < 0):
            rule_idx = np.where(rule_rob[:, 0] < 0)[0][0]
            if other_ids[rule_idx][0] == ():
                return -math.inf, None
            return -math.inf, other_ids[rule_idx][0][0]

        tv_per_rule = np.argmax(rule_rob < 0, axis=-1)
        if np.all(tv_per_rule + world_ego.start_time == world_ego.start_time):
            return math.inf, None

        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if rule_idx == monitor.min_rule_idx:
            if other_ids[rule_idx][min_tv] == ():
                return min_tv * world.dt, self.ego_vehicle.obstacle_id
            return min_tv * world.dt, other_ids[rule_idx][min_tv][0]

        print("Violated rule changed.")
        return min_tv * world.dt, None

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
                # if check_flag:
                #     compliance_check_start_time = time.time()
                #     tv_updated, _ = self.calc_tv_updated(
                #         repaired_traj.state_list,
                #         self.tc,
                #     )
                #     self.runtime_breakdown["compliance_check"] += time.time() - compliance_check_start_time
                #     if tv_updated != math.inf:
                #         print("* \t<VPRepairer>: repaired trajectory still violates the rule")
                #         self.sat_solver.update_formula()
                #         nr += 1
                #         continue

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
            nr += 1

        print(f"*******   Repairing Failed ಠ_ಠ with {nr} iteration(s)  *******")
        return None

    def _repair_with_velocity_planning(self) -> Trajectory:
        if self._uses_in_series_processing():
            return self._repair_with_velocity_planning_in_series()

        self._prepare_fixed_cutoff_time()
        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        trajectory_clcs, ref_path = self._build_trajectory_clcs(all_states)
        cl_trajectory_before = self._convert_states_to_clcs(all_states, lanelet_clcs)

        constraint_extraction_start_time = time.time()
        if self.config.repair.constraint_mode == 2:
            s_min, s_max, v_min, v_max = self._extract_constraints_from_corridor()
        elif self.config.repair.constraint_mode == 1:
            s_min, s_max, v_min, v_max = self._extract_constraints_manually(
                all_states,
                lanelet_clcs,
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
            )
        )
        self.runtime_breakdown["constraint_conversion"] += time.time() - constraint_conversion_start_time

        s_hat = self._build_reference_longitudinal_positions(all_states, trajectory_clcs)
        amin, amax, jmin, jmax = self._get_longitudinal_planning_limits()
        s0, v0 = self._maybe_get_velocity_planning_initial_conditions(
            all_states,
            trajectory_clcs,
        )

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
            # s0=s0,
            # v0=v0,
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

    def _repair_with_velocity_planning_in_series(self) -> Trajectory:
        if self.config.repair.constraint_mode != 1:
            raise NotImplementedError(
                "IN-series VP repair currently supports constraint_mode == 1 only."
            )

        self._prepare_fixed_cutoff_time()
        all_states = self._get_states_with_initial()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        trajectory_clcs, ref_path = self._build_trajectory_clcs(all_states)
        cl_trajectory_before = self._convert_states_to_clcs(all_states, lanelet_clcs)

        constraint_extraction_start_time = time.time()
        s_min, s_max, v_min, v_max, trajectory_s_max_cap = self._extract_constraints_manually_in_series(
            all_states,
            lanelet_clcs,
            trajectory_clcs,
            cl_trajectory_before,
            ref_path,
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
                trajectory_s_max_cap=trajectory_s_max_cap,
            )
        )
        self.runtime_breakdown["constraint_conversion"] += time.time() - constraint_conversion_start_time

        s_hat = self._build_reference_longitudinal_positions(all_states, trajectory_clcs)
        amin, amax, jmin, jmax = self._get_longitudinal_planning_limits()
        s0, v0 = self._maybe_get_velocity_planning_initial_conditions(
            all_states,
            trajectory_clcs,
        )

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
