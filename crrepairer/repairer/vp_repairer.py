import math
import time
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import linprog
from shapely import affinity
import shapely
from z3 import sat
from shapely.geometry import LineString, Polygon

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams, ProcessingOption, ResamplingOption

from crmonitor.common.world import World

from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.constraints import (
    _lateral_position_constraints_reference_point,
    longitudinal_position_constraints,
    longitudinal_velocity_constraints,
)


class VPTrajectoryRepairer(SMTTrajectoryRepairer):
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
        super().__init__(rule_monitor, ego_vehicle, config)
        self.ego_vehicle = ego_vehicle
        self._vp_tc_time_step = 0
        self.domain_dict_time = 0.0
        self.domain_dict = {}
        self._domain_dict_initialized = False
        self.runtime_breakdown = {}
        self.domain_dict_breakdown = {}
        self._domain_predicate_timing = {}

    def repair(self, check_flag=True, *args, **kwargs):
        self._tv = self.rule_monitor.tv_time_step
        if self._tv in (-math.inf, math.inf):
            return None
        if self.config.repair.planner not in (1, 2):
            raise NotImplementedError(
                "VPTrajectoryRepairer currently supports planner == 1 or 2 only."
            )
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

            self.t_solver.assign_proposition(
                select_proposition,
                list(self._model),
                self.config.repair.use_mpr_derivative,
            )
            try:
                repaired_traj = self._repair_with_velocity_planning()
            except Exception as exc:
                print(f"* \t<VPRepairer>: VP repair failed for current SAT model: {exc}")
                repaired_traj = None
            if repaired_traj is not None:
                # if check_flag:
                #     compliance_check_start_time = time.time()
                #     tv_updated, _ = self.t_solver.tc_object.calc_tv_updated(
                #         repaired_traj.state_list,
                #         self.t_solver.tc_object.tc,
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

    def _uses_in_series_processing(self) -> bool:
        return any(rule.startswith("R_IN") for rule in self.config.repair.rules)

    def _uses_rg3_specific_second_lp(self) -> bool:
        return "R_G3" in self.config.repair.rules

    def _should_pin_velocity_planning_initial_conditions(self) -> bool:
        return "R_G3" in self.config.repair.rules

    def _prepare_fixed_cutoff_time(self):
        dt = self._get_dt()
        self._tc = self._vp_tc_time_step
        self.t_solver.tc_object._tc = self._vp_tc_time_step * dt
        if self.t_solver.compliant_maneuvers:
            self.t_solver.tc_object._compliant_maneuver = self.t_solver.compliant_maneuvers[0]

        if self.config.repair.planner == 1:
            self.t_solver._planner.reset(
                scenario=self.t_solver.tc_object.scenario,
                tc_object=self.t_solver.tc_object,
                sel_proposition=self.t_solver._sel_prop,
                full_proposition=self.t_solver._prop_full,
            )
        elif self.config.repair.planner == 2:
            self.t_solver._planner.reset(
                tc_object=self.t_solver.tc_object,
                rule_monitor=self.rule_monitor,
            )
        self.t_solver._planner.construct_constraints(
            self.t_solver._sel_prop,
            self.t_solver._prop_full,
        )

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
        planner = self.t_solver._planner
        if self.config.repair.planner == 1:
            return planner.vehicle_configuration.CLCS, planner._scenario.dt
        if self.config.repair.planner == 2:
            return planner._vehicle_configuration.CLCS, planner.config.scenario.dt
        raise NotImplementedError(
            f"Unsupported planner for CLCS extraction: {self.config.repair.planner}"
        )

    def _get_dt(self) -> float:
        return self._get_lanelet_clcs_and_dt()[1]

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
        planner = self.t_solver._planner
        if self.config.repair.planner == 1:
            planner._rule_constraints.reset(
                start_time_step=self.t_solver.tc_object.tc,
                tc_object=self.t_solver.tc_object,
                rule_monitor=None,
                sel_proposition_full=None,
                proposition_full=None,
            )
            planner._rule_constraints.compute_semantic_reachable_set(
                planner._qp_configuration
            )
            corridor = planner.rule_constraints.corridor
        elif self.config.repair.planner == 2:
            planner._constraints.reset(
                start_time_step=self.t_solver.tc_object.tc,
                tc_object=self.t_solver.tc_object,
                rule_monitor=None,
                sel_proposition_full=None,
                proposition_full=None,
            )
            planner._constraints.compute_semantic_reachable_set(
                planner._vehicle_configuration
            )
            corridor = planner._constraints.corridor
        else:
            raise NotImplementedError(
                f"Unsupported planner for corridor extraction: {self.config.repair.planner}"
            )
        if corridor is None:
            raise RuntimeError("No corridor available for VP constraints.")

        s_min, s_max = longitudinal_position_constraints(corridor)
        v_min, v_max = longitudinal_velocity_constraints(corridor)
        time_steps = list(corridor.keys())
        _lateral_position_constraints_reference_point(corridor, time_steps)
        return s_min, s_max, v_min, v_max

    def _extract_constraints_manually(
        self,
        all_states: List[CustomState],
        lanelet_clcs: CurvilinearCoordinateSystem,
    ):
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

        final_time_step = all_states[-1].time_step
        for time_step in range(int(self._tc) + 1, final_time_step + 1):
            idx = time_step - int(self._tc) - 1
            follow_velocity = all_states[time_step - all_states[0].time_step].velocity
            v_max_list = []
            s_max_list = []
            s_min_list = []

            for prop in self.t_solver._sel_prop:
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
                else:
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
                v_max_list.append(follow_velocity)
                # print('v_max candidates at time step {}: {}'.format(time_step, v_max_list))

            v_max[idx] = min(v_max_list) if v_max_list else math.inf
            s_max[idx] = min(s_max_list) if s_max_list else max(lane_start, lane_end)
            s_min[idx] = max(s_min_list) if s_min_list else min(lane_start, lane_end)

        return s_min, s_max, v_min, v_max

    def _extract_constraints_manually_in_series(
        self,
        all_states: List[CustomState],
        lanelet_clcs: CurvilinearCoordinateSystem,
        trajectory_clcs: CurvilinearCoordinateSystem,
        cl_trajectory_before: List[np.ndarray],
        ref_path: np.ndarray,
    ):
        if "R_IN4" not in self.config.repair.rules and "R_IN1" not in self.config.repair.rules:
            raise NotImplementedError(
                "IN-series VP repair currently supports R_IN1 and R_IN4 only."
            )

        start_idx = int(self._tc - all_states[0].time_step)
        horizon = len(all_states) - start_idx - 1
        if horizon <= 0:
            raise RuntimeError("No horizon available after tc for IN-series VP repair.")

        lane_start = cl_trajectory_before[0][0]
        lane_end = cl_trajectory_before[-1][0]
        s_min = np.ones(horizon) * min(lane_start, lane_end)
        s_max = np.ones(horizon) * max(lane_start, lane_end)
        v_min = np.zeros(horizon)
        v_max = np.ones(horizon) * math.inf

        ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(
            float(ref_path[-1][0]),
            float(ref_path[-1][1]),
        )[0]
        trajectory_s_max_cap = np.ones(horizon) * ct_s_max

        wheelbase = self._get_planner_wheelbase()
        final_time_step = all_states[-1].time_step

        for prop in self.t_solver._sel_prop:
            for time_step in range(int(self._tc) + 1, final_time_step + 1):
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
                    else:
                        print(
                            f"* \t<VPRepairer>: Unsupported predicate {prop.name} "
                            f"for IN1 manual constraint extraction."
                        )
                    continue

                if "in_intersection_conflict_area" in prop.name:
                    prop_assignment = -1 if prop.alphabet.startswith("~") else 1
                    rear_constr_ct = self._constraint_in_intersection_conflict_area_rear_ct(
                        time_step=time_step,
                        prop_assignment=prop_assignment,
                        lanelet_clcs=lanelet_clcs,
                        trajectory_clcs=trajectory_clcs,
                    )
                    if rear_constr_ct is not None and np.isfinite(rear_constr_ct):
                        trajectory_s_max_cap[idx] = min(
                            trajectory_s_max_cap[idx],
                            rear_constr_ct - wheelbase / 2,
                        )
                else:
                    front_constr, _ = self._constraint_in_intersection_conflict_area_ego(
                        time_step=time_step,
                        prop_assignment=-1,
                        lanelet_clcs=lanelet_clcs,
                        cart=False,
                    )
                    s_max[idx] = min(s_max[idx], front_constr)
                    print(
                        f"* \t<VPRepairer>: IN-series manual extraction reuses conflict-area upper bound "
                        f"for unsupported predicate {prop.name}."
                    )

        return s_min, s_max, v_min, v_max, trajectory_s_max_cap

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
        clcs: CurvilinearCoordinateSystem,
    ) -> float:
        upper_bound = np.inf
        for lanelet_id in ego_vehicle.lanelets_dir:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is not None:
                stop_line_s = min(
                    clcs.convert_to_curvilinear_coords(*lanelet.stop_line.start)[0],
                    clcs.convert_to_curvilinear_coords(*lanelet.stop_line.end)[0],
                )
                upper_bound = min(
                    upper_bound,
                    stop_line_s
                    - ego_vehicle.circle_radius
                    - ego_vehicle.shape.length / 3
                    - wheelbase / 2,
                )
        return upper_bound

    def _get_planner_wheelbase(self):
        planner = self.t_solver._planner
        if self.config.repair.planner == 1:
            return planner.vehicle_configuration.wheelbase
        if self.config.repair.planner == 2:
            return planner.vehicle_configuration.qp_veh_config.wheelbase
        raise NotImplementedError(
            f"Unsupported planner for wheelbase extraction: {self.config.repair.planner}"
        )

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
        trajectory_s_max_cap=None,
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

            estimated_s_min.append(min_cart_to_traj)
            s_max_traj = max_cart_to_traj
            if trajectory_s_max_cap is not None:
                s_max_traj = min(s_max_traj, trajectory_s_max_cap[i])
            estimated_s_max.append(s_max_traj)
            estimated_v_max.append(v_max[i] * rmin)
            estimated_v_min.append(v_min[i] * rmax)

        return estimated_s_min, estimated_s_max, estimated_v_min, estimated_v_max

    def _build_reference_longitudinal_positions(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
    ) -> np.ndarray:
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

    def _maybe_get_velocity_planning_initial_conditions(
        self,
        all_states: List[CustomState],
        trajectory_clcs: CurvilinearCoordinateSystem,
    ):
        if not self._should_pin_velocity_planning_initial_conditions():
            return None, None
        return self._get_velocity_planning_initial_conditions(all_states, trajectory_clcs)

    def _build_rg3_second_pass_vmax(self, all_states: List[CustomState], est_v_max):
        refined_vmax = np.asarray(est_v_max, dtype=float).copy()
        first_plan_idx = int(self._tc - all_states[0].time_step) + 1
        for i in range(first_plan_idx, len(all_states)):
            refined_vmax[i - first_plan_idx] = all_states[i].velocity
        return refined_vmax

    def _get_longitudinal_planning_limits(self):
        planner = self.t_solver._planner
        if self.config.repair.planner == 1:
            amin = planner.lon_planner.c_ti.a_long_min
            amax = planner.lon_planner.c_ti.a_long_max
            jmin = planner.lon_planner.c_ti.j_long_min
            jmax = planner.lon_planner.c_ti.j_long_max
            return amin, amax, jmin, jmax
        if self.config.repair.planner == 2:
            qp_veh_config = planner.vehicle_configuration.qp_veh_config
            amin = qp_veh_config.a_min_x
            amax = qp_veh_config.a_max_x
            jmin = qp_veh_config.j_min_x
            jmax = qp_veh_config.j_max_x
            return amin, amax, jmin, jmax
        raise NotImplementedError(
            f"Unsupported planner for longitudinal limits: {self.config.repair.planner}"
        )

    def _get_longitudinal_reachability_limits(self):
        planner = self.t_solver._planner
        if self.config.repair.planner == 1:
            return (
                planner.lon_planner.c_ti.a_long_max,
                planner.lon_planner.c_ti.a_long_min,
                planner.lon_planner.c_ti.v_long_max,
            )
        if self.config.repair.planner == 2:
            qp_veh_config = planner.vehicle_configuration.qp_veh_config
            return (
                qp_veh_config.a_max_x,
                qp_veh_config.a_min_x,
                qp_veh_config.max_speed_x,
            )
        raise NotImplementedError(
            f"Unsupported planner for reachability limits: {self.config.repair.planner}"
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
            delta = np.asarray(cart_pos) - np.asarray(prev_state.position)
            if np.linalg.norm(delta) > 1e-9:
                orientation = math.atan2(delta[1], delta[0])
            else:
                orientation = getattr(prev_state, "orientation", 0.0)

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
        line_center = LineString(traj_xy)
        conflict_circle_center_center = self._find_conflict_points(
            line_center, conflict_linestring
        )
        if conflict_circle_center_center is None:
            return np.array([np.inf, -np.inf])
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

    def _constraint_in_intersection_conflict_area_ego(
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

    def _constraint_in_intersection_conflict_area_rear_ct(
        self,
        time_step: int,
        prop_assignment: float,
        lanelet_clcs: CurvilinearCoordinateSystem,
        trajectory_clcs: CurvilinearCoordinateSystem,
    ):
        front_constr, rear_constr = self._constraint_in_intersection_conflict_area_ego(
            time_step=time_step,
            prop_assignment=prop_assignment,
            lanelet_clcs=lanelet_clcs,
            cart=True,
        )
        if not np.isfinite(np.asarray(rear_constr, dtype=float)).all():
            return None
        rear_constr_ct = trajectory_clcs.convert_to_curvilinear_coords(
            float(rear_constr[0]),
            float(rear_constr[1]),
        )[0]
        return rear_constr_ct

    def _copy_state(self, state) -> CustomState:
        return CustomState(
            time_step=state.time_step,
            position=np.asarray(state.position, dtype=float),
            velocity=getattr(state, "velocity", 0.0),
            orientation=getattr(state, "orientation", 0.0),
            acceleration=getattr(state, "acceleration", 0.0),
        )

    def ensure_domain_dict_initialized(self):
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
        self.domain_dict_time = time.time() - domain_start_time
        self.runtime_breakdown["domain_dict"] = self.domain_dict_time
        self.domain_dict = dict(domain_dict)
        self.sat_solver.set_domain_dict(domain_dict)
        self._domain_dict_initialized = True
        print(f"* \t<VPRepairer>: domain_dict construction time = {self.domain_dict_time:.6f}s")
        if self.domain_dict_breakdown:
            print(f"* \t<VPRepairer>: domain_dict breakdown = {self.domain_dict_breakdown}")
        print(f"* \t<VPRepairer>: domain_dict for DomainDPLL = {domain_dict}")
        return self.domain_dict

    def _build_domain_dict_for_sat(self):
        if self._uses_in_series_processing():
            return self._build_domain_dict_for_sat_in_series()
        if self._tv in (-math.inf, math.inf):
            self.domain_dict_breakdown = {}
            return {}

        breakdown = {}
        all_states = self._get_states_with_initial()
        start = time.time()
        lanelet_clcs, dt = self._get_lanelet_clcs_and_dt()
        breakdown["get_clcs_dt"] = time.time() - start

        start = time.time()
        trajectory_clcs, ref_path = self._build_trajectory_clcs(
            all_states, resampling_factor=2
        )
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
        domain_dict = self._infer_domain_dict_from_predicate_values(
            predicate_values, self.sat_solver._prop_nodes
        )
        breakdown["infer_domain_dict"] = time.time() - start
        self.domain_dict_breakdown = breakdown
        return domain_dict

    def _build_domain_dict_for_sat_in_series(self):
        self.domain_dict_breakdown = {}
        if "R_IN4" in self.config.repair.rules:
            pred_dict = self._evaluate_turning_priority_domains(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._infer_domain_dict_from_in4_priority(
                pred_dict, self.sat_solver._prop_nodes
            )
        if "R_IN1" in self.config.repair.rules:
            pred_dict = self._evaluate_stop_line_domains(
                0,
                len(self.ego_vehicle.prediction.trajectory.state_list),
            )
            return self._infer_domain_dict_from_in1_stop_line(
                pred_dict, self.sat_solver._prop_nodes
            )
        raise NotImplementedError(
            "IN-series DomainDPLL support currently supports R_IN1 and R_IN4 only."
        )

    def _evaluate_stop_line_domains(self, t_start: int, t_end: int) -> dict:
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

    def _infer_domain_dict_from_in1_stop_line(self, pred_dict, prop_nodes):
        domain_dict = {}
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            for key, value in pred_dict.items():
                if key in prop_name and len(value) == 1:
                    domain_dict[prop_node.alphabet[-1]] = set(value)
                    break
        return domain_dict

    def _evaluate_turning_priority_domains(self, t_start: int, t_end: int) -> dict:
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

    def _infer_domain_dict_from_in4_priority(self, pred_dict, prop_nodes):
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
        follow_vehicle = self.ego_vehicle
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
            )
            safe_dist_max, in_front_of_max = self._evaluate_longitudinal_predicates(
                follow_vehicle=follow_vehicle,
                other_context=other_context,
                time_step=time_step,
                follow_cart_pos=cart_reach[t][2],
                follow_velocity=cl_reach[t][4],
                follow_theta=max_theta,
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
                lane_speed_min = self._keep_speed_limit_eval(world, abs_time_step, v_min, speed_limits["lane"])
                lane_speed_max = self._keep_speed_limit_eval(world, abs_time_step, v_max, speed_limits["lane"])
                lane_speed.append((abs_time_step, 2 if lane_speed_min != lane_speed_max else int(lane_speed_min)))
            if speed_limits["type"] is not None:
                type_speed_min = self._keep_speed_limit_eval(world, abs_time_step, v_min, speed_limits["type"])
                type_speed_max = self._keep_speed_limit_eval(world, abs_time_step, v_max, speed_limits["type"])
                type_speed.append((abs_time_step, 2 if type_speed_min != type_speed_max else int(type_speed_min)))
            if speed_limits["fov"] is not None:
                fov_speed_min = self._keep_speed_limit_eval(world, abs_time_step, v_min, speed_limits["fov"])
                fov_speed_max = self._keep_speed_limit_eval(world, abs_time_step, v_max, speed_limits["fov"])
                fov_speed.append((abs_time_step, 2 if fov_speed_min != fov_speed_max else int(fov_speed_min)))
            if speed_limits["brake"] is not None:
                brake_speed_min = self._keep_speed_limit_eval(world, abs_time_step, v_min, speed_limits["brake"])
                brake_speed_max = self._keep_speed_limit_eval(world, abs_time_step, v_max, speed_limits["brake"])
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

    def _collect_prop_tuples(self, predicate_values):
        prop_tuples = set()
        safe_dist = predicate_values["safe_dist"]
        in_same_lane = predicate_values["in_same_lane"]
        in_front_of = predicate_values["in_front_of"]
        cut_in = predicate_values["cut_in"]
        for t in range(len(safe_dist)):
            prop_tuples.add(
                (
                    safe_dist[t][1],
                    in_same_lane[t][1],
                    in_front_of[t][1],
                    cut_in[t][1],
                )
            )
        return prop_tuples

    def _infer_domain_dict_from_prop_tuples(self, prop_tuples, prop_nodes):
        domain_options = {}
        for prop_node in prop_nodes:
            domain_options[prop_node.alphabet] = set()
            for prop_tuple in prop_tuples:
                possible_values = self._possible_values_for_prop(prop_node.name, prop_tuple)
                domain_options[prop_node.alphabet].update(possible_values)

        domain_dict = {}
        for alphabet, possible_values in domain_options.items():
            if possible_values == {0} or possible_values == {1}:
                domain_dict[alphabet[-1]] = set(possible_values)
        return domain_dict

    def _infer_domain_dict_from_predicate_values(self, predicate_values, prop_nodes):
        domain_dict = {}
        for prop_node in prop_nodes:
            possible_values = self._possible_values_for_prop_from_sequences(prop_node.name, predicate_values)
            if possible_values == {0} or possible_values == {1}:
                domain_dict[prop_node.alphabet[-1]] = set(possible_values)
        return domain_dict

    def _possible_values_for_prop_from_sequences(self, prop_name, predicate_values):
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

    def _possible_values_for_prop(self, prop_name, prop_tuple):
        if "distance" in prop_name:
            value = prop_tuple[0]
        elif "lane" in prop_name:
            value = prop_tuple[1]
        elif "front" in prop_name:
            value = prop_tuple[2]
        elif "cut_in" in prop_name:
            value = prop_tuple[3]
        else:
            return {0, 1}
        return {0, 1} if value == 2 else {int(value)}

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

    def _calc_s(self, s, w, l, theta):
        rot_mat_factors = np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
        return (
            rot_mat_factors[0] * l / 2.0 * np.cos(theta)
            - rot_mat_factors[1] * w / 2 * np.sin(theta)
            + s
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
        if not self._vehicle_has_valid_time_step(follow_vehicle, time_step):
            return None
        lane_follow = follow_vehicle.get_lane(time_step)
        if lane_follow is None:
            return None
        follow_width = follow_vehicle.shape.width
        follow_length = follow_vehicle.shape.length
        follow_curvi_pos = lane_follow.clcs.convert_to_curvilinear_coords(
            follow_cart_pos[0], follow_cart_pos[1]
        )
        return np.max(
            self._calc_s(follow_curvi_pos[0], follow_width, follow_length, follow_theta)
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
    ):
        front_s = self._compute_follow_front_s(
            follow_vehicle=follow_vehicle,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_theta=follow_theta,
        )
        if front_s is None:
            return True, True
        if not other_context["valid"] or other_context["rear_s"] is None:
            return True, True

        delta_s = other_context["rear_s"] - front_s
        in_front_of = np.clip(delta_s / max_lon_dist, -1.0, 1.0) > 0.0

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
        return keep_safe_distance, in_front_of

    def _keep_safe_distance_eval(
        self,
        world,
        time_step,
        lead_id,
        follow_id,
        follow_cart_pos,
        follow_velocity,
        follow_theta,
        max_lon_dist=200.0,
    ):
        follow_vehicle = world.vehicle_by_id(follow_id)
        other_context = self._build_other_vehicle_predicate_context(
            road_network=world.road_network,
            other_vehicle=world.vehicle_by_id(lead_id),
            time_step=time_step,
        )
        keep_safe_distance, _ = self._evaluate_longitudinal_predicates(
            follow_vehicle=follow_vehicle,
            other_context=other_context,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_velocity=follow_velocity,
            follow_theta=follow_theta,
            max_lon_dist=max_lon_dist,
        )
        return keep_safe_distance

    def _keep_speed_limit_eval(self, world, time_step, ego_velocity, speed_limit, max_speed=250.0 / 3.6, eps=1e-5):
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

    def _in_front_of_eval(self, world, time_step, lead_id, follow_id, follow_cart_pos, follow_theta, max_lon_dist=200.0):
        follow_vehicle = world.vehicle_by_id(follow_id)
        other_context = self._build_other_vehicle_predicate_context(
            road_network=world.road_network,
            other_vehicle=world.vehicle_by_id(lead_id),
            time_step=time_step,
        )
        _, in_front_of = self._evaluate_longitudinal_predicates(
            follow_vehicle=follow_vehicle,
            other_context=other_context,
            time_step=time_step,
            follow_cart_pos=follow_cart_pos,
            follow_velocity=getattr(follow_vehicle.states_cr.get(time_step), "velocity", 0.0),
            follow_theta=follow_theta,
            max_lon_dist=max_lon_dist,
        )
        return in_front_of
