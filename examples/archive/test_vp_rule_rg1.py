from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)
from crrepairer.utils.constraints import (
    longitudinal_position_constraints,
    longitudinal_velocity_constraints,
    lateral_position_constraints,
    _lateral_position_constraints_reference_point
)

from crrepairer.smt.t_solver.rule_constraints_reach import RuleConstraintsReach
from crrepairer.smt.t_solver.rule_constraints import RuleConstraintsManual
from miqp_planner.miqp_constraints_manual import (
    RuleConstraint as RuleConstraintMIQPManual
)
from crmonitor.common.world import World
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
    PredPreceding,
    PredStopLineInFront,
    PredInIntersectionConflictArea,
    PredOnLaneletWithTypeIntersection,
)
from commonroad_qp_planner.constraints import LonConstraints
from commonroad_clcs.util import compute_curvature_from_polyline_python
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
import numpy as np
import time
from scipy.optimize import linprog

import math


def ConstrInSameLane(
    world: World, time_step, other_id, ego_id, lanelet_clcs, t_c, t_f
) -> bool:
    '''
    Check if the other vehicle and ego vehicle are in the same lane at any time step between t_c and t_f. 
    If yes, return the longitudinal position of the first and last time step they are in the same lane as max and min constraints. If no, return None.
    '''
    other_vehicle = world.vehicle_by_id(other_id)
    ego_vehicle = world.vehicle_by_id(ego_id)
    
    steps = []
    ego_lane = set()
    other_lane = set()
    other_lane.add(other_vehicle.get_lane(time_step))
    for t in range(t_c, t_f+1):
        try:
            ego_lane.add(ego_vehicle.get_lane(t))
        except Exception as e:
            print(f"Error getting ego lane at time step {t}: {e}")
            continue
        common_lanelets = other_lane & ego_lane
        if common_lanelets:
            steps.append(t)
        else:
            pass
    if len(steps) > 0:
        start = steps[0]
        end = steps[-1]
        start_cart_pos = ego_vehicle.states_cr[start].position
        end_cart_pos = ego_vehicle.states_cr[end].position
        start_cl_pos = lanelet_clcs.convert_to_curvilinear_coords(start_cart_pos[0], start_cart_pos[1])
        end_cl_pos = lanelet_clcs.convert_to_curvilinear_coords(end_cart_pos[0], end_cart_pos[1])
        return start_cl_pos[0], end_cl_pos[0]
    else:
        return None, None


def ConstrKeepSafeDist(
    world: World, lanelet_clcs, time_step, lead_id, follow_id, follow_velocity, delta_s=0.5
) -> tuple:
    '''
    Calculate the safe distance between the following vehicle and the leading vehicle at a given time step. 
    Return the longitudinal max position and velocity for the following vehicle in lanelet CLCS. 
    '''
    def calculate_safe_distance(
            v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow
        ):
        d_safe = (
            (v_lead**2) / (-2 * np.abs(a_min_lead))
            - (v_follow**2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )
        return d_safe     
    vehicle_follow = world.vehicle_by_id(follow_id)
    vehicle_lead = world.vehicle_by_id(lead_id)
    follow_length = vehicle_follow.shape.length
    if (
        time_step > vehicle_lead.end_time
        or not vehicle_lead.is_valid(time_step)
        or time_step not in vehicle_lead.lanelet_assignment
    ):
        return math.inf, math.inf
    if vehicle_lead.get_lane(time_step) is None:
        return math.inf, math.inf
    a_min_follow = vehicle_follow.vehicle_param.get("a_min")
    a_min_lead = vehicle_lead.vehicle_param.get("a_min")
    t_react_follow = vehicle_follow.vehicle_param.get("t_react")
    safe_distance = calculate_safe_distance(
        follow_velocity+1.0,
        vehicle_lead.states_cr[time_step].velocity,
        a_min_lead,
        a_min_follow,
        t_react_follow,
    )
    lead_rear_s = lanelet_clcs.convert_to_curvilinear_coords(vehicle_lead.states_cr[time_step].position[0], vehicle_lead.states_cr[time_step].position[1])
    lead_rear_s = lead_rear_s[0] - vehicle_lead.shape.length/2
    s = lead_rear_s - safe_distance - follow_length - delta_s
    v = follow_velocity
    return s, v
                

def solve_velocity_planning_lp(
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
):
    """
    Solve the velocity planning problem as a linear program.

    Variables:
        x = [s_0, ..., s_{T-1}, v_0, ..., v_{T-1}]

    Objective:
        minimize sum_t (s_hat[t] - s[t])
        equivalent to minimize -sum_t s[t]

    Constraints:
        1) s_{t+1} = s_t + 0.5*(v_t + v_{t+1})*dt
        2) vmin_t <= v_t <= vmax_t
        3) amin <= (v_{t+1} - v_t)/dt <= amax
        4) jmin <= (v_{t+2} + v_t - 2*v_{t+1})/dt^2 <= jmax
        5) smin_t <= s_t <= smax_t, and s_t <= s_hat_t

    Optional:
        s0: if not None, enforce s_0 == s0
        v0: if not None, enforce v_0 == v0
    """
    s_hat = np.asarray(s_hat, dtype=float)
    vmin = np.asarray(vmin, dtype=float)
    vmax = np.asarray(vmax, dtype=float)
    smin = np.asarray(smin, dtype=float)
    smax = np.asarray(smax, dtype=float)

    T = len(s_hat)
    assert len(vmin) == T
    assert len(vmax) == T
    assert len(smin) == T
    assert len(smax) == T
    assert dt > 0.0

    n_s = T
    n_v = T
    n_x = n_s + n_v

    def s_idx(t):
        return t

    def v_idx(t):
        return n_s + t

    # ---------------------------
    # Objective: minimize -sum(s_t)
    # ---------------------------
    c = np.zeros(n_x)
    for t in range(T):
        c[s_idx(t)] = -1.0

    # ---------------------------
    # Equality constraints
    # ---------------------------
    A_eq = []
    b_eq = []

    # 1) Dynamics:
    # s_{t+1} - s_t - 0.5*dt*v_t - 0.5*dt*v_{t+1} = 0
    for t in range(T - 1):
        row = np.zeros(n_x)
        row[s_idx(t + 1)] = 1.0
        row[s_idx(t)] = -1.0
        row[v_idx(t)] = -0.5 * dt
        row[v_idx(t + 1)] = -0.5 * dt
        A_eq.append(row)
        b_eq.append(0.0)

    # Optional fixed initial position
    if s0 is not None:
        row = np.zeros(n_x)
        row[s_idx(0)] = 1.0
        A_eq.append(row)
        b_eq.append(float(s0))

    # Optional fixed initial velocity
    if v0 is not None:
        row = np.zeros(n_x)
        row[v_idx(0)] = 1.0
        A_eq.append(row)
        b_eq.append(float(v0))

    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    # ---------------------------
    # Inequality constraints A_ub x <= b_ub
    # ---------------------------
    A_ub = []
    b_ub = []

    # 3) Acceleration constraints
    # v_{t+1} - v_t <= amax * dt
    # v_{t+1} - v_t >= amin * dt
    # -> -(v_{t+1} - v_t) <= -amin*dt
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

    # 4) Jerk constraints
    # v_{t+2} + v_t - 2*v_{t+1} <= jmax * dt^2
    # v_{t+2} + v_t - 2*v_{t+1} >= jmin * dt^2
    # -> -(v_{t+2} + v_t - 2*v_{t+1}) <= -jmin * dt^2
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

    # ---------------------------
    # Variable bounds
    # ---------------------------
    bounds = []

    # s_t bounds: smin_t <= s_t <= min(smax_t, s_hat_t)
    for t in range(T):
        ub = min(smax[t], s_hat[t])
        lb = smin[t]
        if lb > ub:
            raise ValueError(
                f"Infeasible position bounds at t={t}: "
                f"smin={lb}, min(smax, s_hat)={ub}"
            )
        bounds.append((lb, ub))

    # v_t bounds: vmin_t <= v_t <= vmax_t
    for t in range(T):
        lb = vmin[t]
        ub = vmax[t]
        if lb > ub:
            raise ValueError(
                f"Infeasible velocity bounds at t={t}: vmin={lb}, vmax={ub}"
            )
        bounds.append((lb, ub))

    # ---------------------------
    # Solve LP
    # ---------------------------
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


if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_Gar-1_1_T-1"
    # Build configuration object
    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()
    config.debug.show_plots = True
    config.repair.planner = 1  # 1: qp planner 2: miqp
    config.repair.constraint_mode = 1   # 1: Manual, 2: Reach
    config.repair.use_mpr = False

    # scenario_id = "DEU_LocationDLower-8_154_T-1"
    # # Build configuration object
    # config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    # config.update()
    # config.repair.rules = ["R_G1"]
    # config.repair.ego_id = 11
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 1
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False

    # # scenario_id = "DEU_LocationAUpper-54_67_T-1" # id=9
    # # scenario_id = "DEU_LocationELower-18_22_T-1" # id=14
    # scenario_id = "DEU_LocationALower-26_189_T-1" # id=19
    # # Build configuration object
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.rules = ["R_G1"]
    # config.repair.ego_id = 19
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 1
    # config.repair.use_mpr = False

    # scenario_id = "DEU_LocationALower-34_37_T-1" 
    # config = RepairerConfiguration()
    # config.general.path_scenarios = '/data_linux/Lab/highD-cr-scenarios/highD-repair/'
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.rules = ["R_G1"]
    # config.repair.ego_id = 9
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 2
    # config.repair.use_mpr = False


    ego_initial = retrieve_ego_vehicle(config)

    # ========== Velocity Planning Test Demo =========
    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()
        
        # ========== Pre-Processing =========
        resampling_factor = 10
        binary_search_time = 4
        params = CLCSParams()
        params.processing_option = ProcessingOption.SPLINE_SMOOTHING
        params.resampling.option = ResamplingOption.ADAPTIVE
        ref_path_processor = ProcessorFactory.create_processor(params)

        # Obtain tc and trajectories before and after repair and lanelet CLCS
        original_tc = repairer.tc
        # tc = original_tc
        # tc = 12
        cart_trajectory_before_repair = ego_initial.prediction.trajectory.state_list
        cart_trajectory_before_repair = [ego_initial.initial_state] + cart_trajectory_before_repair
        if config.repair.planner == 1:
            print("This is QP planner")
            lanelet_clcs = repairer.t_solver._planner.vehicle_configuration.CLCS
            dt = repairer.t_solver._planner._scenario.dt
        elif config.repair.planner == 2:
            print("This is MIQP planner")
            lanelet_clcs = repairer.t_solver._planner._vehicle_configuration.CLCS
            dt = repairer.t_solver._planner.scenario.dt
        cl_trajectory_before_repair = []
        for state in cart_trajectory_before_repair:
            clcs_state = lanelet_clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
            cl_trajectory_before_repair.append(clcs_state)
        # cart_trajectory_after_repair = repaired_traj.state_list
        # clcs_trajectory_after_repair = repaired_traj.get_positions()

        ct_ref_path = []

        time2 = time.time()
        # Construct a high-resolution reference path and generate trajectory CLCS
        for i in range(0, len(cart_trajectory_before_repair)-1):
            pos = cart_trajectory_before_repair[i].position
            next_pos = cart_trajectory_before_repair[i+1].position
            delta = (np.array(next_pos) - np.array(pos)) / resampling_factor
            for j in range(resampling_factor):
                intermediate_pos = np.array(pos) + j * delta
                ct_ref_path.append(intermediate_pos)    # TODO: this might be faster
        ct_ref_path.append(cart_trajectory_before_repair[-1].position)
        ct_ref_path = np.array(ct_ref_path)
        # processed_ref_path = ref_path_processor(ct_ref_path)
        num_extend_pts = 10
        processed_ref_path = ct_ref_path
        start_dir = processed_ref_path[0] - processed_ref_path[1]
        end_dir = processed_ref_path[-1] - processed_ref_path[-2]
        start_extend = np.array([
            processed_ref_path[0] + start_dir * i
            for i in range(num_extend_pts, 0, -1)
        ])
        end_extend = np.array([
            processed_ref_path[-1] + end_dir * i
            for i in range(1, num_extend_pts + 1)
        ])
        processed_ref_path = np.vstack((start_extend, processed_ref_path, end_extend))
        trajectory_clcs = CurvilinearCoordinateSystem(
            reference_path=processed_ref_path,
            params=params,
            preprocess_path=False
        )
        time3 = time.time()
        print(f"Time to construct trajectory CLCS: {time3 - time2:.3f}s")


        # ========= Fix tc = 1 ========
        time0 = time.time()
        tc = 1
        t_0 = config.repair.t_0
        t_f = ego_initial.prediction.trajectory.state_list[-1].time_step
        # Update repairer tc and corridor, reset planner
        repairer._tc = tc
        tc = tc * dt
        repairer.t_solver.tc_object._tc = tc
        if repairer.t_solver.config.repair.planner == 1:
            repairer.t_solver._planner.reset(scenario=repairer.t_solver.tc_object.scenario,
                                tc_object=repairer.t_solver.tc_object,
                                sel_proposition=repairer.t_solver._sel_prop,
                                full_proposition=repairer.t_solver._prop_full)
            repairer.t_solver._planner.construct_constraints(repairer.t_solver._sel_prop, repairer.t_solver._prop_full)
        time1 = time.time()
        print(f"Time to compute tc: {time1 - time0:.6f}s")
        
        
        # ========= Extract Constraints ========
        # Extract constraints in lanelet CLCS
        time4 = time.time()
        if config.repair.planner == 1:
            repairer.t_solver._planner._rule_constraints.reset(tc_object=repairer.t_solver.tc_object,           
                                                            start_time_step=tc, rule_monitor=None, 
                                                            sel_proposition_full=None, proposition_full=None)
        # Use reachable set for constraint extraction - slow
        if config.repair.constraint_mode == 2:
            repairer.t_solver._planner._rule_constraints.compute_semantic_reachable_set(repairer.t_solver._planner._qp_configuration)
            corridor = repairer.t_solver._planner.rule_constraints.corridor
            if corridor is not None:
                s_min, s_max = longitudinal_position_constraints(corridor)
                v_min, v_max = longitudinal_velocity_constraints(corridor)
                time_steps = list(corridor.keys())
                d_constraints_reference_point = _lateral_position_constraints_reference_point(corridor, time_steps)
                d_reference = np.array(list(d_constraints_reference_point.values()))
                d_min, d_max = d_reference[1:, 0], d_reference[1:, 1]
            else:
                print("No corridor available for constraints.")
        # Use manual constraint extraction - fast but may be conservative, currently only for planner 1
        elif config.repair.constraint_mode == 1:
            if config.repair.planner == 1:
                ct_s_min = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[0][0]), float(ct_ref_path[0][1]))
                ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[-1][0]), float(ct_ref_path[-1][1]))
                s_max = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * ct_s_max[0]
                s_min = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * ct_s_min[0]
                v_min = np.zeros(len(cart_trajectory_before_repair)-int(tc/dt)-1)
                v_max = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * math.inf
                
                # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._prop_full}")
                for prop in repairer.t_solver._sel_prop:
                    # print(f"Extracting constraints for predicate {prop}...")
                    for t in range(int(tc/dt)+1, t_f+1):
                        if 'distance' in prop.name:
                            s_max[t-int(tc/dt)-1], v_max[t-int(tc/dt)-1] = ConstrKeepSafeDist(
                                world=traffic_rule_monitor.world,
                                lanelet_clcs=lanelet_clcs,
                                time_step=t,
                                lead_id=traffic_rule_monitor.other_id,
                                follow_id=ego_initial.obstacle_id,
                                follow_velocity=cart_trajectory_before_repair[t].velocity
                            )
                        elif 'lane' in prop.name:   # TODO: currently only either use lane constraint or safe distance constraint
                            min, max = ConstrInSameLane(
                                world=traffic_rule_monitor.world,
                                lanelet_clcs=lanelet_clcs,
                                time_step=t,
                                other_id=traffic_rule_monitor.other_id,
                                ego_id=ego_initial.obstacle_id,
                                t_c=int(tc/dt),
                                t_f=t_f
                            )
                            if s_min[t-int(tc/dt)-1] is None:
                                print(f"Infeasible lane constraint at time step {t} with prop {prop.name}.")
                            else:
                                s_min[t-int(tc/dt)-1] = min
                                s_max[t-int(tc/dt)-1] = max
                        else:
                            s_max[t-int(tc/dt)-1], v_max[t-int(tc/dt)-1] = ConstrKeepSafeDist(
                                world=traffic_rule_monitor.world,
                                lanelet_clcs=lanelet_clcs,
                                time_step=t,
                                lead_id=traffic_rule_monitor.other_id,
                                follow_id=ego_initial.obstacle_id,
                                follow_velocity=cart_trajectory_before_repair[t].velocity
                            )
                            print(f"Unsupported predicate {prop.name} for manual constraint extraction.")

        time5 = time.time()
        print(f"Time to extract constraints (including reachset computation): {time5 - time4:.3f}s")
        time6 = time.time()
        
        # ======== Constraints Conversion ========
        # Transform positional constrains s_min, s_max, d_min, d_max from lanelet CLCS to trajectory CLCS
        estimated_s_min = []
        estimated_s_max = []
        estimated_v_max = []
        estimated_v_min = []
        ds = np.gradient(ct_ref_path[:, 0])
        dd = np.gradient(ct_ref_path[:, 1])
        eps = 1e-6
        ds_safe = np.where(np.abs(ds) < eps, eps, ds)
        ratio_1_cos = np.sqrt(1.0 + (dd / ds_safe) ** 2)
        rmax = np.max(ratio_1_cos)
        rmin = np.min(ratio_1_cos)
        ct_ref_path_np = np.asarray(ct_ref_path)  # shape (N,2)
        ref_path_curv = compute_curvature_from_polyline_python(ct_ref_path_np)
        max_curv = np.max(np.abs(ref_path_curv))
        a_lat_max = repairer.t_solver._planner.lat_planner.c_ti.a_max 
        v_max_curv = math.sqrt(a_lat_max / max_curv) if max_curv > eps else float('inf')
        ct_s_min = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[0][0]), float(ct_ref_path[0][1]))
        ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[-1][0]), float(ct_ref_path[-1][1]))
        for i in range(len(s_min)):
            d = cl_trajectory_before_repair[i][1]
            try:
                if not np.isfinite(s_min[i]):
                    raise ValueError("s_min outside projection domain")
                min_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(s_min[i], d)
            except Exception:
                min_lane_to_cart = (float(ct_ref_path[0][0]), float(ct_ref_path[0][1]))
            try:
                if not np.isfinite(s_max[i]):
                    raise ValueError("s_max outside projection domain")
                max_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(s_max[i], d)
            except Exception:
                max_lane_to_cart = (float(ct_ref_path[-1][0]), float(ct_ref_path[-1][1]))
            try:
                min_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(min_lane_to_cart[0], min_lane_to_cart[1])
            except Exception as e:
                min_cart_to_traj = ct_s_min
            try:
                max_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(max_lane_to_cart[0], max_lane_to_cart[1])
            except Exception as e:
                max_cart_to_traj = ct_s_max
            estimated_s_min.append(min_cart_to_traj[0])
            estimated_s_max.append(max_cart_to_traj[0])
            # estimated_v_max.append(min(v_max[i] * rmin, v_max_curv))
            estimated_v_max.append(v_max[i] * rmin)
            estimated_v_min.append(v_min[i] * rmax)
            
        time7 = time.time()
        print(f"Time to estimate points in corridor: {time7 - time6:.3f}s") 

        # ======= Solve optimization problem ========
        T = int((t_f - tc / dt))
        s_hat = np.zeros(T)
        for i in range(int(tc/dt), len(cart_trajectory_before_repair)):
            state = cart_trajectory_before_repair[i]
            if i == int(tc/dt):
                s0 = trajectory_clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
                s0 = s0[0]
                v0 = state.velocity
                continue
            ct_pos = trajectory_clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
            s_hat[i-int(tc/dt)-1] = ct_pos[0]
        vmin = np.array(estimated_v_min)
        vmax = np.array(estimated_v_max)
        smin = np.array(estimated_s_min)
        smax = np.array(estimated_s_max)
        amin = repairer.t_solver._planner.lon_planner.c_ti.a_long_min
        # amax = repairer.t_solver._planner.lon_planner.c_ti.a_long_max
        amax = 0
        jmin = repairer.t_solver._planner.lon_planner.c_ti.j_long_min
        jmax = repairer.t_solver._planner.lon_planner.c_ti.j_long_max

        sol = solve_velocity_planning_lp(
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
            # s0=s0,
            # v0=v0,
        )
        s = sol["s"]
        v = sol["v"]

        print("Solved successfully.")
        # print("s =", np.round(s, 4))
        # print("v =", np.round(v, 4))
        print("objective =", sol["objective_min_sum_s_hat_minus_s"])
        
        time8 = time.time()
        print(f"Time to solve optimization problem: {time8 - time7:.3f}s")
        
        # ======= Evaluate repaired trajectory compliance =======
        # Transform back to Cartesian trajectory and recalculate tv
        repaired_state_list = []
        for i in range(int(tc/dt)+1):
            state = cart_trajectory_before_repair[i]
            original_state = CustomState(
                time_step=state.time_step,
                position=state.position,
                velocity=state.velocity,
                orientation=state.orientation,
                acceleration=state.acceleration
            )
            repaired_state_list.append(original_state)
        for i in range(len(s)):
            s_i = s[i]
            d_i = 0.0
            v_i = v[i]
            a_i = (v[i+1] - v[i]) / dt if i < len(v)-1 else (v[i] - v[i-1]) / dt
            cart_pos = trajectory_clcs.convert_to_cartesian_coords(s_i, d_i)
            if repaired_state_list[-1] is not None:
                delta = cart_pos - repaired_state_list[-1].position
                o_i = math.atan2(delta[1], delta[0])
            else:
                o_i = 0.0
            repaired_state = CustomState(
                time_step=int(tc/dt)+i+1,
                position=cart_pos,
                velocity=v_i,
                orientation=o_i,
                acceleration=a_i
            )
            repaired_state_list.append(repaired_state)
        cr_trajectory = Trajectory(t_0, repaired_state_list)
        tv, _ = repairer.t_solver.tc_object.calc_tv_updated(
                    cr_trajectory.state_list, tc
                )
        # tv_original, _ = repairer.t_solver.tc_object.calc_tv_updated(
        #             repaired_traj.state_list, tc
        #         )
        # print(f"Updated tv after post-processing: {tv}, original tv: {tv_original}, new tc: {repairer.t_solver.tc_object._tc},  repairer.tc: {repairer.tc}")

        # # print summary of estimated points, especially that some time steps may have 0 points
        # print("Estimated points in lanelet corridor for each time step after tc:")
        # for i, pts in enumerate(estimated_s_max):
        #     print(f"s: [{estimated_s_min[i]:.3f}, {estimated_s_max[i]:.3f}]")

        if cr_trajectory is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, cr_trajectory
            )
            if config.debug.show_plots:
                # ============= Visualization =============
                visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
        
        # if repaired_traj is not None and config.debug.show_plots:
        #     ego_repaired = repairer.convert_traj_to_ego_vehicle(
        #         ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
        #     )
        #     if config.debug.show_plots:
        #         # ============= Visualization =============
        #         visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
        
