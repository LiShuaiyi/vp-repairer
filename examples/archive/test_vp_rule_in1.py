from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from commonroad.scenario.obstacle import ObstacleType
import math

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
from crmonitor.common.vehicle import Vehicle
from commonroad_qp_planner.constraints import LonConstraints
from commonroad_clcs.util import compute_curvature_from_polyline_python
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.lanelet import LaneletType
import numpy as np
import time
from scipy.optimize import linprog
import shapely
from shapely.geometry import LineString, Polygon
from typing import Union


def ConstrStopLine(world: World, ego_vehicle: Vehicle, wheelbase: float, clcs: CurvilinearCoordinateSystem) -> float:
    world = world
    upper_bound = np.inf
    for lanelet_id in ego_vehicle.lanelets_dir:
        lanelet = world.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet.stop_line is not None:
            stop_line_s = min(
                clcs.convert_to_curvilinear_coords(
                    *lanelet.stop_line.start
                )[0],
                clcs.convert_to_curvilinear_coords(
                    *lanelet.stop_line.end
                )[0],
            )
            upper_bound = min(
                upper_bound,
                stop_line_s
                - ego_vehicle.circle_radius
                - ego_vehicle.shape.length / 3
                - wheelbase / 2,
            )
    return upper_bound


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
    # scenario_id = "DEU_AachenBendplatz-1_152460_T-2479"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10161
    # config.repair.N_r = 20
    # config.update()
    # config.repair.use_mpr = False
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 1
    # config.debug.plot_limits = [40, 69, -45, -17]

    # scenario_id = "DEU_AachenBendplatz-1_151520_T-1539"
    # config = RepairerConfiguration()
    # config.general.set_path_scenario(scenario_id)
    # config.update()
    # config.repair.scenario_type = "intersection"
    # config.repair.rules = ["R_IN1"]
    # config.repair.ego_id = 10108
    # config.repair.N_r = 20
    # config.update()
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 1
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False
    # config.debug.plot_limits = [37, 54, -30, -12]

    scenario_id = "DEU_AachenBendplatz-1_75140_T-5159"
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.rules = ["R_IN1"]
    config.repair.ego_id = 10203
    config.repair.N_r = 20
    config.update()
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.show_plots = True
    config.repair.planner = 2
    config.repair.constraint_mode = 1
    config.debug.plot_limits = [53, 62, -30, -20]

    # Retrieve the ego vehicle
    ego_initial = retrieve_ego_vehicle(config)

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()


        # ========== Pre-Processing =========
        resampling_factor = 2
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
            preprocess_path=False,
        )
        time3 = time.time()
        print(f"Time to construct trajectory CLCS: {time3 - time2:.3f}s")


        # ========= Compute tc Using Binary Search ========
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
        # Extract constraints in lanelet 
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
                wheelbase = repairer.t_solver._planner.vehicle_configuration.wheelbase
            elif config.repair.planner == 2:
                wheelbase = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.wheelbase
            cl_s_min = cl_trajectory_before_repair[0][0]
            cl_s_max = cl_trajectory_before_repair[-1][0]
            s_max = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * cl_s_max
            s_min = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * cl_s_min
            v_min = np.zeros(len(cart_trajectory_before_repair)-int(tc/dt)-1)
            v_max = np.ones(len(cart_trajectory_before_repair)-int(tc/dt)-1) * math.inf

            
            # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._prop_full}")
            conflict_area_flag = False
            for prop in repairer.t_solver._sel_prop:
                for t in range(int(tc/dt)+1, t_f+1):
                    if 'stop_line' in prop.name:
                        world = traffic_rule_monitor.world
                        ego_vehicle = world.vehicle_by_id(config.repair.ego_id)
                        upper_bound = ConstrStopLine(world, ego_vehicle, wheelbase, lanelet_clcs)
                        s_max[t-int(tc/dt)-1] = min(s_max[t-int(tc/dt)-1], upper_bound)
                    else:
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
        if config.repair.planner == 1:
            a_lat_max = repairer.t_solver._planner.lat_planner.c_ti.a_max 
            amin = repairer.t_solver._planner.lon_planner.c_ti.a_long_min
            amax = repairer.t_solver._planner.lon_planner.c_ti.a_long_max
            # amax = 0
            jmin = repairer.t_solver._planner.lon_planner.c_ti.j_long_min
            jmax = repairer.t_solver._planner.lon_planner.c_ti.j_long_max
        elif config.repair.planner == 2:
            a_lat_max = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.a_max
            amin = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.a_min_x
            amax = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.a_max_x
            # amax = 0
            jmin = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.j_min_x
            jmax = repairer.t_solver._planner.vehicle_configuration.qp_veh_config.j_max_x
        v_max_curv = math.sqrt(a_lat_max / max_curv) if max_curv > eps else float('inf')
        ct_s_min = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[0][0]), float(ct_ref_path[0][1]))
        ct_s_max = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[-1][0]), float(ct_ref_path[-1][1]))
        for i in range(len(s_min)):
            d = cl_trajectory_before_repair[i][1]
            min_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(s_min[i], d)
            max_lane_to_cart = lanelet_clcs.convert_to_cartesian_coords(s_max[i], d)
            try:
                min_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(min_lane_to_cart[0], min_lane_to_cart[1])
            except Exception as e:
                dists = np.linalg.norm(processed_ref_path - min_lane_to_cart, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_pt = processed_ref_path[nearest_idx]
                min_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(nearest_pt[0], nearest_pt[1])
                print(f"Warning: Failed to convert min constraint from lanelet CLCS to trajectory CLCS at index {i}, using clostest point as fallback. Error: {e}, with main lane to cart coords: {min_lane_to_cart}")
            try:
                max_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(max_lane_to_cart[0], max_lane_to_cart[1])
            except Exception as e:
                dists = np.linalg.norm(processed_ref_path - max_lane_to_cart, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_pt = processed_ref_path[nearest_idx]
                max_cart_to_traj = trajectory_clcs.convert_to_curvilinear_coords(nearest_pt[0], nearest_pt[1])
                ct_s_small = trajectory_clcs.convert_to_curvilinear_coords(float(ct_ref_path[1][0]), float(ct_ref_path[1][1]))
                max_cart_to_traj[0] = max(max_cart_to_traj[0], ct_s_small[0])  
                print(f"Warning: Failed to convert max constraint from lanelet CLCS to trajectory CLCS at index {i}, using clostest point as fallback. Error: {e}, with main lane to cart coords: {max_lane_to_cart}")
            estimated_s_min.append(min_cart_to_traj[0])
            estimated_s_max.append(max_cart_to_traj[0])
            # estimated_v_max.append(min(v_max[i] * rmin, v_max_curv))
            estimated_v_max.append(v_max[i] * rmin)
            estimated_v_min.append(v_min[i] * rmax)
        time7 = time.time()
        print(f"Time to estimate points in corridor: {time7 - time6:.3f}s") 

        time8 = time.time()
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

        # print("Solved successfully.")
        # print("s =", np.round(s, 4))
        # # print("v =", np.round(v, 4))
        # print("objective =", sol["objective_min_sum_s_hat_minus_s"])
        
        time9 = time.time()
        print(f"Time to solve optimization problem: {time9 - time8:.3f}s")


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
            if delta[0] == 0 and delta[1] == 0:
                o_i = repaired_state_list[-1].orientation
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
        tv_original, _ = repairer.t_solver.tc_object.calc_tv_updated(
                    repaired_traj.state_list, tc
                )
        print(f"Updated tv after post-processing: {tv}, original tv: {tv_original}, new tc: {repairer.t_solver.tc_object._tc},  repairer.tc: {repairer.tc}")


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

        # for state in cart_trajectory_before_repair:
        #     pos = state.position
        #     ct_pos = trajectory_clcs.convert_to_curvilinear_coords(pos[0], pos[1])
        #     print(f"Original trajectory state at time step {state.time_step}: position {pos}, curvilinear position {ct_pos}, velocity {state.velocity}")
        # for ct_pos in cl_trajectory_before_repair:
        #     print(f"Original curvilinear position {ct_pos}")
        
        # if config.repair.planner == 2:
        #     lon_constr = repairer.t_solver._planner._constraints.longitudinal_constraints
        #     rule_constr_dict = lon_constr.rule_constraints
        #     collision_constr = lon_constr.collision_free_constraints
        #     for key,val in rule_constr_dict.items():
        #         print(f"rule: {key}, constr: {vars(val)}")
        #     print(f"collision_constr: {vars(collision_constr)}")
        # elif config.repair.planner == 1:
        #     rule_constr = repairer.t_solver._planner.rule_constraints
        #     lon_constr = rule_constr._lon_dis_constraints
        #     print(f"rule: {lon_constr}")

        print('traffic_rule_monitor.rob_predicate', len(traffic_rule_monitor.rob_predicate[0]), f"example predicate: {traffic_rule_monitor.rob_predicate[0][0]}")
        print('traffic_rule_monitor.rob_abstraction', len(traffic_rule_monitor.rob_abstraction[0]), f"example abstraction: {traffic_rule_monitor.rob_abstraction[0][0]}")
            
        # for proposition in repairer.t_solver._prop_full:
        #     # print(f"repairer.t_solver._sel_prop:{repairer.t_solver._sel_prop}")
        #     print(f"repairer.t_solver._sel_prop:{repairer.t_solver._prop_full}")
        #     predicate = proposition.children[0]
        #     # print(f"predicate: {len(proposition.children)}")
        #     # print(f"predicate: {predicate.name}, {vars(predicate)}")