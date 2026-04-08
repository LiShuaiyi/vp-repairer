from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)
from crrepairer.smt.sat_solver.dpll_domain import DomainDPLL
from crmonitor.common.world import World
from shapely import affinity
import numpy as np
import time

import math

def calculate_safe_distance(
        v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow
    ):
    d_safe = (
        (v_lead**2) / (-2 * np.abs(a_min_lead))
        - (v_follow**2) / (-2 * np.abs(a_min_follow))
        + v_follow * t_react_follow
    )

    return d_safe
def calc_s(s, w, l, theta):
    rot_mat_factors = np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
    s = (
        rot_mat_factors[0] * l / 2.0 * np.cos(theta)
        - rot_mat_factors[1] * w / 2 * np.sin(theta)
        + s
    )
    return s

def KeepSafeDistEval(
        world: World, time_step, lead_id, follow_id, follow_cart_pos, follow_velocity, follow_theta, MAX_LON_DIST=200.0
    ) -> bool:
    
    vehicle_follow = world.vehicle_by_id(follow_id)
    vehicle_lead = world.vehicle_by_id(lead_id)
    lane_follow = vehicle_follow.get_lane(time_step)
    follow_width = vehicle_follow.shape.width
    follow_length = vehicle_follow.shape.length
    follow_curvi_pos = lane_follow.clcs.convert_to_curvilinear_coords(follow_cart_pos[0], follow_cart_pos[1])
    front_s = np.max(calc_s(follow_curvi_pos[0], follow_width, follow_length, follow_theta))
    
    if vehicle_lead.get_lane(time_step) is None:
        return 1.0
    a_min_follow = vehicle_follow.vehicle_param.get("a_min")
    a_min_lead = vehicle_lead.vehicle_param.get("a_min")
    t_react_follow = vehicle_follow.vehicle_param.get("t_react")
    safe_distance = calculate_safe_distance(
        follow_velocity,
        vehicle_lead.states_cr[time_step].velocity,
        a_min_lead,
        a_min_follow,
        t_react_follow,
    )

    delta_s = vehicle_lead.rear_s(time_step) - front_s
    robustness = np.clip((delta_s - safe_distance) / MAX_LON_DIST, -1.0, 1.0) 
    boolean = robustness > 0.0 
    return boolean


def InSameLaneEval(
        world: World, time_step, other_id, ego_id, ego_cart_pos, ego_theta, MAX_LAT_DIST=20.0  # NOTE: ego_theta is lanelet coordinate orientation not cartesian, could be fixed
    ) -> bool:
    lanelet_network = world.road_network.lanelet_network
    other_vehicle = world.vehicle_by_id(other_id)
    ego_vehicle = world.vehicle_by_id(ego_id)
    # ego_vehicle_cart = ego_vehicle.states_cr[time_step].position
    ego_vehicle_shape = ego_vehicle.shape
    ego_vehicle_orientation = ego_theta
    ego_cos = np.cos(ego_vehicle_orientation)
    ego_sin = np.sin(ego_vehicle_orientation)
    ego_mat = [ego_cos, -ego_sin, ego_sin, ego_cos, ego_cart_pos[0], ego_cart_pos[1]]
    ego_vehicle_shapely_object = affinity.affine_transform(ego_vehicle_shape.shapely_object, ego_mat)
    res = []
    for idx in lanelet_network._strtee.query(ego_vehicle_shapely_object):
        lanelet_shapely_polygon = lanelet_network._strtee.geometries[idx]
        if lanelet_shapely_polygon.intersects(ego_vehicle_shapely_object):
            res.append(lanelet_network._get_lanelet_id_by_shapely_polygon(lanelet_shapely_polygon))
    ego_lanelets = set(res)
    # other_vehicle_shape = other_vehicle.shape
    # other_vehicle_orientation = other_vehicle.states_cr[time_step].orientation
    # other_cos = np.cos(other_vehicle_orientation)
    # other_sin = np.sin(other_vehicle_orientation)
    # other_mat = [other_cos, -other_sin, other_sin, other_cos, other_vehicle.states_cr[time_step].position[0], other_vehicle.states_cr[time_step].position[1]]
    # other_vehicle_shapely_object = affinity.affine_transform(other_vehicle_shape.shapely_object, other_mat)
    
    # res = []
    # for idx in lanelet_network._strtee.query(other_vehicle_shapely_object):
    #     lanelet_shapely_polygon = lanelet_network._strtee.geometries[idx]
    #     if lanelet_shapely_polygon.intersects(other_vehicle_shapely_object):
    #         res.append(lanelet_network._get_lanelet_id_by_shapely_polygon(lanelet_shapely_polygon))
    # other_lanelets = set(res)
    other_lanelets = other_vehicle.lanelet_assignment[time_step]
    # ego_lanelets = set(lanelet_network.find_lanelet_by_shape(ego_vehicle_shape))
    
    '''
    Robustness is ommited because this would slow down a lot
    '''
    # distance_to_vertices = []
    # def point_to_polyline_distance_vec(p, verts, return_closest=False):
    #     p = np.asarray(p, dtype=float).reshape(2,)
    #     verts = np.asarray(verts, dtype=float)
    #     if verts.ndim != 2 or verts.shape[1] != 2:
    #         raise ValueError(f"verts must be (N,2), got {verts.shape}")
    #     if len(verts) == 0:
    #         raise ValueError("verts is empty")
    #     if len(verts) == 1:
    #         d = float(np.linalg.norm(p - verts[0]))
    #         if return_closest:
    #             return d, verts[0].copy(), 0, 0.0
    #         return d

    #     a = verts[:-1]          # (M,2)  M=N-1
    #     b = verts[1:]           # (M,2)
    #     v = b - a               # (M,2)
    #     w = p - a               # (M,2)

    #     vv = np.einsum("ij,ij->i", v, v)          # (M,)
    #     wv = np.einsum("ij,ij->i", w, v)          # (M,)

    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         t = np.where(vv > 0.0, wv / vv, 0.0)  # (M,)
    #     t = np.clip(t, 0.0, 1.0)

    #     closest = a + v * t[:, None]              # (M,2)
    #     diff = p - closest                        # (M,2)
    #     dist2 = np.einsum("ij,ij->i", diff, diff) # (M,)
    #     idx = int(np.argmin(dist2))
    #     min_dist = float(np.sqrt(dist2[idx]))
    #     if return_closest:
    #         return min_dist, closest[idx], idx, float(t[idx])
    #     return min_dist
    
    # for lanelet_id in other_lanelets:
    #     lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
    #     left_vertice = lanelet.left_vertices
    #     right_vertice = lanelet.right_vertices
    #     left_dist = point_to_polyline_distance_vec(ego_cart_pos, left_vertice)
    #     right_dist = point_to_polyline_distance_vec(ego_cart_pos, right_vertice)
    #     distance_to_vertices.append(left_dist)
    #     distance_to_vertices.append(right_dist)
    # min_distance = min(distance_to_vertices)
    # robustness = np.clip(min_distance / MAX_LAT_DIST, -1.0, 1.0)  

    other_lane = world.road_network.find_lanes_by_lanelets(other_lanelets)
    ego_lane = world.road_network.find_lanes_by_lanelets(ego_lanelets)
    # print(f"Time for finding lanes by lanelets: {time3

    common_lanelets = other_lane & ego_lane
    if common_lanelets:
        return True
    else:
        # robustness = -robustness
        return False


def CutInEval(
        world: World, time_step, other_id, ego_id, ego_cart_pos, ego_theta, in_same_lane_value=None, eps=1e-5
    ) -> bool:
    # lanelet_network = world.road_network.lanelet_network
    other_vehicle = world.vehicle_by_id(other_id)
    # other_vehicle_shape = other_vehicle.shape
    other_vehicle_cart = other_vehicle.states_cr[time_step].position
    # other_vehicle_shape.center = other_vehicle_cart
    # other_lanelets = set(lanelet_network.find_lanelet_by_shape(other_vehicle_shape))
    other_lanelets = other_vehicle.lanelet_assignment[time_step]
    other_lane = world.road_network.find_lanes_by_lanelets(other_lanelets)
    if len(other_lane) == 1:
        return False
    if in_same_lane_value is not None:
        in_same_lane = bool(in_same_lane_value)
    else:
        in_same_lane = InSameLaneEval(
            world, time_step, other_id, ego_id, ego_cart_pos, ego_theta
        )
    if not in_same_lane:
        return False
    lane = other_vehicle.get_lane(time_step)
    other_d = lane.clcs.convert_to_curvilinear_coords(other_vehicle_cart[0], other_vehicle_cart[1])[1]
    ego_d = lane.clcs.convert_to_curvilinear_coords(ego_cart_pos[0], ego_cart_pos[1])[1]
    other_orient = other_vehicle.get_lat_state(time_step).theta
    result = (other_d < ego_d and other_orient > eps) or (
            other_d > ego_d and other_orient < -eps
        )
    return result


def InFrontOfEval(
        world: World, time_step, lead_id, follow_id, follow_cart_pos, follow_theta, MAX_LON_DIST=200.0
    ) -> bool:
    vehicle_follow = world.vehicle_by_id(follow_id)
    
    vehicle_lead = world.vehicle_by_id(lead_id)
    lane_follow = vehicle_follow.get_lane(time_step)
    follow_width = vehicle_follow.shape.width
    follow_length = vehicle_follow.shape.length
    follow_curvi_pos = lane_follow.clcs.convert_to_curvilinear_coords(follow_cart_pos[0], follow_cart_pos[1])
    front_s = np.max(calc_s(follow_curvi_pos[0], follow_width, follow_length, follow_theta))
    if vehicle_lead.get_lane(time_step) is None:
        robustness = 1.0
    delta_s = vehicle_lead.rear_s(time_step) - front_s
    robustness = np.clip(delta_s / MAX_LON_DIST, -1.0, 1.0) 
    boolean = robustness > 0.0 
    return boolean



if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_Gar-1_1_T-1"
    # Build configuration object
    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()
    config.debug.show_plots = True
    config.repair.planner = 1  # 1: qp planner 2: miqp
    config.repair.constraint_mode = 2   # 1: Manual, 2: Reach
    config.repair.use_mpr = False

    # scenario_id = "DEU_LocationDLower-8_154_T-1"
    # # Build configuration object
    # config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    # config.update()
    # config.repair.rules = ["R_G1"]
    # config.repair.ego_id = 11
    # config.debug.show_plots = True
    # config.repair.planner = 1
    # config.repair.constraint_mode = 2
    # config.repair.use_mpr = False
    # config.repair.use_mpr_derivative = False

    ego_initial = retrieve_ego_vehicle(config)
    t_0 = config.repair.t_0

    # ========== Velocity Planning Feasibility Estimation Test Demo =========
    # ========== Some Pre-processing =========
    traffic_rule_monitor = STLRuleMonitor(config)
    repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
    cart_trajectory_before_repair = ego_initial.prediction.trajectory.state_list    # prediction probably does not include the initial state
    lanelet_clcs = repairer.t_solver._planner.vehicle_configuration.CLCS
    ct_ref_path = []
    resampling_factor = 25
    initial_pos = ego_initial.initial_state.position
    time0 = time.time()
    ct_ref_path.append(initial_pos)
    for j in range(resampling_factor):
        delta = (np.array(cart_trajectory_before_repair[0].position) - np.array(initial_pos)) / resampling_factor
        intermediate_pos = np.array(initial_pos) + j * delta
        ct_ref_path.append(intermediate_pos)
    for i in range(len(cart_trajectory_before_repair)-1):
        pos = cart_trajectory_before_repair[i].position
        next_pos = cart_trajectory_before_repair[i+1].position
        delta = (np.array(next_pos) - np.array(pos)) / resampling_factor
        for j in range(resampling_factor):
            intermediate_pos = np.array(pos) + j * delta
            ct_ref_path.append(intermediate_pos)
    ct_ref_path.append(cart_trajectory_before_repair[-1].position)
    ct_ref_path = np.array(ct_ref_path)
    params = CLCSParams()
    params.processing_option = ProcessingOption.SPLINE_SMOOTHING
    params.resampling.option = ResamplingOption.ADAPTIVE
    ref_path_processor = ProcessorFactory.create_processor(params)
    trajectory_clcs = CurvilinearCoordinateSystem(
        reference_path=ref_path_processor(ct_ref_path),
        params=params,
        preprocess_path=False
    )
    ct_ref_path_np = np.asarray(ct_ref_path)  # shape (N,2)
    lanelet_pts = np.empty((len(ct_ref_path_np), 2), dtype=float)
    for k, (x, y) in enumerate(ct_ref_path_np):
        lanelet_pts[k] = lanelet_clcs.convert_to_curvilinear_coords(x, y)  # [s, d]
    lanelet_s = lanelet_pts[:, 0]
    lanelet_d = lanelet_pts[:, 1]
    ds = np.gradient(lanelet_s)
    dd = np.gradient(lanelet_d)
    eps = 1e-6
    ds_safe = np.where(np.abs(ds) < eps, eps, ds)
    ratio_cos = np.sqrt(ds ** 2 / (ds_safe ** 2 + dd ** 2))
    min_ratio_cos = np.min(ratio_cos)
    max_ratio_cos = np.max(ratio_cos)
    min_theta = np.arccos(max_ratio_cos)
    max_theta = np.arccos(min_ratio_cos)
    time1 = time.time()
    print(f"Time for pre-processing and predicate evaluation: {time1 - time0:.6f}s, with shape of reference path: {ct_ref_path_np.shape}")

    # ========= Reachable Set Along Trajectory Estimation =========
    time0 = time.time()
    ct_initial_pos = trajectory_clcs.convert_to_curvilinear_coords(
        float(ego_initial.initial_state.position[0]),
        float(ego_initial.initial_state.position[1])
    )
    v0 = ego_initial.initial_state.velocity
    s0 = ct_initial_pos[0]
    a_lon_max = repairer.t_solver._planner.lon_planner.c_ti.a_long_max
    a_lon_min = repairer.t_solver._planner.lon_planner.c_ti.a_long_min
    v_lon_max = repairer.t_solver._planner.lon_planner.c_ti.v_long_max
    v_lon_min = 0.0
    maximum = trajectory_clcs.convert_to_curvilinear_coords(
        float(ct_ref_path[-1][0]),
        float(ct_ref_path[-1][1])
    )
    maximum = maximum[0]
    dt = repairer.t_solver._planner._scenario.dt
    ct_reach = []
    for t in range(len(cart_trajectory_before_repair)):
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
            s_max = s_prev_max + v_prev_max * dt + 0.5 * a_lon_max * dt **2
        if v_prev_min + a_lon_min * dt < v_lon_min:
            s_min = s_prev_min + v_prev_min * dt + 0.5 * a_lon_min * dt **2
        s_max = min(s_max, maximum)
        s_min = max(s_min, s0)
        ct_reach.append((t+1+t_0, s_min, s_max, v_min, v_max))
        v_prev_max = v_max
        v_prev_min = v_min
        s_prev_max = s_max
        s_prev_min = s_min
    time1 = time.time()
    print(f"Time for reachable set estimation: {time1 - time0:.6f}s")
        
    # ========== Convert Reachable Set Back to Cartesian Coordinates and Lanelet CLCS ========= 
    time0 = time.time() 
    # # Additionally sample the reference trajectory in  trajectory coordinate and convert back to cartesian coordinates
    # cart_sample = []
    # cl_sample = []
    # ct_sample = []
    # M_sample = 50
    # ds = (maximum - s0) / M_sample
    # for m in range(M_sample):
    #     s = s0 + t * ds
    #     ct_sample.append((s, 0.0))
    #     # convert back to cartesian
    #     pos = trajectory_clcs.convert_to_cartesian_coords(s, 0.0)
    #     cart_sample.append(pos)
    #     pos_cl = lanelet_clcs.convert_to_curvilinear_coords(float(pos[0]), float(pos[1]))
    #     cl_sample.append(pos_cl)
    # Convert reachableset back to cartesian and lanelet CLCS
    cart_reach = []
    cl_reach = []
    for t in range(len(ct_reach)):
        s_min = ct_reach[t][1]
        s_max = ct_reach[t][2]
        # convert back to cartesian
        pos_min = trajectory_clcs.convert_to_cartesian_coords(s_min, 0.0)
        pos_max = trajectory_clcs.convert_to_cartesian_coords(s_max, 0.0)
        cart_reach.append((t, pos_min, pos_max))
        pos_min_cl = lanelet_clcs.convert_to_curvilinear_coords(float(pos_min[0]), float(pos_min[1]))
        pos_max_cl = lanelet_clcs.convert_to_curvilinear_coords(float(pos_max[0]), float(pos_max[1]))
        v_lon_max_cl = ct_reach[t][4] * max_ratio_cos
        v_lon_min_cl = ct_reach[t][3] * min_ratio_cos
        a_lon_max_cl = a_lon_max * max_ratio_cos
        a_lon_min_cl = a_lon_min * max_ratio_cos
        cl_reach.append((t+1+t_0, pos_min_cl, pos_max_cl, v_lon_min_cl, v_lon_max_cl, a_lon_min_cl, a_lon_max_cl))
    time1 = time.time()
    print(f"Time for coordinate conversion: {time1 - time0:.6f}s")

    # ========== Estimate Predicate Range =========
    other_id = traffic_rule_monitor.other_id
    safe_dist = []
    in_same_lane = []
    cut_in = []
    in_front_of = []
    stlmonitor_world = traffic_rule_monitor.world
    time0 = time.time()

    for t in range(len(cart_trajectory_before_repair)):
        # test = lanelet_clcs.convert_to_curvilinear_coords(
        #     float(cart_trajectory_before_repair[t].position[0]),
        #     float(cart_trajectory_before_repair[t].position[1])
        # )
        # s_min = ct_reach[t][1]
        # s_max = ct_reach[t][2]
        # idx_max = int((s_max - s0) / ds)
        # idx_min = min(int((s_min - s0) / ds) + 1, idx_max)
        time11 = time.time()
        safe_dist_min = KeepSafeDistEval(   # Safe distance predicate only considers maximum deceleration
            world=stlmonitor_world,
            time_step=t+1,
            lead_id=other_id,
            follow_id=ego_initial.obstacle_id,
            follow_cart_pos=cart_reach[t][1],
            # follow_cart_pos=cart_trajectory_before_repair[t].position,
            follow_velocity=cl_reach[t][3],
            # follow_velocity=cart_trajectory_before_repair[t].velocity,
            follow_theta=max_theta
        )
        safe_dist_max = KeepSafeDistEval(
            world=stlmonitor_world,
            time_step=t+1,
            lead_id=other_id,
            follow_id=ego_initial.obstacle_id,
            follow_cart_pos=cart_reach[t][2],
            # follow_cart_pos=cart_trajectory_before_repair[t].position,
            follow_velocity=cl_reach[t][4],
            # follow_velocity=cart_trajectory_before_repair[t].velocity,
            follow_theta=max_theta
        )
        time12 = time.time()
        # print(f"Time for evaluation safedist: {time12 - time11:.6f}s")
        if safe_dist_min != safe_dist_max:
            safe_dist.append((t+1+t_0, 2))
        else:
            safe_dist.append((t+1+t_0, int(safe_dist_min)))
        time21 = time.time()
        in_same_lane_min = InSameLaneEval(
            world=stlmonitor_world,
            time_step=t+1,
            other_id=other_id,
            ego_id=ego_initial.obstacle_id,
            ego_cart_pos=cart_reach[t][1],
            ego_theta=max_theta
        )
        in_same_lane_max = InSameLaneEval(
            world=stlmonitor_world,
            time_step=t+1,
            other_id=other_id,
            ego_id=ego_initial.obstacle_id,
            ego_cart_pos=cart_reach[t][2],
            ego_theta=max_theta
        )
        time22 = time.time()
        print(f"Time for evaluation in_same_lane: {time22 - time21:.6f}s")
        if in_same_lane_min != in_same_lane_max:
            in_same_lane.append((t+1+t_0, 2))
        else:
            in_same_lane.append((t+1+t_0, int(in_same_lane_min)))
        time31 = time.time()
        cut_in_min = CutInEval(
            world=stlmonitor_world,
            time_step=t+1,
            other_id=other_id,
            ego_id=ego_initial.obstacle_id,
            ego_cart_pos=cart_reach[t][1],
            ego_theta=max_theta,
            in_same_lane_value=in_same_lane_min
        )
        cut_in_max = CutInEval(
            world=stlmonitor_world,
            time_step=t+1,
            other_id=other_id,
            ego_id=ego_initial.obstacle_id,
            ego_cart_pos=cart_reach[t][2],
            ego_theta=max_theta,
            in_same_lane_value=in_same_lane_max
        )
        time32 = time.time()
        print(f"Time for evaluation cut_in: {time32 - time31:.6f}s")
        if cut_in_min != cut_in_max:
            cut_in.append([t+1+t_0, 2])
        else:
            cut_in.append([t+1+t_0, int(cut_in_min)])
        time41 = time.time()
        in_front_of_min = InFrontOfEval(
            world=stlmonitor_world,
            time_step=t+1,
            lead_id=other_id,
            follow_id=ego_initial.obstacle_id,
            follow_cart_pos=cart_reach[t][1],
            follow_theta=max_theta
        )
        in_front_of_max = InFrontOfEval(
            world=stlmonitor_world,
            time_step=t+1,
            lead_id=other_id,
            follow_id=ego_initial.obstacle_id,
            follow_cart_pos=cart_reach[t][2],
            follow_theta=max_theta
        )
        time42 = time.time()
        print(f"Time for evaluation in_front_of: {time42 - time41:.6f}s")
        if in_front_of_min != in_front_of_max:
            in_front_of.append((t+1+t_0, 2))
        else:
            in_front_of.append((t+1+t_0, int(in_front_of_min)))
    
    time1 = time.time()
    # print(f"Average time to evaluate predicates: {((time1 - time0)/len(cart_trajectory_before_repair)):.6f}s")
    print(f"Time for predicate evaluation: {time1 - time0:.6f}s")
    
    # print(traffic_rule_monitor.rob_rule)
    # print(traffic_rule_monitor.rob_predicate)
    # print(traffic_rule_monitor.other_ids)

    # ========== SAT Solve with Domain Constraints =========
    time0 = time.time()
    prop_nodes = repairer.sat_solver._prop_nodes
    formula = repairer.sat_solver._formula
    def NOT(value_seq: np.ndarray) -> np.ndarray:
        assert len(value_seq.shape) == 1 and value_seq.dtype == int and set(np.unique(value_seq)).issubset({0,1,2})
        lut = np.array([1, 0, 2])
        not_value_seq = lut[value_seq]
        return not_value_seq
    def PREVIOUS(value_seq: np.ndarray, t=0) -> np.ndarray:
        assert len(value_seq.shape) == 1 and value_seq.dtype == int and set(np.unique(value_seq)).issubset({0,1,2})
        if t < -1:
            raise ValueError("t must be >= -1")
        if t == -1:
            return value_seq
        prev_value_seq = np.empty_like(value_seq)
        prev_value_seq[:t+1] = 2  
        prev_value_seq[t+1:] = value_seq[:-1-t]
        return prev_value_seq
    def AND(value_seq1: np.ndarray, value_seq2: np.ndarray) -> np.ndarray:
        assert len(value_seq1.shape) == 1 and len(value_seq2.shape) == 1
        assert value_seq1.shape == value_seq2.shape
        assert value_seq1.dtype == int and value_seq2.dtype == int
        assert set(np.unique(value_seq1)).issubset({0,1,2}) and set(np.unique(value_seq2)).issubset({0,1,2})
        lut = np.array([
            [0, 0, 0],  
            [0, 1, 2],  
            [0, 2, 2],  
        ])
        return lut[value_seq1, value_seq2]
    def OR(value_seq1: np.ndarray, value_seq2: np.ndarray) -> np.ndarray:
        assert len(value_seq1.shape) == 1 and len(value_seq2.shape) == 1
        assert value_seq1.shape == value_seq2.shape
        assert value_seq1.dtype == int and value_seq2.dtype == int
        assert set(np.unique(value_seq1)).issubset({0,1,2}) and set(np.unique(value_seq2)).issubset({0,1,2})
        lut = np.array([
            [0, 1, 2],  
            [1, 1, 1],  
            [2, 1, 2],  
        ])
        return lut[value_seq1, value_seq2]
    def ONCE(t1, t2, value_seq: np.ndarray) -> np.ndarray:
        assert len(value_seq.shape) == 1 and value_seq.dtype == int and set(np.unique(value_seq)).issubset({0,1,2})
        for i in range(t1, t2+1):
            if i == t1:
                once_value_seq = PREVIOUS(value_seq, t=i-1)
            else:
                tmp = PREVIOUS(value_seq, t=i-1)
                once_value_seq = OR(once_value_seq, tmp)
        return once_value_seq 
    for prop_node in prop_nodes:
        prop_name = prop_node.name
        if 'once' in prop_name: # deal with (cut_in__1_0)and(previous(not(cut_in__1_0)))
            time_horizon = []
            time_horizon.append(int(prop_name[5]))
            time_horizon.append(int(prop_name[7]))
            value_seq = []
            for i in range(len(cut_in)):
                value_seq.append(cut_in[i][1])
            cut_seq = np.array(value_seq, dtype=int)
            not_cut_seq = NOT(cut_seq)
            prev_not_cut_seq = PREVIOUS(not_cut_seq)
            cut_and_prev_not_cut_seq = AND(cut_seq, prev_not_cut_seq)
            once_seq = ONCE(time_horizon[0], time_horizon[1], cut_and_prev_not_cut_seq)
            for i in range(len(cut_in)):
                cut_in[i][1] = once_seq[i]
    
    prop_tuples = set()
    for t in range(len(cart_trajectory_before_repair)): 
        prop_tuples.add((
            safe_dist[t][1],
            in_same_lane[t][1],
            in_front_of[t][1],
            cut_in[t][1]
        ))
    print("Unique predicate combinations over time steps:", prop_tuples)

    domain_dict = dict()
    for prop_tuple in prop_tuples:
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            prop_alphabet = prop_node.alphabet
            if 'distance' in prop_name:
                if prop_tuple[0] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[0]])
            elif 'lane' in prop_name:
                if prop_tuple[1] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[1]])
            elif 'front' in prop_name:
                if prop_tuple[2] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[2]])
            elif 'cut_in' in prop_name:
                if prop_tuple[3] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[3]])
        
        solver = DomainDPLL(formula, prop_nodes)
        solver.set_domains(domain_dict)
        sat_result = solver.solve()
        print(f"SAT result for prop tuple {prop_tuple}: true= {solver._assign_true}, false= {solver._assign_false}")
    
    time1 = time.time()
    print(f"Time to solve SAT with domain constraints for all prop tuples: {time1 - time0:.6f}s")
        
