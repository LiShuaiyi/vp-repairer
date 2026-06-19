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
from typing import Callable, Dict, List, Set, Tuple
from shapely import affinity
import numpy as np
import time

import math


def KeepSpeedLimitEval(world: World, time_step, ego_velocity, speed_limit, MAX_SPEED: float = 250.0 / 3.6, eps=1e-5):
    if speed_limit is None:
        robustness = math.inf
    else:
        robustness = speed_limit + eps - ego_velocity
    robustness = np.clip(robustness / MAX_SPEED, -1.0, 1.0)
    boolean = robustness > 0.0 
    return boolean



if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_Muc-4_2_T-1"
    # Build configuration object
    config = RepairerConfiguration.load(f"../config/{scenario_id}.yaml", scenario_id)
    config.update()
    config.repair.planner = 1
    config.repair.constraint_mode = 2

    ego_initial = retrieve_ego_vehicle(config)
    t_0 = config.repair.t_0
    t_f = config.repair.t_f - 1

    # ========== Velocity Planning Feasibility Estimation Test Demo =========
    # ========== Some Pre-processing =========
    traffic_rule_monitor = STLRuleMonitor(config)
    # print(traffic_rule_monitor.rob_rule)
    # print(traffic_rule_monitor.rob_predicate)
    # print(traffic_rule_monitor.other_ids)
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
    processed_ref_path = ref_path_processor(ct_ref_path)
    extend_by_init = processed_ref_path[0] * 2 - processed_ref_path[1]
    processed_ref_path = np.vstack((extend_by_init, processed_ref_path)) 
    trajectory_clcs = CurvilinearCoordinateSystem(
        reference_path=processed_ref_path,
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
    lane_speed_limit = None
    type_speed_limit = None
    fov_speed_limit = None
    brake_speed_limit = None
    stlmonitor_world = traffic_rule_monitor.world
    # extract speed limit values
    for idx, proposition in enumerate(traffic_rule_monitor.proposition_nodes):
        for predicate in proposition.children:
            speed_limit = predicate.evaluator.get_speed_limit(
                                stlmonitor_world, t_0, [ego_initial.obstacle_id]
                            )
            if 'lane' in predicate.base_name:
                lane_speed_limit = speed_limit
            elif 'type' in predicate.base_name:
                type_speed_limit = speed_limit
            elif 'fov' in predicate.base_name:
                fov_speed_limit = speed_limit
            elif 'brake' in predicate.base_name:
                brake_speed_limit = speed_limit
            else:
                print(f"unseen speed limit predicate name: {predicate.base_name} with value of {speed_limit}")
    
    time0 = time.time()
    lane_speed = []
    type_speed = []
    fov_speed = []
    brake_speed = []
    for t in range(len(cart_trajectory_before_repair)):
        # test = lanelet_clcs.convert_to_curvilinear_coords(
        #     float(cart_trajectory_before_repair[t].position[0]),
        #     float(cart_trajectory_before_repair[t].position[1])
        # )
        v_min = cl_reach[t][3]
        v_max = cl_reach[t][4]
        lane_speed_min = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_min, lane_speed_limit
        )
        lane_speed_max = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_max, lane_speed_limit
        )
        if lane_speed_min != lane_speed_max:
            lane_speed.append((t+1+t_0, 2))
        else:
            lane_speed.append((t+1+t_0, int(lane_speed_min)))
        type_speed_min = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_min, type_speed_limit
        )
        type_speed_max = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_max, type_speed_limit
        )
        if type_speed_min != type_speed_max:
            type_speed.append((t+1+t_0, 2))
        else:
            type_speed.append((t+1+t_0, int(type_speed_min)))
        fov_speed_min = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_min, fov_speed_limit
        )
        fov_speed_max = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_max, fov_speed_limit
        )
        if fov_speed_min != fov_speed_max:
            fov_speed.append((t+1+t_0, 2))
        else:
            fov_speed.append((t+1+t_0, int(fov_speed_min)))
        brake_speed_min = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_min, brake_speed_limit
        )
        brake_speed_max = KeepSpeedLimitEval(
            stlmonitor_world, t+1+t_0, v_max, brake_speed_limit
        )
        if brake_speed_min != brake_speed_max:
            brake_speed.append((t+1+t_0, 2))
        else:
            brake_speed.append((t+1+t_0, int(brake_speed_min)))
    
    time1 = time.time()
    print(f"Time to evaluate predicates: {(time1 - time0):.6f}s")

    
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
    
    prop_tuples = set()
    for t in range(len(cart_trajectory_before_repair)): 
        prop_tuples.add((
            lane_speed[t][1],
            type_speed[t][1],
            fov_speed[t][1],
            brake_speed[t][1]
        ))
    print("Unique predicate combinations over time steps:", prop_tuples)

    domain_dict = dict()
    for prop_tuple in prop_tuples:
        for prop_node in prop_nodes:
            prop_name = prop_node.name
            prop_alphabet = prop_node.alphabet
            if 'lane' in prop_name:
                if prop_tuple[0] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[0]])
            elif 'type' in prop_name:
                if prop_tuple[1] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[1]])
            elif 'fov' in prop_name:
                if prop_tuple[2] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[2]])
            elif 'brake' in prop_name:
                if prop_tuple[3] == 2:
                    domain_dict[prop_alphabet] = set([0,1])
                else:
                    domain_dict[prop_alphabet] = set([prop_tuple[3]])
        
        solver = DomainDPLL(formula, prop_nodes)
        solver.set_domains(domain_dict)
        sat_result = solver.solve()
    
    time1 = time.time()
    print(f"Time to solve SAT with domain constraints for all prop tuples: {time1 - time0:.6f}s")
    
