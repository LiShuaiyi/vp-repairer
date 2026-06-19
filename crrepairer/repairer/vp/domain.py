"""DomainDPLL domain estimation helpers for velocity-planning repair."""

import math
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
        domain_dict = self._domain_dict_construct_general(
            predicate_values, self.sat_solver._prop_nodes
        )
        breakdown["infer_domain_dict"] = time.time() - start
        self.domain_dict_breakdown = breakdown
        return domain_dict

    def _build_domain_dict_for_sat_direct(self):
        '''Extract domain dict for propositions directly from original trajectory's stored robustness values. Currently supports R_IN1, R_IN4, R_G2 and R_IN3_hand_draft and R_IN5.'''
        self.domain_dict_breakdown = {}
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
        domain_dict = {}
        for prop_node in prop_nodes:
            predicate_values_key = self._prop_node_name_to_predicate_values_key(prop_node.name, predicate_values)
            if predicate_values_key == {0} or predicate_values_key == {1}:
                domain_dict[prop_node.alphabet[-1]] = set(predicate_values_key)
        return domain_dict

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
