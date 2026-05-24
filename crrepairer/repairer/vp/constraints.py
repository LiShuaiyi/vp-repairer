"""Constraint extraction helpers for velocity-planning repair."""

import math
from typing import List, Union

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.state import CustomState

from commonroad_clcs.clcs import CurvilinearCoordinateSystem

from crmonitor.common.world import World


class VPConstraintExtraction:
    """Extracts longitudinal position and velocity constraints for VP repair."""

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

        final_time_step = all_states[-1].time_step
        for time_step in range(int(self._tc) + 1, final_time_step + 1):
            idx = time_step - int(self._tc) - 1
            follow_velocity = all_states[time_step - all_states[0].time_step].velocity
            v_max_list = []
            s_max_list = []
            s_min_list = []

            for prop in self._sel_prop:
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
                elif "brakes_abruptly" in prop.name:
                    self._constraint_not_break_abruptly(prop)
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
        if "R_IN4" not in self.config.repair.rules and "R_IN1" not in self.config.repair.rules and "R_IN3_hand_draft" not in self.config.repair.rules and "R_IN5" not in self.config.repair.rules:
            raise NotImplementedError(
                "Intersection VP constraints currently support R_IN1, R_IN4, R_IN3_hand_draft and R_IN5 only."
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

        for prop in self._sel_prop:
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
                    _, rear_constr = self._constraint_in_intersection_conflict_area(
                        time_step=time_step,
                        prop_assignment=prop_assignment,
                        lanelet_clcs=lanelet_clcs,
                        cart=True,
                    )
                    if not np.isfinite(np.asarray(rear_constr, dtype=float)).all():
                        rear_constr_ct = None
                    else:
                        rear_constr_ct = trajectory_clcs.convert_to_curvilinear_coords(
                            float(rear_constr[0]),
                            float(rear_constr[1]),
                        )[0]
                    if rear_constr_ct is not None and np.isfinite(rear_constr_ct):
                        trajectory_s_max_cap[idx] = min(
                            trajectory_s_max_cap[idx],
                            rear_constr_ct - wheelbase / 2,
                        )
                else:
                    front_constr, _ = self._constraint_in_intersection_conflict_area(
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

    def _constraint_in_intersection_conflict_area_rear_ct(
        self,
        time_step: int,
        prop_assignment: float,
        lanelet_clcs: CurvilinearCoordinateSystem,
        trajectory_clcs: CurvilinearCoordinateSystem,
    ):
        front_constr, rear_constr = self._constraint_in_intersection_conflict_area(
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
