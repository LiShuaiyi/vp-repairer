import math
import re
from fractions import Fraction
import numpy as np
from typing import List, Union
from collections import defaultdict

from commonroad.scenario.trajectory import Trajectory

from commonroad_crime.utility.simulation import Maneuver
import shapely

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
    PredPreceding,
    PredStopLineInFront,
    PredInIntersectionConflictArea,
    PredOnLaneletWithTypeIntersection,
)

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from commonroad.scenario.lanelet import LaneletType
from shapely.geometry import Polygon, LineString

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory


class BasicConstraint:
    def __init__(self):
        self.var_lat_x_ub = []
        self.var_lat_x_lb = []
        self.var_lat_u_lb = []
        self.var_lat_u_ub = []

        self.var_long_x_ub = []
        self.var_long_x_lb = []
        self.var_long_u_lb = []
        self.var_long_u_ub = []

        self.var_slack_lb = []
        self.var_slack_ub = []

        self.dynamic_matrix_list = []
        self.init_state = []


class LongitudinalConstraint(BasicConstraint):
    def __init__(
            self,
            tc_object: TC,
            rule_monitor: STLRuleMonitor,
            sel_proposition_full: List[PropositionNode],
            proposition_full: List[PropositionNode],
            veh_config: PlanningConfigurationVehicle,
            initial_trajectory: Trajectory,
            start_time_step: int,
    ):
        super().__init__()
        self._foll_veh = None
        self._prec_veh = None
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._start_time_step = start_time_step
        self._world_state = self._rule_monitor.world
        self._other_id = self._rule_monitor.other_id
        self._ego_id = (
            self._rule_monitor.vehicle_id
        )  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full
        self._prop_full = proposition_full
        self._veh_config = veh_config
        self.rule_constraints = {}
        self.collision_free_constraints = {}
        for proposition in proposition_full:
            if "in_intersection_conflict_area__a0_a1" in proposition.name:
                self.s_circle_center_front, self.s_circle_center_rear, self.conflict_line_front, self.conflict_line_rear = self.create_conflict_area_parameter()
                break

        self._target_lanes = defaultdict(List[Lane])

    @property
    def target_lanes(self) -> dict:
        return self._target_lanes

    @property
    def sel_prop_full(self) -> list:
        return self._sel_prop_full

    def add_rule_constraints(self):
        num_time_step = 0
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            num_time_step += 1
            total_assignment = self._rule_monitor.prop_robust_all[:, min(k - self._start_time_step + self._tc_obj.future_time_step, self._tc_obj.N - self._start_time_step)]
            for idx, proposition in enumerate(self._rule_monitor.proposition_nodes):
                try:
                    prop_assignment = total_assignment[
                        total_assignment == total_assignment
                        ][idx]
                except:
                    # no assignment can be found
                    continue
                if (
                    proposition in self._prop_full
                    and k >= self._tc_obj.tv_time_step - self._tc_obj.future_time_step
                ):
                    # proposition to be repaired (greater than the time-to-violation)
                    robs_at_tv = self._rule_monitor.prop_robust_all[
                        :, self._tc_obj.tv_time_step - self._start_time_step
                    ]
                    prop_assignment = robs_at_tv[robs_at_tv == robs_at_tv][idx]
                    if proposition in self._sel_prop_full:
                        prop_assignment = -prop_assignment
                if k < self._tc_obj.tv_time_step - self._tc_obj.future_time_step or proposition in self._prop_full:
                    if(
                            proposition.name[0:5] == "once["
                            and proposition.name[5:6] != proposition.name[7:8]
                    ):
                        pattern = r"once\[(.*?)\]"
                        matches = re.findall(pattern, proposition.name)
                        further_time = Fraction(matches[0].split(",")[1])
                        future_time_step = int(float(further_time) * self._tc_obj.future_time_step)
                        for time_step in range(k, min(k + future_time_step + 1, self._tc_obj.N + 1)):
                            self.add_rule_constraint_time_step(proposition, prop_assignment, time_step)
                    else:
                        self.add_rule_constraint_time_step(proposition, prop_assignment, k)
                    # for predicate in proposition.children:
                    #     if not hasattr(predicate, "base_name"):
                    #         continue
                    #     if (
                    #             predicate.base_name
                    #             == PredInIntersectionConflictArea.predicate_name
                    #     ):
                    #         if predicate.base_name in self.rule_constraints.keys():
                    #             # avoid multiple updates in one time step for the same predicate constraints
                    #             if (
                    #                     len(
                    #                         self.rule_constraints[predicate.base_name][
                    #                             "s_limit_front"
                    #                         ]
                    #                     )
                    #                     == num_time_step
                    #             ):
                    #                 continue
                    #             (
                    #                 s_limit_front,
                    #                 s_limit_behind,
                    #             ) = self.ConstrInIntersectionConflictAreaEgo(
                    #                 k, prop_assignment
                    #             )
                    #             self.rule_constraints[predicate.base_name][
                    #                 "s_limit_front"
                    #             ].append(s_limit_front)
                    #             self.rule_constraints[predicate.base_name][
                    #                 "s_limit_behind"
                    #             ].append(s_limit_behind)
                    #         else:
                    #             self.rule_constraints[predicate.base_name] = {
                    #                 "decision_variable": True,
                    #                 "num_decision_variables": 1,
                    #                 "constraint_name": "conflict_area",
                    #                 "constraint_state": [0],
                    #                 "s_limit_front": [],
                    #                 "s_limit_behind": [],
                    #             }
                    #             (
                    #                 s_limit_front,
                    #                 s_limit_behind,
                    #             ) = self.ConstrInIntersectionConflictAreaEgo(
                    #                 k, prop_assignment
                    #             )
                    #             self.rule_constraints[predicate.base_name][
                    #                 "s_limit_front"
                    #             ].append(s_limit_front)
                    #             self.rule_constraints[predicate.base_name][
                    #                 "s_limit_behind"
                    #             ].append(s_limit_behind)

    def add_rule_constraint_time_step(self, proposition, prop_assignment, time_step: int):
        for predicate in proposition.children:
            if not hasattr(predicate, "base_name"):
                continue
            if predicate.base_name == PredInIntersectionConflictArea.predicate_name and predicate.agent_placeholders == (0, 1):
                if predicate.base_name in self.rule_constraints.keys():
                    (
                        s_limit_front,
                        s_limit_behind,
                    ) = self.ConstrInIntersectionConflictAreaEgo(time_step, prop_assignment)

                    # avoid multiple updates in one time step for the same predicate constraints
                    if (
                        time_step in self.rule_constraints[predicate.base_name]["time_step"]
                    ):
                        index = self.rule_constraints[predicate.base_name]["time_step"].index(time_step)
                        if s_limit_front < self.rule_constraints[predicate.base_name]["s_limit_front"][index]:
                            self.rule_constraints[predicate.base_name]["s_limit_front"][index] = s_limit_front
                        if s_limit_behind > self.rule_constraints[predicate.base_name]["s_limit_behind"][index]:
                            self.rule_constraints[predicate.base_name]["s_limit_behind"][index] = s_limit_behind
                    else:
                        self.rule_constraints[predicate.base_name]["s_limit_front"].append(
                            s_limit_front
                        )
                        self.rule_constraints[predicate.base_name]["s_limit_behind"].append(
                            s_limit_behind
                        )
                        self.rule_constraints[predicate.base_name]["time_step"].append(time_step)
                else:
                    self.rule_constraints[predicate.base_name] = {
                        "decision_variable": True,
                        "num_decision_variables": 1,
                        "constraint_name": "conflict_area",
                        "constraint_state": [0],
                        "s_limit_front": [],
                        "s_limit_behind": [],
                        "time_step": [],
                    }
                    (
                        s_limit_front,
                        s_limit_behind,
                    ) = self.ConstrInIntersectionConflictAreaEgo(time_step, prop_assignment)
                    self.rule_constraints[predicate.base_name]["s_limit_front"].append(
                        s_limit_front
                    )
                    self.rule_constraints[predicate.base_name]["s_limit_behind"].append(
                        s_limit_behind
                    )
                    self.rule_constraints[predicate.base_name]["time_step"].append(time_step)
            elif predicate.base_name == PredStopLineInFront.predicate_name:
                if predicate.base_name in self.rule_constraints.keys():
                    (s_limit_front, s_limit_behind) = self.ConstrStopLineInFront(time_step, prop_assignment)

                    # avoid multiple updates in one time step for the same predicate constraints
                    if (
                            time_step in self.rule_constraints[predicate.base_name]["time_step"]
                    ):
                        index = self.rule_constraints[predicate.base_name]["time_step"].index(time_step)
                        if s_limit_front < self.rule_constraints[predicate.base_name]["s_limit_front"][index]:
                            self.rule_constraints[predicate.base_name]["s_limit_front"][index] = s_limit_front
                        if s_limit_behind > self.rule_constraints[predicate.base_name]["s_limit_behind"][index]:
                            self.rule_constraints[predicate.base_name]["s_limit_behind"][index] = s_limit_behind
                    else:
                        self.rule_constraints[predicate.base_name]["s_limit_front"].append(
                            s_limit_front
                        )
                        self.rule_constraints[predicate.base_name]["s_limit_behind"].append(
                            s_limit_behind
                        )
                        self.rule_constraints[predicate.base_name]["time_step"].append(time_step)
                else:
                    self.rule_constraints[predicate.base_name] = {
                        "decision_variable": False,
                        "num_decision_variables": 0,
                        "constraint_name": "stop_line",
                        "constraint_state": [0],
                        "s_limit_front": [],
                        "s_limit_behind": [],
                        "time_step": [],
                    }
                    (s_limit_front, s_limit_behind) = self.ConstrStopLineInFront(time_step, prop_assignment)
                    self.rule_constraints[predicate.base_name]["s_limit_front"].append(s_limit_front)
                    self.rule_constraints[predicate.base_name]["s_limit_behind"].append(s_limit_behind)
                    self.rule_constraints[predicate.base_name]["time_step"].append(time_step)
            # elif (predicate.base_name == PredOnLaneletWithTypeIntersection.predicate_name and proposition in self._prop_full):
            #     if predicate.base_name in self.rule_constraints.keys():
            #         (s_limit_front, s_limit_behind) = self.ConstrOnLaneletWithTypeIntersection(time_step, prop_assignment)
            #
            #         # avoid multiple updates in one time step for the same predicate constraints
            #         if (
            #                 time_step in self.rule_constraints[predicate.base_name]["time_step"]
            #         ):
            #             index = self.rule_constraints[predicate.base_name]["time_step"].index(time_step)
            #             if s_limit_front < self.rule_constraints[predicate.base_name]["s_limit_front"][index]:
            #                 self.rule_constraints[predicate.base_name]["s_limit_front"][index] = s_limit_front
            #             if s_limit_behind > self.rule_constraints[predicate.base_name]["s_limit_behind"][index]:
            #                 self.rule_constraints[predicate.base_name]["s_limit_behind"][index] = s_limit_behind
            #         else:
            #             self.rule_constraints[predicate.base_name]["s_limit_front"].append(
            #                 s_limit_front
            #             )
            #             self.rule_constraints[predicate.base_name]["s_limit_behind"].append(
            #                 s_limit_behind
            #             )
            #             self.rule_constraints[predicate.base_name]["time_step"].append(time_step)
            #     else:
            #         self.rule_constraints[predicate.base_name] = {
            #             "decision_variable": True,
            #             "num_decision_variables": 1,
            #             "constraint_name": "intersection",
            #             "constraint_state": [0],
            #             "s_limit_front": [],
            #             "s_limit_behind": [],
            #             "time_step": [],
            #         }
            #         (s_limit_front, s_limit_behind) = self.ConstrOnLaneletWithTypeIntersection(time_step, prop_assignment)
            #         self.rule_constraints[predicate.base_name]["s_limit_front"].append(s_limit_front)
            #         self.rule_constraints[predicate.base_name]["s_limit_behind"].append(s_limit_behind)
            #         self.rule_constraints[predicate.base_name]["time_step"].append(time_step)


    def add_collision_free_constraints(self):
        if self._rule_monitor.scenario_type == "interstate":
            self.add_collision_free_interstate()
        else:
            self.add_collision_free_intersection()

    def add_collision_free_interstate(self):
        self.collision_free_constraints["index_lb"] = list()
        self.collision_free_constraints["index_ub"] = list()
        self.collision_free_constraints["collision_free_ub"] = list()
        self.collision_free_constraints["collision_free_lb"] = list()
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            self._prec_veh, self._foll_veh = self._determine_related_veh(
                k, self._ego_vehicle.ref_path_lane
            )
            index = k - self._tc_obj.tc_time_step
            # TODO: fix plus or minus a small number
            if self._prec_veh is not None:
                self.collision_free_constraints["index_ub"].append(index)
                self.collision_free_constraints["collision_free_ub"].append(
                    self._prec_veh.rear_s(k, self._ego_vehicle.ref_path_lane)
                    - self._veh_config.wheelbase / 2
                    - self._veh_config.length / 2
                    - 2
                )
            if self._foll_veh is not None:
                self.collision_free_constraints["index_lb"].append(index)
                self.collision_free_constraints["collision_free_lb"].append(
                    self._prec_veh.front_s(k, self._ego_vehicle.ref_path_lane)
                    + self._veh_config.wheelbase / 2
                    + self._veh_config.length / 2
                    + 2
                )

    def add_collision_free_intersection(self):
        self.collision_free_constraints["index_lb"] = list()
        self.collision_free_constraints["index_ub"] = list()
        self.collision_free_constraints["collision_free_ub"] = list()
        self.collision_free_constraints["collision_free_lb"] = list()
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            self._prec_veh, self._foll_veh = self._determine_related_veh(
                k, self._ego_vehicle.ref_path_lane
            )
            index = k - self._tc_obj.tc_time_step
            if self._prec_veh is not None:
                if k <= self._prec_veh.end_time:
                    conflict_points = self.calculation_circle_approximation(k)
                    if conflict_points is not None:
                        s_min = self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_points[0])[0]
                        self.collision_free_constraints["index_ub"].append(index)
                        self.collision_free_constraints["collision_free_ub"].append(
                            s_min
                            - self._ego_vehicle.shape.length / 3
                            - self._veh_config.wheelbase / 2
                        )

    def calculation_circle_approximation(self, time_step):
        circle_target = self._prec_veh.circle_appr_occupancy_at_time_step(time_step)
        offset_circle_target = shapely.offset_curve(circle_target, self._ego_vehicle.circle_radius)
        reference_lane_center = LineString(self._ego_vehicle.ref_path_lane.new_vertice)
        intersection = reference_lane_center.intersection(offset_circle_target)
        conflict_line_points = list()
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif (
                intersection.geom_type == "LineString"
                or intersection.geom_type == "LinearRing"
        ):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif (
                intersection.geom_type == "MultiPoint"
                or intersection.geom_type == "MultiLineString"
        ):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            conflict_points = None
        else:
            conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
        return conflict_points


    def _determine_related_veh(self, time_step, lane):
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        vehicle_ids.update(lane.lanelet.dynamic_obstacle_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        ego_front_s = self._ego_vehicle.front_s(
            time_step, self._ego_vehicle.ref_path_lane
        )
        for id in vehicle_ids:
            if id in self._world_state.vehicle_ids():
                other_vehicle = self._world_state.vehicle_by_id(id)
            else:
                continue
            if other_vehicle is None:
                continue
            try:
                other_front_s = other_vehicle.front_s(
                    time_step, self._ego_vehicle.ref_path_lane
                )
            except:
                continue
            dist = other_front_s - ego_front_s
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def ConstrStopLineInFront(self, time_step: int, prop_assignment: float):
        wold = self._rule_monitor.world
        upper_bound = np.inf
        for lanelet_id in self._ego_vehicle.lanelets_dir:
            lanelet = wold.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is not None:
                stop_line_center = (lanelet.stop_line.start + lanelet.stop_line.end) / 2
                stop_line_direction = (
                    lanelet.stop_line.start - lanelet.stop_line.end
                ) / np.linalg.norm(lanelet.stop_line.start - lanelet.stop_line.end)
                direction_1 = np.array([stop_line_direction[1], -stop_line_direction[0]])
                direction_2 = np.array([-stop_line_direction[1], stop_line_direction[0]])
                stop_line_bound_1 = (
                    stop_line_center
                    + (self._veh_config.length / 2 + self._veh_config.wheelbase / 2)
                    * direction_1
                )
                stop_line_bound_2 = (
                    stop_line_center
                    + (self._veh_config.length / 2 + self._veh_config.wheelbase / 2)
                    * direction_2
                )
                s_1 = self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *stop_line_bound_1
                )[0]
                s_2 = self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *stop_line_bound_2
                )[0]
                stop_line_s = min(
                    self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.start
                    )[0],
                    self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.end
                    )[0],
                )
                max_distance_to_vehicle = np.sqrt(
                    (self._veh_config.length / 2 + self._veh_config.wheelbase / 2) ** 2
                    + (self._veh_config.width / 2) ** 2
                )
                upper_bound = min(upper_bound, stop_line_s - self._ego_vehicle.circle_radius - self._veh_config.length / 3 - self._veh_config.wheelbase / 2)
        return upper_bound, -math.inf

    def ConstrOnLaneletWithTypeIntersection(self, time_step: int, prop_assignment: float):
        if prop_assignment > 0:
            return math.inf, -math.inf
        else:
            incoming = self._ego_vehicle.incoming_intersection
            turning_lanelets = incoming.successors_left.union(
                incoming.successors_right,
                incoming.successors_straight,
            )
            lanelet_intersection_id = list(
                self._ego_vehicle.ref_path_lane.contained_lanelets.intersection(turning_lanelets)
            )
            lanelet_intersection = (
                self._rule_monitor.world.road_network.lanelet_network.find_lanelet_by_id(
                    lanelet_intersection_id[0]
                )
            )
            start_s = min(
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.right_vertices[0]
                )[0],
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.left_vertices[0]
                )[0],
            )
            end_s = max(
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.right_vertices[-1]
                )[0],
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.left_vertices[-1]
                )[0],
            )
            upper_bound = start_s - self._ego_vehicle.circle_radius - self._veh_config.length / 3 - self._veh_config.wheelbase / 2
            lower_bound = end_s + self._ego_vehicle.circle_radius
            return upper_bound, lower_bound

    def ConstrInIntersectionConflictAreaEgo(self, time_step: int, prop_assignment: float):
        if prop_assignment <= 0:
            front_constr = (
                self.s_circle_center_front
                - self._ego_vehicle.shape.length / 3
                - self._veh_config.wheelbase / 2
            )
            rear_constr = self.s_circle_center_rear
            return front_constr, rear_constr
        else:
            return math.inf, -math.inf

    def create_conflict_area_parameter(self):
        ego_vehicle = self._ego_vehicle
        target_vehicle = self._target_vehicle
        road_network = self._rule_monitor.world.road_network

        # offset conflict lanelets
        conflict_lanelets_shape = list()
        for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                conflict_lanelets_shape.append(lanelet.polygon.shapely_object)
        conflict_area_shape = shapely.unary_union(conflict_lanelets_shape)
        conflict_linestring = shapely.offset_curve(conflict_area_shape, ego_vehicle.circle_radius)

        # find right conflict point
        line_right = LineString(ego_vehicle.lanelets_dir_right_vertices)
        line_right_offset = shapely.offset_curve(line_right, ego_vehicle.circle_radius)
        conflict_circle_center_right = self.find_conflict_points(line_right_offset, conflict_linestring)
        if conflict_circle_center_right is not None:
            s_circle_center_right = [ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_right[0])[0],
                                     ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_right[1])[0]]
            s_circle_center_right = np.sort(s_circle_center_right)
        else:
            s_circle_center_right = np.array([np.inf, -np.inf])

        # find left conflict point
        line_left = LineString(ego_vehicle.lanelets_dir_left_vertices)
        line_left_offset = shapely.offset_curve(line_left, -ego_vehicle.circle_radius)
        conflict_circle_center_left = self.find_conflict_points(line_left_offset, conflict_linestring)
        if conflict_circle_center_left is not None:
            s_circle_center_left = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_left[0])[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_left[1])[0]]
            s_circle_center_left = np.sort(s_circle_center_left)
        else:
            s_circle_center_left = np.array([np.inf, -np.inf])

        # find center conflict point
        line_center = LineString(ego_vehicle.lanelets_dir_center_vertices)
        conflict_circle_center_center = self.find_conflict_points(line_center, conflict_linestring)
        if conflict_circle_center_center is not None:
            s_circle_center_center = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_center[0])[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_circle_center_center[1])[0]]
            s_circle_center_center = np.sort(s_circle_center_center)
        else:
            s_circle_center_center = np.array([np.inf, -np.inf])

        line_center_left_offset = shapely.offset_curve(line_center, ego_vehicle.shape.width / 2)
        conflict_center_left_offset = self.find_conflict_points(line_center_left_offset, conflict_area_shape)
        if conflict_center_left_offset is not None:
            s_center_left_offset = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_center_left_offset[0])[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_center_left_offset[1])[0]]
        line_center_right_offset = shapely.offset_curve(line_center, -ego_vehicle.shape.width / 2)
        conflict_center_right_offset = self.find_conflict_points(line_center_right_offset, conflict_area_shape)
        if conflict_center_right_offset is not None:
            s_center_right_offset = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_center_right_offset[0])[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(*conflict_center_right_offset[1])[0]]
        s_center_offset_front = np.min([s_center_left_offset[0], s_center_right_offset[0]]) - s_circle_center_center[0] - ego_vehicle.shape.length / 6
        s_center_offset_rear = -np.max([s_center_left_offset[1], s_center_right_offset[1]]) + s_circle_center_center[1] - ego_vehicle.shape.length / 6

        # find conflict point for constraints
        s_circle_center_front = np.min([s_circle_center_right[0], s_circle_center_left[0], s_circle_center_center[0]])
        s_circle_center_rear = np.max([s_circle_center_right[1], s_circle_center_left[1], s_circle_center_center[1]])
        # s_circle_center_front = s_circle_center_center[0]
        # s_circle_center_rear = s_circle_center_center[1]

        # find conflict area points in clcs, separate in two parts
        interpolation_distance = 0.5
        interpolated_points = []
        distance_along_line = 0.0
        while distance_along_line <= conflict_linestring.length:
            point = conflict_linestring.interpolate(distance_along_line)
            interpolated_points.append(point.xy)
            distance_along_line += interpolation_distance
        interpolated_points.append(conflict_linestring.interpolate(conflict_linestring.length).xy)
        exterior_arrays = [np.array(coord).reshape(2, 1) for coord in interpolated_points]
        conflict_linestring_clcs = ego_vehicle.ref_path_lane.clcs.convert_list_of_points_to_curvilinear_coords(exterior_arrays, 1)
        conflict_line = [[], []]
        conflict_s_interval = [[np.inf, -np.inf],[np.inf, -np.inf]]
        for i in range(1, len(conflict_linestring_clcs)):
            if conflict_linestring_clcs[i - 1][0] < conflict_linestring_clcs[i][0]:
                conflict_line[0].append(conflict_linestring_clcs[i])
                if conflict_linestring_clcs[i][0] < conflict_s_interval[0][0]:
                    conflict_s_interval[0][0] = conflict_linestring_clcs[i][0]
                if conflict_linestring_clcs[i][0] > conflict_s_interval[0][1]:
                    conflict_s_interval[0][1] = conflict_linestring_clcs[i][0]
            else:
                conflict_line[1].append(conflict_linestring_clcs[i])
                if conflict_linestring_clcs[i][0] < conflict_s_interval[1][0]:
                    conflict_s_interval[1][0] = conflict_linestring_clcs[i][0]
                if conflict_linestring_clcs[i][0] > conflict_s_interval[1][1]:
                    conflict_s_interval[1][1] = conflict_linestring_clcs[i][0]
        if conflict_s_interval[0][0] <= s_circle_center_front <= conflict_s_interval[0][1]:
            conflict_line_front = conflict_line[0]
            conflict_line_rear = conflict_line[1]
        else:
            conflict_line_front = conflict_line[1]
            conflict_line_rear = conflict_line[0]
        # self.test_plot(conflict_area_shape, conflict_linestring, conflict_line_front, conflict_line_rear, ego_vehicle.ref_path_lane, conflict_circle_center_right, conflict_circle_center_left, conflict_circle_center_center)
        conflict_line_front = np.array(conflict_line_front).T
        conflict_line_rear = np.array(conflict_line_rear).T
        return s_circle_center_front, s_circle_center_rear, conflict_line_front, conflict_line_rear

    @staticmethod
    def find_conflict_points(
        curved_line: LineString, conflict_polygon: Union[Polygon, LineString]
    ):
        conflict_line_points = list()
        # Get intersection of line and polygon
        intersection = curved_line.intersection(conflict_polygon)
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif (
            intersection.geom_type == "LineString" or intersection.geom_type == "LinearRing"
        ):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif (
            intersection.geom_type == "MultiPoint"
            or intersection.geom_type == "MultiLineString"
        ):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            conflict_points = None
        else:
            conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
        return conflict_points


class LateralConstraint(BasicConstraint):
    def __init__(
            self,
            tc_object: TC,
            rule_monitor: STLRuleMonitor,
            veh_config: PlanningConfigurationVehicle,
            target_lanes,
            long_traj: QPTrajectory,
            sel_proposition_full: List[PropositionNode],
    ):
        super().__init__()

        # TODO: fix input
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._veh_config = veh_config

        self.theta_r = list()
        self.target_lanes = target_lanes
        self._sel_prop_full = sel_proposition_full
        self.long_traj = long_traj

        self.lat_dis_cons_matrix = list()
        self._lat_dis_constraints = list()
        self.d_min = None
        self.d_max = None
        self.kappa_lim = None

    def create_d_constraints(self, long_traj: QPTrajectory):
        # TODO: fix construction (now copy from rule_constraints)
        self._lat_dis_constraints = list()
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            d_min = -np.inf
            d_max = np.inf
            if k in self.target_lanes:
                target_lanes = self.target_lanes[k]
                index = k - self._tc_obj.tc_time_step
                lane_boundary_left = target_lanes[
                    -1
                ].clcs_left.convert_to_cartesian_coords(
                    long_traj.states[index].position[0], 0.0
                )
                lane_boundary_right = target_lanes[
                    0
                ].clcs_right.convert_to_cartesian_coords(
                    long_traj.states[index].position[0], 0.0
                )
                d_max = min(
                    self._veh_config.curvilinear_coordinate_system.convert_to_curvilinear_coords(
                        lane_boundary_left[0], lane_boundary_left[1]
                    )[
                        1
                    ],
                    d_max,
                )
                d_min = max(
                    self._veh_config.curvilinear_coordinate_system.convert_to_curvilinear_coords(
                        lane_boundary_right[0], lane_boundary_right[1]
                    )[
                        1
                    ],
                    d_min,
                )
            self._lat_dis_constraints.append([d_min, d_max])
        lateral_constraints = np.array(self._lat_dis_constraints)
        d_min = np.array(
            (
                lateral_constraints[1:, 0],
                lateral_constraints[1:, 0],
                lateral_constraints[1:, 0],
            )
        ).transpose()
        d_max = np.array(
            (
                lateral_constraints[1:, 1],
                lateral_constraints[1:, 1],
                lateral_constraints[1:, 1],
            )
        ).transpose()
        self.d_min = d_min
        self.d_max = d_max
