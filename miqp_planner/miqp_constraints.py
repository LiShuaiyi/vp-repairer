import numpy as np
from typing import List
from collections import defaultdict

from commonroad.scenario.trajectory import Trajectory

from commonroad_crime.utility.simulation import Maneuver

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
)

from crmonitor.predicates.velocity import (
    PredLaneSpeedLimit,
    PredFovSpeedLimit,
    PredBrSpeedLimit,
    PredTypeSpeedLimit,
)
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import PredAbruptBreaking, PredRelAbruptBreaking

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.road_network import RoadNetwork
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
    ):
        super().__init__()
        self._foll_veh = None
        self._prec_veh = None
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
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
            total_assignment = self._rule_monitor.prop_robust_all[:, k]
            for idx, proposition in enumerate(self._rule_monitor.proposition_nodes):
                try:
                    prop_assignment = total_assignment[
                        total_assignment == total_assignment
                    ][idx]
                except:
                    # no assignment can be found
                    continue
                for predicate in proposition.children:
                    if (
                        proposition in self._prop_full
                        and k >= self._tc_obj.tv_time_step
                    ):
                        # proposition to be repaired (greater than the time-to-violation)
                        robs_at_tv = self._rule_monitor.prop_robust_all[
                            :, self._tc_obj.tv_time_step
                        ]
                        prop_assignment = robs_at_tv[robs_at_tv == robs_at_tv][idx]
                        if proposition in self._sel_prop_full:
                            prop_assignment = -prop_assignment
                    if k < self._tc_obj.tv_time_step or proposition in self._prop_full:
                        if not hasattr(predicate, "base_name"):
                            continue
                        if (
                            predicate.base_name
                            == PredInIntersectionConflictArea.predicate_name
                        ):
                            if predicate.base_name in self.rule_constraints.keys():
                                # avoid multiple updates in one time step for the same predicate constraints
                                if (
                                    len(
                                        self.rule_constraints[predicate.base_name][
                                            "s_limit_front"
                                        ]
                                    )
                                    == num_time_step
                                ):
                                    continue
                                (
                                    s_limit_front,
                                    s_limit_behind,
                                ) = self.ConstrInIntersectionConflictAreaEgo(
                                    time_step=k
                                )
                                self.rule_constraints[predicate.base_name][
                                    "s_limit_front"
                                ].append(s_limit_front)
                                self.rule_constraints[predicate.base_name][
                                    "s_limit_behind"
                                ].append(s_limit_behind)
                            else:
                                self.rule_constraints[predicate.base_name] = {
                                    "decision_variable": True,
                                    "num_decision_variables": 1,
                                    "constraint_name": "conflict_area",
                                    "constraint_state": [0],
                                    "s_limit_front": [],
                                    "s_limit_behind": [],
                                }
                                (
                                    s_limit_front,
                                    s_limit_behind,
                                ) = self.ConstrInIntersectionConflictAreaEgo(
                                    time_step=k
                                )
                                self.rule_constraints[predicate.base_name][
                                    "s_limit_front"
                                ].append(s_limit_front)
                                self.rule_constraints[predicate.base_name][
                                    "s_limit_behind"
                                ].append(s_limit_behind)

    def add_collision_free_constraints(self):
        self.collision_free_constraints["index_lb"] = list()
        self.collision_free_constraints["index_ub"] = list()
        self.collision_free_constraints["collision_free_ub"] = list()
        self.collision_free_constraints["collision_free_lb"] = list()
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N):
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
            other_vehicle = self._world_state.vehicle_by_id(id)
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

    def ConstrInIntersectionConflictAreaEgo(self, time_step: int):
        def compute_conflict_start_end_points(
            ego_vehicle: Vehicle,
            target_vehicle: Vehicle,
            road_network: RoadNetwork,
            time_step,
        ):
            all_conflict_points = list()
            for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                if LaneletType.INTERSECTION in lanelet.lanelet_type:
                    # find conflict points between center vertices of lanelets_dir of k-th vehicle and reference path
                    # lanelets of p-th vehicle
                    conflict_points = find_conflict_points(
                        ego_vehicle.lanelets_dir_center_vertices,
                        lanelet.polygon.shapely_object,
                    )
                    if conflict_points is not None:
                        all_conflict_points.append(conflict_points)
            if len(all_conflict_points) == 0:
                return [np.inf, -np.inf]
            start_conflict_s = (
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *all_conflict_points[0][0]
                )[0]
            )
            end_conflict_s = (
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *all_conflict_points[-1][-1]
                )[0]
            )
            return [start_conflict_s, end_conflict_s]

        def find_conflict_points(line, conflict_polygon):
            conflict_line_points = list()
            # Create curved line
            curved_line = LineString(line)
            # Get intersection of line and polygon
            intersection = curved_line.intersection(conflict_polygon)
            if intersection.geom_type == "Point":
                conflict_line_points.append(intersection)
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
            if conflict_points is None:
                return [np.inf, -np.inf]
            return conflict_points

        index_prop_conflict_area_target = 0
        rule_monitor = self._rule_monitor
        world = rule_monitor.world
        ego_vehicle = self._ego_vehicle
        target_vehicle = self._target_vehicle
        for i in range(len(rule_monitor.proposition_nodes)):
            if (
                rule_monitor.proposition_nodes[i].name
                == "(once[1,1](in_intersection_conflict_area__a1_a0))>=(0.0)"
            ):
                index_prop_conflict_area_target = i
                break
        prop_conflict_area_target = rule_monitor.prop_robust_all[
            0, time_step, index_prop_conflict_area_target
        ]
        if prop_conflict_area_target >= 0:
            conflict_points = compute_conflict_start_end_points(
                ego_vehicle, target_vehicle, world.road_network, time_step
            )
            # TODO: fix plus or minus a small number
            s_limit_front = (
                conflict_points[0]
                - self._ego_vehicle.shape.length / 2
                - self._veh_config.wheelbase / 2
                - 2
            )
            s_limit_behind = (
                conflict_points[1]
                + self._ego_vehicle.shape.length / 2
                - self._veh_config.wheelbase / 2
                + 2
            )
        else:
            s_limit_front = np.inf
            s_limit_behind = -np.inf
        return s_limit_front, s_limit_behind


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
