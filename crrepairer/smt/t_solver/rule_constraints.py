import numpy as np

from collections import defaultdict

from crrepairer.cut_off.simulation import CutOffAction
from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.position import (PredSafeDistPrec, PredInSameLane, 
                                           PredInFrontOf, PredPreceding)

from crmonitor.predicates.velocity import (PredLaneSpeedLimit, PredFovSpeedLimit,
                                           PredBrSpeedLimit, PredTypeSpeedLimit)
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import (PredAbruptBreaking, PredRelAbruptBreaking)

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle

from typing import List

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints
from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory

from commonroad.scenario.trajectory import Trajectory


class RuleConstraints:
    """
    Class for traffic rule constraints
    """

    def __init__(self,
                 tc_object: TC,
                 rule_monitor: STLRuleMonitor,
                 sel_proposition_full: List[PropositionNode],
                 veh_config: PlanningConfigurationVehicle,
                 initial_trajectory: Trajectory):
        # initialize the needed components
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world
        self._other_id = self._rule_monitor.other_id
        self._ego_id = self._rule_monitor.vehicle_id  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._veh_config = veh_config
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full

        # initialize the elements for rule constraints
        self._target_lanes = defaultdict(List[Lane])
        self._lon_dis_constraints = list()
        self._lon_vel_constraints = list()
        self._lon_acc_constraint = []
        self._lat_dis_constraints = list()

        self._prec_veh = None
        self._foll_veh = None
        # whether safe distance needs to be obeyed
        self._safe_dis_mode = [False for _ in range(self._tc_obj.N - self._tc_obj.tc_time_step + 1)]
        if self._compliant_maneuver in [CutOffAction.LANECHANGELEFT,
                                        CutOffAction.LANECHANGERIGHT]:
            # time for leaving the current lane
            self._tc_obj.simulation_lateral.set_inputs(
                self._ego_vehicle.get_lon_state(self._tc_obj.tc_time_step).v)
            lane_dist = self._ego_vehicle.get_lane(tc_object.tc_time_step).width(
                self._ego_vehicle.get_lon_state(self._tc_obj.tc_time_step).s) / 2 - \
                        abs(self._ego_vehicle.get_lat_state(0).d) - self._veh_config.width / 2
            self._time_leave_lane = int(
                self._tc_obj.simulation_lateral.calc_leave_time(lane_dist) / self._world_state.dt)

    @property
    def safe_distance_modes(self):
        return self._safe_dis_mode

    @property
    def target_lanes(self) -> dict:
        return self._target_lanes

    @property
    def time_leave_lane(self):
        return self._time_leave_lane

    def add(self):
        """
        add rule constraints. Since QP planner is used for longitudinal and lateral motions separately,
        we can only first obtain the numerical values for then longitudinal motion and the lane constraints
        for the lateral motion.
            longitudinal motion: s, v, a
            lateral motion: lane
        """
        # acceleration limit (only one value for all time steps)
        a_limit = [-np.inf, np.inf]
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            total_assignment = self._rule_monitor.prop_robust_all[:, k]
            # longitudinal position and velocity limit
            s_limit = [-np.inf, np.inf]
            v_limit = [0, np.inf]
            for idx, proposition in enumerate(self._rule_monitor.proposition_nodes):
                try:
                    prop_assignment = total_assignment.flatten()[idx]
                except:
                    # no assignment can be found
                    continue
                for predicate in proposition.children:
                    if proposition in self._sel_prop_full and k >= self._tc_obj.tv_time_step:
                        # proposition to be repaired (greater than the time-to-violation)
                        prop_assignment = -prop_assignment
                    if k < self._tc_obj.tv_time_step or proposition in self._sel_prop_full:
                        if not hasattr(predicate, 'base_name'):
                            continue
                        if predicate.base_name == PredInSameLane.predicate_name:
                            self.ConstrInSameLane(k, prop_assignment)
                        elif predicate.base_name == PredInFrontOf.predicate_name:
                            s_constr = self.ConstrInFrontOf(k, prop_assignment)
                            s_limit = self._get_overlap(s_limit, s_constr)
                        elif predicate.base_name == PredPreceding.predicate_name:
                            # precedes = in_front_of  and in_same_lane
                            self.ConstrInSameLane(k, prop_assignment)
                            s_constr = self.ConstrInFrontOf(k, prop_assignment)
                            s_limit = self._get_overlap(s_limit, s_constr)
                        elif predicate.base_name == PredSafeDistPrec.predicate_name:
                            self.ConstrSafeDist(k, prop_assignment)
                        elif predicate.base_name == PredCutIn.predicate_name:
                            self.ConstrCutIn(k, prop_assignment)
                        elif predicate.base_name in (PredFovSpeedLimit.predicate_name,
                                                     PredBrSpeedLimit.predicate_name,
                                                     PredTypeSpeedLimit.predicate_name,
                                                     PredLaneSpeedLimit.predicate_name):
                            speed_limit = predicate.evaluator.get_speed_limit(self._world_state,
                                                                              k, 
                                                                              [self._ego_id])
                            if speed_limit is None:
                                speed_limit = np.inf
                            v_constr = self.ConstrSpeedLimit(speed_limit)
                            v_limit = self._get_overlap(v_limit, v_constr)
                        elif predicate.base_name in (PredAbruptBreaking.predicate_name,
                                                     PredRelAbruptBreaking.predicate_name):
                            a_abruptly = predicate.evaluator.config["a_abrupt"]
                            a_constr = self.ConstrAccNotAbruptly(a_abruptly)
                            a_limit = self._get_overlap(a_constr, a_limit)
                        else:
                            print("<QPRepairer/_rule_constraints>: the provided predicate {} "
                                  "is not supported".format(predicate.name))
            self._lon_dis_constraints.append(s_limit)
            self._lon_vel_constraints.append(v_limit)
        self._lon_acc_constraint = a_limit

    def longitudinal_constraints(self):
        """
        Set the longitudinal constraints
        """
        # add general constraints
        self.add()
        # add collision constraints (implicitly)
        self.ConstrCollisionFree()
        longitudinal_distance_constraints = np.array(self._lon_dis_constraints)
        longitudinal_velocity_constraints = np.array(self._lon_vel_constraints)
        return LonConstraints.construct_constraints(longitudinal_distance_constraints[1:, 0],
                                                    longitudinal_distance_constraints[1:, 1],
                                                    longitudinal_distance_constraints[1:, 0],
                                                    longitudinal_distance_constraints[1:, 1],
                                                    v_min=longitudinal_velocity_constraints[1:, 0],
                                                    v_max=longitudinal_velocity_constraints[1:, 1],
                                                    a_min=self._lon_acc_constraint[0],
                                                    a_max=self._lon_acc_constraint[1],
                                                    prec_veh=self._target_vehicle,
                                                    tc_time_step=self._tc_obj.tc_time_step,
                                                    select_proposition=self._sel_prop_full)

    def lateral_constraints(self, long_traj: QPTrajectory, ):
        """
        Set the lateral constraints (based on the planned longitudinal trajectory and the previously
        added lane constraints - target lanes)
        """
        self._lat_dis_constraints = []
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            d_min = -np.inf
            d_max = np.inf
            if k in self._target_lanes:
                target_lanes = self._target_lanes[k]
                index = k - self._tc_obj.tc_time_step
                lane_boundary_left = target_lanes[-1].clcs_left. \
                    convert_to_cartesian_coords(long_traj.states[index].position[0], 0.)
                lane_boundary_right = target_lanes[0].clcs_right. \
                    convert_to_cartesian_coords(long_traj.states[index].position[0], 0.)
                d_max = min(self._veh_config.curvilinear_coordinate_system.
                            convert_to_curvilinear_coords(lane_boundary_left[0],
                                                          lane_boundary_left[1])[1], d_max)
                d_min = max(self._veh_config.curvilinear_coordinate_system.
                            convert_to_curvilinear_coords(lane_boundary_right[0],
                                                          lane_boundary_right[1])[1], d_min)
            self._lat_dis_constraints.append([d_min,
                                              d_max])
        lateral_constraints = np.array(self._lat_dis_constraints)
        d_min = np.array((lateral_constraints[1:, 0], lateral_constraints[1:, 0],
                          lateral_constraints[1:, 0])).transpose()
        d_max = np.array((lateral_constraints[1:, 1], lateral_constraints[1:, 1],
                          lateral_constraints[1:, 1])).transpose()
        return LatConstraints.construct_constraints(d_min, d_max,
                                                    d_min, d_max)

    def _determine_related_veh(self, time_step: int, lanes: List[Lane]):
        """
        Determines the related vehicles for collision-free constraints
        """
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        if not list(lanes)[0]:
            return None, None
        # find all the vehicles in the target lanes (then remove the ego)
        for lane in lanes:
            vehicle_ids.update(lane.lanelet.dynamic_obstacle_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        for id in vehicle_ids:
            other_vehicle = self._world_state.vehicle_by_id(id)
            if time_step == 0:
                ego_state = self._ego_vehicle.states_cr[0]
            else:
                ego_state = self._ini_traj.state_at_time_step(time_step)
            ego_lon_s = convert_pos_curvilinear(ego_state, self._veh_config)[0]
            #if time_step in other_vehicle.states_lon:
            #    dist = other_vehicle.states_lon[time_step].s - ego_lon_s
            #else:
            #    continue
            try:
                dist = other_vehicle.get_lon_state(time_step).s - ego_lon_s
            except:
                continue
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def ConstrSpeedLimit(self, speed_limit):
        return [0, speed_limit]

    def ConstrAccNotAbruptly(self, a_abrupt):
        return [a_abrupt, np.inf]

    def ConstrCollisionFree(self):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N):
            if k in self._target_lanes:
                self._prec_veh, self._foll_veh = self._determine_related_veh(k, self._target_lanes[k])
            else:
                lanelet = self._world_state.scenario.lanelet_network.find_lanelet_by_position(
                    [self._ego_vehicle.states_cr[k].position])[0]
                lanes = self._world_state.road_network.find_lanes_by_lanelets(set(lanelet))
                if lanes:
                    self._prec_veh, self._foll_veh = self._determine_related_veh(k, list(lanes))
            index = k - self._tc_obj.tc_time_step
            if self._prec_veh is not None:
                if k <= self._prec_veh.end_time:
                    self._lon_dis_constraints[index] = self._get_overlap(self._lon_dis_constraints[index],
                                                                         [-np.inf, self._prec_veh.rear_s(k)
                                                                          - self._veh_config.wheelbase / 2
                                                                          - self._veh_config.length / 2
                                                                          ])
            # discard the following vehicles since the scenario is not interactive
            # if self._foll_veh is not None:
            #     if k <= self._foll_veh.end_time:
            #         self._lon_dis_constraints[index] = self._get_overlap(self._lon_dis_constraints[index],
            #                                                              [self._foll_veh.front_s(k) +
            #                                                            self._veh_config.wheelbase/2,
            #                                                            np.inf])

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        if time_step in self._target_vehicle.lanelet_assignment.keys():
            tar_veh_lanelet = self._target_vehicle.lanelet_assignment[time_step]
            try:
                tar_veh_lane = self._world_state.road_network.find_lane_by_lanelet(list(tar_veh_lanelet)[0])
                #if prop_assignment > 0:
                #    target_lane = [tar_veh_lane]
                if self._compliant_maneuver == CutOffAction.LANECHANGELEFT:
                    target_lane = [tar_veh_lane.adj_right]
                elif self._compliant_maneuver == CutOffAction.LANECHANGERIGHT:
                    target_lane = [tar_veh_lane.adj_left]
                else:
                    target_lane = [tar_veh_lane]
                if self._compliant_maneuver in [CutOffAction.LANECHANGELEFT,
                                                CutOffAction.LANECHANGERIGHT]:
                    if time_step <= self._time_leave_lane:
                        target_lane = [tar_veh_lane]
                    elif self._time_leave_lane < time_step <= self._tc_obj.tv_time_step:
                        target_lane += [tar_veh_lane]
                    target_lane = sorted(target_lane, key=lambda lane: lane.lane_id)
            except:
                tar_veh_lane = [None]
                target_lane = [None]
            
        else:
            target_lane = [None]
        self._target_lanes[time_step] = list(set(target_lane))

    def ConstrInFrontOf(self, time_step: int, prop_assignment: float):
        # preventing KeyError
        if time_step > self._target_vehicle.end_time:
            return [-np.inf, np.inf]
        if prop_assignment > 0:
            rear_s = self._target_vehicle.rear_s(time_step)
            return [-np.inf, rear_s]
        else:
            front_s = self._target_vehicle.front_s(time_step)
            return [front_s, np.inf]

    def ConstrSafeDist(self, time_step: int, prop_assignment: float):
        if prop_assignment > 0:
            self._safe_dis_mode[time_step - self._tc_obj.tc_time_step] = True
        else:
            pass

    def ConstrCutIn(self, time_step: int, prop_assignment: float, ):
        # print("<QPRepairer/_rule_constraints>: we cannot add constraints for cut in")
        return None

    @staticmethod
    def lane_lateral_boundary(lane: Lane):
        pass

    @staticmethod
    def _get_overlap(interval1: list, interval2: list):
        return [max(interval1[0], interval2[0]), min(interval1[1], interval2[1])]
