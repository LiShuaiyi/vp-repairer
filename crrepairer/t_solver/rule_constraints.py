import math

import numpy as np

from collections import defaultdict

from commonroad_repair.crrepairer.cut_off.simulation import CutOffAction
from commonroad_repair.crrepairer.cut_off.tc import TC
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter

from stl_crmonitor.crmonitor.predicates.predicate import (PredInSameLane, PredInFrontOf,
                                                          PredCutIn, PredSafeDistPrec,
                                                          PredLaneSpeedLimit, PredFovSpeedLimit,
                                                          PredBrSpeedLimit, PredTypeSpeedLimit,
                                                          PredAbruptBreaking)
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode
from stl_crmonitor.crmonitor.common.road_network import Lane
from stl_crmonitor.crmonitor.common.vehicle import Vehicle

from typing import Dict, List

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints
from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory

from commonroad.scenario.trajectory import Trajectory


class RuleConstraints:
    def __init__(self,
                 tc_object: TC,
                 rule_abstracter: RuleAbstracter,
                 sel_proposition: List[PropositionNode],
                 veh_config: PlanningConfigurationVehicle,
                 initial_trajectory: Trajectory):
        self._tc_obj = tc_object
        self._rule_abstracter = rule_abstracter
        self._world_state = self._rule_abstracter.world_state
        self._other_id = rule_abstracter.other_veh_id
        self._ego_id = rule_abstracter.vehicle_id  # if no target vehicle, the other_id stands for the ego
        self._ini_traj = initial_trajectory
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._veh_config = veh_config
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop = sel_proposition
        self._target_lanes = defaultdict(List[Lane])
        self._lon_dis_constraints = list()
        self._lon_vel_constraints = list()
        self._lon_acc_constraint = []
        self._lat_dis_constraints = list()
        self._prec_veh = None
        self._foll_veh = None
        self._safe_dis_list = [False for _ in range(self._tc_obj.N + 1 - self._tc_obj.tc_time_step)]
        if self._compliant_maneuver in [CutOffAction.LANECHANGELEFT,
                                        CutOffAction.LANECHANGERIGHT]:
            self._tc_obj.simulation_lateral.set_inputs(self._world_state.ego_vehicle.states_lon[self._tc_obj.tc_time_step].v)
            # self._t_min_change_lane = int(self._tc_obj.simulation_lateral.calc_total_time(
            #     self._world_state.ego_vehicle.lane.width(
            #     self._world_state.ego_vehicle.states_lon[self._tc_obj.tc_time_step].s)) / (2 * self._world_state.dt)) \
            #         + self._tc_obj.tc_time_step

            lane_dist = self._world_state.ego_vehicle.lane.width(
                self._world_state.ego_vehicle.states_lon[self._tc_obj.tc_time_step].s)/2 - \
                abs(self._world_state.ego_vehicle.states_lat[0].d) - self._veh_config.width/2
            self._t_min_change_lane = int(self._tc_obj.simulation_lateral.calc_leave_time(lane_dist)/self._world_state.dt)

    @property
    def safe_distance_modes(self):
        return self._safe_dis_list

    def _add(self):
        a_limit = [-np.inf, np.inf]
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            total_assignment = self._rule_abstracter.rule_monitor.prop_robust_all.query('time_step == @k')
            s_limit = [-np.inf, np.inf]
            v_limit = [0, np.inf]
            for proposition in self._rule_abstracter.rule_monitor.proposition_nodes:
                try:
                    prop_assignment = total_assignment.query('alphabet == @proposition.alphabet')["robustness"].values[0]
                except:
                    continue
                for predicate in proposition.children:
                    if proposition in self._sel_prop and k >= self._tc_obj.tv_time_step:
                        prop_assignment = -prop_assignment

                    if k < self._tc_obj.tv_time_step or proposition in self._sel_prop:
                        if not hasattr(predicate, 'base_name'):
                            continue
                        if predicate.base_name == PredInSameLane.predicate_name:
                            self.ConstrInSameLane(k, prop_assignment)
                        elif predicate.base_name == PredInFrontOf.predicate_name:
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
                            speed_limit = predicate.evaluator.speed_limit
                            print(predicate.name, speed_limit)
                            v_constr = self.ConstrSpeedLimit(speed_limit)
                            v_limit = self._get_overlap(v_limit, v_constr)
                        elif predicate.base_name == PredAbruptBreaking.predicate_name:
                            a_abruptly = predicate.evaluator.a_abrupt
                            a_constr = self.ConstrAccNotAbruptly(a_abruptly)
                            a_limit = self._get_overlap(a_constr, a_limit)
                        else:
                            print("<QPRepairer/_rule_constraints>: the provided predicate {} is not supported".
                                  format(predicate.name))
            self._lon_dis_constraints.append(s_limit)
            self._lon_vel_constraints.append(v_limit)
        self._lon_acc_constraint = a_limit

    def longitudinal_constraints(self):
        self._add()
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
                                                    tc_time_step=self._tc_obj.tc_time_step)

    def lateral_constraints(self, long_traj: QPTrajectory, ):
        ego_lane = self._world_state.ego_vehicle.lane  # todo, fix
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N+1):
            if k in self._target_lanes:
                target_lanes = self._target_lanes[k]
                index = k - self._tc_obj.tc_time_step
                x_curr, y_curr = ego_lane.clcs.convert_to_cartesian_coords(long_traj.states[index].position[0], 0.)
                lane_boundary_left = -target_lanes[-1].clcs_left.convert_to_curvilinear_coords(x_curr, y_curr)
                lane_boundary_right = -target_lanes[0].clcs_right.convert_to_curvilinear_coords(x_curr, y_curr)
                self._lat_dis_constraints.append([lane_boundary_right[1] + self._veh_config.wheelbase / 2.,
                                                  lane_boundary_left[1] - self._veh_config.wheelbase / 2.])
            else:
                self._lat_dis_constraints.append([-np.inf,
                                                  np.inf])
        lateral_constraints = np.array(self._lat_dis_constraints)
        d_min = np.array((lateral_constraints[1:, 0], lateral_constraints[1:, 0], lateral_constraints[1:, 0])).transpose()
        d_max = np.array((lateral_constraints[1:, 1], lateral_constraints[1:, 1], lateral_constraints[1:, 1])).transpose()
        return LatConstraints.construct_constraints(d_min, d_max,
                                                    d_min, d_max)

    def _determine_related_veh(self, time_step: int, lanes: List[Lane]):
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        for lane in lanes:
            vehicle_ids.update(lane.dynamic_obstacles_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        for id in vehicle_ids:
            other_vehicle = self._world_state.vehicle_by_id(id)
            if time_step == 0:
                ego_state = self._world_state.ego_vehicle.states_cr[0]
            else:
                ego_state = self._ini_traj.state_at_time_step(time_step)
            ego_lon_s = convert_pos_curvilinear(ego_state, self._veh_config)[0]
            dist = other_vehicle.states_lon[time_step].s - ego_lon_s
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
        # prec_veh, foll_veh = self._determine_related_veh(self._tc_obj.tc_time_step,
        #                                                  self._target_lanes[self._tc_obj.tc_time_step])
        # num_target_lanes = len(self._target_lanes[self._tc_obj.tc_time_step])
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N+1):
            # if len(self._target_lanes[k]) < num_target_lanes:
            if k in self._target_lanes:
                self._prec_veh, self._foll_veh = self._determine_related_veh(k, self._target_lanes[k])
            # num_target_lanes = len(self._target_lanes[k])
            index = k - self._tc_obj.tc_time_step
            if self._prec_veh is not None:  # todo fix the length
                if k <= self._prec_veh.end_time:
                    self._lon_dis_constraints[index] = self._get_overlap(self._lon_dis_constraints[index],
                                                                         [-np.inf, self._prec_veh.rear_s(k)
                                                                       - self._veh_config.wheelbase/2
                                                                       - self._veh_config.length/2
                                                                       ])
            if self._foll_veh is not None:
                if k <= self._foll_veh.end_time:
                    self._lon_dis_constraints[index] = self._get_overlap(self._lon_dis_constraints[index],
                                                                         [self._foll_veh.front_s(k) +
                                                                       self._veh_config.wheelbase/2,
                                                                       np.inf])

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        # todo: fix in stl monitor
        ego_lane = self._world_state.ego_vehicle.lane
        if self._compliant_maneuver == CutOffAction.LANECHANGELEFT:
            target_lane = [ego_lane.adj_left]
        elif self._compliant_maneuver == CutOffAction.LANECHANGERIGHT:
            target_lane = [ego_lane.adj_right]
        else:
            target_lane = [ego_lane]
        if self._compliant_maneuver in [CutOffAction.LANECHANGELEFT,
                                        CutOffAction.LANECHANGERIGHT]:

            # todo: consider ego-to-distance boundary
            if time_step <= self._t_min_change_lane:
                target_lane = [ego_lane]
            elif self._t_min_change_lane < time_step <= self._tc_obj.tv_time_step:
                target_lane += [ego_lane]
        if None not in target_lane:
            target_lane = sorted(target_lane, key=lambda lane: lane.lane_id)
        self._target_lanes[time_step] = target_lane

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
                self._safe_dis_list[time_step - self._tc_obj.tc_time_step] = True
            # return [-np.inf, self._target_vehicle.rear_s(time_step)]
        else:
            pass
            # if there is no safe distance requirement,
            # return [-np.inf, np.inf]

    def ConstrCutIn(self, time_step: int, prop_assignment: float, ):
        print("<QPRepairer/_rule_constraints>: we cannot add constraints for cut in")
        return None

    @staticmethod
    def lane_lateral_boundary(lane: Lane):
        pass

    @staticmethod
    def _get_overlap(interval1: list, interval2: list):
        return [max(interval1[0], interval2[0]), min(interval1[1], interval2[1])]

    @staticmethod
    def safe_distance(v_follow: float, v_lead: float,
                      a_min_follow: float, a_min_lead: float,
                      t_react_follow: float) -> float:
        """
        Calculates safe distance analytically

        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param t_react_follow: reaction time of following vehicle
        :returns boolean indicating satisfaction
        """

        assert a_min_follow and 0 > a_min_lead, \
            '<BrakingPredicateCollection/safe_distance>: acceleration is not valid'
        d_safe = (
            (v_lead ** 2) / (-2. * np.abs(a_min_lead))
            - (v_follow ** 2) / (-2. * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )
        return d_safe
