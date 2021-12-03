import math

import numpy as np

from collections import defaultdict

from cut_off.simulation import CutOffAction
from cut_off.tc import TC
from lazy_smt.abstracter import RuleAbstracter

from stl_crmonitor.crmonitor.predicates.predicate import (PredInSameLane, PredInFrontOf,
                                                          PredCutIn, PredSafeDistPrec)
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode
from stl_crmonitor.crmonitor.common.road_network import Lane

from typing import Dict

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints


class RuleConstraints:
    def __init__(self,
                 tc_object: TC,
                 rule_abstracter: RuleAbstracter,
                 sel_proposition: PropositionNode):
        self._tc_obj = tc_object
        self._rule_abstracter = rule_abstracter
        self._world_state = self._rule_abstracter.world_state
        self._other_id = rule_abstracter.other_veh_id  # if no target vehicle, the other_id stands for the ego
        self._target_vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop = sel_proposition
        self._target_lanes = defaultdict(Lane)
        self._long_constraints = list()

    def set_target_lanes(self) -> Dict[int, Lane]:
        """
        Set up target lanes for all time steps based on the compliant maneuver.
        """
        target_lanes = defaultdict(Lane)
        # todo: fix this from stl monitor
        cut_off_lane = list(self._world_state.road_network.find_lanes_by_lanelets(
            self._world_state.ego_vehicle.lanelet_assignment[self._tc_obj.tc_time_step]))[0]
        if self._compliant_maneuver == CutOffAction.LANECHANGELEFT:
            violation_target_lane = cut_off_lane.adj_left
        elif self._compliant_maneuver == CutOffAction.LANECHANGERIGHT:
            violation_target_lane = cut_off_lane.adj_right
        elif self._compliant_maneuver in (CutOffAction.BRAKE, CutOffAction.CONSTANT, CutOffAction.KICKDOWN):
            violation_target_lane = cut_off_lane
        else:
            raise ValueError('<RuleConstraints>: provided action {} is not valid'.format(self._compliant_maneuver))
        for time_step in range(self._tc_obj.tc_time_step, self._tc_obj.N):
            if time_step >= self._tc_obj.tv_time_step:
                target_lanes[time_step] = violation_target_lane
            else:
                target_lanes[time_step] = cut_off_lane
        return target_lanes

    def _add(self, veh_config: PlanningConfigurationVehicle):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            total_assignment = self._rule_abstracter.rule_monitor.prop_robust_all.query('time_step == @k')
            s_limit = [-math.inf, math.inf]
            for proposition in self._rule_abstracter.rule_monitor.proposition_nodes:
                prop_assignment = total_assignment.query('alphabet == @proposition.alphabet')["robustness"].values[0]
                for predicate in proposition.children:
                    if proposition in self._sel_prop and k >= self._tc_obj.tv_time_step:
                        prop_assignment = -prop_assignment
                    if predicate.base_name == PredInSameLane.predicate_name:
                        self.ConstrInSameLane(k, prop_assignment)
                    elif predicate.base_name == PredInFrontOf.predicate_name:
                        s_constr = self.ConstrInFrontOf(k, prop_assignment)
                        s_limit = self._get_overlap(s_limit, s_constr)
                    elif predicate.base_name == PredSafeDistPrec.predicate_name:
                        s_constr = self.ConstrSafeDist(k, prop_assignment, veh_config)
                        s_limit = self._get_overlap(s_limit, s_constr)
                    elif predicate.base_name == PredCutIn.predicate_name:
                        self.ConstrCutIn(k, prop_assignment)
                    else:
                        print("<QPRepairer/_rule_constraints>: the provided predicate {} is not supported".
                              format(predicate.name))
            self._long_constraints.append(s_limit)
        pass

    def longitudinal_constraints(self, veh_config: PlanningConfigurationVehicle):
        self._add(veh_config)
        self.ConstrCollisionFree()
        longitudinal_constraints = np.array(self._long_constraints)
        return LonConstraints.construct_constraints(longitudinal_constraints[1:, 0], longitudinal_constraints[1:, 1],
                                                    longitudinal_constraints[1:, 0], longitudinal_constraints[1:, 1])

    def _determine_related_veh(self, time_step: int, lane: Lane):
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        ego_vehicle = self._world_state.ego_vehicle
        vehicle_ids = lane.dynamic_obstacles_by_time_step(time_step)
        vehicle_ids.discard(ego_vehicle.id)
        for id in vehicle_ids:
            other_vehicle = self._world_state.vehicle_by_id(id)
            dist = other_vehicle.states_lon[time_step].s - ego_vehicle.states_lon[time_step].s
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def ConstrCollisionFree(self):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N):
            prec_veh, foll_veh = self._determine_related_veh(k, self._target_lanes[k])
            index = k - self._tc_obj.tc_time_step
            if prec_veh is not None:
                self._long_constraints[index] = self._get_overlap(self._long_constraints[index],
                                                                  [-math.inf, prec_veh.rear_s(k)])
            if foll_veh is not None:
                self._long_constraints[index] = self._get_overlap(self._long_constraints[index],
                                                                  [foll_veh.front_s(k), math.inf])

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        # todo: fix in stl monitor
        other_veh_lane = list(self._world_state.road_network.find_lanes_by_lanelets(
            self._target_vehicle.lanelet_assignment[time_step]
        ))[0]
        if prop_assignment > 0:
            # still in the same lane
            target_lane = other_veh_lane
        elif self._compliant_maneuver == CutOffAction.LANECHANGELEFT:
            target_lane = other_veh_lane.adj_left
        elif self._compliant_maneuver == CutOffAction.LANECHANGERIGHT:
            target_lane = other_veh_lane.adj_right
        else:
            raise ValueError("<QPRepairer/ConstrInSameLane>: the cut off action {} is wrong".format(CutOffAction))
        self._target_lanes[time_step] = target_lane

    def ConstrInFrontOf(self, time_step: int, prop_assignment: float):
        if prop_assignment > 0:
            rear_s = self._target_vehicle.rear_s(time_step)
            return [-math.inf, rear_s]
        else:
            front_s = self._target_vehicle.front_s(time_step)
            return [front_s, math.inf]

    def ConstrSafeDist(self, time_step: int, prop_assignment: float, veh_config: PlanningConfigurationVehicle):
        safe_dist = self.safe_distance(
            veh_config.desired_speed,
            self._target_vehicle.states_lon[time_step].v,
            veh_config.a_min_x,
            self._target_vehicle.vehicle_param.get('a_min'),
            0.0  # todo: t react
        ) - 10
        safe_dist = max(0., safe_dist)
        if prop_assignment > 0:
            return [-math.inf, self._target_vehicle.rear_s(time_step) - safe_dist]
        else:
            return [self._target_vehicle.rear_s(time_step) - safe_dist, math.inf]

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
        d_safe = \
            (v_lead ** 2) / (-2 * abs(a_min_lead)) - (v_follow ** 2) / (-2 * abs(a_min_follow)) \
            + v_follow * t_react_follow

        return d_safe

#
# def add_tv_constraints(tc: int,
#                        N: int,
#                        rule_abstracter: RuleAbstracter,
#                        compliant_maneuver: CutOffAction,
#                        sel_proposition: PropositionNode):
#     # iterate through all propositions and all time steps
#     for k in range(tc, N):
#         s_interval = [-math.inf, math.inf]
#         d_interval = [-math.inf, math.inf]
#         total_assignment = rule_abstracter.prop_robust_all.query('time_step == @k')
#         for proposition in rule_abstracter.propositions:
#             init_assignment = total_assignment.query('alphabet == @proposition.alphabet')
#             for predicate in proposition.children:
#                 if sel_proposition.name == proposition.name:
#                     pass
#                 else:
#                     if predicate.name == PredInSameLane.predicate_name:
#                         lat_constr =


#
#
# def target_lane_assignment(target_predicate: BasePredicateEvaluator,
#                            tstcc: int,
#                            tstv: int,
#                            N: int,
#                            world_state: WorldState,
#                            maneuver: CutOffAction) -> dict:
#     """
#     :param tstcc: time step to compliance
#     :param tstv: time step to violation
#     """
#     target_lanes = defaultdict(Lane)
#     ego_vehicle = world_state.ego_vehicle
#     cut_off_lanelet, violation_lanelet\
#         = world_state.road_network.lanelet_network.find_lanelet_by_position(
#             [ego_vehicle.states_cr[tstcc].position, ego_vehicle.states_cr[tstv].position]
#         )
#     cut_off_lane = world_state.road_network.find_lane_by_lanelet(
#         cut_off_lanelet[0]
#     )
#     # todo: use the center position? (major occupancy)
#     if maneuver == CutOffAction.LANECHANGELEFT:
#         violation_target_lane = world_state.road_network.find_lane_by_lanelet(
#             violation_lanelet[0]
#         ).adj_left
#     elif maneuver == CutOffAction.LANECHANGERIGHT:
#         violation_target_lane = world_state.road_network.find_lane_by_lanelet(
#             violation_lanelet[0]
#         ).adj_right
#     elif maneuver == CutOffAction.BRAKE or maneuver == CutOffAction.CONSTANT or maneuver == CutOffAction.KICKDOWN:
#         violation_target_lane = world_state.road_network.find_lane_by_lanelet(
#             violation_lanelet[0]
#         )
#     else:
#         raise ValueError('<SAT_INTERFACE>: provided actions {} are not valid '
#                                      'for the predicate {}!'.format(maneuver, target_predicate.predicate_name))
#
#     if target_predicate.predicate_name == PredInSameLane.predicate_name:
#         transition_lane = {cut_off_lane}
#         # lane change procedure
#         # if violation_target_lane.lane_id != cut_off_lane.lane_id:
#         #     transition_lane.add(cut_off_lane)
#         for time_step in range(tstcc, N+1):
#             if time_step >= tstv:
#                 target_lanes[time_step] = {violation_target_lane}
#             elif time_step == tstcc:
#                 target_lanes[time_step] = {cut_off_lane}
#             else:
#                 # transition lanes
#                 target_lanes[time_step] = transition_lane
#         # lanelet_assignment: where the shape is on
#         # target_lanes[time_step] = list(world_state.road_network.find_lane_by_obstacle(
#         #     world_state.road_network.lanelet_network.
#         #         find_lanelet_by_position([ego_vehicle.states_cr[time_step].position]),
#         #     list(ego_vehicle.lanelet_assignment[time_step]),
#         # ))
#     elif target_predicate.predicate_name == PredSafeDistPrec.predicate_name:
#         for time_step in range(tstcc, N+1):
#             target_lanes[time_step] = {violation_target_lane}
#     return target_lanes
