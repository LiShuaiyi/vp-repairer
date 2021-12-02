import numpy as np

from collections import defaultdict

from cut_off.simulation import CutOffAction
from cut_off.tc import TC
from lazy_smt.abstracter import RuleAbstracter

from stl_crmonitor.crmonitor.predicates.predicate import PredInSameLane
from stl_crmonitor.crmonitor.predicates.rule import PropositionNode
from stl_crmonitor.crmonitor.common.road_network import Lane

from typing import Dict


class RuleConstraints:
    def __init__(self,
                 tc_object: TC,
                 rule_abstracter: RuleAbstracter,
                 sel_proposition: PropositionNode):
        self._tc_obj = tc_object
        self._rule_abstracter = rule_abstracter
        self._world_state = self._rule_abstracter.world_state
        self._other_id = rule_abstracter.other_veh_id  # if no target vehicle, the other_id stands for the ego
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop = sel_proposition

    def set_target_lanes(self) -> Dict[int, Lane]:
        """
        Set up target lanes for all time steps based on the compliant maneuver.
        """
        target_lanes = defaultdict(Lane)
        # todo: fix this from stl monitor
        cut_off_lane = list(self._world_state.road_network.find_lanes_by_lanelets(
            self._world_state.ego_vehicle.lanelet_assignment[self._tc_obj.tc_time_step]
        ))[0]
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

    def add(self):
        for k in range(self._tc_obj.tc, self._tc_obj.N):
            s_interval = [-np.inf, np.inf]
            d_interval = [-np.inf, np.inf]
            total_assignment = self._rule_abstracter.rule_monitor.prop_robust_all.query('time_step == @k')
            for proposition in self._rule_abstracter.propositions:
                prop_assignment = total_assignment.query('alphabet == @proposition.alphabet')
                for predicate in proposition.children:
                    if self._sel_prop.name == proposition.name:
                        pass
                    else:
                        if predicate.name == PredInSameLane.predicate_name:
                            constr = self.ConstrInSameLane(k, prop_assignment)

    def add_lat_position(self):
        pass

    def add_lon_position(self):
        pass

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        other_veh_lane_id = self._world_state.road_network.find_lane_ids_by_obstacle(self._other_id, time_step)[0]
        other_veh_lane = self._world_state.road_network.lanes[other_veh_lane_id]
        if prop_assignment>0:
            # still in the same lane
            target_lane = other_veh_lane
        elif self._compliant_maneuver == CutOffAction.LANECHANGELEFT:
            target_lane = other_veh_lane.adj_left
        elif self._compliant_maneuver == CutOffAction.LANECHANGERIGHT:
            target_lane = other_veh_lane.adj_right
        else:
            raise ValueError("<ConstrInSameLane>: the cut off action {} is wrong".format(CutOffAction))

    @staticmethod
    def lane_lateral_boundary(lane: Lane):
        pass



    @staticmethod
    def get_overlap(interval1: list, interval2: list):
        return [max(interval1[0], interval2[0]), min(interval1[0], interval2[0])]

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