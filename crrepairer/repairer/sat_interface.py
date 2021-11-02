
from collections import defaultdict
from crmonitor.common.world_state import WorldState
from crmonitor.common.road_network import Lane
from crmonitor.predicates.predicate import (BasePredicateEvaluator,
                                            PredInSameLane,
                                            PredSafeDistPrec)

from cut_off.simulation import CutOffAction


def target_lane_assignment(target_predicate: BasePredicateEvaluator,
                           tstcc: int,
                           tstv: int,
                           world_state: WorldState,
                           action: CutOffAction) -> dict:
    """
    :param tstcc: time step to compliance
    :param tstv: time step to violation
    """
    target_lanes = defaultdict(Lane)
    ego_vehicle = world_state.ego_vehicle
    initial_lanelet, violation_lanelet = world_state.road_network.lanelet_network.find_lanelet_by_position(
        [ego_vehicle.states_cr[tstcc].position,
         ego_vehicle.states_cr[tstv].position])
    initial_lane = world_state.road_network.find_lane_by_lanelet(cut_off_lanelet[0])
    if action == CutOffAction.LANECHANGELEFT:
        compliance_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        ).adj_left
    elif action == CutOffAction.LANECHANGERIGHT:
        compliance_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        ).adj_right
    elif action == CutOffAction.BRAKE or action == CutOffAction.CONSTANT or action == CutOffAction.KICKDOWN:
        compliance_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        )
    else:
        raise ValueError('<SAT_INTERFACE>: provided actions {} are not valid '
                         'for the predicate {}!'.format(action, target_predicate.predicate_name))

    if target_predicate.predicate_name == PredInSameLane.predicate_name:
        for time_step in range(tstcc, world_state.num_time_steps):
            if time_step >= tstv:
                target_lanes[time_step] = {compliance_target_lane}
            else:
                target_lanes[time_step] = transition_lane
        # lanelet_assignment: where the shape is on
        # target_lanes[time_step] = list(world_state.road_network.find_lane_by_obstacle(
        #     world_state.road_network.lanelet_network.
        #         find_lanelet_by_position([ego_vehicle.states_cr[time_step].position]),
        #     list(ego_vehicle.lanelet_assignment[time_step]),
        # ))
    elif target_predicate.predicate_name == PredSafeDistPrec.predicate_name:
        for time_step in range(tstcc, world_state.num_time_steps):
            target_lanes[time_step] = {compliance_target_lane}
    return target_lanes
