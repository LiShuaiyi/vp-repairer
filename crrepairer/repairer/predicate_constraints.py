from stl_crmonitor.crmonitor.predicates.rule import PropositionNode

from cut_off.simulation import CutOffAction


def add_tv_constraints(compliant_maneuver: CutOffAction,
                       initial_assignment: list,
                       sel_proposition: PropositionNode):
    pass

def target_lane_assignment(target_predicate: BasePredicateEvaluator,
                           tstcc: int,
                           tstv: int,
                           N: int,
                           world_state: WorldState,
                           maneuver: CutOffAction) -> dict:
    """
    :param tstcc: time step to compliance
    :param tstv: time step to violation
    """
    target_lanes = defaultdict(Lane)
    ego_vehicle = world_state.ego_vehicle
    cut_off_lanelet, violation_lanelet\
        = world_state.road_network.lanelet_network.find_lanelet_by_position(
            [ego_vehicle.states_cr[tstcc].position, ego_vehicle.states_cr[tstv].position]
        )
    cut_off_lane = world_state.road_network.find_lane_by_lanelet(
        cut_off_lanelet[0]
    )
    # todo: use the center position? (major occupancy)
    if maneuver == CutOffAction.LANECHANGELEFT:
        violation_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        ).adj_left
    elif maneuver == CutOffAction.LANECHANGERIGHT:
        violation_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        ).adj_right
    elif maneuver == CutOffAction.BRAKE or maneuver == CutOffAction.CONSTANT or maneuver == CutOffAction.KICKDOWN:
        violation_target_lane = world_state.road_network.find_lane_by_lanelet(
            violation_lanelet[0]
        )
    else:
        raise ValueError('<SAT_INTERFACE>: provided actions {} are not valid '
                                     'for the predicate {}!'.format(maneuver, target_predicate.predicate_name))

    if target_predicate.predicate_name == PredInSameLane.predicate_name:
        transition_lane = {cut_off_lane}
        # lane change procedure
        # if violation_target_lane.lane_id != cut_off_lane.lane_id:
        #     transition_lane.add(cut_off_lane)
        for time_step in range(tstcc, N+1):
            if time_step >= tstv:
                target_lanes[time_step] = {violation_target_lane}
            elif time_step == tstcc:
                target_lanes[time_step] = {cut_off_lane}
            else:
                # transition lanes
                target_lanes[time_step] = transition_lane
        # lanelet_assignment: where the shape is on
        # target_lanes[time_step] = list(world_state.road_network.find_lane_by_obstacle(
        #     world_state.road_network.lanelet_network.
        #         find_lanelet_by_position([ego_vehicle.states_cr[time_step].position]),
        #     list(ego_vehicle.lanelet_assignment[time_step]),
        # ))
    elif target_predicate.predicate_name == PredSafeDistPrec.predicate_name:
        for time_step in range(tstcc, N+1):
            target_lanes[time_step] = {violation_target_lane}
    return target_lanes