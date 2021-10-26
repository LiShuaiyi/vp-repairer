import numpy as np
from operator import itemgetter
from collections import defaultdict
from optimization.configuration import RepairingConfigurationVehicle
from optimization.constraints import LonConstraints, LatConstraints
from typing import Dict, Union, List
from crmonitor.common.world_state import WorldState
from crmonitor.common.road_network import RoadNetwork, Lane

# commonroad-io
from commonroad.common.util import Interval

class SafetyConstraints:
    def __init__(self,
                 world_state: WorldState,
                 N: int):
        self._world_state: WorldState = world_state
        self.time_steps = None
        self.N: int = N

    @property
    def world_state(self):
        return self._world_state

    @world_state.setter
    def world_state(self, world_state):
        self._world_state = world_state

    def lateral_position_constraints(self, tstcc: int, target_lanes: List[Lane],
                                     longitudinal_trajectory_s_coordinates: List[float],
                                     vehicle_configuration:RepairingConfigurationVehicle) -> LatConstraints:
        def convert(d_constraints: Dict[int, Interval]):
            d_min = list()
            d_max = list()
            d_constraints = dict(sorted(d_constraints.items()))
            for i in d_constraints.values():
                d_min.append(i.start)
                d_max.append(i.end)
            return d_min, d_max
        s_coordinates_dict = dict(zip(self.time_steps, longitudinal_trajectory_s_coordinates))
        lateral_constraints = list()
        d_constraints_reference_point = self._lateral_position_constraints_reference_point(
            tstcc, target_lanes, longitudinal_trajectory_s_coordinates, vehicle_configuration
        )
        d_constraints_rear_or_center, d_constraints_front = self._lateral_position_constraints_other_vehicles(
            longitudinal_trajectory_s_coordinates, vehicle_configuration, d_constraints_reference_point
        )
        d_min_reference, d_max_reference = convert(d_constraints_reference_point)
        d_min_front, d_max_front = convert(d_constraints_front)
        d_min_rear_or_center, d_max_rear_or_center = convert(d_constraints_rear_or_center)

        d_min = np.array((d_min_reference[1:], d_min_rear_or_center[1:], d_min_front[1:])).transpose()
        d_max = np.array((d_max_reference[1:], d_max_rear_or_center[1:], d_max_front[1:])).transpose()

        return LatConstraints.construct_constraints(d_min, d_max,
                                                    d_min, d_max)

    def _lateral_position_constraints_reference_point(self,
                                                      tstcc: int,
                                                      target_lanes: Union[Dict, None],
                                                      longitudinal_trajectory_s_coordinates: Dict[int, float],
                                                      vehicle_configuration: RepairingConfigurationVehicle):
        d_constraints_reference_point = defaultdict()
        for time_step, lanes in target_lanes.items():
            time_step -= tstcc
            s_current = longitudinal_trajectory_s_coordinates[time_step]
            x_current, y_current = self.world_state.ego_vehicle.lane.clcs.convert_to_cartesian_coords(s_current, 0.0)
            lanes = list(lanes)
            if len(lanes) == 1:
                # single lane constraint
                # todo: left or right
                coord_lane_boundary_left = lanes[0].clcs_left.convert_to_curvilinear_coords(x_current, y_current)
                coord_lane_boundary_right = lanes[0].clcs_right.convert_to_curvilinear_coords(x_current, y_current)
            else:
                # sort the lanes according to their ids (ascending order)
                lanes = sorted(lanes, key=lambda lane: lane.lane_id)
                coord_lane_boundary_left = lanes[-1].clcs_left.convert_to_curvilinear_coords(x_current, y_current)
                coord_lane_boundary_right = lanes[0].clcs_right.convert_to_curvilinear_coords(x_current, y_current)
            d_lane_boundary_left = -coord_lane_boundary_left[-1] # coord_lane_boundary_left[-1] - d_current
            d_lane_boundary_right = -coord_lane_boundary_right[-1] # coord_lane_boundary_right[-1] - d_current
            d_constraints_reference_point[time_step] = \
                Interval(d_lane_boundary_right + vehicle_configuration.wheelbase / 2.0,
                         d_lane_boundary_left - vehicle_configuration.wheelbase / 2.0) # o
        return d_constraints_reference_point

    def _lateral_position_constraints_other_vehicles(self,
                                                     longitudinal_trajectory_s_coordinates: Dict[int, float],
                                                     vehicle_configuration: RepairingConfigurationVehicle,
                                                     d_constraints_reference_point: Dict[int, Interval]):
        d_constraints_rear_or_center = defaultdict()
        d_constraints_front = defaultdict()
        for time_step in d_constraints_reference_point.keys():
            # get current longitudinal position of reference point
            s = longitudinal_trajectory_s_coordinates[time_step]

            # get normal and tangent vectors at s
            # normal = vehicle_configuration.curvilinear_coordinate_system.normal(s)
            # tangent = vehicle_configuration.curvilinear_coordinate_system.tangent(s)
            #
            # # get Cartesian position coordinates of reference point
            # cartesian_position = vehicle_configuration.curvilinear_coordinate_system.\
            #     convert_to_cartesian_coords(s, 0.0)

            # reference point: rear
            # cartesian_position_rear_or_center = cartesian_position + (
            #             vehicle_configuration.wheelbase / 2.0) * tangent
            # cartesian_position_front = cartesian_position + vehicle_configuration.wheelbase * tangent
            d_constraints_rear_or_center[time_step] = d_constraints_reference_point[time_step]
            d_constraints_front[time_step] = d_constraints_reference_point[time_step]
            # d_constraints_rear_or_center[time_step] = self.\
            #     find_other_circle_centers(cartesian_position_rear_or_center,
            #                               normal,
            #                               d_constraints_reference_point[time_step]
            #                               )
            # # todo: add other vehicles
        return d_constraints_rear_or_center, d_constraints_front


    @staticmethod
    def find_other_circle_centers(position: np.ndarray,
                                  normal_vector: np.ndarray,
                                  d_reference: Interval):
        pass


    def longitudinal_position_constraints(self, veh_config: RepairingConfigurationVehicle,
                                          target_lanes: Union[Dict, None], tstcc: int, tstv: int) -> LonConstraints:
        # set time steps
        self.time_steps = sorted(list(target_lanes.keys()))
        # surrounding vehicles
        longitudinal_constraints = list()
        # iterate through the time horizon
        current_lanes = target_lanes[tstcc]
        FLAG_LANEFIXED = False
        prec_veh, foll_veh = self._determine_related_vehicles(0, lane=list(current_lanes)[0])
        FLAG_LANEFIXED = all(len(lanes) == 1 for lanes in target_lanes.values())
        for time_step, lanes in target_lanes.items():
            s_min_current = -np.inf #self.world_state.ego_vehicle.states_lon[0].s
            s_max_current = np.inf

            if len(current_lanes) == 2 and len(lanes) == 1:
                FLAG_LANEFIXED = True
                prec_veh, foll_veh = self._determine_related_vehicles(time_step, lane=list(lanes)[0])

            for lane in lanes:
                if not FLAG_LANEFIXED:
                    prec_veh, foll_veh = self._determine_related_vehicles(time_step, lane=lane)
                if prec_veh is not None:
                    safe_dist = safe_distance(
                        veh_config.desired_speed,
                        prec_veh.states_lon[time_step].v,
                        veh_config.a_min_x,
                        prec_veh.vehicle_param.get('a_min'),
                        0.0 # todo: t react
                    )
                    # after lane change - keep safe distance
                    if FLAG_LANEFIXED or time_step >= tstv:
                        safe_dist = max(0, safe_dist)
                        s_max_current = prec_veh.rear_s(time_step) - safe_dist
                    else:
                        s_max_current = prec_veh.rear_s(time_step) # - safe_dist todo: safe distance included
                if foll_veh is not None:
                    # todo: safe distance
                    s_min_current = foll_veh.front_s(time_step)
            current_lanes = lanes
            # print(time_step, s_min_current, s_max_current, lane)
            longitudinal_constraints.append([s_min_current, s_max_current])
        longitudinal_constraints = np.array(longitudinal_constraints)
        # start from the time step right after the cut-off state
        return LonConstraints.construct_constraints(longitudinal_constraints[1:, 0], longitudinal_constraints[1:, 1],
                                                    longitudinal_constraints[1:, 0], longitudinal_constraints[1:, 1])

    def _determine_related_vehicles(self, time_step:int, lane: Lane):
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        ego_vehicle = self.world_state.ego_vehicle
        vehicle_ids = lane.dynamic_obstacles_by_time_step(time_step)
        vehicle_ids.discard(ego_vehicle.id)
        for id in vehicle_ids:
            other_vehicle = self.world_state.vehicle_by_id(id)
            # todo: not the states_long of the ego
            dist = other_vehicle.states_lon[time_step].s - ego_vehicle.states_lon[time_step].s
            if dist > 0 and dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif dist < 0 and dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle


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
