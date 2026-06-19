import numpy as np

from crmonitor.common.vehicle import Vehicle, Lane
from crmonitor.common.world import World

from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=None)
def compute_lane_bounds(lane: Lane, ref_lane: Lane):
    top_left_s, top_left_d = ref_lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.left_vertices[0]
    )
    top_right_s, top_right_d = ref_lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.left_vertices[-1]
    )
    bottom_left_s, bottom_left_d = ref_lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.right_vertices[0]
    )
    bottom_right_s, bottom_right_d = ref_lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.right_vertices[-1]
    )
    s_left = max(top_left_s, bottom_left_s)
    s_right = min(top_right_s, bottom_right_s)
    d_left = min(top_left_d, top_right_d)
    d_right = max(bottom_right_d, bottom_left_d)
    return s_left, s_right, d_right, d_left


class InSameLaneConstraint:
    def __init__(self):
        self.constraint_dict = defaultdict(tuple)

    def compute(self, ego_vehicle: Vehicle, target_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            lane = target_vehicle.get_lane(initial_time_step)
            self.constraint_dict[time_step] = compute_lane_bounds(lane, ego_vehicle.get_lane(initial_time_step))
        return self.constraint_dict


class InFrontOfConstraint:
    def __init__(self):
        self.constraint_dict = defaultdict(tuple)

    def compute(self, ego_vehicle: Vehicle, target_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            if time_step > target_vehicle.end_time:
                s_max = np.inf
            else:
                s_max = target_vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step))
            self.constraint_dict[time_step] = (-np.inf, s_max)
        return self.constraint_dict


class KeepsSafeDistanceConstraint:
    def __init__(self):
        self.constraint_dict = defaultdict(tuple)

    def compute(self, ego_vehicle: Vehicle, target_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            if time_step > target_vehicle.end_time:
                real_s_l = np.inf
                velocity_l = 0
            else:
                real_s_l = target_vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step))
                velocity_l = target_vehicle.get_lon_state(time_step, ego_vehicle.get_lane(initial_time_step)).v
            self.constraint_dict[time_step] = (real_s_l, velocity_l)
        return self.constraint_dict

class CollisionFreeConstraint:
    def __init__(self):
        self.constraint_dict = defaultdict(tuple)

    def compute(self, world: World, ego_vehicle: Vehicle, target_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            s_min, s_max = -np.inf, np.inf
            for vid in world.vehicle_ids_for_time_step(time_step):
                vehicle = world.vehicle_by_id(vid)
                if ego_vehicle.lanes_at_state(time_step).intersection(vehicle.lanes_at_state(time_step)):
                    if (vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step)) >
                            ego_vehicle.front_s(time_step, ego_vehicle.get_lane(initial_time_step))):
                        s_max = min(s_max, vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step)))
                    elif (vehicle.front_s(time_step, ego_vehicle.get_lane(initial_time_step)) <
                            ego_vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step))):
                        s_min = max(s_min, vehicle.front_s(time_step, ego_vehicle.get_lane(initial_time_step)))
            self.constraint_dict[time_step] = (s_min, s_max)
        return self.constraint_dict

class CollisionFreeConstraintIntersection:
    def __init__(self):
        self.constraint_dict = defaultdict(tuple)

    def compute(self, world: World, ego_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            s_min, s_max = -np.inf, np.inf
            for vid in world.vehicle_ids_for_time_step(time_step):
                vehicle = world.vehicle_by_id(vid)
                if ego_vehicle.lanes_at_state(time_step).intersection(vehicle.lanes_at_state(time_step)):
                    if (vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step)) >
                            ego_vehicle.front_s(time_step, ego_vehicle.ref_path_lane)):
                        s_max = min(s_max, vehicle.rear_s(time_step, ego_vehicle.ref_path_lane))
                    elif (vehicle.front_s(time_step, ego_vehicle.ref_path_lane) <
                            ego_vehicle.rear_s(time_step, ego_vehicle.get_lane(initial_time_step))):
                        s_min = max(s_min, vehicle.front_s(time_step, ego_vehicle.ref_path_lane))
            self.constraint_dict[time_step] = (s_min, s_max)
        return self.constraint_dict