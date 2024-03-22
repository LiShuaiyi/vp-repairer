
from crmonitor.common.vehicle import Vehicle, Lane

from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=None)
def compute_lane_bounds(lane: Lane):
    top_left_s, top_left_d = lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.left_vertices[0]
    )
    top_right_s, top_right_d = lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.left_vertices[-1]
    )
    bottom_left_s, bottom_left_d = lane.clcs.convert_to_curvilinear_coords(
        *lane.lanelet.right_vertices[0]
    )
    bottom_right_s, bottom_right_d = lane.clcs.convert_to_curvilinear_coords(
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

    def compute(self, target_vehicle: Vehicle, initial_time_step: int, final_time_step: int):
        for time_step in range(initial_time_step, final_time_step + 1):
            lane = target_vehicle.get_lane(time_step)
            self.constraint_dict[time_step] = compute_lane_bounds(lane)
        return self.constraint_dict
