import bezier
import numpy as np
from typing import List, Tuple, Set
import matplotlib.pyplot as plt

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.lanelet import Lanelet, LineMarking, LaneletType, RoadUser
from commonroad.scenario.scenario import Scenario, Tag, Location


def create_access_ramp_start(lanelet_length: int, l_id: int) -> Lanelet:
    x = [
        0,
        lanelet_length // 4,
        lanelet_length // 2,
        lanelet_length // (4 / 3),
        lanelet_length,
    ]
    right = np.asfortranarray([x, [-5, -4.25, -3.75, -3.5, -3.5]])
    right_curve = bezier.Curve(right, degree=4)
    ax = right_curve.plot(num_pts=50)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    left = np.asfortranarray([x, [-1.5, -0.5, -0.25, 0.0, 0.0]])
    left_curve = bezier.Curve(left, degree=4)
    ax = left_curve.plot(num_pts=50)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    center = np.asfortranarray([x, [-3.25, -2.5, -2.9, -1.75, -1.75]])
    center_curve = bezier.Curve(center, degree=4)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    plt.close()

    lanelet = Lanelet(
        left_vertices=left_vertices,
        center_vertices=center_vertices,
        right_vertices=right_vertices,
        lanelet_id=l_id,
        successor=[l_id + 1],
        adjacent_left=None,
        adjacent_left_same_direction=None,
        line_marking_left_vertices=LineMarking.SOLID,
        line_marking_right_vertices=LineMarking.SOLID,
        lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP},
        user_one_way={RoadUser.VEHICLE},
    )

    return lanelet


def create_access_ramp_end(
    x_start: int, l_id: int, lanelet_length: int, number_lanelets_lane: int
):
    x = [x_start + 0, x_start + 5, x_start + 10, x_start + 15, x_start + 20]
    right = np.asfortranarray([x, [-3.5, -3.25, -1, 0, 0]])
    right_curve = bezier.Curve(right, degree=4)
    ax = right_curve.plot(num_pts=20)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    left = np.asfortranarray([x, [0, 0.25, 2.5, 3.5, 3.5]])
    left_curve = bezier.Curve(left, degree=4)
    ax = left_curve.plot(num_pts=20)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    center = np.asfortranarray([x, [-1.75, -1.5, -1.25, 1.75, 1.75]])

    center_curve = bezier.Curve(center, degree=4)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    plt.close()

    lanelet = Lanelet(
        left_vertices=left_vertices,
        center_vertices=center_vertices,
        right_vertices=right_vertices,
        lanelet_id=l_id,
        predecessor=[l_id - 1],
        successor=None,
        adjacent_left=number_lanelets_lane + l_id + 1,
        adjacent_left_same_direction=True,
        line_marking_left_vertices=None,
        line_marking_right_vertices=LineMarking.SOLID,
        lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP},
        user_one_way={RoadUser.VEHICLE},
    )

    return lanelet


def create_access_ramp(
    road_length: int, num_lanelets_per_lane: int, lanelet_length: int
):
    lanelets = []
    lane_width = 3.5

    # create straight lanelets
    lanelet = create_access_ramp_start(lanelet_length, 1)
    lanelets.append(lanelet)

    l_id_start = 1
    predecessor = [l_id_start]
    successor = [l_id_start + 2]
    adj_left = num_lanelets_per_lane + 2
    for lanelet_idx in range(l_id_start + 1, l_id_start + num_lanelets_per_lane - 2):
        left_vertices_point_list = []
        center_vertices_point_list = []
        right_vertices_point_list = []

        for i in range(lanelet_length + 1):
            left_vertices_point_list.append(
                np.array([i + lanelet_length * (lanelet_idx - l_id_start), 0])
            )
            center_vertices_point_list.append(
                np.array(
                    [i + lanelet_length * (lanelet_idx - l_id_start), -0.5 * lane_width]
                )
            )
            right_vertices_point_list.append(
                np.array([i + lanelet_length * (lanelet_idx - l_id_start), -lane_width])
            )

        left_vertices = np.array(left_vertices_point_list)
        center_vertices = np.array(center_vertices_point_list)
        right_vertices = np.array(right_vertices_point_list)

        lanelet = Lanelet(
            left_vertices=left_vertices,
            center_vertices=center_vertices,
            right_vertices=right_vertices,
            lanelet_id=lanelet_idx,
            predecessor=predecessor,
            successor=successor,
            adjacent_left=adj_left,
            adjacent_left_same_direction=True,
            line_marking_left_vertices=LineMarking.DASHED,
            line_marking_right_vertices=LineMarking.SOLID,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP},
            user_one_way={RoadUser.VEHICLE},
        )
        lanelets.append(lanelet)
        adj_left += 1
        predecessor = [lanelet_idx]
        if lanelet_idx % num_lanelets_per_lane != 0.0:
            successor = [successor[0] + 1]
        else:
            successor = None
    lanelet_end = create_access_ramp_end(
        road_length - 2 * lanelet_length,
        l_id_start + num_lanelets_per_lane - 2,
        lanelet_length,
        num_lanelets_per_lane,
    )
    lanelets.append(lanelet_end)

    return lanelets


def create_exit_ramp_start(
    lanelet_lenght: int, l_id: int, num_lanelets_per_lane: int
) -> Lanelet:
    x_start = lanelet_lenght
    right = np.asfortranarray(
        [
            [
                x_start,
                x_start + 2.5,
                x_start + 5,
                x_start + 7.5,
                x_start + 9.0,
                x_start + 10.0,
            ],
            [0.0, -1, -3, -3.25, -3.5, -3.5],
        ]
    )
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=50)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    for idx in range(10):
        point = np.array([x_right[-1] + (idx + 1), y_right[-1]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    left = np.asfortranarray(
        [
            [
                x_start,
                x_start + 2.5,
                x_start + 5,
                x_start + 7.5,
                x_start + 9.0,
                x_start + 10.0,
            ],
            [3.5, 2.5, 0.5, 0.25, 0, 0],
        ]
    )
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=50)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    for idx in range(10):
        point = np.array([x_left[-1] + (idx + 1), y_left[-1]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    center = np.asfortranarray(
        [[10.0, 12.5, 15, 17.5, 19, 20.0], [1.75, 0.75, -1.25, -1.5, -1.75, -1.75]]
    )
    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    for idx in range(10):
        point = np.array([x_center[-1] + (idx + 1), y_center[-1]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    plt.close()

    lanelet = Lanelet(
        left_vertices=left_vertices,
        center_vertices=center_vertices,
        right_vertices=right_vertices,
        lanelet_id=l_id,
        predecessor=[num_lanelets_per_lane + 1],
        successor=[l_id + 1],
        adjacent_left=l_id + num_lanelets_per_lane + 1,
        adjacent_left_same_direction=True,
        line_marking_left_vertices=None,
        line_marking_right_vertices=LineMarking.SOLID,
        lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP},
        user_one_way={RoadUser.VEHICLE},
    )
    return lanelet


def create_exit_ramp_end(x_start, l_id: int) -> Lanelet:
    right = np.asfortranarray(
        [
            [
                x_start,
                x_start + 2.5,
                x_start + 5,
                x_start + 7.5,
                x_start + 9.0,
                x_start + 10.0,
            ],
            [-3.5, -3.75, -4.5, -5.5, -6.75, -8.25],
        ]
    )
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=20)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    left = np.asfortranarray(
        [
            [
                x_start,
                x_start + 2.5,
                x_start + 5,
                x_start + 7.5,
                x_start + 9.0,
                x_start + 10.0,
            ],
            [0.0, -0.25, -1.0, -2.0, -3.0, -4.0],
        ]
    )
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=20)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    center = np.asfortranarray(
        [
            [
                x_start,
                x_start + 2.5,
                x_start + 5,
                x_start + 7.5,
                x_start + 9.0,
                x_start + 10.0,
            ],
            [-1.75, -2.0, -2.75, -3.75, -5.0, -6.5],
        ]
    )

    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    plt.close()

    lanelet = Lanelet(
        left_vertices=left_vertices,
        center_vertices=center_vertices,
        right_vertices=right_vertices,
        lanelet_id=l_id,
        predecessor=[l_id - 1],
        successor=None,
        adjacent_left=None,
        adjacent_left_same_direction=None,
        line_marking_left_vertices=LineMarking.SOLID,
        line_marking_right_vertices=LineMarking.SOLID,
        lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP},
        user_one_way={RoadUser.VEHICLE},
    )
    return lanelet


def create_exit_ramp(road_length: int, num_lanelets_per_lane: int, lanelet_length: int):
    lanelets = []
    lane_width = 3.5

    # create straight lanelets
    lanelet = create_exit_ramp_start(lanelet_length, 1, num_lanelets_per_lane)
    lanelets.append(lanelet)

    l_id_start = 2
    predecessor = [l_id_start - 1]
    successor = [l_id_start + 1]
    adj_left = num_lanelets_per_lane + 3
    for lanelet_idx in range(l_id_start, num_lanelets_per_lane - 1):
        left_vertices_point_list = []
        center_vertices_point_list = []
        right_vertices_point_list = []
        for i in range(lanelet_length + 1):
            left_vertices_point_list.append(
                np.array([i + lanelet_length * lanelet_idx, 0])
            )
            center_vertices_point_list.append(
                np.array([i + lanelet_length * lanelet_idx, -0.5 * lane_width])
            )
            right_vertices_point_list.append(
                np.array([i + lanelet_length * lanelet_idx, -lane_width])
            )

        left_vertices = np.array(left_vertices_point_list)
        center_vertices = np.array(center_vertices_point_list)
        right_vertices = np.array(right_vertices_point_list)

        lanelet = Lanelet(
            left_vertices=left_vertices,
            center_vertices=center_vertices,
            right_vertices=right_vertices,
            lanelet_id=lanelet_idx,
            predecessor=predecessor,
            successor=successor,
            adjacent_left=adj_left,
            adjacent_left_same_direction=True,
            line_marking_left_vertices=LineMarking.DASHED,
            line_marking_right_vertices=LineMarking.SOLID,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP},
            user_one_way={RoadUser.VEHICLE},
        )
        lanelets.append(lanelet)
        adj_left += 1
        predecessor = [lanelet_idx]
        if lanelet_idx % num_lanelets_per_lane != 0.0:
            successor = [successor[0] + 1]
        else:
            successor = None
    lanelet = create_exit_ramp_end(
        road_length - lanelet_length, num_lanelets_per_lane - 1
    )
    lanelets.append(lanelet)

    return lanelets


def create_straight_scenario(
    commonroad_benchmark_id: str,
    dt: float,
    num_straight_lanes: int,
    num_lanelets_per_lane: int,
    road_length: int,
    obstacles: List[DynamicObstacle],
    markings: List[Tuple[LineMarking, LineMarking]],
    lanelet_types: List[Set[LaneletType]],
    lane_width: List[float],
):
    # desired number of lanes and parameters
    lanelet_length = int(road_length / num_lanelets_per_lane)
    scenario = create_scenario(commonroad_benchmark_id, dt)

    # create straight lanelets
    lanelet_id_list = range(1, num_straight_lanes * num_lanelets_per_lane + 1)
    lanelet_id_idx = 0
    for lane in range(num_straight_lanes):
        predecessor = None
        successor = [lanelet_id_list[lanelet_id_idx + 1]]
        for lanelet_idx in range(1, num_lanelets_per_lane + 1):
            left_vertices_point_list = []
            center_vertices_point_list = []
            right_vertices_point_list = []

            for i in range(lanelet_length + 1):
                left_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 1) * lane_width[lane],
                        ]
                    )
                )
                center_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 0.5) * lane_width[lane],
                        ]
                    )
                )
                right_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            lane * lane_width[lane],
                        ]
                    )
                )

            left_vertices = np.array(left_vertices_point_list)
            center_vertices = np.array(center_vertices_point_list)
            right_vertices = np.array(right_vertices_point_list)

            # setting adjecent lanes to correct object ID
            # first lane: no adjecent right lane
            if num_straight_lanes == 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types[lane],
                    user_one_way={RoadUser.VEHICLE},
                )
            elif lane == 0:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types[lane],
                    user_one_way={RoadUser.VEHICLE},
                )
            # last lane: no adjecent left lane
            elif lane == num_straight_lanes - 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types[lane],
                    user_one_way={RoadUser.VEHICLE},
                )
            else:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types[lane],
                    user_one_way={RoadUser.VEHICLE},
                )
            predecessor = [lanelet_id_list[lanelet_id_idx]]
            lanelet_id_idx += 1
            if (
                lanelet_id_idx + 1 < len(lanelet_id_list)
                and (lanelet_id_idx + 1) % num_lanelets_per_lane != 0.0
            ):
                successor = [lanelet_id_list[lanelet_id_idx + 1]]
            else:
                successor = None
            scenario.lanelet_network.add_lanelet(lanelet)
    for obs in obstacles:
        scenario.add_objects(obs)

    return scenario


def create_exit_ramp_scenario(
    commonroad_benchmark_id: str,
    dt: float,
    num_straight_lanes: int,
    num_lanelets_per_lane: int,
    road_length: int,
    obstacles: List[DynamicObstacle],
    markings: List[Tuple[LineMarking, LineMarking]],
):
    # desired number of lanes and parameters
    lane_width = 3.5
    lanelet_types = {LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY}
    lanelet_length = int(road_length / num_lanelets_per_lane)
    scenario = create_scenario(commonroad_benchmark_id, dt)

    # create straight lanelets
    lanelet_id_list = range(1, (num_straight_lanes + 1) * num_lanelets_per_lane + 1)
    lanelet_id_idx = num_lanelets_per_lane

    exit_ramp_lanelets = create_exit_ramp(
        road_length, num_lanelets_per_lane, lanelet_length
    )
    for lanelet in exit_ramp_lanelets:
        scenario.lanelet_network.add_lanelet(lanelet)

    for lane in range(num_straight_lanes):
        predecessor = []
        successor = [lanelet_id_list[lanelet_id_idx + 1]]
        for lanelet_idx in range(1, num_lanelets_per_lane + 1):
            left_vertices_point_list = []
            center_vertices_point_list = []
            right_vertices_point_list = []

            for i in range(lanelet_length + 1):
                left_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 1) * lane_width,
                        ]
                    )
                )
                center_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 0.5) * lane_width,
                        ]
                    )
                )
                right_vertices_point_list.append(
                    np.array(
                        [i + lanelet_length * (lanelet_idx - 1), lane * lane_width]
                    )
                )

            left_vertices = np.array(left_vertices_point_list)
            center_vertices = np.array(center_vertices_point_list)
            right_vertices = np.array(right_vertices_point_list)

            # setting adjecent lanes to correct object ID
            if num_straight_lanes == 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            elif lane == 0:
                if (
                    lanelet_id_list[lanelet_id_idx] == 1 + num_lanelets_per_lane
                    or lanelet_id_list[lanelet_id_idx] == 2 * num_lanelets_per_lane
                ):
                    line_marking_right = LineMarking.SOLID
                    adjacent_right = None
                    adjacent_right_same_direction = False
                else:
                    line_marking_right = LineMarking.DASHED
                    adjacent_right = (
                        lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane - 1
                    )
                    adjacent_right_same_direction = True
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=line_marking_right,
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            # last lane: no adjecent left lane
            elif lane == num_straight_lanes - 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            else:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            predecessor = [lanelet_id_list[lanelet_id_idx]]
            lanelet_id_idx += 1
            if (
                lanelet_id_idx + 1 < len(lanelet_id_list)
                and (lanelet_id_idx + 1) % num_lanelets_per_lane != 0.0
            ):
                successor = [lanelet_id_list[lanelet_id_idx + 1]]
            else:
                successor = None
            scenario.lanelet_network.add_lanelet(lanelet)
    for obs in obstacles:
        scenario.add_objects(obs)

    return scenario


def create_access_ramp_scenario(
    commonroad_benchmark_id: str,
    dt: float,
    num_straight_lanes: int,
    num_lanelets_per_lane: int,
    road_length: int,
    obstacles: List[DynamicObstacle],
    markings: List[Tuple[LineMarking, LineMarking]],
):
    # desired number of lanes and parameters
    lane_width = 3.5
    lanelet_types = {LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY}
    lanelet_length = int(road_length / num_lanelets_per_lane)
    scenario = create_scenario(commonroad_benchmark_id, dt)

    # create straight lanelets
    lanelet_id_list = range(1, (num_straight_lanes + 1) * num_lanelets_per_lane + 1)
    lanelet_id_idx = num_lanelets_per_lane

    access_ramp_lanelets = create_access_ramp(
        road_length, num_lanelets_per_lane, lanelet_length
    )
    for lanelet in access_ramp_lanelets:
        scenario.lanelet_network.add_lanelet(lanelet)

    for lane in range(num_straight_lanes):
        predecessor = None
        successor = [lanelet_id_list[lanelet_id_idx + 1]]
        for lanelet_idx in range(1, num_lanelets_per_lane + 1):
            left_vertices_point_list = []
            center_vertices_point_list = []
            right_vertices_point_list = []

            for i in range(lanelet_length + 1):
                left_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 1) * lane_width,
                        ]
                    )
                )
                center_vertices_point_list.append(
                    np.array(
                        [
                            i + lanelet_length * (lanelet_idx - 1),
                            (lane + 0.5) * lane_width,
                        ]
                    )
                )
                right_vertices_point_list.append(
                    np.array(
                        [i + lanelet_length * (lanelet_idx - 1), lane * lane_width]
                    )
                )

            left_vertices = np.array(left_vertices_point_list)
            center_vertices = np.array(center_vertices_point_list)
            right_vertices = np.array(right_vertices_point_list)

            # setting adjecent lanes to correct object ID
            if num_straight_lanes == 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            elif lane == 0:
                if (
                    lanelet_id_list[lanelet_id_idx] == 1 + num_lanelets_per_lane
                    or lanelet_id_list[lanelet_id_idx] == 2 * num_lanelets_per_lane
                ):
                    line_marking_right = LineMarking.SOLID
                    adjacent_right = None
                    adjacent_right_same_direction = False
                else:
                    line_marking_right = LineMarking.DASHED
                    adjacent_right = (
                        lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane - 1
                    )
                    adjacent_right_same_direction = True
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=line_marking_right,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            # last lane: no adjecent left lane
            elif lane == num_straight_lanes - 1:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            else:
                lanelet = Lanelet(
                    left_vertices,
                    center_vertices,
                    right_vertices,
                    lanelet_id_list[lanelet_id_idx],
                    predecessor=predecessor,
                    successor=successor,
                    adjacent_left=lanelet_id_list[lanelet_id_idx]
                    + num_lanelets_per_lane,
                    adjacent_left_same_direction=True,
                    adjacent_right=lanelet_id_list[lanelet_id_idx]
                    - num_lanelets_per_lane,
                    adjacent_right_same_direction=True,
                    line_marking_left_vertices=markings[lane][0],
                    line_marking_right_vertices=markings[lane][1],
                    lanelet_type=lanelet_types,
                    user_one_way={RoadUser.VEHICLE},
                )
            predecessor = [lanelet_id_list[lanelet_id_idx]]
            lanelet_id_idx += 1
            if (
                lanelet_id_idx + 1 < len(lanelet_id_list)
                and (lanelet_id_idx + 1) % num_lanelets_per_lane != 0.0
            ):
                successor = [lanelet_id_list[lanelet_id_idx + 1]]
            else:
                successor = None
            scenario.lanelet_network.add_lanelet(lanelet)
    for obs in obstacles:
        scenario.add_objects(obs)

    return scenario


def create_scenario(commonroad_benchmark_id, dt):
    author = "Sebastian Maierhofer"
    affiliation = "Technical University of Munich, Germany"
    source = "CommonRoad Monitor"
    tags = {Tag.INTERSTATE, Tag.MULTI_LANE, Tag.NO_ONCOMING_TRAFFIC, Tag.PARALLEL_LANES}
    location = Location(-999, 0, 0)
    scenario = Scenario(
        dt,
        "DEU_" + commonroad_benchmark_id,
        author,
        tags,
        affiliation,
        source,
        location,
    )
    return scenario
