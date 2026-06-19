from scipy.integrate import odeint

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion, Interval, AngleInterval
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.traffic_sign import (
    TrafficSign,
    TrafficSignElement,
    TrafficSignIDGermany,
)
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from vehicleDynamics_ST import vehicleDynamics_ST

from crmonitor.common.helper import *
from scenarios.test_interstate.road_network_generation import *

CONSTANT_DRIVING_50 = [0] * 50
CONSTANT_DRIVING_200 = [0] * 200


def write_to_file(scenario):
    # create planing problem set (goal state is arbitrarily chosen, since it is not needed for this purpose)
    goal_position_shape = Rectangle(10, 3.5, np.array([50, 1.75]))
    goal_state = State(
        position=goal_position_shape,
        velocity=Interval(0, 50),
        orientation=AngleInterval(-0.01, 0.01),
        time_step=Interval(0, 100),
    )
    goal_region = GoalRegion([goal_state])
    init_state = State(
        position=np.array([0.0, 1.75]),
        orientation=0,
        velocity=0,
        yaw_rate=0,
        slip_angle=0,
        time_step=0,
    )
    planning_problem = PlanningProblem(1, init_state, goal_region)
    planning_problem_set = PlanningProblemSet([planning_problem])
    # write new scenario
    fw = CommonRoadFileWriter(scenario, planning_problem_set)
    filename = "./" + scenario.benchmark_id + ".xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)


def func_st(x, t, u, p):
    f = vehicleDynamics_ST(x, u, p)
    return f


def create_obstacle_by_acceleration(
    acceleration_profile, v_init, p_init, obs_id, steering_velocity_profile=None
):
    initial_state = State(
        position=p_init,
        velocity=v_init,
        orientation=0,
        yaw_rate=0,
        slip_angle=0,
        time_step=0,
    )
    dt = 0.1
    state_list = []
    time_step = 1
    x = [p_init[0], p_init[1], 0, v_init, 0, 0, 0]
    p = parameters_vehicle2()
    t = np.arange(0, 2 * dt, dt)
    for idx in range(len(acceleration_profile)):
        a = acceleration_profile[idx]
        if steering_velocity_profile is not None:
            v_s = steering_velocity_profile[idx]
        else:
            v_s = 0

        u = [v_s, a]
        x = odeint(func_st, x, t, args=(u, p))
        x = x[1]
        state_list.append(
            State(
                position=np.array([x[0], x[1]]),
                velocity=x[3],
                orientation=x[4],
                yaw_rate=x[5],
                slip_angle=x[6],
                time_step=time_step,
            )
        )
        time_step += 1
    obstacle_length = 4.508
    obstacle_width = 1.610
    shape = Rectangle(obstacle_length, obstacle_width)
    trajectory = Trajectory(initial_time_step=0, state_list=state_list)
    prediction = TrajectoryPrediction(trajectory, shape)

    return DynamicObstacle(obs_id, ObstacleType.CAR, shape, initial_state, prediction)


def create_max_speed_limit_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
        ],
        31.5,
        np.array([3.0, 1.75]),
        1000,
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 35, np.array([10.0, 5.25]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 36, np.array([3.0, 8.75]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 30, np.array([3.0, 5.25]), 1003
    )
    for i in range(4):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 3
    num_lanelets = 5
    road_length = 200
    scenario = create_straight_scenario(
        "test_max_speed_limit",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5],
    )
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, [str(35)])
    traffic_sign = TrafficSign(
        201,
        [traffic_sign_elem],
        {1, 6, 11},
        scenario.lanelet_network.find_lanelet_by_id(1).right_vertices[0],
    )
    scenario.lanelet_network.add_traffic_sign(
        traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 13, 14, 15}
    )

    write_to_file(scenario)


def create_min_speed_limit_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22, np.array([3.0, 1.75]), 1000
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 35, np.array([3.0, 5.25]), 1001
    )

    obstacles.append(obs1)
    obstacles.append(obs2)
    num_lanes = 2
    num_lanelets = 5
    road_length = 200
    scenario = create_straight_scenario(
        "test_min_speed_limit",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5],
    )
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MIN_SPEED, [str(30)])
    traffic_sign = TrafficSign(
        202,
        [traffic_sign_elem],
        {1, 6},
        scenario.lanelet_network.find_lanelet_by_id(1).right_vertices[0],
    )
    scenario.lanelet_network.add_traffic_sign(
        traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    )

    write_to_file(scenario)


def create_preserves_traffic_flow_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 35, np.array([3.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 35, np.array([13.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 5, np.array([3.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 8, np.array([13.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 35, np.array([22.0, 5.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 5, np.array([100.0, 8.75]), 1005
    )
    for i in range(6):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 3
    num_lanelets = 5
    road_length = 200
    scenario = create_straight_scenario(
        "test_preserve_traffic_flow",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5],
    )
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, [str(40)])
    traffic_sign = TrafficSign(
        201,
        [traffic_sign_elem],
        {1, 6, 11},
        scenario.lanelet_network.find_lanelet_by_id(1).right_vertices[0],
    )
    scenario.lanelet_network.add_traffic_sign(
        traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    )

    write_to_file(scenario)


def create_safe_distance_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            -2,
            0,
            0,
            -2,
            0,
            -2,
            0,
            0,
            0,
            0,
            -5,
            0,
            0,
            -5,
            0,
            -2,
            0,
            -5.9,
            -3.1,
            0.1,
            -2,
            -5,
            -4.4,
            0,
            -10,
            -1,
            0,
            -2,
            -0,
            -5.1,
            -2,
            -0.5,
            -2,
            -2.8,
            -8,
            -8.3,
            -2,
            0,
            0,
            0,
            -6,
            0,
            -1,
            0,
            -8,
            -1,
            -2,
        ],
        40,
        np.array([3.0, 1.75]),
        1000,
    )
    obs1 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -2.3,
            2,
            -4,
            -3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2.2,
            -2,
            0,
            0,
            0.8,
            0,
            0,
            1.1,
            1.5,
            0,
            1,
            0,
            2.2,
            1,
            0.4,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
        ],
        30,
        np.array([61.50, 1.75]),
        1001,
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 30, np.array([3.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 30, np.array([13.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([75.0, 5.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([3.0, 8.75]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([40.0, 8.75]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([85.0, 6.25]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
        ],
        30,
        np.array([3.0, 12.25]),
        1008,
    )
    obs9 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        20,
        np.array([30.0, 12.25]),
        1009,
        [
            0.05,
            0.05,
            0.05,
            0,
            0,
            0,
            0,
            -0.05,
            -0.05,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs10 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        20,
        np.array([45.0, 15.75]),
        1010,
        [
            -0.05,
            -0.05,
            -0.05,
            0,
            0,
            0,
            0,
            0.05,
            0.05,
            0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    for i in range(11):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 5
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario(
        "test_safe_distance",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5, 3.5],
    )
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, [str(22.22)])
    traffic_sign = TrafficSign(
        201,
        [traffic_sign_elem],
        {1, 6, 11},
        scenario.lanelet_network.find_lanelet_by_id(1).right_vertices[0],
    )
    scenario.lanelet_network.add_traffic_sign(
        traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    )

    write_to_file(scenario)


def create_unnecessary_braking_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        20,
        np.array([3.0, 1.75]),
        1000,
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 30, np.array([3.0, 5.25]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        [
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.9,
            -1.0,
            -1.1,
            -1.2,
            -1.3,
            -1.4,
            -1.5,
            -1.6,
            -1.7,
            -1.8,
            -1.9,
            -2.0,
            -2.1,
            -2.2,
            -2.3,
            -2.4,
            -2.5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        30,
        np.array([75.0, 8.75]),
        1002,
    )
    obs3 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.9,
            -1.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        20,
        np.array([40.0, 12.25]),
        1005,
    )
    obs4 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.9,
            -1.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        20,
        np.array([3.0, 15.75]),
        1006,
    )
    obs5 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.9,
            -1.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        20,
        np.array([40.0, 15.75]),
        1007,
    )
    for i in range(6):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 5
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario(
        "test_unnecessary_braking",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_standstill_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([3.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([13.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([23.0, 1.75]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([3.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([13.0, 5.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([33.0, 5.25]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([43.0, 5.25]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([53.0, 5.25]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([63.0, 5.25]), 1008
    )
    obs9 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([73.0, 5.25]), 1009
    )
    obs10 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 3, np.array([73.0, 5.25]), 1010
    )
    for i in range(11):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario(
        "test_standstill",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5],
    )

    write_to_file(scenario)


def create_reverse_and_u_turn_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        2,
        np.array([50.0, 1.75]),
        1000,
    )
    obs1 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        -10,
        np.array([100.0, 1.75]),
        1001,
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        5,
        np.array([125.0, 1.75]),
        1002,
        [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
        ],
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([140.0, 1.75]), 1003
    )
    for i in range(4):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 4
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario(
        "test_reversing_and_u_turn",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_overtaking_right_congestion_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 5, np.array([3.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([13.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([3.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([13.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([33.0, 5.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([43.0, 5.25]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([53.0, 5.25]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([63.0, 5.25]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([73.0, 5.25]), 1008
    )
    for i in range(9):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 150
    scenario = create_straight_scenario(
        "test_overtaking_right_congestion",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_overtaking_exit_ramp_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 5, np.array([45.0, -1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([55.0, -1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([43.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([53.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([73.0, 5.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([83.0, 5.25]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([93.0, 5.25]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([103.0, 5.25]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([113.0, 5.25]), 1008
    )
    for i in range(9):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 200
    scenario = create_exit_ramp_scenario(
        "test_overtaking_exit_ramp",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
    )

    write_to_file(scenario)


def create_overtaking_access_ramp_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 25, np.array([25.0, 5.25]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([40.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 25, np.array([25.0, -1.75]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([3.0, 1.75]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([10.0, 5.25]), 1004
    )
    for i in range(5):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 200
    scenario = create_access_ramp_scenario(
        "test_overtaking_access_ramp",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
    )

    write_to_file(scenario)


def create_overtaking_broad_lane_marking_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([5.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 25, np.array([20.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([5.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([40.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 28.5, np.array([5.0, 8.75]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 10, np.array([50.0, 12.25]), 1005
    )
    for i in range(6):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 4
    num_lanelets = 10
    road_length = 200
    scenario = create_straight_scenario(
        "test_overtaking_right_broad_lane_marking",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.BROAD_DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.BROAD_DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_overtaking_normal_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([25.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 25, np.array([40.0, 1.75]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([25.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([60.0, 5.25]), 1003
    )
    for i in range(4):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 200
    scenario = create_straight_scenario(
        "test_overtaking_right_normal",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5],
    )

    write_to_file(scenario)


def create_emergency_three_lanes_with_shoulder_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([13.0, 5.35]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([25.0, 4.35]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([32.0, 4.35]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2.5, np.array([40.0, 4.35]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.75, np.array([51.0, 5.5]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([59.0, 5.25]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([13.0, 13.15]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([23.0, 6.3]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([43.0, 7.9]), 1008
    )
    obs9 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([53.0, 7.25]), 1009
    )
    obs10 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([63.0, 8.75]), 1010
    )
    obs11 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([73.0, 6.4]), 1011
    )
    obs12 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([83.0, 7.0]), 1012
    )
    obs13 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([23.0, 13.15]), 1013
    )
    obs14 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([43.0, 13.15]), 1014
    )
    obs15 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([53.0, 13.15]), 1015
    )
    obs16 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([63.0, 12.15]), 1016
    )
    obs17 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([72.0, 13.15]), 1017
    )
    obs18 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2.25, np.array([67.0, 4.35]), 1018
    )
    obs19 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([76.0, 4.35]), 1019
    )
    obs20 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([83.0, 13.15]), 1020
    )
    obs21 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2.25, np.array([4.0, 3.25]), 1021
    )
    obs22 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([5.0, 7.75]), 1022
    )
    obs23 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([3.0, 13.15]), 1023
    )
    obs24 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([23.0, 11.00]), 1024
    )
    obs25 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([70.0, 8.5]), 1025
    )
    obs26 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([77.5, 13.15]), 1026
    )
    for i in range(27):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 4
    num_lanelets = 10
    road_length = 100
    scenario = create_straight_scenario(
        "test_emergency_three_lanes_with_shoulder",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.SOLID, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.SHOULDER, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_emergency_two_lanes_not_broad_enough_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([13.0, 2.05]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([25.0, 0.95]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([32.0, 0.95]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2.5, np.array([40.0, 0.95]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.75, np.array([51.0, 2.0]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([59.0, 1.75]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([23.0, 3.3]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0, np.array([43.0, 4.25]), 1007
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([53.0, 4.25]), 1008
    )
    obs9 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([63.0, 5.25]), 1009
    )
    obs10 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1, np.array([73.0, 3.4]), 1010
    )
    obs11 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 1.5, np.array([83.0, 3.5]), 1011
    )
    obs12 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2.25, np.array([67.0, 1.75]), 1012
    )
    obs13 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([76.0, 1.75]), 1013
    )
    obs14 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 2, np.array([5.0, 4.25]), 1014
    )
    obs15 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 0.5, np.array([93.5, 5.25]), 1015
    )

    for i in range(16):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 100
    scenario = create_straight_scenario(
        "test_emergency_two_lanes_not_broad_enough",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3, 3],
    )

    write_to_file(scenario)


def create_consider_entering_vehicles_for_lane_change_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        25,
        np.array([25.0, 5.25]),
        1000,
        [
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.0275,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 20, np.array([5.0, 5.25]), 1001
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        11,
        np.array([50.0, -1.75]),
        1002,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    for i in range(3):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 200
    scenario = create_access_ramp_scenario(
        "test_consider_entering_vehicles_for_lane_change",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
    )

    write_to_file(scenario)


def create_recapture_safe_distance():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22.5, np.array([5.0, 5.25]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([100.0, 1.75]),
        1001,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22.5, np.array([12.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22, np.array([40.0, 5.25]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        [
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
        ],
        22.6,
        np.array([5.0, 1.75]),
        1004,
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22.5, np.array([12.0, 1.75]), 1005
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([60.0, 8.75]),
        1006,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs7 = create_obstacle_by_acceleration(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
        ],
        25,
        np.array([10.0, 12.25]),
        1007,
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([60.0, 15.75]),
        1008,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs9 = create_obstacle_by_acceleration(
        [
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
            -5,
        ],
        25,
        np.array([10.0, 19.25]),
        1009,
    )
    for i in range(10):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 6
    num_lanelets = 10
    road_length = 200
    scenario = create_straight_scenario(
        "test_recapture_safe_distance",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_safe_distance_lane_change_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([5.0, 5.25]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([100.0, 1.75]),
        1001,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs2 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 22.5, np.array([40.0, 5.25]), 1002
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 25, np.array([40.0, 1.75]), 1003
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 28, np.array([40.0, 1.75]), 1003
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 11.5, np.array([30.0, 12.25]), 1004
    )
    obs5 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([20.0, 8.75]),
        1005,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs6 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 11.5, np.array([100.0, 19.25]), 1006
    )
    obs7 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([50.0, 15.75]),
        1007,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs8 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 11.5, np.array([10.0, 19.25]), 1008
    )
    for i in range(9):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 6
    num_lanelets = 10
    road_length = 200
    scenario = create_straight_scenario(
        "test_safe_distance_lane_change",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.DASHED, LineMarking.DASHED),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
    )

    write_to_file(scenario)


def create_driving_rightmost_scenario():
    obstacles = []
    obs0 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([100.0, 1.75]), 1000
    )
    obs1 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50, 15, np.array([110.0, 3.5]), 1001
    )
    obs3 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        12,
        np.array([5.0, 1.75]),
        1002,
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.05,
            -0.035,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -0.05,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    obs4 = create_obstacle_by_acceleration(
        CONSTANT_DRIVING_50,
        25,
        np.array([10.0, 1.75]),
        1003,
        [
            0.1,
            0.075,
            0.05,
            0.025,
            0.0,
            -0.1,
            -0.075,
            -0.05,
            -0.025,
            -0.0,
            -0.1,
            -0.075,
            -0.05,
            -0.025,
            0.0,
            0,
            0,
            0.0,
            0.1,
            0.1,
            0.1,
            0.1,
            0.0,
            0,
            0,
            -0.1,
            -0.075,
            -0.05,
            -0.025,
            -0.1,
            0.1,
            0.01,
            0.1,
            0.00,
            0.02,
            0.02,
            0.075,
            0.05,
            0.025,
            0.0,
            0.01,
            0.075,
            -0.05,
            -0.025,
            0.0,
            -0.1,
            -0.075,
            -0.05,
            -0.025,
            -0.0,
        ],
    )
    for i in range(5):
        obs = locals().get("obs" + str(i))
        if obs is not None:
            obstacles.append(obs)
    num_lanes = 2
    num_lanelets = 10
    road_length = 200
    scenario = create_straight_scenario(
        "test_driving_rightmost",
        0.1,
        num_lanes,
        num_lanelets,
        road_length,
        obstacles,
        [
            (LineMarking.DASHED, LineMarking.SOLID),
            (LineMarking.SOLID, LineMarking.DASHED),
        ],
        [
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
            {LaneletType.MAIN_CARRIAGE_WAY, LaneletType.INTERSTATE},
        ],
        [3.5, 3.5],
    )

    write_to_file(scenario)


def main():
    create_max_speed_limit_scenario()
    create_min_speed_limit_scenario()
    create_preserves_traffic_flow_scenario()
    create_safe_distance_scenario()
    create_unnecessary_braking_scenario()
    create_standstill_scenario()
    create_reverse_and_u_turn_scenario()
    create_overtaking_right_congestion_scenario()
    create_emergency_three_lanes_with_shoulder_scenario()
    create_overtaking_exit_ramp_scenario()
    create_overtaking_access_ramp_scenario()
    create_overtaking_broad_lane_marking_scenario()
    create_overtaking_normal_scenario()
    create_emergency_two_lanes_not_broad_enough_scenario()
    create_safe_distance_lane_change_scenario()
    create_recapture_safe_distance()
    create_consider_entering_vehicles_for_lane_change_scenario()
    create_driving_rightmost_scenario()


if __name__ == "__main__":
    main()
