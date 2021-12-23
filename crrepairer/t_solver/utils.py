from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle

import copy

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns

# commonroad-io
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
# from commonroad_dc.collision.visualization.draw_dispatch import draw_object

import commonroad_dc.pycrcc as pycrcc

TUMblue = [0, 101/255, 189/255]
TUMgreen = [162/255, 173/255, 0]
TUMgray = [156/255, 157/255,  159/255]
TUMdarkgray = [88/255, 88/255, 99/255]
TUMorange = [227/255, 114/255, 34/255]
TUMdarkblue = [0, 82/255, 147/255]
TUMwhite = [1, 1, 1]
TUMblack = [0, 0, 0]
TUMlightgray = [217/255, 218/255, 219/255]

ego_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "facecolor": TUMblue,
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "facecolor": TUMblue,
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}
ego_occupancy_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "opacity": 0.1,
                        "facecolor": TUMblue,
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "opacity": 1.0,
                        "facecolor": TUMblue,
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}
preceding_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "facecolor": TUMorange,
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "facecolor": TUMorange,
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}
ego_violation_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "facecolor": [1, 0, 0],
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "opacity": 1.0,
                        "facecolor": [1, 0, 0],
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}
ego_repaired_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "facecolor": TUMgreen,
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "opacity": 1.0,
                        "facecolor": TUMgreen,
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}
other_params = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {
                "shape": {
                    "polygon": {
                        "facecolor": TUMblack,
                        'edgecolor': TUMblack,
                    },
                    "rectangle": {
                        "facecolor": TUMblack,
                        'edgecolor': TUMblack,
                    },
                }
            },
        },
    },
}

def plot_reference_path(reference_path: Union[np.ndarray]):
    x_list = []
    y_list = []
    for point in reference_path:
        x_list.append(point[0])
        y_list.append(point[1])
    plt.plot(x_list, y_list)
    plt.show()

def merge_trajectories(tstcc:int, initial_trajectory, repaired_ego_vehicle: DynamicObstacle):
    updated_state_list = initial_trajectory.state_list[0:tstcc-1]
    for state in repaired_ego_vehicle.prediction.trajectory.state_list:
        state.time_step += tstcc
        updated_state_list.append(state)
    repaired_ego_vehicle.prediction.trajectory.state_list = updated_state_list
    return repaired_ego_vehicle

def update_trajectory_cartesian(tstcc:int, initial_trajectory, repaired_cartesian):
    updated_cartesian_x = []
    updated_cartesian_y = []
    for state in initial_trajectory.state_list[0: tstcc-1]:
        updated_cartesian_x.append(state.position[0])
        updated_cartesian_y.append(state.position[1])
    updated_cartesian_x += repaired_cartesian.cartesian_ptsX()
    updated_cartesian_y += repaired_cartesian.cartesian_ptsY()
    return [updated_cartesian_x, updated_cartesian_y]

def plot_trajectory_new(ego_vehicle, preceding_vehicle_id,
                        plot_limits,
                        scenario: Scenario, time_step: int,
                        flag_occupancy_ego, tstv: int,
                        filename, tag,
                        flag_trajectory=True):
    rnd = MPRenderer(plot_limits=plot_limits, figsize=(8, 4.5))
    scenario_copied = copy.deepcopy(scenario)
    if preceding_vehicle_id is not None:
        preceding_vehicle = scenario_copied.obstacle_by_id(preceding_vehicle_id)
        preceding_vehicle.draw(rnd, draw_params={'time_begin': time_step,
                                                 'trajectory': {'draw_trajectory': False},
                                                 **preceding_params})
        scenario_copied.remove_obstacle(preceding_vehicle)
    scenario_copied.remove_obstacle(ego_vehicle)

    if flag_occupancy_ego:
        ego_vehicle.draw(rnd,
                         draw_params={'time_begin': 0,
                                      'trajectory': {'draw_trajectory': True,
                                                     'draw_continuous': True},
                                      **ego_params})
        for i in range(len(ego_vehicle.prediction.trajectory.state_list)):
            preceding_vehicle.draw(rnd, draw_params={'time_begin': i, 'trajectory': {'draw_trajectory': False},
                                        **preceding_params})
            scenario_copied.draw(rnd, draw_params={'time_begin': i,
                                                   # 'lanelet_network': {"lanelet":
                                                   #                         {"facecolor": TUMlightgray,
                                                   #                          'fill_lanelet': False}},
                                                   'trajectory': {'draw_trajectory': False,
                                                                  'draw_continuous': False},
                                                   **other_params})
            if i < tstv:
                ego_vehicle.draw(rnd,
                                 draw_params={'time_begin': i,
                                              'trajectory': {'draw_trajectory': False},
                                              **ego_occupancy_params})
            else:
                ego_vehicle.draw(rnd,
                                 draw_params={'time_begin': i,
                                              'trajectory': {'draw_trajectory': False},
                                              **ego_repaired_params})  # ego_violation_params
            safe_distance = calculate_safe_distance(ego_vehicle.prediction.trajectory.state_list[i].velocity,
                                                    preceding_vehicle.prediction.trajectory.state_list[i].velocity,
                                                    -10.5, -10.0, 0.4)
            box_center = preceding_vehicle.prediction.trajectory.state_list[i].position-\
                [safe_distance/2, 0] - [preceding_vehicle.obstacle_shape.length, 0] + [0.25, 0]
            # -preceding_vehicle.obstacle_shape.width/2
            # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
            obb = pycrcc.RectOBB(safe_distance/2, 3.5/2, 0.0, box_center[0], box_center[1])
            obb.draw(rnd, draw_params={"opacity": 0.2,
                                        "facecolor": TUMorange,
                                        'edgecolor': TUMorange})
            rnd.render()
            plt.axis('off')
            # plt.show()
            plt.savefig(filename + '{}'.format(tag) + '{:05d}.png'.format(i),
                        format='png',dpi=300, )  # , transparent=True)
    else:
        ego_vehicle.draw(rnd,
                         draw_params={'time_begin': time_step,
                                      'trajectory': {'draw_trajectory': True,
                                                     'draw_continuous': True},
                                      **ego_params})
    # ego_vehicle.draw(rnd,
    #                  draw_params={'time_begin': 0,
    #                               'trajectory': {'draw_trajectory': True,
    #                                              'draw_continuous': True},
    #                               **ego_params})
    rnd.render()
    plt.axis('off')
    # plt.show()
    plt.savefig(filename + '{}.png'.format(tag),
                format='svg', bbox_inches='tight')  # , transparent=True)

def calculate_safe_distance(
        v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow
    ):
        d_safe = (
            (v_lead ** 2) / (-2 * np.abs(a_min_lead))
            - (v_follow ** 2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )

        return d_safe


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    # todo: complete the function
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2)
    goal_velocity = Interval(0, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = State(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step)
    goal_region = GoalRegion([goal_state])
    return goal_region


def convert_to_cr_ego_vehicle(
                                  width: float, length: float,
                                  wheelbase: float, vehicle_id: int = 0) -> DynamicObstacle:
    """
    Converts trajectory object to CommonRoad obstacle with specified width and length
    :param width: The width of the ego vehicle
    :param length: The length of the ego vehicle
    :param vehicle_id: ID of ego vehicle
    :return: The CommonRoad DynamicObstacle object containing the current trajectory
    """
    assert isinstance(width, (
        int, float)) and width > 0, '<Trajectory>: Provided width of vehicle invalid! width = {}'.format(width)
    assert isinstance(length, (int, float)) and length > 0, '<Trajectory>: Provided width of vehicle ' \
                                                            'invalid! width = {}'.format(length)

    # get trajectory
    traj = self.convert_to_cr_trajectory(wheelbase)
    for state in traj.state_list:
        state.time_step += tstcc
    traj.initial_time_step = initial_trajectory.state_list[0].time_step
    traj.state_list = initial_trajectory.state_list[0:tstcc - 1] + traj.state_list
    shape = Rectangle(length, width)
    pred = TrajectoryPrediction(traj, shape)

    # create new object
    ego = DynamicObstacle(obstacle_id=vehicle_id,
                          obstacle_type=ObstacleType.CAR,
                          prediction=pred,
                          obstacle_shape=shape,
                          initial_state=traj.state_list[0])
    return ego
