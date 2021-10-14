import math
from abc import ABC, abstractmethod
from typing import Union

# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState

# CommonRoad Toolbox
from commonroad.scenario.obstacle import DynamicObstacle, State
from commonroad.scenario.scenario import Scenario
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle


class CutOffBase(ABC):
    """
        Abstract base class for calculating cut-off states
    """
    def __init__(self, world_state: WorldState,
                 scenario: Scenario,
                 ego_vehicle_cr: DynamicObstacle,
                 dT: float):
        self._world_state = world_state
        self._ego_vehicle = ego_vehicle_cr
        self._dT = dT
        if scenario.obstacle_by_id(ego_vehicle_cr.obstacle_id) is not None:
            scenario.remove_obstacle(ego_vehicle_cr)
        # create collision checker (incl. road boundary)
        road_boundary_obstacle, road_boundary_sg_triangles = create_road_boundary_obstacle(scenario,
                                                                                           method='obb_rectangles')
        self._collision_checker = create_collision_checker(scenario)
        self._collision_checker.add_collision_object(road_boundary_sg_triangles)


    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def ego_vehicle(self) -> DynamicObstacle:
        return self._ego_vehicle

    @property
    def dT(self) -> float:
        return self._dT

    @dT.setter
    def dT(self, dT: float):
        raise Exception("You are not allowed to change the time step of the planner!")

    @abstractmethod
    def generate(self):
        """
        generates the cut off state: time-to-react or time-to-compliance
        """
        pass

    def _calc_ttc(self, state_list: Union[State]):
        """
        Detects the collision time given the trajectory of ego_vehicle using a for loop over
        the state list.
        """
        for state in state_list:
            ego = pycrcc.TimeVariantCollisionObject(state.time_step)
            ego.append_obstacle(pycrcc.RectOBB(0.5 * self._ego_vehicle.obstacle_shape.length,
                                               0.5 * self._ego_vehicle.obstacle_shape.width,
                                               state.orientation,
                                               state.position[0],
                                               state.position[1]))
            if self._collision_checker.collide(ego):
                return state.time_step * self._dT
        return math.inf


