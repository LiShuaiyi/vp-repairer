from abc import ABC, abstractmethod

from commonroad.scenario.scenario import State
"""
Class for trajectory repairer.
"""


class TrajectoryRepairer(ABC):
    """
        Abstract base class for a trajectory repairer. Contains basic methods and properties every repairer has to offer,
        e.g., time step and horizon
    """

    def __init__(self, vehicle_id: int):
        self._vehicle_id = vehicle_id

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @vehicle_id.setter
    def vehicle_id(self, N: int):
        raise Exception("You are not allowed to change the vehicle id of the repairer!")

    @abstractmethod
    def repair(self):
        """
        Calculates the trajectory which is based on the initially planned trajectory
        :return: The trajectory of the maneuver with respect to the initial state and environment map
        """
        pass

    @abstractmethod
    def cutting_off(self) -> State:
        """
        Detects the cut off state
        """
        pass