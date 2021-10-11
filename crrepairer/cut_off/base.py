from abc import ABC, abstractmethod

# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState


class CutOffBase(ABC):
    """
        Abstract base class for calculating cut-off states
    """
    def __init__(self, world_state: WorldState,
                 dT: float):
        self._world_state = world_state
        self._dT = dT

    @property
    def world_state(self) -> float:
        return self._world_state

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