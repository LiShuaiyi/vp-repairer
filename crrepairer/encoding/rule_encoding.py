from typing import List, Dict, Union, Iterable

from encoding.monitor import RuleMonitor, MonitorType
# CommonRoad STL monitor
from crmonitor.common.world_state import WorldState

# CommonRoad Toolbox
from commonroad.scenario.obstacle import DynamicObstacle, Shape
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario


class RuleEncoder:
    """
    Wrapper class to wrap rule monitors with evaluation functionalities
    """
    def __init__(self,
                 scenario: Scenario,
                 vehicle_id: int,
                 rule_str: Union[str, Iterable[str]],
                 monitor_type: MonitorType = MonitorType.STL):
        self._world_state = self.construct_world_state(scenario, vehicle_id)
        self._rule_monitor = RuleMonitor(self._world_state, rule_str, monitor_type)
        self._prop_abs = self.construct_prop_abs()  #propositional abstraction
        self._abs_robustness = self._rule_monitor.rob_abstraction

    def construct_world_state(self,
                              scenario: Scenario,
                              ego_id: int) -> WorldState:
        world_state = WorldState.create_from_scenario(scenario, ego_id)
        return world_state

    def construct_prop_abs(self):
        return self._rule_monitor.abstraction_nodes

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def ego_vehicle(self) -> DynamicObstacle:
        return self._ego_vehicle

    @property
    def prop_abs(self):
        return self._prop_abs

    @property
    def abs_robustness(self):
        return self._abs_robustness

