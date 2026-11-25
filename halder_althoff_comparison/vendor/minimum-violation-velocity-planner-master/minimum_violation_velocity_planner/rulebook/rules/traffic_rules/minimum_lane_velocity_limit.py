from typing import Dict, Tuple, Union

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from minimum_violation_velocity_planner.common.utils import getVelocityLimit
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class MinimumLaneVelocityLimit(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, ego_lane_station_map: Dict,
                 traffic_sign_interpreter: TrafficSigInterpreter):
        super().__init__("MINIMUM_LANE_VELOCITY_LIMIT", "The ego shall drive faster than the lane min speed limit.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.ego_lane_station_map = ego_lane_station_map
        self.traffic_sign_interpreter = traffic_sign_interpreter

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        velocity_limit = getVelocityLimit(self.traffic_sign_interpreter, tuple(self.ego_lane_station_map[state2.s]))
        if velocity_limit is None:
            return 0.0, 0.0, None
        else:
            return max(0.0, velocity_limit - state2.v), 0.0, None
