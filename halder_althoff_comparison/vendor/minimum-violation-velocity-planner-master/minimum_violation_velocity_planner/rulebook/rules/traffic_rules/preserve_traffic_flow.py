from typing import Dict, Tuple, Union

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from minimum_violation_velocity_planner.common.utils import getVelocityLimit
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class PreserveTrafficFlow(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, ego_lane_station_map: Dict,
                 traffic_sign_interpreter: TrafficSigInterpreter, params: Dict):
        super().__init__("PRESERVE_TRAFFIC_FLOW", "The ego shall not impede the traffic flow.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.ego_lane_station_map = ego_lane_station_map

        self.params = params.get('traffic_rules').get('preserve_traffic_flow')
        self.admissible_lane_v_max_offset = self.params.get('admissible_lane_v_max_offset')

        self.traffic_sign_interpreter = traffic_sign_interpreter

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        velocity_limit = getVelocityLimit(self.traffic_sign_interpreter, tuple(self.ego_lane_station_map[state2.s]))
        if velocity_limit is None:
            return 0.0, 0.0, None
        else:
            return max(0.0, velocity_limit - self.admissible_lane_v_max_offset - state2.v), 0.0, None
