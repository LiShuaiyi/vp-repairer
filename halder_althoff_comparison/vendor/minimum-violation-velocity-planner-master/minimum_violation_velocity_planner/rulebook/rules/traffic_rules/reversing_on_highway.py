import functools
from typing import Dict, Tuple, Union

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletType
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class ReversingOnHighway(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, ego_lane_station_map: Dict,
                 params: Dict):
        super().__init__("REVERSING_ON_HIGHWAY", "The ego shall not reverse on highways.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy
        self.ego_lane_station_map = ego_lane_station_map

        self.params = params.get('traffic_rules').get('reversing_on_highway')
        self.v_reversing_threshold = self.params.get('v_reversing_threshold')

        self.lanelet_types = {LaneletType.HIGHWAY,
                              LaneletType.ACCESS_RAMP,
                              LaneletType.EXIT_RAMP,
                              LaneletType.SHOULDER,
                              LaneletType.INTERSTATE}

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # If velocity is in stop interval
        if self.isEgoOnLanelet(state2.s) and state2.v < self.v_reversing_threshold:
            return 1.0, 0.0, None
        return 0.0, 0.0, None

    @functools.lru_cache(maxsize=None)
    def isEgoOnLanelet(self, station):
        for lanelet_id in self.ego_lane_station_map[station]:
            # If one of the lanelets is of the respective types
            lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            nof_same_lanelet_types = len(self.lanelet_types.intersection(lanelet.lanelet_type))
            if nof_same_lanelet_types > 0:
                return True
        return False
