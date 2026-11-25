import functools
from typing import Dict, List, Tuple, Union

from commonroad.scenario.lanelet import StopLine
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import determineStationOfStopLine
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class StopAtStopSign(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, vehicle_parameters: VehicleParameters,
                 params: Dict):
        super().__init__("STOP_AT_STOP_SIGN", "The ego shall stop at stop signs.")

        self.scenario = scenario
        self.ccosy = ccosy

        self.vehicle_parameters = vehicle_parameters
        self.params = params.get('traffic_rules').get('stop_at_stop_sign')
        self.v_stop_threshold = self.params.get('v_stop_threshold')

        self.stop_line_stations = self.getStopLineStations()

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float, List]]:
        instant_violations = [0.0]

        for stop_line_station in self.stop_line_stations:
            if state1.s < stop_line_station <= state2.s:
                instant_violations.append(max([transfer_value, state2.v]) - self.v_stop_threshold)  # TODO: THis only condisers the last time step, make this a variable intervall!!

        return max(instant_violations), 0.0, state2.v

    def getStopLineStations(self) -> List[float]:
        s_coordinates = []

        for lanelet in self.scenario.lanelet_network.lanelets:
            # If there exists a stop line, and it is referenced with a stop sign
            if lanelet.stop_line is not None and self.existsReferencedStopSign(lanelet.stop_line):
                # determine s-coordinate
                s = determineStationOfStopLine(lanelet.stop_line, self.ccosy)
                # Shift s by half ego length, because ego ref point is at center
                if s is not None:
                    s_coordinates.append(s - self.vehicle_parameters.l/2)

        return s_coordinates

    @functools.lru_cache(maxsize=None)
    def existsReferencedStopSign(self, stop_line: StopLine) -> bool:
        if stop_line.traffic_sign_ref is not None:
            for sign_ref in stop_line.traffic_sign_ref:
                for traffic_sign_element in self.scenario.lanelet_network.find_traffic_sign_by_id(sign_ref).traffic_sign_elements:
                    if traffic_sign_element.traffic_sign_element_id == TrafficSignIDGermany.STOP:
                        return True
        return False
