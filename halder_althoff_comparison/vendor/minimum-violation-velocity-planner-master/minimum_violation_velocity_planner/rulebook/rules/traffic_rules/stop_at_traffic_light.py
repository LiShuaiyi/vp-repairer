import logging
from typing import Dict, List, Tuple, Union

from commonroad.scenario.lanelet import StopLine
from commonroad.scenario.traffic_sign import TrafficLight, TrafficLightState
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import determineStationOfStopLine, getIntervalOverlap
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule

from commonroad.scenario.scenario import Scenario

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class StopAtTrafficLight(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict,
                 vehicle_parameters: VehicleParameters, params: Dict):
        super().__init__("STOP_AT_TRAFFIC_LIGHT", "The ego shall stop at red traffic lights.")

        self.scenario = scenario
        self.ccosy = ccosy
        self.time_step_map = time_step_map

        self.vehicle_parameters = vehicle_parameters
        self.params = params.get('traffic_rules').get('stop_at_traffic_light')

        self.traffic_light_tuples = self.getTrafficLights()

        if len(self.traffic_light_tuples) > 1:
            raise Warning('There are several traffic lights on our route. Currently, we can only handle one!')

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # TODO: Extend for multiple traffic lights

        if len(self.traffic_light_tuples) > 0:
            # We only consider ONE traffic light!
            traffic_light_tuple = self.traffic_light_tuples[0]

            # if stopline was already crossed, or will be crossed
            if transfer_value == 1 or self.stopLineIsCrossedDuringRedPhase(state1, state2, traffic_light_tuple):
                s_min, _, traffic_light = traffic_light_tuple

                # If traffic light phase is red, return distance between ego center and s_min as violation -> The more you
                # drive over the stop line (= into the junction), the worse it is. Note: We use the time of state1 since this
                # is the phase which holds during the transition from state1 to state2
                time_step = self.time_step_map[state1.t]
                if traffic_light.get_state_at_time_step(time_step) == TrafficLightState.RED:
                    return abs(state2.s - s_min), 0.0, 1
        return 0.0, 0.0, 0


    def getTrafficLights(self) -> List[Tuple]:
        traffic_lights = []

        for lanelet in self.scenario.lanelet_network.lanelets:
            # If there exists a stop line
            if lanelet.stop_line is not None:
                referenced_traffic_light = self.getReferencedTrafficLight(lanelet.stop_line)
                # If there exists a referenced traffic light
                if referenced_traffic_light is not None:
                    # determine s-coordinate
                    s = determineStationOfStopLine(lanelet.stop_line, self.ccosy)
                    # Shift s by half ego length, because ego ref point is at center
                    if s is not None:
                        # Tuple: (s_min, s_max, traffic_light)
                        traffic_lights.append((s - self.vehicle_parameters.l/2, s, referenced_traffic_light))

        return traffic_lights

    def getReferencedTrafficLight(self, stop_line: StopLine) -> Union[None, TrafficLight]:
        if stop_line.traffic_light_ref is not None:
            # TODO: Can there be really more than one traffic lights?
            if len(stop_line.traffic_light_ref) > 1:
                logging.warning('Multiple traffic lights are assigned for one stop line. We can not handle this yet.')

            # Attention: We return the first traffic light!
            return self.scenario.lanelet_network.find_traffic_light_by_id(next(iter(stop_line.traffic_light_ref)))
        return None

    def stopLineIsCrossedDuringRedPhase(self, state1: LongitudinalState, state2: LongitudinalState,
                                        traffic_light_tuple: Tuple) -> bool:
        s_min, s_max, traffic_light = traffic_light_tuple

        # Note: We use the time of state1 since this is the phase which holds during the transition
        # from state1 to state2
        time_step = self.time_step_map[state1.t]
        if traffic_light.get_state_at_time_step(time_step) == TrafficLightState.RED:
            # If ego overlaps with stop line area
            if getIntervalOverlap((s_min, s_max), (state1.s, state2.s)) != 0.0:
                return True
        return False
