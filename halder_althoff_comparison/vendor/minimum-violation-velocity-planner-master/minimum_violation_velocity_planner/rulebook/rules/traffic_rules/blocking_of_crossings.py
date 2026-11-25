from typing import Dict, List, Tuple, Union

from commonroad.scenario.lanelet import LaneletType, RoadUser
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.utils import getIntervalOverlap, determineStationRangeOfLanelet
from minimum_violation_velocity_planner.rulebook.rules.rule import Rule
from commonroad.scenario.scenario import Scenario

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState


class BlockingOfCrossings(Rule):
    def __init__(self, scenario: Scenario, ccosy: CurvilinearCoordinateSystem, time_step_map: Dict,
                 vehicle_parameters: VehicleParameters, params: Dict, time_horizon: List):
        super().__init__("BLOCKING_OF_CROSSINGS", "The ego shall not block intersections, pedestrian crossings or train crossings.", is_path_independent=True)

        self.scenario = scenario
        self.ccosy = ccosy

        self.time_step_map = time_step_map

        self.vehicle_params = vehicle_parameters
        self.params = params.get('traffic_rules').get('blocking_of_crossings')
        self.velocity_threshold = self.params.get('velocity_threshold')

        self.time_horizon = time_horizon[1]

        self.intersection_intervals = self.determineIntersectionStationInterval()
        self.pedestrian_crossing_intervals = self.determinePedestrianCrossingStationInterval()
        self.railroad_crossing_intervals = self.determineRailroadCrossingStationInterval()

    def determineViolation(self, state1: LongitudinalState, state2: LongitudinalState, transfer_value=None) -> Tuple[float, float, Union[None, bool, int, float]]:
        # ToDO: What was the reason for this?
        #if self.time_step_map[state2.t] == self.time_horizon:
        #    return 999.9, 0.0, None

        ego_interval = (state2.s - self.vehicle_params.l / 2, state2.s + self.vehicle_params.l / 2)
        instant_violations = [0.0]

        # For each intersection
        for intersection_interval in self.intersection_intervals:
            overlap = getIntervalOverlap(intersection_interval, ego_interval)
            if overlap > 0.0 and state2.v < self.velocity_threshold:
                instant_violations.append(overlap)
        # For each pedestrian crossing
        for pedestrian_crossing_interval in self.pedestrian_crossing_intervals:
            overlap = getIntervalOverlap(pedestrian_crossing_interval, ego_interval)
            if overlap > 0.0 and state2.v < self.velocity_threshold:
                instant_violations.append(overlap)
        # For each railroad crossing
        for railroad_crossing_interval in self.railroad_crossing_intervals:
            overlap = getIntervalOverlap(railroad_crossing_interval, ego_interval)
            if overlap > 0.0 and state2.v < self.velocity_threshold:
                instant_violations.append(overlap)

        return max(instant_violations), 0.0, None

    def determineIntersectionStationInterval(self) -> List:
        station_intervals = []

        for intersection in self.scenario.lanelet_network.intersections:
            # Get all lanelets inside the intersection
            intersection_lanelets = set()
            for incoming in intersection.incomings:
                intersection_lanelets.update(incoming.successors_straight)
                intersection_lanelets.update(incoming.successors_left)
                intersection_lanelets.update(incoming.successors_right)

            station_values = []
            for lanelet_id in list(intersection_lanelets):
                lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                station_interval = determineStationRangeOfLanelet(lanelet, self.ccosy)
                # We only consider the lanelet, if we have an intersection with the left and right boundary!
                if station_interval:
                    station_values.extend(station_interval)

            if len(station_values) > 0:
                station_intervals.append((min(station_values), max(station_values)))

        return station_intervals

    def determinePedestrianCrossingStationInterval(self) -> List:
        station_intervals = []

        for lanelet in self.scenario.lanelet_network.lanelets:
            if LaneletType.CROSSWALK in lanelet.lanelet_type:
                station_interval = determineStationRangeOfLanelet(lanelet, self.ccosy)
                # We only consider the lanelet, if we have an intersection with the left and right boundary!
                if station_interval:
                    station_intervals.append(station_interval)

        return station_intervals

    def determineRailroadCrossingStationInterval(self) -> List:
        station_intervals = []

        for lanelet in self.scenario.lanelet_network.lanelets:
            if RoadUser.TRAIN in lanelet.user_bidirectional or RoadUser.TRAIN in lanelet.user_one_way:
                station_interval = determineStationRangeOfLanelet(lanelet, self.ccosy)
                # We only consider the lanelet, if we have an intersection with the left and right boundary!
                if station_interval:
                    station_intervals.append(station_interval)

        return station_intervals
