from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.STL.formula import STLTree
from stlpy.systems.linear import DoubleIntegrator
from stlpy.benchmarks.common import make_rectangle_patch

from comparison.micp.formula import (in_front_of_formula, in_same_lane_formula, keeps_safe_distance_formula,
                                     not_in_front_of_formula, not_in_same_lane_formula, no_backwards_driving,
                                     linearized_keeps_safe_distance_formula, keeps_speed_limit,
                                     not_braking_formula, not_braking_abruptly_formula, relative_braking_abruptly_formula)
from comparison.micp.constraints import InSameLaneConstraint, InFrontOfConstraint, KeepsSafeDistanceConstraint
from comparison.micp.vehicle_models import VehicleModel
from crmonitor.common.vehicle import Vehicle, Lane

from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter


class RG1(BenchmarkScenario):
    """Rule RG1"""
    def __init__(self, T: int, ego_vehicle: Vehicle, other_vehicle: Vehicle):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.other_vehicle = other_vehicle

        self.in_same_lane_constr = InSameLaneConstraint()
        self.in_same_lane_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.in_front_of_constr = InFrontOfConstraint()
        self.in_front_of_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.keeps_safe_distance_constr = KeepsSafeDistanceConstraint()
        self.keeps_safe_distance_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

    def GetSpecification(self):

        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            not_in_front_of = not_in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 6, self.ego_vehicle.shape.length, 2.578
            )
            not_in_same_lane = not_in_same_lane_formula(
                self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 6
            )
            # in_same_lane = in_same_lane_formula(
            #     self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 6
            # )
            # keeps_safe_distance = keeps_safe_distance_formula(
            #     self.keeps_safe_distance_constr.constraint_dict[time_step][0],
            #     self.keeps_safe_distance_constr.constraint_dict[time_step][1],
            #     0, 2, 6,  self.ego_vehicle.shape.length, 2.578
            # )
            keeps_safe_distance = linearized_keeps_safe_distance_formula(
                self.keeps_safe_distance_constr.constraint_dict[time_step][0],
                self.keeps_safe_distance_constr.constraint_dict[time_step][1],
                0, 2, 10,  self.ego_vehicle.shape.length, 2.578
            )
            subformula_list.append(
                not_in_front_of | not_in_same_lane | keeps_safe_distance
            )
        formula = STLTree(subformula_list, "and", time_interval)
        formula.name = "RG1"
        return formula

    def GetSystem(self):
        sys = DoubleIntegrator(4)
        return sys

    def add_to_plot(self, ax):
        boundary = make_rectangle_patch(
            * self.in_same_lane_constr.constraint_dict[0], facecolor="white", edgecolor='black'
        )
        time_interval = [t for t in range(0, self.T + 1)]
        ax.add_patch(boundary)

        for i in time_interval:
            s_other = self.other_vehicle.get_lon_state(i, self.ego_vehicle.get_lane(0)).s
            obstacle = make_rectangle_patch(
            *(s_other, s_other+2, -1, 1), color="blue", alpha=0.1
        )
            ax.add_patch(obstacle)
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')


class RG2(BenchmarkScenario):
    """Rule RG2"""
    def __init__(self, T: int, ego_vehicle: Vehicle, other_vehicle: Vehicle):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.other_vehicle = other_vehicle

        self.in_same_lane_constr = InSameLaneConstraint()
        self.in_same_lane_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.in_front_of_constr = InFrontOfConstraint()
        self.in_front_of_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.keeps_safe_distance_constr = KeepsSafeDistanceConstraint()
        self.keeps_safe_distance_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

    def GetSpecification(self):
        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            not_in_front_of = not_in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 10, self.ego_vehicle.shape.length, 2.578
            )
            in_front_of = in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 10, self.ego_vehicle.shape.length, 2.578
            )
            not_in_same_lane = not_in_same_lane_formula(
                self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 10
            )
            in_same_lane = in_same_lane_formula(
               self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 10
            )

            keeps_safe_distance = linearized_keeps_safe_distance_formula(
                self.keeps_safe_distance_constr.constraint_dict[time_step][0],
                self.keeps_safe_distance_constr.constraint_dict[time_step][1],
                0, 2, 10,  self.ego_vehicle.shape.length, 2.578
            )
            not_braking = not_braking_formula(5, 10)
            not_braking_abruptly = not_braking_abruptly_formula(5, 10)

            relative_braking_abruptly = relative_braking_abruptly_formula(
                self.other_vehicle.get_lon_state(time_step, self.ego_vehicle.get_lane(0)).a,
                5, 10
            )
            subformula_list.append(
                not_braking | not_in_front_of | not_in_same_lane | not_braking_abruptly
                | (keeps_safe_distance & in_front_of & in_same_lane & relative_braking_abruptly)
            )
        formula = STLTree(subformula_list, "and", time_interval)
        formula.name = "RG2"
        return formula

    def GetSystem(self):
        sys = VehicleModel()
        return sys

    def add_to_plot(self, ax):
        pass



class RG3(BenchmarkScenario):
    """Rule RG3"""
    def __init__(self, T: int, ego_vehicle: Vehicle, lanelet_network):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.lanelet_network = lanelet_network

    def GetSpecification(self):

        # lane speed limit
        country = SupportedTrafficSignCountry.GERMANY
        lanelet_ids = self.ego_vehicle.lanelet_assignment[0]
        ts_interpreter = TrafficSignInterpreter(
            country, self.lanelet_network
        )
        lane_speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        if lane_speed_limit is None:
            lane_speed_limit = 60
        keeps_lane_speed_limit = keeps_speed_limit(lane_speed_limit, 2, 10)

        # type speed_limit
        type_speed_limit = 60
        keeps_type_speed_limit = keeps_speed_limit(type_speed_limit, 2, 10)

        # FOV speed limit
        fov_speed_limit = 50
        keeps_foc_speed_limit = keeps_speed_limit(fov_speed_limit, 2, 10)

        # braking speed limit
        br_speed_limit = 43
        keeps_br_speed_limit = keeps_speed_limit(br_speed_limit, 2, 10)

        keeps_speed_limit_one_step = (keeps_lane_speed_limit & keeps_type_speed_limit &
                                      keeps_foc_speed_limit & keeps_br_speed_limit)

        spec = keeps_speed_limit_one_step.always(0, self.T)
        spec.name = "RG2"

        return spec

    def GetSystem(self):
        sys = DoubleIntegrator(2)
        return sys

    def add_to_plot(self, ax):
        pass


class RG123(BenchmarkScenario):
    """Rule RG1-3"""
    def __init__(self, T: int, ego_vehicle: Vehicle, other_vehicle: Vehicle, lanelet_network):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.other_vehicle = other_vehicle
        self.lanelet_network = lanelet_network

        self.in_same_lane_constr = InSameLaneConstraint()
        self.in_same_lane_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.in_front_of_constr = InFrontOfConstraint()
        self.in_front_of_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

        self.keeps_safe_distance_constr = KeepsSafeDistanceConstraint()
        self.keeps_safe_distance_constr.compute(
            self.ego_vehicle, self.other_vehicle, 0, self.T
        )

    def GetSpecification(self):

        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            not_in_front_of = not_in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 8, self.ego_vehicle.shape.length, 2.578
            )
            in_front_of = in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 8, self.ego_vehicle.shape.length, 2.578
            )
            not_in_same_lane = not_in_same_lane_formula(
                self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 8
            )
            in_same_lane = in_same_lane_formula(
               self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 8
            )
            # keeps_safe_distance = keeps_safe_distance_formula(
            #    self.keeps_safe_distance_constr.constraint_dict[time_step][0],
            #     self.keeps_safe_distance_constr.constraint_dict[time_step][1],
            #     0, 2, 6,  self.ego_vehicle.shape.length, 2.578
            # )
            keeps_safe_distance = linearized_keeps_safe_distance_formula(
                self.keeps_safe_distance_constr.constraint_dict[time_step][0],
                self.keeps_safe_distance_constr.constraint_dict[time_step][1],
                0, 2, 8,  self.ego_vehicle.shape.length, 2.578
            )
            not_braking = not_braking_formula(5, 8)
            not_braking_abruptly = not_braking_abruptly_formula(5, 8)

            relative_braking_abruptly = relative_braking_abruptly_formula(
                self.other_vehicle.get_lon_state(time_step, self.ego_vehicle.get_lane(0)).a,
                5, 8
            )
            subformula_list.append(
                (not_braking | not_in_front_of | not_in_same_lane | not_braking_abruptly
                | (keeps_safe_distance & in_front_of & in_same_lane & relative_braking_abruptly))
                 & (not_in_front_of | not_in_same_lane | keeps_safe_distance)
            )
        formula = STLTree(subformula_list, "and", time_interval)

        # lane speed limit
        country = SupportedTrafficSignCountry.GERMANY
        lanelet_ids = self.ego_vehicle.lanelet_assignment[0]
        ts_interpreter = TrafficSignInterpreter(
            country, self.lanelet_network
        )
        lane_speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        if lane_speed_limit is None:
            lane_speed_limit = 60
        keeps_lane_speed_limit = keeps_speed_limit(lane_speed_limit, 2, 8)

        # type speed_limit
        type_speed_limit = 60
        keeps_type_speed_limit = keeps_speed_limit(type_speed_limit, 2, 8)

        # FOV speed limit
        fov_speed_limit = 50
        keeps_foc_speed_limit = keeps_speed_limit(fov_speed_limit, 2, 8)

        # braking speed limit
        br_speed_limit = 43
        keeps_br_speed_limit = keeps_speed_limit(br_speed_limit, 2, 8)


        no_backwards = no_backwards_driving(2, 8)
        keeps_speed_limit_one_step = (keeps_lane_speed_limit & keeps_type_speed_limit & no_backwards &
                                      keeps_foc_speed_limit & keeps_br_speed_limit)

        spec = keeps_speed_limit_one_step.always(0, self.T)

        spec.name = "RG123"
        return spec

    def GetSystem(self):
        sys = VehicleModel()
        return sys

    def add_to_plot(self, ax):
        boundary = make_rectangle_patch(
            * self.in_same_lane_constr.constraint_dict[0], facecolor="white", edgecolor='black'
        )
        time_interval = [t for t in range(0, self.T + 1)]
        ax.add_patch(boundary)

        for i in time_interval:
            s_other = self.other_vehicle.get_lon_state(i, self.ego_vehicle.get_lane(0)).s
            obstacle = make_rectangle_patch(
            *(s_other, s_other+2, -1, 1), color="blue", alpha=0.1
        )
            ax.add_patch(obstacle)
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')