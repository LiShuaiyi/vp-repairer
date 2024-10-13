import numpy as np
import shapely

from typing import List

from matplotlib import pyplot as plt
from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.STL.formula import STLTree
from stlpy.systems.linear import DoubleIntegrator
from stlpy.benchmarks.common import make_rectangle_patch, inside_rectangle_formula, outside_rectangle_formula

from comparison.micp.formula import (in_front_of_formula, in_same_lane_formula, keeps_safe_distance_formula,
                                     collision_free_formula,
                                     not_in_front_of_formula, not_in_same_lane_formula, no_backwards_driving,
                                     linearized_keeps_safe_distance_formula, keeps_speed_limit,
                                     not_braking_formula, not_braking_abruptly_formula,
                                     relative_braking_abruptly_formula, inside_interval_formula,
                                     outside_interval_formula)
from comparison.micp.constraints import (InSameLaneConstraint, InFrontOfConstraint,
                                         KeepsSafeDistanceConstraint, CollisionFreeConstraint, CollisionFreeConstraintIntersection)
# from comparison.micp.vehicle_models import VehicleModel
from comparison.micp.vehicle_models_dt import VehicleModel
from crmonitor.common.vehicle import Vehicle, Lane
from crmonitor.common.world import World

from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter
from commonroad.scenario.lanelet import LaneletType, Lanelet

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
    def __init__(self, T: int, world: World, ego_vehicle: Vehicle, other_vehicle: Vehicle, lanelet_network):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.other_vehicle = other_vehicle
        self.lanelet_network = lanelet_network

        self.world = world


    def GetSpecification(self):

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

        self.collision_avoidance_constr = CollisionFreeConstraint()
        self.collision_avoidance_constr.compute(
            self.world, self.ego_vehicle, self.other_vehicle, 0, self.T
        )
        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            not_in_front_of = not_in_front_of_formula(
                self.in_front_of_constr.constraint_dict[time_step], 0, 10, self.ego_vehicle.shape.length, 2.578
            )
            # in_front_of = in_front_of_formula(
            #     self.in_front_of_constr.constraint_dict[time_step], 0, 10, self.ego_vehicle.shape.length, 2.578
            # )
            not_in_same_lane = not_in_same_lane_formula(
                self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 10
            )
            # in_same_lane = in_same_lane_formula(
            #    self.in_same_lane_constr.constraint_dict[time_step], 0, 1, 10
            # )
            # keeps_safe_distance = keeps_safe_distance_formula(
            #    self.keeps_safe_distance_constr.constraint_dict[time_step][0],
            #     self.keeps_safe_distance_constr.constraint_dict[time_step][1],
            #     0, 2, 6,  self.ego_vehicle.shape.length, 2.578
            # )
            keeps_safe_distance = linearized_keeps_safe_distance_formula(
                self.keeps_safe_distance_constr.constraint_dict[time_step][0],
                self.keeps_safe_distance_constr.constraint_dict[time_step][1],
                0, 2, 10,  self.ego_vehicle.shape.length, 2.578
            )

            collision_free = collision_free_formula(
                self.collision_avoidance_constr.constraint_dict[time_step], 0, 10,
                self.ego_vehicle.shape.length, 2.578
            )
            # not_braking = not_braking_formula(5, 10)
            # not_braking_abruptly = not_braking_abruptly_formula(5, 10)
            #
            # relative_braking_abruptly = relative_braking_abruptly_formula(
            #     self.other_vehicle.get_lon_state(time_step, self.ego_vehicle.get_lane(0)).a,
            #     5, 10
            # )

            # subformula_list.append(
            #     (not_braking | not_in_front_of | not_in_same_lane | not_braking_abruptly
            #     | (keeps_safe_distance & in_front_of & in_same_lane & relative_braking_abruptly))
            #      & (not_in_front_of | not_in_same_lane | keeps_safe_distance) & collision_free
            # )
            subformula_list.append(
                (not_in_front_of | not_in_same_lane | keeps_safe_distance) & collision_free
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


        no_backwards = no_backwards_driving(2, 10)
        keeps_speed_limit_one_step = (keeps_lane_speed_limit & keeps_type_speed_limit & no_backwards &
                                      keeps_foc_speed_limit & keeps_br_speed_limit)

        spec = keeps_speed_limit_one_step.always(0, self.T) & formula

        spec.name = "RG123"
        return spec

    def GetSystem(self):
        sys = VehicleModel(0.2)
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

class RIN1(BenchmarkScenario):
    """Rule RIN1"""
    def __init__(self, T: int, world: World, ego_vehicle: Vehicle, lanelet_network):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.lanelet_network = lanelet_network

        self.world = world

    def obtain_stop_line_s(self):

        lanelets_with_stop_line = [
            l.lanelet_id
            for l in self.lanelet_network.lanelets
            if l.stop_line is not None
        ]

        # Get the set of lanelets in the current path, that have a stop line
        intersection_lanelets = list(
            self.ego_vehicle.ref_path_lane.contained_lanelets.intersection(
                lanelets_with_stop_line
            )
        )

        stop_line_s = np.array(
            [
                min(
                    self.ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *self.lanelet_network.find_lanelet_by_id(
                            l
                        ).stop_line.start
                    )[0],
                    self.ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *self.lanelet_network.find_lanelet_by_id(
                            l
                        ).stop_line.end
                    )[0],
                )
                for l in intersection_lanelets
            ]
        )

        return stop_line_s

    def GetSpecification(self):

        self.stop_line_s = self.obtain_stop_line_s() - self.ego_vehicle.shape.length/2

        self.collision_avoidance_constr = CollisionFreeConstraintIntersection()
        self.collision_avoidance_constr.compute(
            self.world, self.ego_vehicle, 0, self.T
        )

        # inside the conflict area
        stop_line_in_front = inside_interval_formula(
            [self.stop_line_s - 1.0, self.stop_line_s], 0, 10
        )
        stop_line_in_front_not = outside_interval_formula(
            [self.stop_line_s - 1.0, self.stop_line_s], 0, 10
        )
        # passing_stop_line = stop_line_in_front & stop_line_in_front_not.eventually(0, 1)
        # spec = passing_stop_line.negation().always(0, self.T)

        # not a or not X(not a) = not a or X a
        not_passing_stop_line = stop_line_in_front_not | stop_line_in_front.eventually(1, 1)

        spec = not_passing_stop_line.always(0, self.T - 1)

        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            collision_free = collision_free_formula(
                self.collision_avoidance_constr.constraint_dict[time_step], 0, 10,
                self.ego_vehicle.shape.length, 2.578
            )
            no_backwards = no_backwards_driving(2, 10)
            if collision_free:
                subformula_list.append(collision_free & no_backwards)
            else:
                subformula_list.append(no_backwards)
        formula = STLTree(subformula_list, "and", time_interval)

        spec = spec & formula
        return spec

    def GetSystem(self):
        sys = VehicleModel(0.2)
        return sys

    def add_to_plot(self, ax):
        pass

class RIN4(BenchmarkScenario):
    """Rule RIN4"""
    def __init__(self, T: int, world: World, ego_vehicle: Vehicle, other_vehicle: Vehicle, lanelet_network):
        self.T: int = T
        self.ego_vehicle = ego_vehicle
        self.other_vehicle = other_vehicle
        self.lanelet_network = lanelet_network

        self.world = world


    def obtain_conflict_area(self):
        lanelets_dir_ego = self.ego_vehicle.lanelets_dir
        lanelets_dir_other = self.other_vehicle.lanelets_dir

        # Calculate the union of the ego lanelets and other vehicle's lanelets at intersections
        lanelets_ego_intersection = self.get_intersection_lanelets(self.lanelet_network, lanelets_dir_ego)
        lanelets_other_intersection = self.get_intersection_lanelets(self.lanelet_network, lanelets_dir_other)

        ego_intersection_region = self.get_lanelet_union(lanelets_ego_intersection)
        other_intersection_region = self.get_lanelet_union(lanelets_other_intersection)

        conflict_region_CART = ego_intersection_region.intersection(other_intersection_region)

        vehicle_length_add = self.ego_vehicle.shape.length / 2

        conflict_region_enlarged = shapely.offset_curve(
            conflict_region_CART, vehicle_length_add
        )

        conflict_region_enl_polygon = shapely.Polygon(conflict_region_enlarged)

        conflict_region_enl_clcs = [self.ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
            vertex[0], vertex[1]) for vertex in conflict_region_enl_polygon.exterior.coords]

        conflict_region_enl_clcs_polygon = shapely.Polygon(conflict_region_enl_clcs)

        # # Plotting both the Cartesian and Curvilinear conflict regions
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        #
        # # Plot Cartesian conflict region
        # ax[0].set_title("Conflict Region in Cartesian Coordinates")
        # ax[0].plot(*conflict_region_CART.exterior.xy, color='blue', label='Original Region')
        # ax[0].plot(*conflict_region_enl_polygon.exterior.xy, color='red', linestyle='--', label='Enlarged Region')
        # ax[0].set_aspect('equal', 'box')
        # ax[0].legend()
        #
        # # Plot Curvilinear conflict region
        # ax[1].set_title("Conflict Region in Curvilinear Coordinates")
        # ax[1].plot(*zip(*conflict_region_enl_clcs), color='green', label='Enlarged CLCS Region')
        # ax[1].set_aspect('equal', 'box')
        # ax[1].legend()
        #
        # # Show the plots
        # plt.show()

        return conflict_region_enl_clcs_polygon


    @staticmethod
    def get_lanelet_union(lanelets: List[Lanelet]) -> shapely.geometry.base.BaseGeometry:
        # Create a union of all lanelet geometries
        return shapely.unary_union([lanelet.polygon.shapely_object for lanelet in lanelets])

    @staticmethod
    def get_intersection_lanelets(lanelet_network, lanelet_ids):
        return [
            lanelet_network.find_lanelet_by_id(lanelet_id)
            for lanelet_id in lanelet_ids
            if LaneletType.INTERSECTION in lanelet_network.find_lanelet_by_id(lanelet_id).lanelet_type
        ]

    def GetSpecification(self):
        for _ in range(0, self.T):
            self.conflict_area = self.obtain_conflict_area()
        xmin, ymin, xmax, ymax = self.conflict_area.bounds
        self.conflict_area_bound = (xmin, xmax, ymin, ymax) # (xmin, xmax, ymin, ymax)
        print(f"Conflict Area Bounds: {self.conflict_area_bound}")

        self.collision_avoidance_constr = CollisionFreeConstraintIntersection()
        self.collision_avoidance_constr.compute(
            self.world, self.ego_vehicle, 0, self.T
        )
        # inside the conflict area
        inside_conflict_area = inside_rectangle_formula(self.conflict_area_bound, 0, 1, 10)

        # outside the conflict area
        outside_conflict_area = outside_rectangle_formula(self.conflict_area_bound, 0, 1, 10)

        # Put all the constraints together in one specification
        # vehicle 10180 stays in the conflict area from time step 9 to 13
        #     (
        #         (
        #             (in_intersection_conflict_area(a0, a1)) implies
        #         (
        #         not (causes_braking_intersection(a0, a1))
        #     and not (eventually[0, 1s](in_intersection_conflict_area(a1, a0)))
        #     )
        #     ) and
        #     (
        #         (in_intersection_conflict_area(a1, a0)) implies
        #     not (eventually[0, 0.6s](in_intersection_conflict_area(a0, a1)))
        #     )
        #     ) or not (on_lanelet_with_type_intersection(a0))
        # )
        # other_veh_in_conflict_area_time = [9, 13]
        # spec = (outside_conflict_area.always(other_veh_in_conflict_area_time[0], other_veh_in_conflict_area_time[1] + 3) &
        #         outside_conflict_area.always(other_veh_in_conflict_area_time[0] - 5, other_veh_in_conflict_area_time[0]))
        spec = (outside_conflict_area.always(0, self.T))

        time_interval = [t for t in range(0, self.T + 1)]
        subformula_list = []
        for time_step in time_interval:
            collision_free = collision_free_formula(
                self.collision_avoidance_constr.constraint_dict[time_step], 0, 10,
                self.ego_vehicle.shape.length, 2.578
            )
            no_backwards = no_backwards_driving(2, 10)
            if collision_free:
                subformula_list.append(collision_free & no_backwards)
            else:
                subformula_list.append(no_backwards)
        formula = STLTree(subformula_list, "and", time_interval)
        spec = spec & formula
        return spec

    def GetSystem(self):
        sys = VehicleModel(0.2)
        return sys

    def add_to_plot(self, ax):
        pass