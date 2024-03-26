from stlpy.benchmarks.base import BenchmarkScenario
from stlpy.STL.formula import STLTree
from stlpy.systems.linear import DoubleIntegrator
from stlpy.benchmarks.common import make_rectangle_patch

from comparison.micp.formula import (in_front_of_formula, in_same_lane_formula, keeps_safe_distance_formula,
                                     not_in_front_of_formula, not_in_same_lane_formula,
                                     linearized_keeps_safe_distance_formula)
from comparison.micp.constraints import InSameLaneConstraint, InFrontOfConstraint, KeepsSafeDistanceConstraint

from crmonitor.common.vehicle import Vehicle, Lane


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
                0, 2, 6,  self.ego_vehicle.shape.length, 2.578
            )
            subformula_list.append(
                not_in_front_of | not_in_same_lane | keeps_safe_distance
            )
        formula = STLTree(subformula_list, "and", time_interval)
        formula.name = "RG1"
        return formula

    def GetSystem(self):
        sys = DoubleIntegrator(2)
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

