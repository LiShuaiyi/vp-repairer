from stlpy.benchmarks.base import BenchmarkScenario

from comparison.micp.formula import (in_front_of_formula, in_same_lane_formula, keeps_safe_distance_formula)


class RG1(BenchmarkScenario):
    """Rule RG1"""
    def __init__(self, T):
        self.T: int = T

    def GetSpecification(self):

        time_interval = [t for t in range(0, self.T)]
        subformula_list = []



    def GetSystem(self):
        pass

    def add_to_plot(self, ax):
        pass

