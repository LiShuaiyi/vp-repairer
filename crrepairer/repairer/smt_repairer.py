from abc import ABC
from enum import Enum
import math

from commonroad.scenario.scenario import DynamicObstacle

from commonroad_repair.crrepairer.repairer.base import TrajectoryRepair
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.sat_solver.sat_solver import SATSolver
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver

from z3 import sat, unsat
from enum import Enum


class RepairingRule(Enum):
    COLLISION_FREE = "collision free"
    R_G1 = "R_G1"
    R_G2 = "R_G2"
    R_G3 = "R_G3"


class SMTTrajectoryRepairer(TrajectoryRepair, ABC):
    def __init__(self,
                 rule_abstracter: RuleAbstracter,
                 ego_vehicle: DynamicObstacle):
        super().__init__(ego_vehicle.prediction.trajectory)
        self.rule_abstracter = rule_abstracter


    def repair(self, *args, **kwargs):
        if self.rule_abstracter.rule_monitor.tv_time_step == -math.inf:
            return None
        sat_solver = SATSolver(self.rule_abstracter)
        t_solver = TSolver(self.rule_abstracter)
        while sat_solver.solve() == sat:
            if self.rule_abstracter.propositions is None:
                return None
            select_proposition, model = sat_solver.model()
            repairability, repaired_traj = t_solver.check(select_proposition, list(model))
            if repairability:
                return repaired_traj
            else:
                sat_solver.update_formula()
                # todo: check feasibility
        return None


