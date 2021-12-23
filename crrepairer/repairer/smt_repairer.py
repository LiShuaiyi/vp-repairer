from abc import ABC
from enum import Enum

from commonroad.scenario.scenario import DynamicObstacle

from commonroad_repair.crrepairer.repairer.base import TrajectoryRepair
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.sat_solver.sat_solver import SATSolver, SATISFIABILITY
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver


class RepairingRule(Enum, str):
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
        self.sat_solver = SATSolver(rule_abstracter.sat_encoding,
                                    rule_abstracter.propositions,
                                    rule_abstracter.prop_robust_ttv)
        self.t_solver = TSolver(rule_abstracter.rule_monitor)

    def repair(self, *args, **kwargs):
        while self.sat_solver.solve() == SATISFIABILITY.SAT:
            select_proposition = self.sat_solver.model()
            repairability, repaired_traj = self.t_solver.check(select_proposition)
            if repairability:
                return repaired_traj
            else:
                # todo: use propositions with dpll
                self.sat_solver.update_formula(select_proposition)
                # todo: check feasibility
        return None


