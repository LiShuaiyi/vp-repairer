from abc import ABC
import math

from commonroad.scenario.scenario import DynamicObstacle

from commonroad_repair.crrepairer.repairer.base import TrajectoryRepair
from commonroad_repair.crrepairer.abstraction.abstracter import RuleAbstracter
from commonroad_repair.crrepairer.sat_solver.sat_solver import SATSolver
from commonroad_repair.crrepairer.t_solver.t_solver import TSolver

from z3 import sat
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
        self._model = None
        self._tc = -math.inf
        self._tv = -math.inf

    @property
    def tv(self):
        return self._tv

    @property
    def tc(self):
        return self._tc

    @property
    def model(self):
        """
        SAT models
        """
        return self._model

    def repair(self, *args, **kwargs):
        self._tv = self.rule_abstracter.rule_monitor.tv_time_step
        if self._tv == -math.inf:
            return None
        # initialize Solvers for SMT paradigm
        sat_solver = SATSolver(self.rule_abstracter)
        t_solver = TSolver(self.rule_abstracter)
        while sat_solver.solve() == sat:
            if self.rule_abstracter.propositions is None:
                return None
            select_proposition, self._model = sat_solver.model()
            repairability, repaired_traj = t_solver.check(select_proposition, list(self._model))
            self._tc = t_solver.tc_object.tc_time_step
            if repairability:
                tv, _ = t_solver.tc_object.calc_tv_updated(repaired_traj.state_list)
                if tv == math.inf:
                    return repaired_traj
                else:
                    print("<Repairer>: reparable but the solver failed")
            sat_solver.update_formula()
        return None
