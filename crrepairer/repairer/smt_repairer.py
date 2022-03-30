from abc import ABC
import math

from commonroad.scenario.scenario import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import TrajectoryPrediction, ObstacleType

from commonroad_repair.crrepairer.repairer.base import TrajectoryRepair
from commonroad_repair.crrepairer.monitor.monitor_wrapper import STLRuleMonitor
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
                 rule_monitor: STLRuleMonitor,
                 ego_vehicle: DynamicObstacle):
        super().__init__(ego_vehicle.prediction.trajectory)
        self.rule_monitor = rule_monitor
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

    def repair(self, check_flag=True, *args, **kwargs):
        self._tv = self.rule_monitor.tv_time_step
        if self._tv == -math.inf:
            return None
        # initialize Solvers for SMT paradigm
        sat_solver = SATSolver(self.rule_monitor)
        t_solver = TSolver(self.rule_monitor)
        while sat_solver.solve() == sat:
            if self.rule_monitor.proposition_nodes is None:
                return None
            select_proposition, self._model = sat_solver.model()
            repairability, repaired_traj = t_solver.check(select_proposition, list(self._model))
            self._tc = t_solver.tc_object.tc_time_step
            if repairability and repaired_traj is not None:
                tv, _ = t_solver.tc_object.calc_tv_updated(repaired_traj.state_list)
                if tv == math.inf or not check_flag:
                    return repaired_traj
                else:
                    print("<Repairer>: reparable but the solver failed")
            sat_solver.update_formula()
        return None

    @staticmethod
    def convert_traj_to_ego_vehicle(shape,
                                    initial_state,
                                    cr_trajectory: Trajectory,
                                    vehicle_id: int = 0) -> DynamicObstacle:
        """
        Converts trajectory object to CommonRoad obstacle with specified width and length
        :param width: The width of the ego vehicle
        :param length: The length of the ego vehicle
        :param vehicle_id: ID of ego vehicle
        :return: The CommonRoad DynamicObstacle object containing the current trajectory
        """
        # get trajectory
        pred = TrajectoryPrediction(cr_trajectory, shape)

        # create new object
        ego = DynamicObstacle(obstacle_id=vehicle_id,
                              obstacle_type=ObstacleType.CAR,
                              prediction=pred,
                              obstacle_shape=shape,
                              initial_state=initial_state)
        return ego

