from repairer.abstract import TrajectoryRepairer

from typing import List, Dict, Union, Iterable
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from collections import defaultdict

# commonroad-io
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

# trajectory planning tools
from optimizer.constraints import LonConstraints, LatConstraints
from optimizer.configuration import PlanningConfigurationVehicle
from optimizer.trajectory import Trajectory, TrajPoint, TrajectoryType
from optimizer.qp_lat_planner import QPLatPlanner, QPLatReference, QPLatState, QPLatPARAMS
from optimizer.qp_long_planner import QPLongPlanner, QPLongReference, QPLongState, QPLongPARAMS
from optimizer.constraints import TIConstraints

from cut_off.tc import TC
from cut_off.simulation import CutOffAction


class RuleRepairer(TrajectoryRepairer):
    def __init__(self,
                 scenario: Scenario,
                 vehicle_id: int,
                 rule: Union[str, Iterable[str]],
                 ):
        super().__init__(vehicle_id)
        self.scenario = scenario
        if not hasattr(self.scenario, 'dt'):
            self.dt = 0.1  # default time step
        else:
            self.dt = scenario.dt
        self.ego_vehicle = self.scenario.obstacle_by_id(vehicle_id)
        if self.ego_vehicle is None:
            raise ValueError('<RuleRepairer>: the given vehicle id {} is not existed.'.format(vehicle_id))
        self.rule = rule

    def repair(self):
        print('\t\t -------- Trajectory Repairing --------')
        cut_off_state = 0

    def cutting_off(self, action: CutOffAction) -> State:
        """
        Computes cut-off time step and the corresponding cut-off state based on the action.
        """
        ttcc_obj = TC(self.scenario,
                      self.ego_vehicle,
                      self.rule)
        ttcc = ttcc_obj.generate(action)
        print('\t\t\t Time-to-compliance: {}s \n'.format(ttcc))
        return self.ego_vehicle.state_at_time(int(ttcc/self.dt))


