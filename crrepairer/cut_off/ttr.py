from abc import ABC
from typing import Iterable, Union, List
import math

import numpy as np
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world_state import WorldState
from cut_off.base import CutOffBase
from cut_off.monitor import RuleMonitor
from cut_off.utils import update_ego_vehicle, visualize_state_list, int_round
from cut_off.simulation import CutOffAction, SimulationLateral, SimulationLong


class TTR(CutOffBase, ABC):
    """
    Time-To-React.
    """
    def __init__(self,
                 scenario: Scenario,
                 ego_vehicle_cr: DynamicObstacle,
                 dT: float = 0.1):
        super().__init__(scenario, ego_vehicle_cr, dT)
        # calculate the time-to-collision as default value
        self._ttc = self._calc_ttc(ego_vehicle_cr.prediction.trajectory.state_list)
        self._visualize = True

    @property
    def ttc(self):
        return self._ttc

    def generate(self, emergency_maneuvers):
        """
        Computes the time-to-react(TTR).
        :param emergency_maneuvers: the given set of emergency maneuvers
        :return: TTR, corresponding maneuver
        """
        # time to execute certain evasive maneuver
        ttm = dict()
        if self._ttc == 0:
            return -math.inf
        elif self._ttc == math.inf:
            return math.inf
        else:
            for maneuver in emergency_maneuvers:
                ttm[maneuver] = self.search_ttm(maneuver)
            return int_round(max(ttm.values()), 1) #, max(ttm, key=ttm.get)
        return ttr

    def search_ttm(self, maneuver):
        """
        Finds the TTM.
        """
        ttm = 0
        low = 0
        high = int(self._ttc / self.dT)
        while low < high:
            mid = int((low + high)/2)
            if maneuver in [CutOffAction.BRAKE, CutOffAction.KICKDOWN, CutOffAction.STEADYSPEED]:
                SL = SimulationLong(maneuver,
                                    self.ego_vehicle,
                                    mid)
            elif maneuver in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT]:
                SL = SimulationLateral(maneuver,
                                       self.ego_vehicle,
                                       mid,
                                       self.world_state)
            else:
                raise ValueError("<TTR>: given compliant maneuver {} is not supported".format(maneuver))
            state_list = SL.simulate_state_list()
            if self._visualize:
                visualize_state_list(state_list, self.scenario, SL.vehicle_dynamics.shape)
            flag_collision = self._detect_collision(state_list)  # bool value
            # if violation-free and collision-free
            if not flag_collision:
                low = mid + 1
            else:
                high = mid
        if low != 0:
            ttm = (low - 1) * self.dT
        return ttm