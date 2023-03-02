import numpy as np

from collections import defaultdict

from crrepairer.cut_off.simulation import CutOffAction
from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.position import (PredSafeDistPrec, PredInSameLane,
                                           PredInFrontOf, PredPreceding)

from crmonitor.predicates.velocity import (PredLaneSpeedLimit, PredFovSpeedLimit,
                                           PredBrSpeedLimit, PredTypeSpeedLimit)
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import (PredAbruptBreaking, PredRelAbruptBreaking)

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle

from typing import List

from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints
from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory

from commonroad.scenario.trajectory import Trajectory

# specification-compliant reachable set


class RuleConstraintsReach:
    """
    Class for traffic rule constraints (manual definition)
    """

    def __init__(self,
                 tc_object: TC,
                 rule_monitor: STLRuleMonitor,
                 sel_proposition_full: List[PropositionNode],
                 proposition_full: List[PropositionNode],
                 veh_config: PlanningConfigurationVehicle,
                 initial_trajectory: Trajectory):
        # initialize the needed components
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world

        # ego vehicle
        self._ego_id = self._rule_monitor.vehicle_id  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory

        # other vehicle (rule-relevant)
        self._other_id = self._rule_monitor.other_id
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)

        # configuration
        self._veh_config = veh_config
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full
        self._prop_full = proposition_full

        # initialize the elements for rule constraints (from reach)
        # longitudinal components
        self._lon_dis_constraints = list()
        self._lon_vel_constraints = list()
        self._lon_acc_constraints = list()
        # lateral components
        self._lat_dis_constraints = list()
        self._lat_vel_constraints = list()
        self._lat_acc_constraints = list()

    def initialize_reach_interface(self):
        pass