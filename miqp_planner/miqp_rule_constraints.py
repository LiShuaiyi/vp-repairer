import numpy as np
from typing import List
from collections import defaultdict

from commonroad.scenario.trajectory import Trajectory

from commonroad_crime.utility.simulation import Maneuver

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
    PredPreceding,
    PredStopLineInFront,
)

from crmonitor.predicates.velocity import (
    PredLaneSpeedLimit,
    PredFovSpeedLimit,
    PredBrSpeedLimit,
    PredTypeSpeedLimit,
)
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import PredAbruptBreaking, PredRelAbruptBreaking

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle


from commonroad_qp_planner.configuration import PlanningConfigurationVehicle
from commonroad_qp_planner.constraints import LonConstraints, LatConstraints
from commonroad_qp_planner.initialization import convert_pos_curvilinear
from commonroad_qp_planner.trajectory import Trajectory as QPTrajectory


class MIQPRuleComstraints:
    def __init__(self):
        pass
