import math
from typing import Dict, Union
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

# commonroad-io
from commonroad.scenario.trajectory import State, Trajectory as CRTrajectory
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.util import make_valid_orientation

from commonroad_qp_planner.trajectory import TrajPoint
from commonroad_qp_planner.configuration import PlanningConfigurationVehicle, ReferencePoint
from commonroad_qp_planner.initialization import compute_initial_state

from miqp_long_planner import MIQPLongPlanner


class MIQPPlanner:
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, time_horizon: float, vehicle_configuration: PlanningConfigurationVehicle, setting):
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.vehicle_configuration = vehicle_configuration
        self.setting = setting
        if not hasattr(scenario, 'dt'):
            self.dt = 0.1  # default time step
        else:
            if Decimal(str(time_horizon)) % Decimal(str(scenario.dt)) != Decimal('0.0'):
                raise ValueError('<QPPlanner>: the given time step {} is inapproparite,'
                                 'since time horizon is {}.'.format(scenario.dt, time_horizon))
            self.dt = scenario.dt
        self.t_h = time_horizon

        self.N = round(time_horizon/self.dt)
        if isinstance(planning_problem.initial_state, State):
            self.initial_state = compute_initial_state(planning_problem, vehicle_configuration)
        elif not isinstance(planning_problem.initial_state, TrajPoint):
            raise ValueError('<QPPlanner/__init__>: Initial state must be of type {} or '
                             'of type {}. Got type {}.'.format(type(State),
                                                               type(TrajPoint),
                                                               type(planning_problem.initial_state)))
        if vehicle_configuration.reference_point != ReferencePoint.REAR:
            raise ValueError('<QPPlanner>: Reference point must be rear axis!')

        if planning_problem.goal.state_list:
            if self.initial_state.v > planning_problem.goal.state_list[0].velocity.end:
                self.vehicle_configuration.desired_speed = planning_problem.goal.state_list[0].velocity.end
            else:
                self.vehicle_configuration.desired_speed = self.initial_state.v
        else:
            self.vehicle_configuration.desired_speed = self.initial_state.v

    def longitudinal_trajectory_planning(self, reference):
        long_planner = MIQPLongPlanner()
        traj_long = long_planner.plan()

    def lateral_trajectory_planning(self):
        pass