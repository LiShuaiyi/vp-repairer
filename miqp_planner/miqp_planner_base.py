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

from commonroad_qp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType
from commonroad_qp_planner.configuration import (
    PlanningConfigurationVehicle,
    ReferencePoint,
)
from commonroad_qp_planner.initialization import compute_initial_state

from miqp_planner.miqp_long_planner import MIQPLongPlanner
from miqp_planner.miqp_lat_planner import MIQPLatPlanner, MIQPLatState, MIQPLatReference
from miqp_planner.miqp_constraints import LongitudinalConstraint, LateralConstraint


class MIQPPlanner:
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        time_horizon: float,
        vehicle_configuration: PlanningConfigurationVehicle,
    ):
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.vehicle_configuration = vehicle_configuration
        if not hasattr(scenario, "dt"):
            self.dt = 0.1  # default time step
        else:
            if Decimal(str(time_horizon)) % Decimal(str(scenario.dt)) != Decimal("0.0"):
                raise ValueError(
                    "<QPPlanner>: the given time step {} is inapproparite,"
                    "since time horizon is {}.".format(scenario.dt, time_horizon)
                )
            self.dt = scenario.dt
        self.t_h = time_horizon

        self.N = round(time_horizon / self.dt)
        if isinstance(planning_problem.initial_state, State):
            # this state is in curvilinear coordinate system
            self.initial_state = compute_initial_state(
                planning_problem.initial_state, vehicle_configuration
            )
        elif not isinstance(planning_problem.initial_state, TrajPoint):
            raise ValueError(
                "<QPPlanner/__init__>: Initial state must be of type {} or "
                "of type {}. Got type {}.".format(
                    type(State), type(TrajPoint), type(planning_problem.initial_state)
                )
            )
        if vehicle_configuration.reference_point != ReferencePoint.REAR:
            raise ValueError("<QPPlanner>: Reference point must be rear axis!")

        if planning_problem.goal.state_list:
            if self.initial_state.v > planning_problem.goal.state_list[0].velocity.end:
                self.vehicle_configuration.desired_speed = (
                    planning_problem.goal.state_list[0].velocity.end
                )
            else:
                self.vehicle_configuration.desired_speed = self.initial_state.v
        else:
            self.vehicle_configuration.desired_speed = self.initial_state.v

        # TODO: for bug at lateral planner initial state
        self.initial_state_lat_orientation = self.initial_state.orientation

    def longitudinal_trajectory_planning(
        self, reference_path, long_constraints: LongitudinalConstraint, slack=False
    ):
        long_planner = MIQPLongPlanner(
            horizon=self.t_h,
            N=self.N,
            dT=self.dt,
            qp_long_params=None,  # TODO: need to add
            scenario=self.scenario,
            vehicle_configuration=self.vehicle_configuration,
            initial_state=self.initial_state,
            long_constraints=long_constraints,
            slack=slack,
        )
        traj_long = long_planner.plan(reference_path)
        return traj_long

    def lateral_trajectory_planning(
        self,
        longitudinal_trajectory: Trajectory,
        lat_con: LateralConstraint,
        d_reference=None,
    ):
        x_init_lat = MIQPLatState(
            d=self.initial_state.position[1],
            theta=self.initial_state_lat_orientation,
            kappa=self.initial_state.kappa,
            kappa_dot=self.initial_state.kappa_dot,
            t=0.0,
            s=longitudinal_trajectory.states[0].position[0],
            v=longitudinal_trajectory.states[0].v,
            a=longitudinal_trajectory.states[0].a,
            j=longitudinal_trajectory.states[0].j,
            u_lon=longitudinal_trajectory.u_lon,
        )
        # TODO: check the length of reference
        x_ref_lat = MIQPLatReference.construct_from_lon_traj_and_reference(
            lon_traj=longitudinal_trajectory,
            reference=self.vehicle_configuration.reference_path,
            vehicle_configuration=self.vehicle_configuration,
            ti=None,
        )  # TODO: fix time invariant constraints
        lat_planner = MIQPLatPlanner(
            horizon=self.t_h,
            N=self.N,
            dT=self.dt,
            length=self.vehicle_configuration.wheelbase,
            lateral_constraints=lat_con,
            x_init_lat=x_init_lat,
            x_ref_lat=x_ref_lat,
            vehicle_configuration=self.vehicle_configuration,
            miqp_lat_params=None,
        )  # TODO: need to add
        trajectory = lat_planner.plan()
        return trajectory
