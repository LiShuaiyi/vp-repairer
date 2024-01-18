from decimal import Decimal

# commonroad-io
from commonroad.scenario.trajectory import State

from commonroad_qp_planner.trajectory import Trajectory, TrajPoint
from commonroad_qp_planner.configuration import (
    ReferencePoint,
)
from commonroad_qp_planner.initialization import compute_initial_state

from miqp_planner.miqp_long_planner import MIQPLongPlanner
from miqp_planner.miqp_lat_planner import MIQPLatPlanner, MIQPLatState, MIQPLatReference
from miqp_planner.miqp_constraints import (
    LongitudinalConstraint,
    LateralConstraint,
    TIConstraint,
)

from crrepairer.utils.configuration import RepairerConfiguration


class MIQPPlanner:
    def __init__(self, config: RepairerConfiguration):
        self.scenario = config.scenario
        self.planning_problem = config.planning_problem
        self.vehicle_configuration = config.vehicle
        if not hasattr(self.scenario, "dt"):
            self.dt = 0.1  # default time step
        else:
            if Decimal(str(config.miqp_planner.horizon)) % Decimal(
                str(self.scenario.dt)
            ) != Decimal("0.0"):
                raise ValueError(
                    "<MIQPPlanner>: the given time step {} is inappropriate,"
                    "since time horizon is {}.".format(
                        self.scenario.dt, config.miqp_planner.horizon
                    )
                )
            self.dt = self.scenario.dt
        self.t_h = config.miqp_planner.horizon

        config.miqp_planner.N_p = round(config.miqp_planner.horizon / self.dt)
        if isinstance(self.planning_problem.initial_state, State):
            # this state is in curvilinear coordinate system
            self.initial_state = compute_initial_state(
                self.planning_problem.initial_state, config.vehicle.qp_veh_config
            )
        elif not isinstance(self.planning_problem.initial_state, TrajPoint):
            raise ValueError(
                "<MIQPPlanner/__init__>: Initial state must be of type {} or "
                "of type {}. Got type {}.".format(
                    type(State),
                    type(TrajPoint),
                    type(self.planning_problem.initial_state),
                )
            )
        if (
            self.vehicle_configuration.qp_veh_config.reference_point
            != ReferencePoint.REAR
        ):
            raise ValueError("<MIQPPlanner>: Reference point must be rear axis!")

        if self.planning_problem.goal.state_list:
            if (
                self.initial_state.v
                > self.planning_problem.goal.state_list[0].velocity.end
            ):
                self.vehicle_configuration.desired_speed = (
                    self.planning_problem.goal.state_list[0].velocity.end
                )
            else:
                self.vehicle_configuration.desired_speed = self.initial_state.v
        else:
            self.vehicle_configuration.desired_speed = self.initial_state.v

        # initial orientation for the lateral planner
        self.initial_state_lat_orientation = self.initial_state.orientation
        self.config = config
        self.time_invariant_constraints = self._set_time_invariant_constraints()

    def _set_time_invariant_constraints(self):
        ti_constraint = TIConstraint()
        ti_constraint.v_x_max = self.config.vehicle.qp_veh_config.max_speed_x
        ti_constraint.v_x_min = self.config.vehicle.qp_veh_config.min_speed_x
        ti_constraint.a_x_max = self.config.vehicle.qp_veh_config.a_max_x
        ti_constraint.a_x_min = self.config.vehicle.qp_veh_config.a_min_x
        ti_constraint.j_x_max = self.config.vehicle.qp_veh_config.j_max_x
        ti_constraint.j_x_min = self.config.vehicle.qp_veh_config.j_min_x

        ti_constraint.kappa_max = self.config.vehicle.kappa_max
        ti_constraint.kappa_min = self.config.vehicle.kappa_min
        ti_constraint.kappa_dot_max = self.config.vehicle.kappa_dot_max
        ti_constraint.kappa_dot_min = self.config.vehicle.kappa_dot_min
        ti_constraint.kappa_dot_dot_max = self.config.vehicle.kappa_dot_dot_max
        ti_constraint.kappa_dot_dot_min = self.config.vehicle.kappa_dot_dot_min
        return ti_constraint

    def longitudinal_trajectory_planning(
        self, reference_lon, long_constraints: LongitudinalConstraint
    ):
        long_planner = MIQPLongPlanner(
            config=self.config,
            initial_state=self.initial_state,
        )
        traj_long = long_planner.plan(
            reference_lon, self.time_invariant_constraints, long_constraints
        )
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
        x_ref_lat = MIQPLatReference.construct_from_lon_traj_and_reference(
            lon_traj=longitudinal_trajectory,
            reference=self.vehicle_configuration.qp_veh_config.reference_path,
            vehicle_configuration=self.vehicle_configuration,
        )
        lat_planner = MIQPLatPlanner(
            config=self.config,
            x_init_lat=x_init_lat,
            x_ref_lat=x_ref_lat,
            d_reference=d_reference,
        )
        trajectory = lat_planner.plan(
            lateral_constraints=lat_con,
            ti_constraints=self.time_invariant_constraints,
        )
        return trajectory
