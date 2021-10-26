from typing import List, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from collections import defaultdict

# commonroad-io
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

# trajectory planning tools
from optimizer.safety_constraints import SafetyConstraints
from optimizer.trajectory import Trajectory, TrajPoint, TrajectoryType
from optimizer.qp_lat_planner import QPLatPlanner, QPLatReference, QPLatState, QPLatPARAMS
from optimizer.qp_long_planner import QPLongPlanner, QPLongReference, QPLongState, QPLongPARAMS
from optimizer.constraints import TIConstraints

from crmonitor.common.world_state import WorldState

class QPPlanner:
    def __init__(self,
                 initial_state: Union[State, TrajPoint],
                 scenario: Scenario,
                 ego_id: int,
                 time_horizon: float,
                 tstcc: int,
                 tstv: int,
                 vehicle_configuration,
                 qp_long_parameters: Union[Dict, None] = None,
                 qp_lat_parameters: Union[Dict, None] = None,
                 slack_usage: bool = False,
                 verbose: bool = True):
        if not hasattr(scenario, 'dt'):
            self.dt = 0.1 # default time step
        else:
            self.dt = scenario.dt
        self.t_h = time_horizon
        if Decimal(str(time_horizon)) % Decimal(str(self.dt)) != Decimal('0.0'):
            raise ValueError('<QPPlanner>: the given time step {} is inapproparite,'
                             'since time horizon is {}.'.format(dt, time_horizon))
        self.tstcc = tstcc
        self.tstv = tstv
        self.N = int(time_horizon/self.dt) - self.tstcc
        self.initial_state = initial_state
        self.vehicle_configuration = vehicle_configuration

        # todo: set desired speed (goal?)
        self.vehicle_configuration.desired_speed = initial_state.velocity
        # if vehicle_configuration.reference_point != pycrreach.ReferencePoint.REAR:
        #     raise ValueError('<QPPlanner>: Reference point must be rear axis!')

        # self.collision_avoidance_constraints = CollisionAvoidanceConstraints(self.reachable_set,
        #                                                                      self.vehicle_configuration.reference_point)
        self.world_state = WorldState.create_from_scenario(scenario, ego_id)
        ego_vehicle = scenario.obstacle_by_id(ego_id)
        self.initial_trajectory = ego_vehicle.prediction.trajectory
        scenario.remove_obstacle(ego_vehicle) # remove the initial trajectory (already constained in the world state)
        self.collision_avoidance_constraints = SafetyConstraints(self.world_state, self.N)

        self.slack_usage = slack_usage
        self.verbose = verbose

        self.time_invariant_constraints = self._set_time_invariant_constraints(vehicle_configuration)

        if qp_long_parameters is not None:
            self.qp_long_params = self._set_qp_longitudinal_parameters(qp_long_parameters)
        else:
            self.qp_long_params = QPLongPARAMS()

        if qp_lat_parameters is not None:
            self.qp_lat_params = self._set_qp_lateral_parameters(qp_lat_parameters)
        else:
            self.qp_lat_params = QPLatPARAMS()

    def plan_trajectories(self,
                          target_lanes: [Dict, None],
                          find_all_trajectories: bool = False):
        print('\t\t Longitudinal optimization')
        traj_lon, status = self._longitudinal_trajectory_planning(target_lanes)
        if status is not 'optimal':
            raise ValueError('<QPPlanner/_longitudinal_trajectory_planning>: failed')
        print('\t\t Lateral optimization')
        traj_lat, status = self._lateral_trajectory_planning(traj_lon, target_lanes)
        # convert trajectory to cartesian space
        if status is not 'optimal':
            raise ValueError('<QPPlanner/_lateral_trajectory_planning>: failed')
        # except Exception as e:
        #     print(e)
        return traj_lat

    @classmethod
    def _set_time_invariant_constraints(cls, configuration) -> TIConstraints:
        time_invariant_constraints = TIConstraints()
        time_invariant_constraints.j_x_min = configuration.j_min_x
        time_invariant_constraints.j_x_max = configuration.j_max_x
        time_invariant_constraints.j_y_min = configuration.j_min_y
        time_invariant_constraints.j_y_max = configuration.j_max_y
        time_invariant_constraints.a_x_min = configuration.a_min_x
        time_invariant_constraints.a_x_max = configuration.a_max_x
        time_invariant_constraints.a_y_min = configuration.a_min_y
        time_invariant_constraints.a_y_max = configuration.a_max_y
        time_invariant_constraints.a_max = configuration.a_max
        time_invariant_constraints.v_min = configuration.min_speed_x
        time_invariant_constraints.v_max = configuration.max_speed_x
        return time_invariant_constraints

    @classmethod
    def _set_qp_longitudinal_parameters(cls, qp_long_parameters: Dict) -> QPLongPARAMS:
        qp_long_params = QPLongPARAMS()
        qp_long_params.W_S = qp_long_parameters['W_S']
        qp_long_params.W_V = qp_long_parameters['W_V']
        qp_long_params.W_A = qp_long_parameters['W_A']
        qp_long_params.W_J = qp_long_parameters['W_J']
        qp_long_params.W_U = qp_long_parameters['W_U']
        qp_long_params.W_S_Q = qp_long_parameters['W_S_Q']
        qp_long_params.W_S_L = qp_long_parameters['W_S_L']
        qp_long_params.W_C_Q = qp_long_parameters['W_C_Q']
        qp_long_params.W_C_L = qp_long_parameters['W_C_L']
        qp_long_params.W_S_A_L = qp_long_parameters['W_S_A_L']
        qp_long_params.W_S_A_U = qp_long_parameters['W_S_A_U']
        qp_long_params.L_ENLARGE = qp_long_parameters['L_ENLARGE']
        return qp_long_params

    @classmethod
    def _set_qp_lateral_parameters(cls, qp_lat_parameters: Dict) -> QPLatPARAMS:
        qp_lat_params = QPLatPARAMS()
        qp_lat_params.W_D = qp_lat_parameters['W_D']
        qp_lat_params.W_THETA = qp_lat_parameters['W_THETA']
        qp_lat_params.W_KAPPA = qp_lat_parameters['W_KAPPA']
        qp_lat_params.W_D_N = qp_lat_parameters['W_D_N']
        qp_lat_params.W_THETA_N = qp_lat_parameters['W_THETA_N']
        qp_lat_params.W_KAPPA_N = qp_lat_parameters['W_KAPPA_N']
        qp_lat_params.W_KAPPA_DOT = qp_lat_parameters['W_KAPPA_DOT']
        qp_lat_params.W_U = qp_lat_parameters['W_U']
        qp_lat_params.W_SLACK_L = qp_lat_parameters['W_SLACK_L']
        qp_lat_params.W_SLACK_Q = qp_lat_parameters['W_SLACK_Q']
        qp_lat_params.KAPPA_DOT_MIN = qp_lat_parameters['KAPPA_DOT_MIN']
        qp_lat_params.KAPPA_DOT_MAX = qp_lat_parameters['KAPPA_DOT_MAX']
        qp_lat_params.KAPPA_DOT_DOT_MIN = qp_lat_parameters['KAPPA_DOT_DOT_MIN']
        qp_lat_params.KAPPA_DOT_DOT_MAX = qp_lat_parameters['KAPPA_DOT_DOT_MAX']
        qp_lat_params.KAPPA_MAX = qp_lat_parameters['KAPPA_MAX']
        return qp_lat_params

    def transform_trajectory_to_cartesian_coordinates(
            self, trajectory: Trajectory):
        cartesian_traj_points = list()
        for state in trajectory.states:
            cart_pos = self.vehicle_configuration.curvilinear_coordinate_system.convert_to_cartesian_coords(
                state.position[0], state.position[1])
            cartesian_traj_points.append(TrajPoint(
               t=state.t, x=cart_pos[0], y=cart_pos[1], theta=state.orientation, v=state.v, a=state.a,
               kappa=state.kappa, kappa_dot=state.kappa_dot, j=state.j, lane=state.lane))
        traj = Trajectory(cartesian_traj_points, TrajectoryType.CARTESIAN)

        traj._u_lon = trajectory.u_lon
        traj._u_lat = trajectory.u_lat

        return traj

    def _longitudinal_trajectory_planning(self, target_lanes):
        # get longitudinal position constraints from reachable set
        c_long = self.collision_avoidance_constraints.\
            longitudinal_position_constraints(
            self.vehicle_configuration,
            target_lanes,
            self.tstcc,
            self.tstv)
        #####
        # Plan longitudinal trajectory
        #####
        # Create long planner with t_h, N, dT, slack usage, zero velocity at t_h, longitudinal parameters
        lon_planner = QPLongPlanner(self.tstcc, self.t_h, self.N, self.dt, slack=False,
                                    qp_long_params=self.qp_long_params)
        lon_planner.verbose = self.verbose  # turn on diagnose solver output

        # initial state s,v,a,j,t
        if isinstance(self.initial_state, State):
            if hasattr(self.initial_state, 'acceleration'):
                a = self.initial_state.acceleration
            else:
                a = 0.0

            x_init = QPLongState(self.vehicle_configuration.initial_position_x,
                                 self.vehicle_configuration.initial_speed_x,
                                 a, 0., 0.)
        elif isinstance(self.initial_state, TrajPoint):
            x_init = QPLongState(self.initial_state.position[0],
                                 self.initial_state.v,
                                 self.initial_state.a,
                                 self.initial_state.j,
                                 0.)
        else:
            raise ValueError('<QPPlanner/_longitudinal_trajectory_planning>: Initial state must be of type {} or '
                             'of type {}. Got type {}.'.format(type(State), type(TrajPoint),
                                                               type(self.initial_state)))
        x_ref = list()
        # for s in s_ref: todo: change the reference
        # states_long = self.world_state.ego_vehicle.states_lon
        # self.vehicle_configuration.desired_speed =  self.world_state.ego_vehicle.states_lon[0].v
        # for i in range(len(c_long.s_hard_min)):
        #     # reference state s,v,a,j,t
        #     x_ref.append(QPLongState(c_long.s_hard_min[i], self.vehicle_configuration.desired_speed, 0., 0., 0.))

        states_long = self.world_state.ego_vehicle.states_lon
        for state in states_long.values():
            x_ref.append(QPLongState(state.s, state.v, 0., 0., 0.))
        reference = QPLongReference(x_ref[self.tstcc + 1:]) # initial state not included?

        traj, status = lon_planner.plan(x_init, reference, self.time_invariant_constraints, c_long)

        if status == 'optimal':
            if self.verbose:
                print('x_init: {}\n'.format(x_init))
                for i, s in enumerate(traj.cartesian_ptsX()[1:]):
                    print('\t\t\t Longitudinal constraints: time step= {}, s = {}, s_min = {}, s_max = {} \n'.format(
                        i+1, s, c_long.s_hard_min[i], c_long.s_hard_max[i]))
        else:
            if self.verbose:
                print('x_init: {}\n'.format(x_init))
                for i, s in enumerate(c_long.s_hard_min):
                    print('\t\t\t Longitudinal constraints: time step= {}, s_min = {}, s_max = {} \n'.format(
                            i+1, c_long.s_hard_min[i], c_long.s_hard_max[i]))

        if not status == 'optimal' and self.verbose:
            # plot state variables
            print('\t\t\t Lon state: {} \n'.format(x_init))
            plt.figure(figsize=(15, 10))
            plt.plot(0, x_init.s, '*r')
            plt.plot(np.array(range(c_long.N)) + 1,
                     x_init.s + x_init.v * self.dt * (np.array(range(c_long.N)) + 1)
                     + 0.5*x_init.a*((np.array(range(self.N)) + 1)*self.dt)**2
                     + (1/6)*x_init.j*((np.array(range(self.N)) + 1)*self.dt)**3
                     , '*g', linewidth=5)
            plt.plot(np.array(range(c_long.N)) + 1,
                     x_init.s + x_init.v * self.dt * (np.array(range(self.N)) + 1)
                     + 0.5*self.vehicle_configuration.a_min_x*((np.array(range(self.N)) + 1)*self.dt)**2,
                     '*r', linewidth=5)
            plt.plot(np.array(range(c_long.N)) + 1, c_long.s_hard_min)
            plt.plot(np.array(range(c_long.N)) + 1, c_long.s_hard_max)
            plt.autoscale()
            plt.show(block=True)
        return traj, status

    def _lateral_trajectory_planning(self,
                                     longitudinal_trajectory: Trajectory,
                                     target_lanes):
        ######
        # Plan lateral trajectory
        c_lat = self.collision_avoidance_constraints.lateral_position_constraints(
            self.tstcc,
            target_lanes,
            longitudinal_trajectory.cartesian_ptsX(),
            self.vehicle_configuration)
        ######
        # create lateral planner with t_h, N, dt, wheelbase, slack usage, lateral parameters
        lat_planner = QPLatPlanner(self.tstcc, self.tstv, self.t_h, c_lat.N, self.dt,
                                    self.vehicle_configuration.wheelbase,
                                    self.slack_usage, self.qp_lat_params)
        lat_planner.verbose = self.verbose

        # d: float, theta: float, kappa: float, theta_ref: float, t = 0., s= None, v= None, a= None
        if isinstance(self.initial_state, State):
            x_init = QPLatState(d=self.vehicle_configuration.initial_position_y,
                                theta=self.initial_state.orientation,
                                kappa=0.,
                                kappa_dot=0.0,
                                t=0.0,
                                s=longitudinal_trajectory.states[0].position[0],
                                v=longitudinal_trajectory.states[0].v,
                                a=longitudinal_trajectory.states[0].a,
                                j=longitudinal_trajectory.states[0].j,
                                u_lon=longitudinal_trajectory.u_lon)
        elif isinstance(self.initial_state, TrajPoint):
            x_init = QPLatState(d=self.initial_state.position[1],
                                theta=self.initial_state.orientation,
                                kappa=self.initial_state.kappa,
                                kappa_dot=self.initial_state.kappa_dot,
                                t=0.0,
                                s=longitudinal_trajectory.states[0].position[0],
                                v=longitudinal_trajectory.states[0].v,
                                a=longitudinal_trajectory.states[0].a,
                                j=longitudinal_trajectory.states[0].j,
                                u_lon=longitudinal_trajectory.u_lon)
        else:
            raise ValueError('<QPPlanner/_longitudinal_trajectory_planning>: Initial state must be of type {} or '
                             'of type {}. Got type {}.'.format(type(State), type(TrajPoint),
                                                               type(self.initial_state)))

        # create reference
        x_ref = QPLatReference.construct_from_lon_traj_and_reference(
              longitudinal_trajectory, self.vehicle_configuration.reference_path,
              self.time_invariant_constraints)
        self.x_ref = x_ref

        # plan trajectory
        traj_lat, status = lat_planner.plan(x_init, x_ref, self.time_invariant_constraints, c_lat)

        if not status == 'optimal' and self.verbose:
            for s in longitudinal_trajectory.states:
                print('\t\t\t Lon state: {} \n'.format(s))
            print('Lat state: {} \n'.format(x_init))
            for s in x_ref.reference:
                print('\t\t\t Ref state: {} \n'.format(s))

            plt.figure(figsize=(15, 10))
            plt.plot(0, x_init.d, '*-r', linewidth=5)
            plt.plot(0, x_init.d + self.vehicle_configuration.wheelbase / 2.0 *
                     np.sin(self.initial_state.orientation - x_ref.reference[0].theta), '*g')
            plt.plot(0, x_init.d + self.vehicle_configuration.wheelbase *
                     np.sin(self.initial_state.orientation - x_ref.reference[0].theta), '*b')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_min[:, 0], '*-r')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_max[:, 0], '*-r')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_min[:, 1], '-g')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_max[:, 1], '-g')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_min[:, 2], '-b')
            plt.plot(np.array(range(1, self.N + 1)), c_lat.d_soft_max[:, 2], '-b')
            plt.show(block=True)

        return traj_lat, status

