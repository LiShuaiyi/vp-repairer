import math
from enum import Enum
from math import sqrt, atan
from abc import ABC, abstractmethod

import numpy as np

from commonroad.common.solution import VehicleType
from commonroad.scenario.obstacle import DynamicObstacle, State
from commonroad_dc.feasibility.vehicle_dynamics import (KinematicSingleTrackDynamics,
                                                        PointMassDynamics)
from commonroad_dc.feasibility.feasibility_checker import input_vector_feasibility, state_transition_feasibility
from crmonitor.common.world_state import WorldState
from cut_off.utils import check_steering_angle_feasibility, check_velocity_feasibility


class CutOffAction(str, Enum):
    BRAKE = "brake"
    CONSTANT = "constant velocity"
    KICKDOWN = "kick-down"
    LANECHANGELEFT = "lane change to the left"
    LANECHANGERIGHT = "lane change to the right"
    STEADYSPEED = "steady speed"
    STEERLEFT = "steer to the left"
    STEERRIGHT = "steer to the right"


class SimulationBase(ABC):
    def __init__(self, action: CutOffAction,
                 simulated_vehicle: DynamicObstacle,
                 start_time: int,
                 dt: float = 0.1):
        assert isinstance(action, CutOffAction), "provided action is not supported"
        self._dt = dt
        self._time_horizon = simulated_vehicle.prediction.final_time_step
        self._action = action
        self._simulated_vehicle = simulated_vehicle
        self._cut_off_state = simulated_vehicle.state_at_time(start_time)
        if simulated_vehicle.prediction.trajectory.state_list[0].time_step != 0:
            self._state_list = [simulated_vehicle.initial_state] + \
                               simulated_vehicle.prediction.trajectory.state_list[:start_time]
        else:
            self._state_list = [simulated_vehicle.initial_state] + \
                               simulated_vehicle.prediction.trajectory.state_list[1:start_time]
        for state in self._state_list:
            if not hasattr(state, "velocity_y"):
                state.velocity_y = state.velocity * math.sin(state.orientation)
        self._input: State = State(steering_angle_speed=0,
                                   acceleration=0)
        self._vehicle_dynamics = PointMassDynamics(VehicleType.BMW_320i)
        self._parameters = self._vehicle_dynamics.parameters

    @property
    def action(self):
        return self._action

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_list(self):
        return self._state_list

    @property
    def vehicle_dynamics(self):
        return self._vehicle_dynamics

    @property
    def input(self):
        return self._input

    @property
    def start_time(self):
        return self._start_time

    @abstractmethod
    def set_inputs(self):
        pass

    @abstractmethod
    def simulate_state_list(self):
        pass


class SimulationLong(SimulationBase, ABC):
    def __init__(self, action: CutOffAction,
                 simulated_vehicle: DynamicObstacle,
                 start_time: int):
        assert action == CutOffAction.BRAKE or action == CutOffAction.KICKDOWN \
               or action == CutOffAction.STEADYSPEED, "<SimulationLong>: provided action {} is not supported".format(
            action)
        super().__init__(action, simulated_vehicle, start_time, dt=0.1)

    def set_inputs(self, velocity):
        self._input.acceleration_y = 0
        v_switch = self._vehicle_dynamics.parameters.longitudinal.v_switch
        if velocity > v_switch:
            a_max = self._vehicle_dynamics.parameters.longitudinal.a_max * v_switch / velocity
        else:
            a_max = self._vehicle_dynamics.parameters.longitudinal.a_max
        if self.action == CutOffAction.BRAKE:
            self._input.acceleration = - a_max
        elif self.action == CutOffAction.KICKDOWN:
            self._input.acceleration = a_max
        else:
            self._input.acceleration = 0

    def simulate_state_list(self):
        pre_state = self._cut_off_state
        self.set_inputs(pre_state.velocity)
        while pre_state.time_step < self._time_horizon:
            self._input.time_step = pre_state.time_step
            suc_state = self._vehicle_dynamics.simulate_next_state(pre_state, self._input, self._dt, throw=False)
            if suc_state and check_velocity_feasibility(suc_state, self._vehicle_dynamics.parameters):
                check_elements(suc_state)
                # if abs(suc_state.orientation) > np.pi/2:
                #     suc_state.orientation = np.sign(suc_state.orientation)*abs(suc_state.orientation-np.pi/2)
                self._state_list.append(suc_state)
                pre_state = suc_state
                self.set_inputs(pre_state.velocity)
            else:
                self._input.acceleration = 0
        return self._state_list


class SimulationLateral(SimulationBase, ABC):
    def __init__(self, action: CutOffAction,
                 simulated_vehicle: DynamicObstacle,
                 start_time: int,
                 world_state: WorldState):
        assert action in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT,
                          CutOffAction.STEERLEFT, CutOffAction.STEERRIGHT], \
            "<SimulationLateral>: provided action {} is not supported".format(action)
        super().__init__(action, simulated_vehicle, start_time, dt=0.1)
        self._world_state = world_state

    def set_target_lane(self):
        if self.action == CutOffAction.LANECHANGELEFT:
            return self._world_state.ego_vehicle.lane.adj_left
        else:
            return self._world_state.ego_vehicle.lane.adj_right

    def calc_total_time(self, lat_dist):
        """
        Modified from Eq. (11) in Pek, C., Zahn, P. and Althoff, M., Verifying the safety of lane change maneuvers of
         self-driving vehicles based on formalized traffic rules. In IV 2017 (pp. 1477-1483). IEEE.
        """
        return sqrt(4 * lat_dist / abs(self._input.acceleration_y))

    def calc_leave_time(self, lat_dist):
        """
        Miller, Christina, Christian Pek, and Matthias Althoff. "Efficient mixed-integer programming for longitudinal
        and lateral motion planning of autonomous vehicles." 2018 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2018.
        """
        return sqrt(2 * lat_dist / abs(self._input.acceleration_y))

    def set_inputs(self, velocity):
        self._input.acceleration = 0
        v_switch = self._vehicle_dynamics.parameters.longitudinal.v_switch
        if velocity > v_switch:
            a_max = self._vehicle_dynamics.parameters.longitudinal.a_max * v_switch / velocity
        else:
            a_max = self._vehicle_dynamics.parameters.longitudinal.a_max
        if self.action in [CutOffAction.LANECHANGELEFT, CutOffAction.STEERLEFT]:
            self._input.acceleration_y = a_max
        elif self.action in [CutOffAction.LANECHANGERIGHT, CutOffAction.STEERRIGHT]:
            self._input.acceleration_y = - a_max
        else:
            self._input.acceleration_y = 0

    def set_bang_bang_time(self, ego_s, ego_d, target_lane):
        if self.action in [CutOffAction.LANECHANGELEFT, CutOffAction.LANECHANGERIGHT]:
            # todo: fix the lane of the ego
            ego_lane_width = self._world_state.ego_vehicle.lane.width(ego_s)
            ego_to_lane_boundary = ego_lane_width/2 - abs(ego_d)
            lateral_distance = ego_to_lane_boundary + target_lane.width(ego_s) / 2
        else:
            # from paper: A flexible method for criticality assessment in driver assistance systems
            lateral_distance = 0.8
        total_time = self.calc_total_time(lateral_distance)
        bang_bang_time = int(total_time / (2 * self._dt))
        return bang_bang_time

    def set_maximal_orientation(self, lane_orientation):
        if self.action in [CutOffAction.LANECHANGELEFT, CutOffAction.STEERLEFT]:
            return lane_orientation + math.pi/4
        elif self.action in [CutOffAction.LANECHANGERIGHT, CutOffAction.STEERRIGHT]:
            return lane_orientation - math.pi/4
        else:
            pass

    def bang_bang_simulation(self, state, time_horizon, max_orientation):
        pre_state = state
        while pre_state.time_step < state.time_step + time_horizon:
            self._input.time_step = pre_state.time_step
            suc_state = self._vehicle_dynamics.simulate_next_state(pre_state, self._input, self._dt, throw=False)
            if suc_state:  # and check_steering_angle_feasibility(suc_state, self._vehicle_dynamics.parameters):
                check_elements(suc_state)
                self._state_list.append(suc_state)
                pre_state = suc_state
                if abs(suc_state.orientation) > abs(max_orientation):
                    break
            else:
                self._input.acceleration_y = 0
        return suc_state

    def simulate_state_list(self):
        self.set_inputs(self._cut_off_state.velocity)
        target_lane = self.set_target_lane()
        if target_lane is None:
            return None
        current_ego_s, current_ego_d = self._world_state.ego_vehicle.lane.clcs.convert_to_curvilinear_coords(
            self._cut_off_state.position[0], self._cut_off_state.position[1])
        # current_ego_s = self._world_state.ego_vehicle.states_lon[self._cut_off_state.time_step].s
        # current_ego_d = self._world_state.ego_vehicle.states_lat[self._cut_off_state.time_step].d
        bang_bang_time = self.set_bang_bang_time(current_ego_s, current_ego_d, target_lane)
        lane_orientation = self._world_state.ego_vehicle.lane.orientation(current_ego_s)
        max_orientation = self.set_maximal_orientation(lane_orientation)
        current_state = self._cut_off_state
        for i in range(2):
            current_state = self.bang_bang_simulation(current_state, bang_bang_time, max_orientation)
            self._input.acceleration_y = - self._input.acceleration_y
        while current_state.time_step < self._time_horizon:
            self._input.acceleration_y = 0
            self._input.time_step = current_state.time_step
            current_state.velocity_y = current_state.velocity*math.sin(lane_orientation)
            current_state = self._vehicle_dynamics.simulate_next_state(current_state, self._input, self._dt,
                                                                       throw=False)
            self._state_list.append(current_state)
        return self._state_list


def check_elements(state: State):
    if not hasattr(state, "slip_angle"):
        state.slip_angle = 0
    if not hasattr(state, "yaw_rate"):
        state.yaw_rate = 0
    if not hasattr(state, "velocity_y"):
        state.velocity_y = state.velocity * math.cos(state.orientation)


if __name__ == '__main__':
    SL = SimulationLong(CutOffAction.LANECHANGELEFT, None, 1, None)
