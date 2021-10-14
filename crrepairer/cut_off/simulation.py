import math
from enum import Enum
from math import sqrt, atan
from abc import ABC, abstractmethod

from commonroad.common.solution import VehicleType
from commonroad.scenario.obstacle import DynamicObstacle, State
from commonroad_dc.feasibility.vehicle_dynamics import (KinematicSingleTrackDynamics,
                                                        PointMassDynamics)
from commonroad_dc.feasibility.feasibility_checker import input_vector_feasibility, state_transition_feasibility
from crmonitor.common.world_state import WorldState
from cut_off.utils import check_steering_angle_feasibility, check_velocity_feasibility


class CutOffAction(Enum):
    BRAKE = "brake"
    CONSTANT = "constant velocity"
    KICKDOWN = "kick-down"
    LANECHANGELEFT = "lane change to the left"
    LANECHANGERIGHT = "lane change to the right"
    STEADYSPEED = "steady speed"


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
        self._state_list = [simulated_vehicle.initial_state] + \
                           simulated_vehicle.prediction.trajectory.state_list[:start_time]
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

    def set_inputs(self):
        self._input.acceleration_y = 0
        if self.action == CutOffAction.BRAKE:
            self._input.acceleration = - self._vehicle_dynamics.parameters.longitudinal.a_max
        elif self.action == CutOffAction.KICKDOWN:
            self._input.acceleration = self._vehicle_dynamics.parameters.longitudinal.a_max
        else:
            self._input.acceleration = 0

    def simulate_state_list(self):
        self.set_inputs()
        pre_state = self._cut_off_state
        while pre_state.time_step < self._time_horizon:
            self._input.time_step = pre_state.time_step
            suc_state = self._vehicle_dynamics.simulate_next_state(pre_state, self._input, self._dt, throw=False)
            if suc_state and check_velocity_feasibility(suc_state, self._vehicle_dynamics.parameters):
                self._state_list.append(suc_state)
                pre_state = suc_state
            else:
                self._input.acceleration = 0
        return self._state_list


class SimulationLateral(SimulationBase, ABC):
    def __init__(self, action: CutOffAction,
                 simulated_vehicle: DynamicObstacle,
                 start_time: int,
                 world_state: WorldState):
        assert action == CutOffAction.LANECHANGELEFT or action == CutOffAction.LANECHANGERIGHT, \
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
        return sqrt(4 * lat_dist / self.parameters.longitudinal.a_max)

    def set_inputs(self):
        self._input.acceleration = 0
        if self.action == CutOffAction.LANECHANGELEFT:
            self._input.acceleration_y = self._vehicle_dynamics.parameters.longitudinal.a_max
        elif self.action == CutOffAction.LANECHANGERIGHT:
            self._input.acceleration_y = - self._vehicle_dynamics.parameters.longitudinal.a_max
        else:
            self._input.acceleration_y = 0

    def bang_bang_simulation(self, state, time_horizon):
        pre_state = state
        while pre_state.time_step < state.time_step + time_horizon:
            self._input.time_step = pre_state.time_step
            suc_state = self._vehicle_dynamics.simulate_next_state(pre_state, self._input, self._dt, throw=False)
            if suc_state:  # and check_steering_angle_feasibility(suc_state, self._vehicle_dynamics.parameters):
                self._state_list.append(suc_state)
                pre_state = suc_state
            else:
                self._input.acceleration_y = 0
        return suc_state

    def simulate_state_list(self):
        self.set_inputs()
        target_lane = self.set_target_lane()
        lateral_distance = self._world_state.ego_vehicle.lane.width(self._world_state.ego_vehicle.
                                                                    states_lon[self._cut_off_state.time_step].s) / 2 + \
                           target_lane.width(self._world_state.ego_vehicle.
                                             states_lon[self._cut_off_state.time_step].s) / 2
        total_time = self.calc_total_time(lateral_distance)
        bang_bang_time = int(total_time / (2 * self._dt)) + 1
        current_state = self._cut_off_state
        for i in range(2):
            current_state = self.bang_bang_simulation(current_state, bang_bang_time)
            self._input.acceleration_y = - self._input.acceleration_y
        while current_state.time_step < self._time_horizon:
            self._input.acceleration_y = 0
            self._input.time_step = current_state.time_step
            current_state = self._vehicle_dynamics.simulate_next_state(current_state, self._input, self._dt,
                                                                       throw=False)
            self._state_list.append(current_state)
        return self._state_list


if __name__ == '__main__':
    SL = SimulationLong(CutOffAction.LANECHANGELEFT, None, 1, None)
