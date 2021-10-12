from enum import Enum
from abc import ABC, abstractmethod

from commonroad.common.solution import VehicleType
from commonroad.scenario.obstacle import DynamicObstacle, State
from commonroad_dc.feasibility.vehicle_dynamics import (KinematicSingleTrackDynamics,
                                                        PointMassDynamics)
from commonroad_dc.feasibility.feasibility_checker import state_transition_feasibility
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
        self._state_list = simulated_vehicle.prediction.trajectory.state_list[:start_time]
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
    def __init__(self,  action: CutOffAction,
                 simulated_vehicle: DynamicObstacle,
                 start_time: int):
        assert action == CutOffAction.BRAKE or action == CutOffAction.KICKDOWN\
            or action == CutOffAction.STEADYSPEED, "<SimulateLong>: provided action {} is not supported".format(action)
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


if __name__ == '__main__':
    SL = SimulationLong(CutOffAction.LANECHANGELEFT, None, 1, None)