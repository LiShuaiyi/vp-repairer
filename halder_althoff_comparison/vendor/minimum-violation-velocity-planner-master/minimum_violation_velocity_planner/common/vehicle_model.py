import math
from typing import Dict, List

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_parameters import VehicleParameters

from minimum_violation_velocity_planner.common.longitudinal_state import LongitudinalState
from minimum_violation_velocity_planner.common.utils import roundToGrid


class VehicleModel:
    def __init__(self, vehicle_parameters: VehicleParameters, params: Dict):
        self.veh_params = vehicle_parameters
        self.params = params

        # Note: The extensions are necessary, so that a violation of the real limits can be performed
        self.min_velocity = 0.0  # self.veh_params.longitudinal.v_min - params.get('vehicle_model').get('v_min_extension')
        self.max_velocity = self.veh_params.longitudinal.v_max + params.get('vehicle_model').get('v_max_extension')

        self.min_acceleration = -self.veh_params.longitudinal.a_max - params.get('vehicle_model').get('a_min_extension')
        self.max_acceleration = self.veh_params.longitudinal.a_max + params.get('vehicle_model').get('a_max_extension')

        self.state_rounding_precision = params.get('general').get('state_rounding_precision')

    def interpolate(self, state: LongitudinalState, model_input: float, dt: float) -> LongitudinalState:
        t = state.t + dt
        s = 1 / 2 * model_input * dt ** 2 + state.v * dt + state.s
        v = model_input * dt + state.v
        a = model_input
        j = state.j
        return LongitudinalState(s, v, a, j, t)

    def interpolateJerkModel(self, state: LongitudinalState, model_input: float, dt: float) -> LongitudinalState:
        t = state.t + dt
        s = 1 / 6 * model_input * dt ** 3 + 1/2 * state.a * dt ** 2 + state.v * t + state.s
        v = 1 / 2 * model_input * dt ** 2 + state.a * dt + state.v
        a = model_input * dt + state.a
        j = model_input
        return LongitudinalState(s, v, a, j, t)

    def propagate(self, state: LongitudinalState, model_input: float, dt: float, ds: float, dv: float, s_init) -> LongitudinalState:
        t = state.t + dt
        s = 1 / 2 * model_input * dt ** 2 + state.v * dt + state.s
        v = model_input * dt + state.v
        a = model_input
        j = (a - state.a)/dt
        return LongitudinalState(t=roundToGrid(t, self.state_rounding_precision, dt),
                                 s=roundToGrid(s, self.state_rounding_precision, ds, s_init),
                                 v=roundToGrid(v, self.state_rounding_precision, dv),
                                 a=a, j=j)

    def newProperties(self, old_state: LongitudinalState,  new_state: LongitudinalState) -> LongitudinalState:
        dt = new_state.t - old_state.t
        a = (new_state.s - old_state.s - old_state.v * dt) * 2 / (dt ** 2)
        v = a * dt + old_state.v
        j = (a - old_state.a)/dt
        return LongitudinalState(new_state.s, v, a, j, new_state.t)

    def lowerBound(self, state: LongitudinalState, nof_time_steps: int, dt: float) -> List[float]:
        prev_state = [state.s, state.v]
        for i in range(nof_time_steps):
            new_state = [0., 0.]
            a = self.min_acceleration
            if math.isclose(prev_state[1], self.min_velocity):
                v = self.min_velocity
                s = v * dt + prev_state[0]
            else:
                v = a * dt + prev_state[1]
                if v >= self.min_velocity:
                    s = 1 / 2 * a * dt ** 2 + prev_state[1] * dt + prev_state[0]
                else:
                    # dt_prime: Time until velocity is below minimum velocity
                    dt_prime = (self.min_velocity - prev_state[1]) / a
                    v = self.min_velocity
                    s = (1 / 2 * a * dt_prime ** 2 + prev_state[1] * dt_prime + prev_state[0]) + v * (dt - dt_prime)

            new_state[1] = v
            new_state[0] = s
            prev_state = new_state

        return prev_state

    def upperBound(self, state: LongitudinalState, nof_time_steps: int, dt: float) -> List[float]:
        prev_state = [state.s, state.v]
        for i in range(nof_time_steps):
            new_state = [0., 0.]
            a = self.max_acceleration
            if math.isclose(prev_state[1], self.max_velocity):
                v = self.max_velocity
                s = v * dt + prev_state[0]
            else:
                v = a * dt + prev_state[1]
                if v <= self.max_velocity:
                    s = 1 / 2 * a * dt ** 2 + prev_state[1] * dt + prev_state[0]
                else:
                    # dt_prime: Time until velocity is above minimum velocity
                    dt_prime = (self.max_velocity - prev_state[1]) / a
                    v = self.max_velocity
                    s = (1 / 2 * a * dt_prime ** 2 + prev_state[1] * dt_prime + prev_state[0]) + v * (dt - dt_prime)

            new_state[1] = v
            new_state[0] = s
            prev_state = new_state
        return prev_state
