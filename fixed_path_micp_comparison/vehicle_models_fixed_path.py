"""Longitudinal dynamics for the standalone fixed-path MICP comparison."""

import numpy as np
from stlpy.systems.linear import LinearSystem


class FixedPathVehicleModel(LinearSystem):
    """Fourth-order longitudinal model on a fixed reference path.

    The state is ``[s, v, a, j]`` and the input is longitudinal snap.  The
    output is ``[s, v, a, j, snap]`` so that an STL predicate can refer to
    either a state or the input.  This is exactly the longitudinal block of
    :class:`comparison.micp.vehicle_models_dt.VehicleModel`; the lateral
    ``[d, v_d, a_d, j_d]`` block and lateral input are eliminated rather than
    retained with equality constraints.
    """

    STATE_DIM = 4
    INPUT_DIM = 1
    OUTPUT_DIM = 5

    S = 0
    V = 1
    A = 2
    J = 3
    SNAP = 4

    def __init__(self, dt: float):
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = dt

        A = np.array(
            [
                [1.0, dt, dt**2 / 2.0, dt**3 / 6.0],
                [0.0, 1.0, dt, dt**2 / 2.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        B = np.array(
            [
                [dt**4 / 24.0],
                [dt**3 / 6.0],
                [dt**2 / 2.0],
                [dt],
            ]
        )

        # y = [s, v, a, j, snap]
        C = np.vstack((np.eye(self.STATE_DIM), np.zeros((1, self.STATE_DIM))))
        D = np.vstack((np.zeros((self.STATE_DIM, self.INPUT_DIM)), np.ones((1, 1))))

        super().__init__(A, B, C, D)
