from stlpy.systems.linear import LinearSystem
import numpy as np


class VehicleModel(LinearSystem):
    """
    A linear system describing a double integrator in 2D (x, y) with state variables:
    [s_x, v_x, a_x, j_x; s_y, v_y, a_y, j_y]

    .. math::

        A = \\begin{bmatrix}
        0 & I_{2 \\times 2} & 0 & 0 \\\
        0 & 0 & I_{2 \\times 2} & 0 \\\
        0 & 0 & 0 & I_{2 \\times 2} \\\
        0 & 0 & 0 & 0
        \\end{bmatrix}
        \quad
        B = \\begin{bmatrix}
        0_{2 \\times 2} \\\
        0_{2 \\times 2} \\\
        0_{2 \\times 2} \\\
        I_{2 \\times 2}
        \\end{bmatrix}

    .. math::
        C = I_{8 \\times 8}
        \quad
        D = 0_{8 \\times 2}

    :param d: Integer describing the dimensionality of the system (in this case, it's fixed at 2 for x and y).
    """
    def __init__(self):
        d = 2  # Fixed for 2D case (x, y dimensions)
        I = np.eye(d)
        z = np.zeros((d, d))

        # A matrix for the state space  [s_x,s_y, v_x, v_x, a_x, a_y, j_x, j_y]
        A = np.block([
            [I, I, z, z],
            [z, I, I, z],
            [z, z, I, I],
            [z, z, z, I]
        ])

        # B matrix where control input affects only jerk
        B = np.block([
            [z],
            [z],
            [z],
            [I]
        ])

        # C matrix for selective state observation with direct feedthrough
        C = np.block([
            [I, z, z, z],  # output position
            [z, I, z, z],  # output velocity
            [z, z, I, z],
            [z, z, z, I],
            [z, z, z, z],
        ])

        # D matrix for direct feedthrough from control input to acceleration
        D = np.block([
            [z],  # no direct feedthrough to position
            [z],  # no direct feedthrough to velocity
            [z],
            [z],
            [I]   # direct feedthrough to acceleration from jerk's derivative
        ])

        super().__init__(A, B, C, D)