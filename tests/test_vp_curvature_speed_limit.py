import math
import sys
import unittest
from pathlib import Path

import numpy as np
from commonroad.scenario.state import CustomState

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
try:
    sys.path.remove(repo_root_str)
except ValueError:
    pass
sys.path.insert(0, repo_root_str)

from crrepairer.repairer.vp.constraints import VPConstraintExtraction
from crrepairer.repairer.vp.trajectory_context import VPTrajectoryContext


class _PiecewiseCurvatureCLCS:
    _s = np.arange(5.0)
    _curvature = np.array([0.0, 0.5, 2.0, 0.5, 0.0])

    @staticmethod
    def length():
        return 4.0

    def curvature_range(self, lower, upper):
        inside = (self._s >= lower) & (self._s <= upper)
        values = [
            np.interp(lower, self._s, self._curvature),
            np.interp(upper, self._s, self._curvature),
            *self._curvature[inside],
        ]
        return 0.0, max(values)


class _StraightCLCS:
    @staticmethod
    def length():
        return 10.0

    @staticmethod
    def curvature_range(lower, upper):
        return 0.0, 0.0


class TestVPCurvatureSpeedLimit(unittest.TestCase):
    def test_trajectory_clcs_resamples_near_stationary_segments(self):
        states = [
            CustomState(time_step=0, position=np.array([0.0, 0.0])),
            CustomState(time_step=1, position=np.array([0.0004, 0.0004])),
            CustomState(time_step=2, position=np.array([0.0010, 0.0008])),
            CustomState(time_step=3, position=np.array([0.5, 0.02])),
            CustomState(time_step=4, position=np.array([1.0, 0.08])),
        ]

        clcs, _ = VPTrajectoryContext()._build_trajectory_clcs(states)
        segment_lengths = np.linalg.norm(np.diff(clcs.ref_path, axis=0), axis=1)

        self.assertGreaterEqual(float(np.min(segment_lengths)), 0.009)
        for state in states:
            clcs.convert_to_curvilinear_coords(*state.position)

    def test_uses_maximum_absolute_curvature_in_each_interval(self):
        limits = VPConstraintExtraction._curvature_velocity_limits(
            _PiecewiseCurvatureCLCS(),
            s_min=[0.0, 1.0, 3.0, -math.inf],
            s_max=[1.0, 3.0, 1.0, math.inf],
            a_lat_max=2.0,
        )

        np.testing.assert_allclose(limits, [2.0, 1.0, 1.0, 1.0])

    def test_straight_path_does_not_add_a_speed_limit(self):
        limits = VPConstraintExtraction._curvature_velocity_limits(
            _StraightCLCS(),
            s_min=[0.0],
            s_max=[10.0],
            a_lat_max=2.0,
        )

        self.assertTrue(math.isinf(limits[0]))

    def test_rejects_non_positive_lateral_acceleration(self):
        with self.assertRaises(ValueError):
            VPConstraintExtraction._curvature_velocity_limits(
                _StraightCLCS(),
                s_min=[0.0],
                s_max=[10.0],
                a_lat_max=0.0,
            )


if __name__ == "__main__":
    unittest.main()
