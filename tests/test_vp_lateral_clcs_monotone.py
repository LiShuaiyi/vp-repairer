import math

import numpy as np
import pytest

from crmonitor.common.helper import get_curvilinear_coordinate_system
from crrepairer.repairer.vp.lateral_clcs_monotone import MonitorPolylineFrame


def _random_polyline(rng):
    count = int(rng.integers(3, 9))
    angles = np.cumsum(rng.normal(0.0, 0.15, count))
    steps = rng.uniform(2.0, 8.0, count)
    return np.cumsum(
        np.stack([np.cos(angles) * steps, np.sin(angles) * steps], axis=1),
        axis=0,
    ), angles, steps


@pytest.mark.parametrize("seed", range(5))
def test_frame_matches_monitor_lateral_coordinate(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        vertices, _, _ = _random_polyline(rng)
        frame = MonitorPolylineFrame(vertices)
        reference = get_curvilinear_coordinate_system(vertices, True)
        points = vertices.mean(axis=0) + rng.normal(0.0, 8.0, (500, 2))
        lateral, _ = frame.evaluate(points)
        expected = reference(points)[:, 1]
        np.testing.assert_allclose(
            np.nan_to_num(lateral, nan=1e9),
            np.nan_to_num(expected, nan=1e9),
            atol=1e-9,
        )


@pytest.mark.parametrize("seed", range(5))
def test_lateral_coordinate_is_monotone_between_breakpoints(seed):
    rng = np.random.default_rng(100 + seed)
    checked = 0
    for _ in range(40):
        vertices, angles, steps = _random_polyline(rng)
        frame = MonitorPolylineFrame(vertices)
        heading0 = np.array([math.cos(angles[0]), math.sin(angles[0])])
        origin = vertices[0] + rng.normal(0.0, 3.0, 2) - 3.0 * heading0
        theta = angles[0] + rng.normal(0.0, 0.3)
        direction = np.array([math.cos(theta), math.sin(theta)])
        length = float(steps.sum()) * 1.2
        t = np.linspace(0.0, length, 3001)
        lateral, count = frame.evaluate(origin + t[:, None] * direction)
        breaks = np.concatenate(
            ([0.0], frame.motion_breakpoints(origin, direction, length), [length])
        )
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            inside = (t > lower + 1e-6) & (t < upper - 1e-6)
            if inside.sum() < 3 or np.any(count[inside] != 1):
                continue
            delta = np.diff(lateral[inside])
            assert np.all(delta >= -1e-9) or np.all(delta <= 1e-9)
            checked += 1
    assert checked > 50


def test_shared_vertex_line_counts_once():
    vertices = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 2.0]])
    frame = MonitorPolylineFrame(vertices)
    # Point on the bisector line through the middle vertex.
    tangent = np.array([20.0, 2.0]) / np.linalg.norm([20.0, 2.0])
    normal = np.array([-tangent[1], tangent[0]])
    point = np.array([10.0, 0.0]) + 1.5 * normal
    lateral, count = frame.evaluate(point[None, :], tolerance=1e-9)
    assert count[0] == 1
    assert lateral[0] == pytest.approx(1.5, abs=1e-6)
