"""Vectorised, bitwise-identical polyline helpers for ``commonroad_clcs``.

``commonroad_clcs.util`` computes the path length and the orientation of a
polyline in Python loops (one ``np.linalg.norm`` / ``np.arctan2`` call per
segment).  The CLCS constructor, its reference-path validity check, curvature
computation and polyline resampling call them on dense (0.1 m) reference
paths, so building one trajectory CLCS spends several milliseconds there.

* Path length: for a float64 ``(n, 2)`` array, ``np.linalg.norm(v)`` is
  ``sqrt(v.dot(v))`` and ``v.dot(v) == v[0] * v[0] + v[1] * v[1]`` exactly
  (no fused multiply-add for two-element dot products), while ``np.cumsum``
  adds sequentially like the loop.
* Orientation: the loop evaluates ``np.arctan2(dy, dx)`` of each segment and
  repeats the last value; the vectorised call evaluates the same ufunc on the
  same operands.

Both replacements are verified bit for bit against the library loops on a
fixed sample when they are installed and are only enabled if every value
matches (an ``arctan2`` SIMD dispatch that differs from the scalar path would
disable the orientation fast path).  Every other input (other dtypes or
dimensions, fewer than two points, lists) is delegated to the library.

Set ``CRREPAIR_VP_CLCS_PATHLENGTH_FASTPATH=0`` or
``CRREPAIR_VP_CLCS_ORIENTATION_FASTPATH=0`` to keep the library loops.
"""

import os

import numpy as np

FASTPATH_ENV = "CRREPAIR_VP_CLCS_PATHLENGTH_FASTPATH"
ORIENTATION_FASTPATH_ENV = "CRREPAIR_VP_CLCS_ORIENTATION_FASTPATH"
_ORIGINAL = None
_ORIGINAL_ORIENTATION = None


def _is_dense_polyline(polyline):
    return (
        isinstance(polyline, np.ndarray)
        and polyline.dtype == np.float64
        and polyline.ndim == 2
        and polyline.shape[1] == 2
        and len(polyline) >= 2
    )


def _fast_compute_pathlength_from_polyline(polyline):
    """Return the cumulative segment length exactly as the library loop."""
    if _is_dense_polyline(polyline):
        delta = polyline[1:] - polyline[:-1]
        distance = np.empty(len(polyline), dtype=np.float64)
        distance[0] = 0.0
        np.cumsum(
            np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]),
            out=distance[1:],
        )
        return distance
    return _ORIGINAL(polyline)


def _fast_compute_orientation_from_polyline(polyline):
    """Return the segment orientations exactly as the library loop."""
    if _is_dense_polyline(polyline):
        delta = polyline[1:] - polyline[:-1]
        orientation = np.arctan2(delta[:, 1], delta[:, 0])
        return np.concatenate((orientation, orientation[-1:]))
    return _ORIGINAL_ORIENTATION(polyline)


def _sample_polylines():
    rng = np.random.default_rng(20261125)
    samples = []
    for count, scale in ((2, 1.0), (3, 1.0e-3), (257, 0.1), (1031, 1.0), (4099, 10.0)):
        points = np.cumsum(rng.normal(size=(count, 2)) * scale, axis=0)
        samples.append(points + rng.normal(size=2) * 1.0e4)
    # Axis-aligned, opposite and tiny segments.
    samples.append(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                             [0.0, 0.0], [-1.0e-12, -1.0e-12], [-1.0, 0.0]]))
    return samples


def _matches(fast, original):
    for polyline in _sample_polylines():
        expected = np.asarray(original(polyline))
        result = fast(polyline)
        if not (
            result.dtype == expected.dtype
            and result.shape == expected.shape
            and np.array_equal(result, expected, equal_nan=True)
        ):
            return False
    return True


def _disabled(env_name):
    return os.environ.get(env_name, "1").strip().lower() in {
        "0", "false", "no", "off"
    }


def install():
    """Install the verified fast paths into the modules using the loops."""
    global _ORIGINAL, _ORIGINAL_ORIENTATION
    try:
        import commonroad_clcs.clcs as clcs_module
        import commonroad_clcs.util as util_module
    except ImportError:
        return False
    installed = False

    current = getattr(util_module, "compute_pathlength_from_polyline", None)
    if current is not None and not _disabled(FASTPATH_ENV):
        if current is not _fast_compute_pathlength_from_polyline:
            _ORIGINAL = current
        if _matches(_fast_compute_pathlength_from_polyline, _ORIGINAL):
            for module in (util_module, clcs_module):
                if getattr(module, "compute_pathlength_from_polyline", None) is _ORIGINAL:
                    module.compute_pathlength_from_polyline = (
                        _fast_compute_pathlength_from_polyline
                    )
            installed = True

    current = getattr(util_module, "compute_orientation_from_polyline", None)
    if current is not None and not _disabled(ORIENTATION_FASTPATH_ENV):
        if current is not _fast_compute_orientation_from_polyline:
            _ORIGINAL_ORIENTATION = current
        if _matches(_fast_compute_orientation_from_polyline, _ORIGINAL_ORIENTATION):
            for module in (util_module, clcs_module):
                if (
                    getattr(module, "compute_orientation_from_polyline", None)
                    is _ORIGINAL_ORIENTATION
                ):
                    module.compute_orientation_from_polyline = (
                        _fast_compute_orientation_from_polyline
                    )
            installed = True
    return installed
