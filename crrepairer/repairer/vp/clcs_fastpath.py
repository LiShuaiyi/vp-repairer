"""Vectorised, bitwise-identical path length for ``commonroad_clcs``.

``commonroad_clcs.util.compute_pathlength_from_polyline`` accumulates segment
lengths in a Python loop with one ``np.linalg.norm`` call per segment.  The
CLCS constructor, curvature computation and polyline resampling call it on
dense (0.1 m) reference paths, so building one acceleration trajectory CLCS
spends several milliseconds in that loop.

For a float64 ``(n, 2)`` array, ``np.linalg.norm(v)`` is ``sqrt(v.dot(v))``
and ``v.dot(v) == v[0] * v[0] + v[1] * v[1]`` exactly (no fused multiply-add
for two-element dot products), while ``np.cumsum`` adds sequentially like the
loop.  The replacement below therefore returns the same array bit for bit.
Every other input (other dtypes, dimensions, fewer than two points, lists)
is delegated to the original library function.

Set ``CRREPAIR_VP_CLCS_PATHLENGTH_FASTPATH=0`` to keep the library loop.
"""

import os

import numpy as np

FASTPATH_ENV = "CRREPAIR_VP_CLCS_PATHLENGTH_FASTPATH"
_ORIGINAL = None


def _fast_compute_pathlength_from_polyline(polyline):
    """Return the cumulative segment length exactly as the library loop."""
    if (
        isinstance(polyline, np.ndarray)
        and polyline.dtype == np.float64
        and polyline.ndim == 2
        and polyline.shape[1] == 2
        and len(polyline) >= 2
    ):
        delta = polyline[1:] - polyline[:-1]
        distance = np.empty(len(polyline), dtype=np.float64)
        distance[0] = 0.0
        np.cumsum(
            np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]),
            out=distance[1:],
        )
        return distance
    return _ORIGINAL(polyline)


def install():
    """Install the fast path into the modules that call the library loop."""
    global _ORIGINAL
    if os.environ.get(FASTPATH_ENV, "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return False
    try:
        import commonroad_clcs.clcs as clcs_module
        import commonroad_clcs.util as util_module
    except ImportError:
        return False
    current = getattr(util_module, "compute_pathlength_from_polyline", None)
    if current is None:
        return False
    if current is not _fast_compute_pathlength_from_polyline:
        _ORIGINAL = current
    for module in (util_module, clcs_module):
        if getattr(module, "compute_pathlength_from_polyline", None) is _ORIGINAL:
            module.compute_pathlength_from_polyline = (
                _fast_compute_pathlength_from_polyline
            )
    return True
