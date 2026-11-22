"""Monotonicity partitions of monitor lateral coordinates along straight motion.

The STL monitor measures lateral lane distances with the lanelet
transformation routine of Bender et al. (crmonitor.common.helper.
``get_curvilinear_coordinate_system``), the routine also used by Würsching and
Althoff, "Robust and Efficient Curvilinear Coordinate Transformation with
Guaranteed Map Coverage for Motion Planning" (IV 2024, Sec. IV).  For a
polyline segment ``k`` with local frame ``(x, y)`` (tangent, left normal),
length ``L`` and boundary-line slopes ``m_b``/``m_t`` (from the averaged
vertex tangents) the routine uses

    lambda = (x + m_b y) / (L - y (m_t - m_b))
    d      = sign(y) * || (x, y) - (lambda L, 0) ||.

Inside the unique projection domain exactly one segment satisfies
``0 <= lambda <= 1`` and ``d`` is continuous.  For a point moving on a straight
line ``q(t) = q0 + t u`` the local coordinates are affine in ``t``; hence

* the segment-cell boundaries ``lambda = 0``, ``lambda = 1`` and the singular
  line ``L - y (m_t - m_b) = 0`` are crossed at roots of affine functions, and
* inside one cell ``d(t)^2 = P(t) / Q(t)`` with ``deg P = 4``, ``deg Q = 2``,
  so every interior extremum of ``d`` is a real root of the degree-5
  polynomial ``P'Q - PQ'``.

Splitting the motion at all these parameters yields intervals on which the
monitor lateral coordinate is monotone; its range on such an interval is
attained at the interval endpoints.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class MonitorPolylineFrame:
    """Exact replica of the monitor's Bender lanelet transformation."""

    def __init__(self, vertices: np.ndarray) -> None:
        ref_path = np.asarray(vertices, dtype=float)[:, :2]
        seg_dir = ref_path[1:] - ref_path[:-1]
        seg_length = np.sqrt(np.sum(seg_dir * seg_dir, axis=-1))
        keep = seg_length > 0
        seg_dir = seg_dir[keep]
        seg_start = ref_path[:-1][keep]
        seg_end = ref_path[1:][keep]
        seg_length = seg_length[keep]
        if len(seg_length) == 0:
            raise ValueError("degenerate polyline")
        tangents = np.append(seg_end, seg_end[-1:], axis=0) - np.insert(
            seg_start, 0, seg_start[0], axis=0
        )
        tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None]
        self.seg_start = seg_start
        self.seg_dir = seg_dir
        self.seg_length = seg_length
        self.cumsum = np.concatenate((np.zeros(1), np.cumsum(seg_length)))
        self.dir_n = seg_dir / seg_length[:, None]
        normal = self.dir_n[:, ::-1].copy()
        normal[:, 0] *= -1
        self.normal = normal
        rot_mats = np.stack((self.dir_n, normal), axis=-1)
        self._rot_mats = rot_mats

        def to_local(pts):
            return np.squeeze(np.matmul(pts[..., None, :], rot_mats), axis=-2)

        m_b = to_local(tangents[:-1])
        self.m_b = m_b[..., 1] / m_b[..., 0]
        m_t = to_local(tangents[1:])
        self.m_t = m_t[..., 1] / m_t[..., 0]

    def evaluate(
        self, points: np.ndarray, tolerance: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return monitor lateral coordinates and valid-segment counts.

        Identical to ``get_curvilinear_coordinate_system(..., True)(points)``
        for the lateral component (NaN outside the polyline), plus the number
        of segments with ``0 <= lambda <= 1`` (1 inside the unique domain).
        A positive ``tolerance`` widens the segment test to absorb floating
        point noise exactly on a segment boundary line (continuous extension
        instead of the monitor's isolated NaN).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        rel = points[:, None, :] - self.seg_start[None, :, :]
        local = np.squeeze(np.matmul(rel[..., None, :], self._rot_mats), axis=-2)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (local[..., 0] + self.m_b * local[..., 1]) / (
                self.seg_length - local[..., 1] * (self.m_t - self.m_b)
            )
        base = s[..., None] * self.seg_dir + self.seg_start
        distance = np.linalg.norm(points[:, None] - base, axis=-1)
        signed = np.copysign(distance, local[..., 1])
        order = np.argsort(distance, axis=-1)
        s_sorted = np.take_along_axis(s, order, axis=-1)
        rows = np.arange(len(points))
        tol = float(tolerance)
        valid_sorted = (-tol <= s_sorted) & (s_sorted <= 1 + tol)
        chosen = order[rows, np.argmax(valid_sorted, axis=-1)]
        lateral = signed[rows, chosen]
        self.last_chosen = chosen
        count = np.sum((-tol <= s) & (s <= 1 + tol), axis=1)
        lateral = np.where(count == 0, math.nan, lateral)
        # A point on the shared boundary line of segments k and k+1 has
        # lambda_k = 1 and lambda_{k+1} = 0; both describe the same
        # (continuous) coordinate and are counted once.
        if s.shape[1] > 1:
            shared_tol = max(tol, 1.0e-12)
            shared = (np.abs(s[:, :-1] - 1.0) <= shared_tol) & (
                np.abs(s[:, 1:]) <= shared_tol
            )
            count = count - np.sum(shared, axis=1)
        return lateral, count

    def evaluate_with_segment(
        self, points: np.ndarray, tolerance: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``evaluate`` plus the index of the selected segment."""
        lateral, count = self.evaluate(points, tolerance)
        return lateral, count, np.asarray(self.last_chosen, dtype=int)

    def lateral_on_segments(
        self, points: np.ndarray, segments: np.ndarray
    ) -> np.ndarray:
        """Closed-form lateral coordinate of ``points[i]`` on ``segments[i]``.

        Inside a partition interval the selected segment is fixed, so the
        coordinate (and its continuous limit at the interval ends) follows
        directly from the segment formula.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        k = np.asarray(segments, dtype=int).reshape(-1)
        rel = points - self.seg_start[k]
        x = np.sum(rel * self.dir_n[k], axis=1)
        y = np.sum(rel * self.normal[k], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (x + self.m_b[k] * y) / (
                self.seg_length[k] - y * (self.m_t[k] - self.m_b[k])
            )
        base = self.seg_start[k] + lam[:, None] * self.seg_dir[k]
        return np.copysign(np.linalg.norm(points - base, axis=1), y)

    def motion_breakpoints(
        self, q0: np.ndarray, u: np.ndarray, length: float
    ) -> np.ndarray:
        """Critical parameters ``t`` in ``(0, length)`` of ``d(q0 + t u)``."""
        q0 = np.asarray(q0, dtype=float)
        u = np.asarray(u, dtype=float)
        rel = q0[None, :] - self.seg_start
        a0 = np.sum(rel * self.dir_n, axis=1)
        b0 = np.sum(rel * self.normal, axis=1)
        a1 = self.dir_n @ u
        b1 = self.normal @ u
        L = self.seg_length
        c = self.m_t - self.m_b
        n0 = a0 + self.m_b * b0
        n1 = a1 + self.m_b * b1
        d0 = L - c * b0
        d1 = -c * b1
        candidates: List[float] = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for numerator0, numerator1 in (
                (n0, n1),  # lambda = 0
                (n0 - d0, n1 - d1),  # lambda = 1
                (d0, d1),  # singular line
            ):
                t = -numerator0 / numerator1
                mask = np.isfinite(t) & (t > 0.0) & (t < length)
                candidates.extend(float(item) for item in t[mask])
            lam0 = n0 / d0
            lam1 = (n0 + n1 * length) / (d0 + d1 * length)
        # Segments which can be selected somewhere on the motion.
        crossing_any = np.zeros(len(L), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for numerator0, numerator1 in ((n0, n1), (n0 - d0, n1 - d1)):
                t = -numerator0 / numerator1
                crossing_any |= np.isfinite(t) & (t > 0.0) & (t < length)
        relevant = (
            ((lam0 >= 0) & (lam0 <= 1))
            | ((lam1 >= 0) & (lam1 <= 1))
            | crossing_any
        )
        for k in np.flatnonzero(relevant):
            xl = np.polynomial.Polynomial((a0[k], a1[k]))
            yl = np.polynomial.Polynomial((b0[k], b1[k]))
            N = np.polynomial.Polynomial((n0[k], n1[k]))
            D = np.polynomial.Polynomial((d0[k], d1[k]))
            X = xl * D - N * L[k]
            Y = yl * D
            P = X * X + Y * Y
            Q = D * D
            F = P.deriv() * Q - P * Q.deriv()
            coefficients = F.coef
            scale = np.max(np.abs(coefficients)) if len(coefficients) else 0.0
            if scale == 0.0:
                continue
            trimmed = np.polynomial.Polynomial(coefficients / scale).trim(1e-14)
            if trimmed.degree() < 1:
                continue
            for root in trimmed.roots():
                if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real)):
                    t = float(root.real)
                    if 0.0 < t < length:
                        candidates.append(t)
        return np.unique(np.asarray(candidates, dtype=float))
