"""Piecewise-monotone robustness profiles of ``in_same_lane`` and ``cut_in``.

Along the VP reference path the ego pose is a function of progress ``s``:
the centre moves on the trajectory CLCS polyline and the heading is the chord
direction of that polyline over ``s +- 0.25`` (VPOptimization).  The
monitor robustness of

* ``in_same_lane(ego, other)``:  rho(s) = fmin(D(ego(s), L_o), D(other, L_e(s)))
  with D the monitor ``distance_to_lanes`` (Bender lateral coordinates of the
  footprint corners w.r.t. boundary polylines) and L_e(s) the lanes of the
  ego shape lanelet assignment, and
* ``cut_in(other, ego)``:
  alpha(s) = min(-single(other), S(rho(s)),
                 max(min(S(d_e(s) - d_o), o1), min(S(d_o - d_e(s)), o2)))
  with d_e(s) the lateral coordinate of the ego centre in the cutting lane's
  C++ CLCS (same Bender segment formula)

is evaluated on a partition of the progress axis whose points are

1. the CLCS polyline vertices and the heading-chord zone limits (vertex
   +-0.25 m) -- between them the footprint moves rigidly or (inside a zone)
   with a monotone heading,
2. for every footprint corner and every relevant boundary polyline, the
   crossings of the Bender segment-cell lines and the real roots of the
   degree-5 derivative numerator (``MonitorPolylineFrame.motion_breakpoints``),
3. the contact events at which the ego rectangle starts/stops intersecting a
   lanelet polygon (vertex-edge contacts under translation), i.e. all changes
   of the ego lanelet assignment,
4. for ``cut_in``: the critical points of d_e(s) for the cutting lane.

On each open piece every component (corner/boundary lateral coordinate,
lanelet assignment, d_e) is monotone or constant, so its range is attained at
the piece ends; at partition points the robustness is evaluated exactly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import shapely

from crrepairer.repairer.vp.lateral_clcs_monotone import MonitorPolylineFrame

CHORD_STEPS = (0.25, 0.1, 0.05)


class RangeTable:
    """Sparse tables for O(1) range minimum/maximum queries."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        self.size = len(lower)
        self._min = [lower]
        self._max = [upper]
        width = 1
        while 2 * width <= self.size:
            previous_min, previous_max = self._min[-1], self._max[-1]
            self._min.append(
                np.minimum(previous_min[:-width], previous_min[width:])
            )
            self._max.append(
                np.maximum(previous_max[:-width], previous_max[width:])
            )
            width *= 2

    def query(self, first: int, last: int) -> Tuple[float, float]:
        """Range over the inclusive index interval; empty -> (inf, -inf)."""
        if last < first or self.size == 0:
            return math.inf, -math.inf
        level = int(last - first + 1).bit_length() - 1
        span = 1 << level
        return (
            float(min(self._min[level][first], self._min[level][last - span + 1])),
            float(max(self._max[level][first], self._max[level][last - span + 1])),
        )


class EgoPath:
    """Ego pose on the trajectory CLCS polyline exactly as VP builds it."""

    def __init__(
        self,
        vertices: np.ndarray,
        s_vertices: np.ndarray,
        local_corners: np.ndarray,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=float)[:, :2]
        self.s_vertices = np.asarray(s_vertices, dtype=float)
        direction = np.diff(self.vertices, axis=0)
        self.length = np.linalg.norm(direction, axis=1)
        self.unit = direction / np.where(self.length > 0, self.length, 1.0)[:, None]
        self.headings = np.arctan2(direction[:, 1], direction[:, 0])
        self.local_corners = np.asarray(local_corners, dtype=float)
        self.radius = float(np.max(np.linalg.norm(self.local_corners, axis=1)))

    def segment_index(self, s: np.ndarray) -> np.ndarray:
        return np.clip(
            np.searchsorted(self.s_vertices, s, side="right") - 1,
            0,
            len(self.length) - 1,
        )

    def center(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        index = self.segment_index(s)
        points = self.vertices[index] + (s - self.s_vertices[index])[:, None] * (
            self.unit[index]
        )
        outside = (s < self.s_vertices[0]) | (s > self.s_vertices[-1])
        points[outside] = math.nan
        return points

    def heading(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        heading = np.full(len(s), math.nan)
        for ds in CHORD_STEPS:
            missing = ~np.isfinite(heading)
            if not np.any(missing):
                break
            delta = self.center(s[missing] + ds) - self.center(s[missing] - ds)
            norm = np.linalg.norm(delta, axis=1)
            heading[missing] = np.where(
                np.isfinite(norm) & (norm > 1.0e-6),
                np.arctan2(delta[:, 1], delta[:, 0]),
                math.nan,
            )
        return heading

    def corners(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        center = self.center(s)
        heading = self.heading(s)
        cosine, sine = np.cos(heading)[:, None], np.sin(heading)[:, None]
        local = self.local_corners
        corners = np.empty((len(s), len(local), 2))
        corners[:, :, 0] = local[None, :, 0] * cosine - local[None, :, 1] * sine + center[:, 0, None]
        corners[:, :, 1] = local[None, :, 0] * sine + local[None, :, 1] * cosine + center[:, 1, None]
        return center, heading, corners

    def zone_points(self, s_min: float, s_max: float) -> np.ndarray:
        # The heading chord spans s +- 0.25; it bends only while a chord end
        # passes a vertex.  (The shorter fallback chords are used only within
        # 0.25 of the path ends, where these points are added as well.)
        chord = CHORD_STEPS[0]
        points = np.concatenate(
            [self.s_vertices, self.s_vertices + chord, self.s_vertices - chord]
            + [
                np.asarray([self.s_vertices[0] + ds, self.s_vertices[-1] - ds])
                for ds in CHORD_STEPS
            ]
        )
        return points[(points > s_min) & (points < s_max)]

    def rigid_pieces(self, s_min: float, s_max: float):
        """(start, end, centre at start, direction, segment heading)."""
        for index in range(len(self.length)):
            start = max(float(self.s_vertices[index]), s_min)
            end = min(float(self.s_vertices[index + 1]), s_max)
            if end - start <= 1.0e-9 or self.length[index] <= 0.0:
                continue
            center = self.vertices[index] + (start - self.s_vertices[index]) * self.unit[index]
            yield start, end, center, self.unit[index], float(self.headings[index])

    def heading_deviation(self, s_left: np.ndarray, s_right: np.ndarray) -> np.ndarray:
        """Max |heading - rigid segment heading| on each piece (monotone)."""
        middle = 0.5 * (s_left + s_right)
        segment_heading = self.headings[self.segment_index(middle)]

        def gap(values):
            return np.abs(np.arctan2(np.sin(values - segment_heading), np.cos(values - segment_heading)))

        return np.maximum(gap(self.heading(s_left)), gap(self.heading(s_right)))


def _rigid_corner_origins(local_corners, center, heading):
    rotation = np.array(
        [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
    )
    return center[None, :] + local_corners @ rotation.T


def contact_events(
    rectangle: np.ndarray, direction: np.ndarray, length: float, polygon: np.ndarray
) -> np.ndarray:
    """Translation parameters at which a moving convex rectangle touches a polygon.

    Intersection status can only change when a rectangle vertex crosses a
    polygon edge or a polygon vertex crosses a rectangle edge.
    """
    events = []
    u = np.asarray(direction, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    a = poly
    e = np.roll(poly, -1, axis=0) - poly

    def cross(p, q):
        return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        # rectangle vertex r + t u = a + mu e
        rel = a[None, :, :] - rectangle[:, None, :]
        det = cross(u[None, None, :], -e[None, :, :])
        t = cross(rel, -e[None, :, :]) / det
        mu = cross(u[None, None, :], rel) / det
        mask = np.isfinite(t) & (mu >= -1e-12) & (mu <= 1 + 1e-12) & (t > 0) & (t < length)
        events.extend(t[mask].tolist())
        # polygon vertex p = r_i + t u + nu f_i
        f = np.roll(rectangle, -1, axis=0) - rectangle
        rel = a[None, :, :] - rectangle[:, None, :]
        det = cross(u[None, None, :], f[:, None, :])
        t = cross(rel, f[:, None, :]) / det
        nu = cross(u[None, None, :], rel) / det
        mask = np.isfinite(t) & (nu >= -1e-12) & (nu <= 1 + 1e-12) & (t > 0) & (t < length)
        events.extend(t[mask].tolist())
    return np.unique(np.asarray(events, dtype=float))


def polygon_crossings(
    origin: np.ndarray, direction: np.ndarray, length: float, polygon: np.ndarray
) -> np.ndarray:
    """Parameters at which ``origin + t direction`` crosses a polygon edge."""
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 2:
        return np.asarray([], dtype=float)
    a = poly
    e = np.roll(poly, -1, axis=0) - poly
    u = np.asarray(direction, dtype=float)
    rel = a - np.asarray(origin, dtype=float)[None, :]
    det = u[0] * (-e[:, 1]) - u[1] * (-e[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[:, 0] * (-e[:, 1]) - rel[:, 1] * (-e[:, 0])) / det
        mu = (u[0] * rel[:, 1] - u[1] * rel[:, 0]) / det
    mask = np.isfinite(t) & (mu >= 0.0) & (mu <= 1.0) & (t > 0.0) & (t < length)
    return np.unique(t[mask])


class LaneletAssignment:
    """Shape lanelet assignment (find_lanelet_by_shape) and lane unions."""

    def __init__(self, world: Any) -> None:
        self.world = world
        network = world.road_network.lanelet_network
        lanelets = list(network.lanelets)
        self.ids = np.asarray([int(item.lanelet_id) for item in lanelets], dtype=int)
        self.polygons = [np.asarray(item.polygon.vertices, dtype=float) for item in lanelets]
        self.tree = shapely.STRtree([item.polygon.shapely_object for item in lanelets])
        self._union_cache: Dict[frozenset, frozenset] = {}

    def assign(self, corners: np.ndarray, buffer: Optional[np.ndarray] = None) -> List[frozenset]:
        geometries = shapely.polygons(np.where(np.isfinite(corners), corners, 0.0))
        if buffer is not None:
            geometries = shapely.buffer(geometries, buffer, join_style="mitre")
        result = [set() for _ in range(len(corners))]
        pairs = self.tree.query(geometries, predicate="intersects")
        for geometry_index, tree_index in zip(pairs[0], pairs[1]):
            result[int(geometry_index)].add(int(self.ids[int(tree_index)]))
        return [frozenset(item) for item in result]

    def lane_union(self, lanelet_set: frozenset) -> frozenset:
        cached = self._union_cache.get(lanelet_set)
        if cached is None:
            lanes = self.world.road_network.find_lanes_by_lanelets(set(lanelet_set))
            cached = frozenset(int(item) for lane in lanes for item in lane.contained_lanelets)
            self._union_cache[lanelet_set] = cached
        return cached

    def events(self, path: EgoPath, s_min: float, s_max: float) -> np.ndarray:
        result = []
        for start, end, center, direction, heading in path.rigid_pieces(s_min, s_max):
            rectangle = _rigid_corner_origins(path.local_corners, center, heading)
            swept = shapely.convex_hull(
                shapely.multipoints(np.concatenate([rectangle, rectangle + (end - start) * direction]))
            )
            for tree_index in self.tree.query(swept):
                for t in contact_events(rectangle, direction, end - start, self.polygons[int(tree_index)]):
                    result.append(start + float(t))
        return np.unique(np.asarray(result, dtype=float))


def boundary_polylines(world: Any, lanelet_ids: frozenset) -> Tuple[List[Any], List[Any]]:
    """Monitor ``distance_to_bounds`` boundary polylines (left, right)."""
    network = world.road_network.lanelet_network
    left, right = [], []
    for lanelet_id in sorted(lanelet_ids):
        lanelet = network.find_lanelet_by_id(int(lanelet_id))
        if lanelet.adj_left is not None and lanelet.adj_left not in lanelet_ids:
            left.append((int(lanelet_id), "left", lanelet.left_vertices))
        if lanelet.adj_right is not None and lanelet.adj_right not in lanelet_ids:
            right.append((int(lanelet_id), "right", lanelet.right_vertices))
    return left, right


class SameLaneGeometry:
    """Partition and ego-only component data for one target lane union.

    ``extra_frames`` add point frames (ego centre, e.g. cutting-lane CLCS)
    whose critical points must also be partition points.
    """

    def __init__(
        self,
        path: EgoPath,
        assignment: LaneletAssignment,
        s_min: float,
        s_max: float,
        left_frames: Sequence[MonitorPolylineFrame],
        right_frames: Sequence[MonitorPolylineFrame],
        lanelet_events: np.ndarray,
        center_frames: Sequence[Tuple[MonitorPolylineFrame, float]] = (),
        extra_breakpoints: Optional[np.ndarray] = None,
    ) -> None:
        self.path = path
        self.assignment = assignment
        self.left_frames = list(left_frames)
        self.right_frames = list(right_frames)
        self.center_frames = list(center_frames)
        frames = self.left_frames + self.right_frames
        breakpoints: List[float] = []
        center_breakpoints: List[float] = []
        for start, end, center, direction, heading in path.rigid_pieces(s_min, s_max):
            origins = _rigid_corner_origins(path.local_corners, center, heading)
            for origin in origins:
                for frame in frames:
                    breakpoints.extend(start + frame.motion_breakpoints(origin, direction, end - start))
            for frame, _ in self.center_frames:
                center_breakpoints.extend(start + frame.motion_breakpoints(center, direction, end - start))
        grid = np.concatenate(
            [
                [s_min, s_max],
                path.zone_points(s_min, s_max),
                np.asarray(breakpoints, dtype=float),
                np.asarray(center_breakpoints, dtype=float),
                lanelet_events[(lanelet_events > s_min) & (lanelet_events < s_max)],
                np.asarray(
                    [] if extra_breakpoints is None else extra_breakpoints,
                    dtype=float,
                ),
            ]
        )
        grid = np.unique(grid)
        grid = grid[np.concatenate(([True], np.diff(grid) > 1.0e-9))]
        grid = grid[(grid >= s_min) & (grid <= s_max)]
        self.grid = grid
        self.counts = {
            "corner_boundary_breakpoints": len(breakpoints),
            "center_breakpoints": len(center_breakpoints),
            "lanelet_events": int(np.count_nonzero((lanelet_events > s_min) & (lanelet_events < s_max))),
            "projection_domain_crossings": int(0 if extra_breakpoints is None else len(extra_breakpoints)),
            "zone_points": int(len(path.zone_points(s_min, s_max))),
            "partition_points": int(len(grid)),
        }
        middle = 0.5 * (grid[:-1] + grid[1:])
        # Exact poses at partition points and piece midpoints.
        _, _, point_corners = path.corners(grid)
        mid_center, _, mid_corners = path.corners(middle)
        point_center = path.center(grid)
        self.margin = 2.0 * path.radius * path.heading_deviation(grid[:-1], grid[1:])
        # Diagnostics only: 0 evaluates the heading-zone pieces as if the
        # motion were rigid (used to count decisions that depend on the
        # zone margin).
        self.margin_scale = 1.0
        corner_count = point_corners.shape[1]

        def frame_data(frame):
            # point values (monitor-exact) and piece data (segment fixed)
            values, count = frame.evaluate(point_corners.reshape(-1, 2), tolerance=0.0)
            mid_values, mid_count, mid_segment = frame.evaluate_with_segment(
                mid_corners.reshape(-1, 2), tolerance=0.0
            )
            segment = mid_segment.reshape(-1, corner_count)
            left_limit = frame.lateral_on_segments(
                point_corners[:-1].reshape(-1, 2), segment.reshape(-1)
            ).reshape(-1, corner_count)
            right_limit = frame.lateral_on_segments(
                point_corners[1:].reshape(-1, 2), segment.reshape(-1)
            ).reshape(-1, corner_count)
            return {
                "point": np.where(count.reshape(-1, corner_count) >= 1, values.reshape(-1, corner_count), math.nan),
                "present": mid_count.reshape(-1, corner_count) >= 1,
                "ambiguous": mid_count.reshape(-1, corner_count) > 1,
                "segment": segment,
                "left": left_limit,
                "right": right_limit,
            }

        self.left_data = [frame_data(frame) for frame in self.left_frames]
        self.right_data = [frame_data(frame) for frame in self.right_frames]

        def center_data(frame):
            values, count = frame.evaluate(point_center, tolerance=1.0e-7)
            mid_values, mid_count, mid_segment = frame.evaluate_with_segment(mid_center, tolerance=1.0e-7)
            return {
                "point": np.where(count >= 1, values, math.nan),
                "present": mid_count >= 1,
                "ambiguous": mid_count > 1,
                "segment": mid_segment,
                "left": frame.lateral_on_segments(point_center[:-1], mid_segment),
                "right": frame.lateral_on_segments(point_center[1:], mid_segment),
            }

        self.center_data = [center_data(frame) for frame, _ in self.center_frames]
        # Lanelet assignment: exact at points, constant on rigid pieces.
        self.point_sets = assignment.assign(point_corners)
        self.mid_sets = assignment.assign(mid_corners)
        stable = np.ones(len(middle), dtype=bool)
        zone = self.margin > 1.0e-12
        if np.any(zone):
            # Inside heading-chord zones the footprint deviates from the
            # rigid motion by at most r * deviation; certify that this does
            # not change the lane union.
            index = np.flatnonzero(zone)
            buffer = 0.5 * self.margin[index] + 1.0e-9
            for corners_at in (point_corners[:-1][index], mid_corners[index], point_corners[1:][index]):
                dilated = assignment.assign(corners_at, buffer)
                eroded = assignment.assign(corners_at, -buffer)
                for position, item in enumerate(index):
                    target = assignment.lane_union(self.mid_sets[item])
                    if not (
                        assignment.lane_union(dilated[position]) == target
                        and assignment.lane_union(eroded[position]) == target
                    ):
                        stable[item] = False
        self.assignment_stable = stable
        self.point_unions = [assignment.lane_union(item) for item in self.point_sets]
        self.mid_unions = [assignment.lane_union(item) for item in self.mid_sets]
        self.ambiguous = np.zeros(len(middle), dtype=bool)
        for data in self.left_data + self.right_data:
            self.ambiguous |= np.any(data["ambiguous"], axis=1)
        self.center_ambiguous = np.zeros(len(middle), dtype=bool)
        for data in self.center_data:
            self.center_ambiguous |= data["ambiguous"] | ~data["present"]

    # -- component reductions --------------------------------------------

    @staticmethod
    def _side_reduce(values: List[np.ndarray], present: List[np.ndarray], reducer: str) -> np.ndarray:
        if not values:
            return np.full(values[0].shape[0] if values else 0, math.inf)
        stacked = np.concatenate(values, axis=1)
        mask = np.concatenate(present, axis=1) & np.isfinite(stacked)
        if reducer == "left":
            reduced = -np.min(np.where(mask, stacked, math.inf), axis=1)
        else:
            reduced = np.max(np.where(mask, stacked, -math.inf), axis=1)
        return np.where(np.any(mask, axis=1), reduced, math.inf)

    def ego_term_points(self) -> np.ndarray:
        """D(ego(s), L_target) at partition points (monitor exact)."""
        n = len(self.grid)
        left = (
            self._side_reduce([d["point"] for d in self.left_data], [np.isfinite(d["point"]) for d in self.left_data], "left")
            if self.left_data else np.full(n, math.inf)
        )
        right = (
            self._side_reduce([d["point"] for d in self.right_data], [np.isfinite(d["point"]) for d in self.right_data], "right")
            if self.right_data else np.full(n, math.inf)
        )
        return np.fmin(left, right)

    def ego_term_pieces(
        self,
        left_override: Optional[Tuple[int, np.ndarray]] = None,
        right_override: Optional[Tuple[int, np.ndarray]] = None,
        pieces: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bounds of D(ego(s), L_target) on open pieces from end values.

        Each corner/boundary function is monotone on a piece: its range is
        spanned by its two (limit) end values, widened only by the heading
        zone margin.  -min / max over functions are bounded by interval
        arithmetic.  Overrides replace the left/right end values of the
        given pieces (reachable-interval endpoints).
        """
        if pieces is None:
            pieces = np.arange(len(self.grid) - 1)
        margin = self.margin_scale * self.margin[pieces][:, None]

        def side(data_list, reducer):
            if not data_list:
                inf = np.full(len(pieces), math.inf)
                return inf, inf
            lows, highs, presents = [], [], []
            for position, data in enumerate(data_list):
                left = data["left"][pieces]
                right = data["right"][pieces]
                if left_override is not None:
                    left = left_override[1][reducer][position]
                if right_override is not None:
                    right = right_override[1][reducer][position]
                lows.append(np.fmin(left, right) - margin)
                highs.append(np.fmax(left, right) + margin)
                presents.append(data["present"][pieces])
            low = np.concatenate(lows, axis=1)
            high = np.concatenate(highs, axis=1)
            present = np.concatenate(presents, axis=1)
            any_present = np.any(present, axis=1)
            if reducer == "left":
                m_low = np.min(np.where(present, low, math.inf), axis=1)
                m_high = np.min(np.where(present, high, math.inf), axis=1)
                return np.where(any_present, -m_high, math.inf), np.where(any_present, -m_low, math.inf)
            low_max = np.max(np.where(present, low, -math.inf), axis=1)
            high_max = np.max(np.where(present, high, -math.inf), axis=1)
            return np.where(any_present, low_max, math.inf), np.where(any_present, high_max, math.inf)

        left_lower, left_upper = side(self.left_data, "left")
        right_lower, right_upper = side(self.right_data, "right")
        return np.fmin(left_lower, right_lower), np.fmin(left_upper, right_upper)

    def endpoint_values(self, s: float, piece: int) -> Dict[str, List[np.ndarray]]:
        """Closed-form corner values at ``s`` with the piece's fixed segments."""
        _, _, corners = self.path.corners(np.asarray([s]))
        corners = corners[0]
        result: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
        for key, data_list, frames in (
            ("left", self.left_data, self.left_frames),
            ("right", self.right_data, self.right_frames),
        ):
            for data, frame in zip(data_list, frames):
                result[key].append(
                    frame.lateral_on_segments(corners, data["segment"][piece])[None, :]
                )
        center = self.path.center(np.asarray([s]))[0]
        result["center"] = [
            frame.lateral_on_segments(center[None, :], np.asarray([data["segment"][piece]]))[0]
            for (frame, _), data in zip(self.center_frames, self.center_data)
        ]
        return result


class SameLaneProfile:
    """rho(s) of in_same_lane for one target frame on a SameLaneGeometry."""

    def __init__(self, geometry: SameLaneGeometry, target_term: Mapping[frozenset, float]) -> None:
        self.geometry = geometry
        self.target_term = target_term
        g = geometry
        ego_points = g.ego_term_points()
        self.point_target = np.asarray([target_term[u] for u in g.point_unions])
        self.piece_target = np.asarray([target_term[u] for u in g.mid_unions])
        self.points = np.fmin(ego_points, self.point_target)
        lower, upper = g.ego_term_pieces()
        self.piece_ego_upper = upper
        stable = g.assignment_stable
        self.piece_lower = np.where(stable, np.fmin(lower, self.piece_target), -math.inf)
        self.piece_upper = np.where(stable, np.fmin(upper, self.piece_target), upper)
        self.piece_lower = np.where(g.ambiguous, -math.inf, self.piece_lower)
        self.piece_upper = np.where(g.ambiguous, math.inf, self.piece_upper)
        self.points_table = RangeTable(self.points, self.points)
        self.pieces_table = RangeTable(self.piece_lower, self.piece_upper)

    def partial_piece(self, piece: int, s_from: float, s_to: float) -> Tuple[float, float]:
        """Bounds of rho on the open sub-piece (s_from, s_to)."""
        g = self.geometry
        if g.ambiguous[piece]:
            return -math.inf, math.inf
        grid = g.grid
        overrides = {}
        for which, s in (("left", s_from), ("right", s_to)):
            if abs(s - grid[piece + (0 if which == "left" else 1)]) <= 1.0e-12:
                continue
            overrides[which] = (piece, g.endpoint_values(s, piece))
        lower, upper = g.ego_term_pieces(
            left_override=overrides.get("left"),
            right_override=overrides.get("right"),
            pieces=np.asarray([piece]),
        )
        target = self.piece_target[piece]
        if not g.assignment_stable[piece]:
            return -math.inf, float(upper[0])
        return float(min(lower[0], target)), float(min(upper[0], target))


def locate(grid: np.ndarray, s_lo: float, s_hi: float):
    """Pieces/points touched by [s_lo, s_hi] within the grid span."""
    if s_lo < grid[0] - 1e-9 or s_hi > grid[-1] + 1e-9 or s_hi < s_lo:
        return None
    s_lo = max(s_lo, grid[0])
    s_hi = min(s_hi, grid[-1])
    first_point = int(np.searchsorted(grid, s_lo, side="left"))
    last_point = int(np.searchsorted(grid, s_hi, side="right")) - 1
    first_piece = min(max(int(np.searchsorted(grid, s_lo, side="right")) - 1, 0), len(grid) - 2)
    last_piece = min(max(int(np.searchsorted(grid, s_hi, side="left")) - 1, 0), len(grid) - 2)
    return s_lo, s_hi, first_point, last_point, first_piece, last_piece


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def classify(lower: float, upper: float):
    """Three-valued sign classification of a robustness range."""
    if lower > 0.0:
        return True
    if upper < 0.0:
        return False
    return None


class _ProfileQuery:
    grid: np.ndarray
    points_table: RangeTable
    pieces_table: RangeTable

    def partial_piece(self, piece: int, s_from: float, s_to: float) -> Tuple[float, float]:
        raise NotImplementedError

    def point_value(self, s: float) -> Optional[Tuple[float, float]]:
        return None

    def query(self, s_lo: float, s_hi: float, stats: Optional[Dict[str, int]] = None):
        """Sound [min, max] of the robustness over [s_lo, s_hi].

        Interior pieces and interior partition points use precomputed
        endpoint ranges; the two boundary pieces are first taken whole and,
        only if that is not decisive, re-evaluated exactly at s_lo/s_hi.
        """
        located = locate(self.grid, float(s_lo), float(s_hi))
        if located is None:
            return None
        s_lo, s_hi, first_point, last_point, first_piece, last_piece = located
        if s_hi - s_lo <= 1.0e-12:
            point = int(np.searchsorted(self.grid, s_lo))
            if point < len(self.grid) and abs(self.grid[point] - s_lo) <= 1.0e-12:
                value = self.points_table.query(point, point)
                return value[0], value[1]
            piece = min(max(point - 1, 0), len(self.grid) - 2)
            return self.partial_piece(piece, s_lo, s_hi)
        lower, upper = self.points_table.query(first_point, last_point)
        whole = self.pieces_table.query(first_piece, last_piece)
        coarse = (min(lower, whole[0]), max(upper, whole[1]))
        if classify(*coarse) is not None:
            return coarse
        if stats is not None:
            stats["endpoint_refinements"] = stats.get("endpoint_refinements", 0) + 1
        inner = self.pieces_table.query(first_piece + 1, last_piece - 1)
        lower, upper = min(lower, inner[0]), max(upper, inner[1])
        if first_piece == last_piece:
            parts = [self.partial_piece(first_piece, s_lo, s_hi)]
        else:
            parts = [
                self.partial_piece(first_piece, s_lo, self.grid[first_piece + 1]),
                self.partial_piece(last_piece, self.grid[last_piece], s_hi),
            ]
        for part in parts:
            lower, upper = min(lower, part[0]), max(upper, part[1])
        return lower, upper


class SameLaneQuery(_ProfileQuery):
    def __init__(self, profile: SameLaneProfile) -> None:
        self.profile = profile
        self.grid = profile.geometry.grid
        self.points_table = profile.points_table
        self.pieces_table = profile.pieces_table

    def partial_piece(self, piece, s_from, s_to):
        return self.profile.partial_piece(piece, s_from, s_to)


def _crossing(h1, h2, low=-1.0e3, high=1.0e3):
    """inf{d : h1(d) >= h2(d)} for nondecreasing h1 / nonincreasing h2."""
    if h1(low) >= h2(low):
        return low
    if h1(high) < h2(high):
        return high
    for _ in range(200):
        middle = 0.5 * (low + high)
        if h1(middle) >= h2(middle):
            high = middle
        else:
            low = middle
        if high - low <= 1.0e-12:
            break
    return high


class CutInProfile(_ProfileQuery):
    """alpha(s) of cut_in(other, ego) on a SameLaneGeometry with centre frame."""

    def __init__(
        self,
        same_lane: SameLaneProfile,
        choice_points: np.ndarray,
        choice_pieces: np.ndarray,
        scale_lat,
        single_term: float,
        cutting_d: float,
        orient_right_left: float,
        orient_left_right: float,
    ) -> None:
        """``choice_*``: centre frame index per point/piece (monitor uses the
        lane CLCS inside its projection domain, else ``clcs_large_step``);
        -1 marks positions outside both domains."""
        self.same_lane = same_lane
        g = same_lane.geometry
        self.geometry = g
        self.grid = g.grid
        self.choice_points = np.asarray(choice_points, dtype=int)
        self.choice_pieces = np.asarray(choice_pieces, dtype=int)
        self.scale_lat = scale_lat
        self.single_term = float(single_term)
        self.cutting_d = float(cutting_d)
        self.o1 = float(orient_right_left)
        self.o2 = float(orient_left_right)
        h1 = lambda d: min(scale_lat(d - self.cutting_d), self.o1)
        h2 = lambda d: min(scale_lat(self.cutting_d - d), self.o2)
        self._h1, self._h2 = h1, h2
        self.d_star = _crossing(h1, h2, self.cutting_d - 1.0e3, self.cutting_d + 1.0e3)
        points = []
        for index, (rho, choice) in enumerate(zip(same_lane.points, self.choice_points)):
            d = g.center_data[choice]["point"][index] if choice >= 0 else math.nan
            points.append(self._alpha_point(rho, d) if math.isfinite(d) else math.nan)
        self.points = np.asarray(points)
        point_low = np.where(np.isnan(self.points), -math.inf, self.points)
        point_high = np.where(np.isnan(self.points), math.inf, self.points)
        lower, upper, regular = [], [], []
        for piece in range(len(g.grid) - 1):
            choice = self.choice_pieces[piece]
            data = g.center_data[choice] if choice >= 0 else None
            ok = data is not None and bool(data["present"][piece]) and not bool(data["ambiguous"][piece])
            regular.append(ok)
            if not ok:
                lower.append(-math.inf)
                upper.append(math.inf)
                continue
            low, high = self._alpha_range(
                same_lane.piece_lower[piece], same_lane.piece_upper[piece],
                data["left"][piece], data["right"][piece],
            )
            lower.append(low)
            upper.append(high)
        self.regular = np.asarray(regular, dtype=bool)
        self.piece_lower = np.asarray(lower)
        self.piece_upper = np.asarray(upper)
        self.points_table = RangeTable(point_low, point_high)
        self.pieces_table = RangeTable(self.piece_lower, self.piece_upper)

    def _g(self, d: float) -> float:
        return max(self._h1(d), self._h2(d))

    def _g_range(self, d_a: float, d_b: float) -> Tuple[float, float]:
        low_d, high_d = min(d_a, d_b), max(d_a, d_b)
        g_min = self._g(min(max(self.d_star, low_d), high_d))
        g_max = max(self._g(low_d), self._g(high_d))
        return g_min, g_max

    def _alpha_point(self, rho: float, d: float) -> float:
        return min(self.single_term, self._scale(rho), self._g(d))

    def _alpha_range(self, rho_low, rho_high, d_a, d_b):
        g_low, g_high = self._g_range(d_a, d_b)
        return (
            min(self.single_term, self._scale(rho_low), g_low),
            min(self.single_term, self._scale(rho_high), g_high),
        )

    def _scale(self, value: float) -> float:
        if math.isinf(value):
            return float(self.scale_lat(math.copysign(1.0e12, value)))
        return float(self.scale_lat(value))

    def partial_piece(self, piece, s_from, s_to):
        if not self.regular[piece]:
            return -math.inf, math.inf
        rho_low, rho_high = self.same_lane.partial_piece(piece, s_from, s_to)
        g = self.geometry
        choice = int(self.choice_pieces[piece])
        data = g.center_data[choice]
        d_a, d_b = data["left"][piece], data["right"][piece]
        grid = g.grid
        if abs(s_from - grid[piece]) > 1.0e-12:
            d_a = g.endpoint_values(s_from, piece)["center"][choice]
        if abs(s_to - grid[piece + 1]) > 1.0e-12:
            d_b = g.endpoint_values(s_to, piece)["center"][choice]
        return self._alpha_range(rho_low, rho_high, d_a, d_b)
