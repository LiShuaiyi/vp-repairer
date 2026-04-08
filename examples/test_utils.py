import matplotlib.pyplot as plt
from commonroad.scenario.lanelet import LaneletType
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString
from typing import Union


def find_conflict_points(
        curved_line: LineString, conflict_polygon: Union[Polygon, LineString]
    ):
        conflict_line_points = list()
        # Get intersection of line and polygon
        intersection = curved_line.intersection(conflict_polygon)
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif (
            intersection.geom_type == "LineString"
            or intersection.geom_type == "LinearRing"
        ):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif (
            intersection.geom_type == "MultiPoint"
            or intersection.geom_type == "MultiLineString"
        ):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            conflict_points = None
        else:
            conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
        return conflict_points


def _plot_geometry(ax, geom, label=None, linewidth=2, alpha=0.6, linestyle="-", marker=None):
    """Generic shapely geometry plotter."""
    if geom is None:
        return

    geom_type = geom.geom_type

    if geom.is_empty:
        return

    if geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, linewidth=1, alpha=alpha, linestyle="--")
    elif geom_type == "MultiPolygon":
        first = True
        for g in geom.geoms:
            _plot_geometry(
                ax, g,
                label=label if first else None,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                marker=marker,
            )
            first = False
    elif geom_type in ["LineString", "LinearRing"]:
        x, y = geom.xy
        ax.plot(x, y, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
    elif geom_type == "MultiLineString":
        first = True
        for g in geom.geoms:
            _plot_geometry(
                ax, g,
                label=label if first else None,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                marker=marker,
            )
            first = False
    elif geom_type == "Point":
        ax.plot(geom.x, geom.y, marker=marker or "o", label=label)
    elif geom_type == "MultiPoint":
        xs = [p.x for p in geom.geoms]
        ys = [p.y for p in geom.geoms]
        ax.plot(xs, ys, linestyle="None", marker=marker or "o", label=label)
    elif geom_type == "GeometryCollection":
        first = True
        for g in geom.geoms:
            _plot_geometry(
                ax, g,
                label=label if first else None,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                marker=marker,
            )
            first = False
    else:
        print(f"[WARN] Unsupported geometry type: {geom_type}")


def _plot_points(ax, pts, label=None, marker="x"):
    if pts is None:
        return
    xs, ys = [], []
    for p in pts:
        xs.append(p[0])
        ys.append(p[1])
    ax.plot(xs, ys, linestyle="None", marker=marker, label=label)


def debug_plot_conflict_area(ego_vehicle, target_vehicle, world, clcs=None, figsize=(10, 10)):
    road_network = world.road_network

    # 1) target intersection lanelets
    conflict_lanelets_shape = []
    for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if LaneletType.INTERSECTION in lanelet.lanelet_type:
            conflict_lanelets_shape.append(lanelet.polygon.shapely_object)

    if len(conflict_lanelets_shape) == 0:
        print("[WARN] No intersection lanelets found for target vehicle.")
        return

    # 2) union
    conflict_area_shape = shapely.unary_union(conflict_lanelets_shape)

    # 3) offset conflict area
    conflict_linestring = shapely.offset_curve(conflict_area_shape, ego_vehicle.circle_radius)

    # 4) ego lines
    line_right = LineString(ego_vehicle.lanelets_dir_right_vertices)
    line_left = LineString(ego_vehicle.lanelets_dir_left_vertices)
    line_center = LineString(ego_vehicle.lanelets_dir_center_vertices)

    line_right_offset = shapely.offset_curve(line_right, ego_vehicle.circle_radius)
    line_left_offset = shapely.offset_curve(line_left, -ego_vehicle.circle_radius)

    # 5) intersections
    inter_right = line_right_offset.intersection(conflict_linestring)
    inter_left = line_left_offset.intersection(conflict_linestring)
    inter_center = line_center.intersection(conflict_linestring)

    conflict_circle_center_right = find_conflict_points(line_right_offset, conflict_linestring)
    conflict_circle_center_left = find_conflict_points(line_left_offset, conflict_linestring)
    conflict_circle_center_center = find_conflict_points(line_center, conflict_linestring)

    # 6) plot
    fig, ax = plt.subplots(figsize=figsize)

    # raw intersection lanelets
    for i, geom in enumerate(conflict_lanelets_shape):
        _plot_geometry(ax, geom, label="target intersection lanelets" if i == 0 else None, linewidth=1)

    # union area
    _plot_geometry(ax, conflict_area_shape, label="conflict_area_shape", linewidth=3, linestyle="-")

    # offset conflict geometry
    _plot_geometry(ax, conflict_linestring, label="conflict_linestring", linewidth=3, linestyle="--")

    # ego reference lines
    _plot_geometry(ax, line_left, label="ego left", linewidth=2)
    _plot_geometry(ax, line_center, label="ego center", linewidth=2)
    _plot_geometry(ax, line_right, label="ego right", linewidth=2)

    # offset lines
    _plot_geometry(ax, line_left_offset, label="ego left offset", linewidth=2, linestyle="--")
    _plot_geometry(ax, line_right_offset, label="ego right offset", linewidth=2, linestyle="--")

    # raw intersection geometries
    _plot_geometry(ax, inter_right, label="intersection right", marker="o")
    _plot_geometry(ax, inter_left, label="intersection left", marker="s")
    _plot_geometry(ax, inter_center, label="intersection center", marker="^")

    # start/end points extracted by find_conflict_points
    _plot_points(ax, conflict_circle_center_right, label="conflict pts right", marker="x")
    _plot_points(ax, conflict_circle_center_left, label="conflict pts left", marker="x")
    _plot_points(ax, conflict_circle_center_center, label="conflict pts center", marker="x")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title("Conflict Area Debug Plot")
    plt.show()