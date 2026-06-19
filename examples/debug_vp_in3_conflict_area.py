"""Visualize VP conflict-area constraint geometry for R_IN3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString

from commonroad.scenario.lanelet import LaneletType

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_config():
    scenario_id = "DEU_TestIntersectionInteract-3_1_T-1"
    config = RepairerConfiguration.load(
        REPO_ROOT / "config" / f"{scenario_id}.yaml",
        scenario_id,
    )
    config.update()
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.repair.constraint_mode = 1
    config.repair.sat_solver_mode = "domain_dpll"
    config.debug.show_plots = False
    return scenario_id, config


def _plot_geometry(
    ax,
    geom,
    label=None,
    color=None,
    linewidth=2,
    alpha=0.8,
    linestyle="-",
):
    if geom is None or geom.is_empty:
        return
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color=color, linewidth=1, alpha=alpha, linestyle=":")
    elif geom_type == "MultiPolygon":
        for i, sub_geom in enumerate(geom.geoms):
            _plot_geometry(
                ax,
                sub_geom,
                label=label if i == 0 else None,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )
    elif geom_type in ("LineString", "LinearRing"):
        x, y = geom.xy
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )
    elif geom_type in ("MultiLineString", "GeometryCollection"):
        for i, sub_geom in enumerate(geom.geoms):
            _plot_geometry(
                ax,
                sub_geom,
                label=label if i == 0 else None,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )
    elif geom_type == "Point":
        ax.scatter([geom.x], [geom.y], label=label, color=color)
    elif geom_type == "MultiPoint":
        ax.scatter(
            [point.x for point in geom.geoms],
            [point.y for point in geom.geoms],
            label=label,
            color=color,
        )


def _plot_point(ax, point, label, color, marker="x", size=90, annotate=True):
    if point is None:
        return
    point = np.asarray(point, dtype=float)
    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
        return
    ax.scatter(
        [point[0]],
        [point[1]],
        label=label,
        color=color,
        marker=marker,
        s=size,
        zorder=10,
    )
    if annotate:
        ax.annotate(
            label,
            (point[0], point[1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )


def _trajectory_xy(vehicle):
    time_steps = sorted(vehicle.states_cr.keys())
    xy = np.asarray([vehicle.states_cr[t].position for t in time_steps], dtype=float)
    return time_steps, xy


def _try_cartesian(clcs, s_value, d_value=0.0):
    try:
        return clcs.convert_to_cartesian_coords(float(s_value), float(d_value)), None
    except Exception as exc:
        return None, exc


def _find_conflict_points(curved_line, conflict_polygon):
    conflict_line_points = []
    intersection = curved_line.intersection(conflict_polygon)
    if intersection.geom_type == "Point":
        conflict_line_points.append([intersection.x, intersection.y])
    elif intersection.geom_type in ("LineString", "LinearRing"):
        for point in intersection.coords:
            conflict_line_points.append(np.asarray(point, dtype=float))
    elif intersection.geom_type in ("MultiPoint", "MultiLineString"):
        for geom in intersection.geoms:
            for point in geom.coords:
                conflict_line_points.append(np.asarray(point, dtype=float))
    if len(conflict_line_points) == 0:
        return None
    return [conflict_line_points[0], conflict_line_points[-1]]


def _monitor_like_conflict_points(world, ego_vehicle, target_vehicle, time_step):
    road_network = world.road_network
    lanelets_assignment = ego_vehicle.lanelet_assignment[time_step]
    lanelets_ego_intersection = [
        lanelet_id
        for lanelet_id in lanelets_assignment
        if LaneletType.INTERSECTION
        in road_network.lanelet_network.find_lanelet_by_id(lanelet_id).lanelet_type
    ]
    conflict_lanelets = target_vehicle.ref_path_lane.contained_lanelets.intersection(
        lanelets_ego_intersection
    )
    conflict_lanelets = conflict_lanelets.difference(ego_vehicle.lanelets_dir)

    if len(conflict_lanelets) != 0:
        current_lanelets = list(lanelets_assignment.intersection(ego_vehicle.lanelets_dir))
        center_vertices = None
        for lanelet_id in current_lanelets:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if center_vertices is None:
                center_vertices = lanelet.center_vertices
            else:
                center_vertices = np.append(center_vertices, lanelet.center_vertices, axis=0)

        conflict_polygon = None
        for lanelet_id in conflict_lanelets:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if conflict_polygon is None:
                conflict_polygon = lanelet.polygon.shapely_object
            else:
                conflict_polygon = conflict_polygon.union(lanelet.polygon.shapely_object)
        if center_vertices is None or conflict_polygon is None:
            return None, None, None
        points = _find_conflict_points(LineString(center_vertices), conflict_polygon)
        return points, conflict_polygon, LineString(center_vertices)

    all_conflict_points = []
    conflict_polygons = []
    ego_line = LineString(ego_vehicle.lanelets_dir_center_vertices)
    for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if LaneletType.INTERSECTION not in lanelet.lanelet_type:
            continue
        points = _find_conflict_points(ego_line, lanelet.polygon.shapely_object)
        if points is not None:
            all_conflict_points.append(points)
            conflict_polygons.append(lanelet.polygon.shapely_object)
    if len(all_conflict_points) == 0:
        return None, None, ego_line
    monitor_points = [all_conflict_points[0][0], all_conflict_points[-1][-1]]
    return monitor_points, shapely.unary_union(conflict_polygons), ego_line


def main():
    scenario_id, config = build_config()
    config.debug.show_plots = False
    config.update()

    ego_initial = retrieve_ego_vehicle(config)
    rule_monitor = STLRuleMonitor(config)
    repairer = VPTrajectoryRepairer(rule_monitor, ego_initial, config)
    repairer._tv = rule_monitor.tv_time_step
    repairer._tc = 0

    all_states = repairer._get_states_with_initial()
    lanelet_clcs, _ = repairer._get_lanelet_clcs_and_dt()
    trajectory_clcs, ref_path = repairer._build_trajectory_clcs(all_states)

    world = rule_monitor.world
    ego_vehicle = world.vehicle_by_id(config.repair.ego_id)
    target_vehicle = world.vehicle_by_id(rule_monitor.other_id)
    wheelbase = repairer._get_planner_wheelbase()
    tv = int(repairer.tv)

    conflict_lanelets = []
    for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
        lanelet = world.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if LaneletType.INTERSECTION in lanelet.lanelet_type:
            conflict_lanelets.append(lanelet.polygon.shapely_object)
    conflict_area = shapely.unary_union(conflict_lanelets)
    conflict_offset = shapely.offset_curve(conflict_area, ego_vehicle.circle_radius)

    vp_points = repairer._create_conflict_area_parameter(
        ego_vehicle,
        target_vehicle,
        world,
        lanelet_clcs,
        cart=True,
    )
    vp_front_cart = np.asarray(vp_points[0], dtype=float)
    vp_rear_cart = np.asarray(vp_points[1], dtype=float)

    front_lane_s, rear_lane_s = repairer._constraint_in_intersection_conflict_area(
        time_step=tv,
        prop_assignment=-1,
        lanelet_clcs=lanelet_clcs,
        cart=False,
    )
    front_lane_cart, front_lane_cart_error = _try_cartesian(lanelet_clcs, front_lane_s)
    rear_lane_cart, rear_lane_cart_error = _try_cartesian(lanelet_clcs, rear_lane_s)

    vp_front_traj_s = trajectory_clcs.convert_to_curvilinear_coords(
        float(vp_front_cart[0]),
        float(vp_front_cart[1]),
    )[0]
    vp_rear_traj_s = trajectory_clcs.convert_to_curvilinear_coords(
        float(vp_rear_cart[0]),
        float(vp_rear_cart[1]),
    )[0]
    vp_rear_cap_traj_s = vp_rear_traj_s - wheelbase / 2
    vp_rear_cap_cart, vp_rear_cap_error = _try_cartesian(
        trajectory_clcs,
        vp_rear_cap_traj_s,
    )

    monitor_points, monitor_polygon, monitor_line = _monitor_like_conflict_points(
        world,
        ego_vehicle,
        target_vehicle,
        tv,
    )
    monitor_front_cart = None
    monitor_rear_cart = None
    monitor_front_traj_s = None
    monitor_rear_traj_s = None
    monitor_before_cap_traj_s = None
    monitor_before_cap_cart = None
    if monitor_points is not None:
        monitor_front_cart = np.asarray(monitor_points[0], dtype=float)
        monitor_rear_cart = np.asarray(monitor_points[1], dtype=float)
        monitor_front_traj_s = trajectory_clcs.convert_to_curvilinear_coords(
            float(monitor_front_cart[0]),
            float(monitor_front_cart[1]),
        )[0]
        monitor_rear_traj_s = trajectory_clcs.convert_to_curvilinear_coords(
            float(monitor_rear_cart[0]),
            float(monitor_rear_cart[1]),
        )[0]
        monitor_before_cap_traj_s = (
            min(monitor_front_traj_s, monitor_rear_traj_s)
            - ego_vehicle.shape.length / 3
            - wheelbase / 2
        )
        monitor_before_cap_cart, _ = _try_cartesian(
            trajectory_clcs,
            monitor_before_cap_traj_s,
        )

    ego_times, ego_xy = _trajectory_xy(ego_vehicle)
    target_times, target_xy = _trajectory_xy(target_vehicle)
    ego_tv = ego_vehicle.states_cr[tv].position if tv in ego_vehicle.states_cr else None
    target_tv = (
        target_vehicle.states_cr[tv].position if tv in target_vehicle.states_cr else None
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, geom in enumerate(conflict_lanelets):
        _plot_geometry(
            ax,
            geom,
            label="target intersection lanelets" if i == 0 else None,
            color="#808080",
            linewidth=1,
            alpha=0.45,
        )
    _plot_geometry(ax, conflict_area, "VP conflict area union", "#d62728", 2.5)
    _plot_geometry(
        ax,
        conflict_offset,
        "VP conflict area offset by ego radius",
        "#ff7f0e",
        2,
        linestyle="--",
    )
    _plot_geometry(
        ax,
        monitor_polygon,
        "monitor conflict polygon at tv",
        "#17becf",
        2,
        linestyle="-.",
    )
    _plot_geometry(ax, LineString(ego_xy), "ego original trajectory", "#1f77b4", 2.2)
    _plot_geometry(ax, LineString(target_xy), "target trajectory", "#2ca02c", 2.2)
    _plot_geometry(
        ax,
        LineString(ref_path),
        "VP trajectory CLCS reference",
        "#9467bd",
        1.6,
        linestyle=":",
    )
    _plot_geometry(
        ax,
        monitor_line,
        "monitor ego lane center line",
        "#17becf",
        1.5,
        alpha=0.7,
        linestyle=":",
    )

    _plot_point(ax, vp_front_cart, "VP raw front point", "#d62728", marker="x")
    _plot_point(ax, vp_rear_cart, "VP raw rear point", "#d62728", marker="X")
    _plot_point(ax, front_lane_cart, "VP front bound in lane CLCS", "#8c564b", marker="^")
    _plot_point(ax, rear_lane_cart, "VP rear bound in lane CLCS", "#8c564b", marker="v")
    _plot_point(
        ax,
        vp_rear_cap_cart,
        "current VP rear cap on trajectory CLCS",
        "#000000",
        marker="s",
    )
    _plot_point(
        ax,
        monitor_front_cart,
        "monitor conflict start point",
        "#17becf",
        marker="P",
    )
    _plot_point(
        ax,
        monitor_rear_cart,
        "monitor conflict end point",
        "#17becf",
        marker="D",
    )
    _plot_point(
        ax,
        monitor_before_cap_cart,
        "monitor before-conflict cap with margin",
        "#bcbd22",
        marker="s",
    )
    _plot_point(ax, ego_tv, f"ego at tv={tv}", "#1f77b4", marker="o")
    _plot_point(ax, target_tv, f"target at tv={tv}", "#2ca02c", marker="o")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{scenario_id}: RIN3 conflict-area debug")
    ax.legend(loc="best", fontsize=8)

    plot_limits = config.debug.plot_limits
    if plot_limits:
        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])

    output_dir = Path(config.general.path_figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vp_in3_conflict_area_debug.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved conflict-area debug plot to: {output_path}")
    print("Important values:")
    print(f"  scenario_id: {scenario_id}")
    print(f"  ego_id: {config.repair.ego_id}")
    print(f"  target_id: {rule_monitor.other_id}")
    print(f"  tv: {repairer.tv}, tc: {repairer.tc}")
    print(f"  VP raw front cart: {vp_front_cart.tolist()}")
    print(f"  VP raw rear cart: {vp_rear_cart.tolist()}")
    print(f"  VP lane CLCS front bound s: {front_lane_s}")
    print(f"  VP lane CLCS rear bound s: {rear_lane_s}")
    print(f"  VP trajectory CLCS front point s: {vp_front_traj_s}")
    print(f"  VP trajectory CLCS rear point s: {vp_rear_traj_s}")
    print(f"  current VP rear cap s on trajectory CLCS: {vp_rear_cap_traj_s}")
    print(
        "  monitor conflict start/end cart: "
        f"{None if monitor_front_cart is None else monitor_front_cart.tolist()} / "
        f"{None if monitor_rear_cart is None else monitor_rear_cart.tolist()}"
    )
    print(
        "  monitor trajectory CLCS start/end s: "
        f"{monitor_front_traj_s} / {monitor_rear_traj_s}"
    )
    print(f"  monitor before-conflict cap with margin s: {monitor_before_cap_traj_s}")
    if front_lane_cart_error is not None:
        print(f"  front lane cart projection error: {front_lane_cart_error}")
    if rear_lane_cart_error is not None:
        print(f"  rear lane cart projection error: {rear_lane_cart_error}")
    if vp_rear_cap_error is not None:
        print(f"  VP rear cap trajectory cart projection error: {vp_rear_cap_error}")
    print(f"  wheelbase: {wheelbase}")
    print(f"  ego length: {ego_vehicle.shape.length}")
    print(f"  ego circle radius: {ego_vehicle.circle_radius}")
    print(f"  ego position at tv: {None if ego_tv is None else np.asarray(ego_tv).tolist()}")
    print(
        f"  target position at tv: "
        f"{None if target_tv is None else np.asarray(target_tv).tolist()}"
    )


if __name__ == "__main__":
    main()
