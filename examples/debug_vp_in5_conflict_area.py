"""Visualize VP conflict-area constraint geometry for R_IN5."""

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
    scenario_id = "DEU_AAH1-2_7900_T-1049"
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


def _plot_geometry(ax, geom, label=None, color=None, linewidth=2, alpha=0.8, linestyle="-"):
    if geom is None or geom.is_empty:
        return
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
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
        ax.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
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
    ax.scatter([point[0]], [point[1]], label=label, color=color, marker=marker, s=size, zorder=10)
    if annotate:
        ax.annotate(label, (point[0], point[1]), xytext=(6, 6), textcoords="offset points", fontsize=8)


def _trajectory_xy(vehicle):
    time_steps = sorted(vehicle.states_cr.keys())
    xy = np.asarray([vehicle.states_cr[t].position for t in time_steps], dtype=float)
    return time_steps, xy


def _try_cartesian(clcs, s_value, d_value=0.0):
    try:
        return clcs.convert_to_cartesian_coords(float(s_value), float(d_value)), None
    except Exception as exc:
        return None, exc


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

    conflict_lanelets = []
    for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
        lanelet = world.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if LaneletType.INTERSECTION in lanelet.lanelet_type:
            conflict_lanelets.append(lanelet.polygon.shapely_object)
    conflict_area = shapely.unary_union(conflict_lanelets)
    conflict_offset = shapely.offset_curve(conflict_area, ego_vehicle.circle_radius)

    conflict_points_cart = repairer._create_conflict_area_parameter(
        ego_vehicle,
        target_vehicle,
        world,
        lanelet_clcs,
        cart=True,
    )
    front_cart = np.asarray(conflict_points_cart[0], dtype=float)
    rear_cart = np.asarray(conflict_points_cart[1], dtype=float)

    front_lane_s, rear_lane_s = repairer._constraint_in_intersection_conflict_area(
        time_step=int(repairer.tv),
        prop_assignment=-1,
        lanelet_clcs=lanelet_clcs,
        cart=False,
    )
    front_lane_cart, front_lane_cart_error = _try_cartesian(lanelet_clcs, front_lane_s)
    rear_lane_cart, rear_lane_cart_error = _try_cartesian(lanelet_clcs, rear_lane_s)

    rear_traj_s = trajectory_clcs.convert_to_curvilinear_coords(float(rear_cart[0]), float(rear_cart[1]))[0]
    rear_cap_traj_s = rear_traj_s - wheelbase / 2
    rear_cap_traj_cart, rear_cap_traj_cart_error = _try_cartesian(
        trajectory_clcs,
        rear_cap_traj_s,
    )

    front_traj_s = trajectory_clcs.convert_to_curvilinear_coords(float(front_cart[0]), float(front_cart[1]))[0]
    ego_times, ego_xy = _trajectory_xy(ego_vehicle)
    target_times, target_xy = _trajectory_xy(target_vehicle)

    tv = int(repairer.tv)
    ego_tv = ego_vehicle.states_cr[tv].position if tv in ego_vehicle.states_cr else None
    target_tv = target_vehicle.states_cr[tv].position if tv in target_vehicle.states_cr else None

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
    _plot_geometry(ax, conflict_area, label="conflict area union", color="#d62728", linewidth=2.5)
    _plot_geometry(ax, conflict_offset, label="conflict area offset by ego radius", color="#ff7f0e", linewidth=2, linestyle="--")
    _plot_geometry(ax, LineString(ego_xy), label="ego original trajectory", color="#1f77b4", linewidth=2.2)
    _plot_geometry(ax, LineString(target_xy), label="target trajectory", color="#2ca02c", linewidth=2.2)
    _plot_geometry(ax, LineString(ref_path), label="VP trajectory CLCS reference", color="#9467bd", linewidth=1.6, linestyle=":")

    _plot_point(ax, front_cart, "raw front point", "#d62728", marker="x")
    _plot_point(ax, rear_cart, "raw rear point", "#d62728", marker="X")
    _plot_point(ax, front_lane_cart, "front bound in lane CLCS", "#8c564b", marker="^")
    _plot_point(ax, rear_lane_cart, "rear bound in lane CLCS", "#8c564b", marker="v")
    _plot_point(ax, rear_cap_traj_cart, "VP rear cap on trajectory CLCS", "#000000", marker="s")
    _plot_point(ax, ego_tv, f"ego at tv={tv}", "#1f77b4", marker="o")
    _plot_point(ax, target_tv, f"target at tv={tv}", "#2ca02c", marker="o")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{scenario_id}: VP conflict-area constraint debug")
    ax.legend(loc="best", fontsize=8)

    plot_limits = config.debug.plot_limits
    if plot_limits:
        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])

    output_dir = Path(config.general.path_figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vp_in5_conflict_area_debug.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved conflict-area debug plot to: {output_path}")
    print("Important values:")
    print(f"  scenario_id: {scenario_id}")
    print(f"  ego_id: {config.repair.ego_id}")
    print(f"  target_id: {rule_monitor.other_id}")
    print(f"  tv: {repairer.tv}, tc: {repairer.tc}")
    print(f"  raw front cart: {front_cart.tolist()}")
    print(f"  raw rear cart: {rear_cart.tolist()}")
    print(f"  lane CLCS front bound s: {front_lane_s}")
    print(f"  lane CLCS rear bound s: {rear_lane_s}")
    print(f"  trajectory CLCS front point s: {front_traj_s}")
    print(f"  trajectory CLCS rear point s: {rear_traj_s}")
    print(f"  current VP rear cap s on trajectory CLCS: {rear_cap_traj_s}")
    if front_lane_cart_error is not None:
        print(f"  front lane cart projection error: {front_lane_cart_error}")
    if rear_lane_cart_error is not None:
        print(f"  rear lane cart projection error: {rear_lane_cart_error}")
    if rear_cap_traj_cart_error is not None:
        print(f"  rear cap trajectory cart projection error: {rear_cap_traj_cart_error}")
    print(f"  wheelbase: {wheelbase}")
    print(f"  ego position at tv: {None if ego_tv is None else np.asarray(ego_tv).tolist()}")
    print(f"  target position at tv: {None if target_tv is None else np.asarray(target_tv).tolist()}")


if __name__ == "__main__":
    main()
