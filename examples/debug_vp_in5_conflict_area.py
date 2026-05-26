"""Visualize VP conflict-area constraint geometry for R_IN5."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString
from z3 import sat

from commonroad.scenario.lanelet import LaneletType

from crrepairer.repairer.vp_repairer import VPTrajectoryRepairer
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.repair import retrieve_ego_vehicle

from test_vp_repairer_in5 import build_config, patch_rtamt_bound_alignment

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _trajectory_s_values(trajectory_clcs, ref_path):
    s_values = []
    for point in ref_path:
        try:
            s_values.append(
                trajectory_clcs.convert_to_curvilinear_coords(
                    float(point[0]),
                    float(point[1]),
                )[0]
            )
        except Exception:
            s_values.append(np.nan)
    return np.asarray(s_values, dtype=float)


def _extract_trajectory_constraints_by_time(
    repairer,
    all_states,
    lanelet_clcs,
    trajectory_clcs,
    ref_path,
):
    if repairer.sat_solver.solver_mode == "domain_dpll":
        repairer.ensure_domain_dict_initialized()
    if repairer.sat_solver.solve() != sat:
        raise RuntimeError("SAT solver could not find a model for debug constraint extraction.")

    selected_propositions, model = repairer.sat_solver.model()
    repairer._model = model
    repairer._assign_proposition(selected_propositions, list(model))

    cl_trajectory_before = repairer._convert_states_to_clcs(all_states, lanelet_clcs)
    (
        s_min,
        s_max,
        v_min,
        v_max,
        trajectory_s_min_cap,
        trajectory_s_max_cap,
    ) = repairer._extract_intersection_constraints_manually(
        all_states,
        lanelet_clcs,
        trajectory_clcs,
        cl_trajectory_before,
        ref_path,
    )
    est_s_min, est_s_max, est_v_min, est_v_max = (
        repairer._convert_lanelet_constraints_to_trajectory_constraints(
            s_min,
            s_max,
            v_min,
            v_max,
            all_states,
            ref_path,
            lanelet_clcs,
            trajectory_clcs,
            cl_trajectory_before,
            trajectory_s_min_cap=trajectory_s_min_cap,
            trajectory_s_max_cap=trajectory_s_max_cap,
        )
    )

    time_steps = list(range(int(repairer.tc) + 1, all_states[-1].time_step + 1))
    return {
        "time_steps": time_steps,
        "s_min": np.asarray(est_s_min, dtype=float),
        "s_max": np.asarray(est_s_max, dtype=float),
        "v_min": np.asarray(est_v_min, dtype=float),
        "v_max": np.asarray(est_v_max, dtype=float),
        "trajectory_s_min_cap": np.asarray(trajectory_s_min_cap, dtype=float),
        "trajectory_s_max_cap": np.asarray(trajectory_s_max_cap, dtype=float),
    }


def _plot_allowed_trajectory_interval(
    ax,
    ref_path,
    ref_path_s,
    s_min,
    s_max,
    color="#00a6d6",
):
    finite = np.isfinite(ref_path_s) & np.isfinite(s_min) & np.isfinite(s_max)
    mask = finite & (ref_path_s >= min(s_min, s_max)) & (ref_path_s <= max(s_min, s_max))
    if np.any(mask):
        ax.plot(
            ref_path[mask, 0],
            ref_path[mask, 1],
            color=color,
            linewidth=5,
            alpha=0.85,
            solid_capstyle="round",
            label="allowed trajectory range",
        )


def _save_per_time_step_constraint_plots(
    output_dir,
    scenario_id,
    config,
    repairer,
    rule_monitor,
    conflict_lanelets,
    conflict_area,
    conflict_offset,
    ego_vehicle,
    target_vehicle,
    ego_xy,
    target_xy,
    ref_path,
    trajectory_clcs,
    constraints_by_time,
):
    frame_dir = output_dir / "vp_in5_trajectory_constraints_by_time"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ref_path_s = _trajectory_s_values(trajectory_clcs, ref_path)
    plot_limits = config.debug.plot_limits

    for idx, time_step in enumerate(constraints_by_time["time_steps"]):
        s_min = constraints_by_time["s_min"][idx]
        s_max = constraints_by_time["s_max"][idx]
        s_min_cart, s_min_error = _try_cartesian(trajectory_clcs, s_min)
        s_max_cart, s_max_error = _try_cartesian(trajectory_clcs, s_max)

        ego_pos = ego_vehicle.states_cr[time_step].position if time_step in ego_vehicle.states_cr else None
        target_pos = (
            target_vehicle.states_cr[time_step].position
            if time_step in target_vehicle.states_cr
            else None
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, geom in enumerate(conflict_lanelets):
            _plot_geometry(
                ax,
                geom,
                label="target intersection lanelets" if i == 0 else None,
                color="#808080",
                linewidth=1,
                alpha=0.35,
            )
        _plot_geometry(ax, conflict_area, label="conflict area union", color="#d62728", linewidth=2.2)
        _plot_geometry(ax, conflict_offset, label="conflict offset", color="#ff7f0e", linewidth=1.8, linestyle="--")
        _plot_geometry(ax, LineString(ego_xy), label="ego trajectory", color="#1f77b4", linewidth=1.8)
        _plot_geometry(ax, LineString(target_xy), label="target trajectory", color="#2ca02c", linewidth=1.8)
        _plot_geometry(ax, LineString(ref_path), label="trajectory CLCS reference", color="#9467bd", linewidth=1.4, linestyle=":")
        _plot_allowed_trajectory_interval(ax, ref_path, ref_path_s, s_min, s_max)

        _plot_point(ax, s_min_cart, "s_min", "#005f73", marker="|", size=180)
        _plot_point(ax, s_max_cart, "s_max", "#005f73", marker="|", size=180)
        _plot_point(ax, ego_pos, f"ego t={time_step}", "#1f77b4", marker="o")
        _plot_point(ax, target_pos, f"target t={time_step}", "#2ca02c", marker="o")

        info_lines = [
            f"scenario={scenario_id}",
            f"t={time_step}, tc={repairer.tc}, tv={repairer.tv}",
            f"ego={config.repair.ego_id}, target={rule_monitor.other_id}",
            f"trajectory s range=[{s_min:.3f}, {s_max:.3f}]",
            f"trajectory_s_min_cap={constraints_by_time['trajectory_s_min_cap'][idx]:.3f}",
            f"trajectory_s_max_cap={constraints_by_time['trajectory_s_max_cap'][idx]:.3f}",
            f"v range=[{constraints_by_time['v_min'][idx]:.3f}, {constraints_by_time['v_max'][idx]:.3f}]",
        ]
        if s_min_error is not None:
            info_lines.append(f"s_min projection error={s_min_error}")
        if s_max_error is not None:
            info_lines.append(f"s_max projection error={s_max_error}")
        ax.text(
            0.02,
            0.02,
            "\n".join(info_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "#777777"},
        )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{scenario_id}: trajectory constraints at t={time_step}")
        ax.legend(loc="best", fontsize=8)
        if plot_limits:
            ax.set_xlim(plot_limits[0], plot_limits[1])
            ax.set_ylim(plot_limits[2], plot_limits[3])

        frame_path = frame_dir / f"step_{time_step:03d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=180)
        plt.close(fig)

    print(f"Saved per-time-step trajectory constraint plots to: {frame_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VP conflict-area geometry and optional trajectory constraints for R_IN5."
    )
    parser.add_argument(
        "--plot-trajectory-constraints-by-time",
        action="store_true",
        help="Save one extra image per repaired horizon step with the trajectory s_min/s_max range.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_id, config = build_config()
    config.debug.show_plots = False
    patch_rtamt_bound_alignment(config)

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
    front_traj_s = trajectory_clcs.convert_to_curvilinear_coords(float(front_cart[0]), float(front_cart[1]))[0]
    lower_interval_upper = min(front_traj_s, rear_traj_s)
    upper_interval_lower = max(front_traj_s, rear_traj_s)

    start_plan_idx = min(int(repairer.tc - all_states[0].time_step) + 1, len(all_states) - 1)
    start_state = all_states[start_plan_idx]
    start_s = trajectory_clcs.convert_to_curvilinear_coords(
        float(start_state.position[0]),
        float(start_state.position[1]),
    )[0]

    lower_interval_upper_cart, lower_interval_upper_cart_error = _try_cartesian(
        trajectory_clcs,
        lower_interval_upper,
    )
    upper_interval_lower_cart, upper_interval_lower_cart_error = _try_cartesian(
        trajectory_clcs,
        upper_interval_lower,
    )
    start_s_cart, start_s_cart_error = _try_cartesian(trajectory_clcs, start_s)
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
    _plot_point(ax, lower_interval_upper_cart, "trajectory lower interval upper", "#000000", marker="s")
    _plot_point(ax, upper_interval_lower_cart, "trajectory upper interval lower", "#000000", marker="D")
    _plot_point(ax, start_s_cart, "VP start on trajectory CLCS", "#17becf", marker="P")
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

    important_values = "\n".join(
        [
            f"ego={config.repair.ego_id}, target={rule_monitor.other_id}, tv={repairer.tv}, tc={repairer.tc}",
            f"front cart=({front_cart[0]:.3f}, {front_cart[1]:.3f})",
            f"rear cart=({rear_cart[0]:.3f}, {rear_cart[1]:.3f})",
            f"lane s front/rear=({front_lane_s:.3f}, {rear_lane_s:.3f})",
            f"traj s front/rear=({front_traj_s:.3f}, {rear_traj_s:.3f})",
            f"traj forbidden interval=[{lower_interval_upper:.3f}, {upper_interval_lower:.3f}]",
            f"start_s={start_s:.3f}, wheelbase={wheelbase:.3f}",
        ]
    )
    ax.text(
        0.02,
        0.02,
        important_values,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.78, "edgecolor": "#777777"},
    )

    output_dir = Path(config.general.path_figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vp_in5_conflict_area_debug.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    if args.plot_trajectory_constraints_by_time:
        constraints_by_time = _extract_trajectory_constraints_by_time(
            repairer,
            all_states,
            lanelet_clcs,
            trajectory_clcs,
            ref_path,
        )
        _save_per_time_step_constraint_plots(
            output_dir=output_dir,
            scenario_id=scenario_id,
            config=config,
            repairer=repairer,
            rule_monitor=rule_monitor,
            conflict_lanelets=conflict_lanelets,
            conflict_area=conflict_area,
            conflict_offset=conflict_offset,
            ego_vehicle=ego_vehicle,
            target_vehicle=target_vehicle,
            ego_xy=ego_xy,
            target_xy=target_xy,
            ref_path=ref_path,
            trajectory_clcs=trajectory_clcs,
            constraints_by_time=constraints_by_time,
        )

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
    print(f"  trajectory forbidden interval lower upper s: {lower_interval_upper}")
    print(f"  trajectory forbidden interval upper lower s: {upper_interval_lower}")
    print(f"  VP start s on trajectory CLCS: {start_s}")
    if front_lane_cart_error is not None:
        print(f"  front lane cart projection error: {front_lane_cart_error}")
    if rear_lane_cart_error is not None:
        print(f"  rear lane cart projection error: {rear_lane_cart_error}")
    if lower_interval_upper_cart_error is not None:
        print(f"  lower interval trajectory cart projection error: {lower_interval_upper_cart_error}")
    if upper_interval_lower_cart_error is not None:
        print(f"  upper interval trajectory cart projection error: {upper_interval_lower_cart_error}")
    if start_s_cart_error is not None:
        print(f"  start trajectory cart projection error: {start_s_cart_error}")
    print(f"  wheelbase: {wheelbase}")
    print(f"  ego position at tv: {None if ego_tv is None else np.asarray(ego_tv).tolist()}")
    print(f"  target position at tv: {None if target_tv is None else np.asarray(target_tv).tolist()}")


if __name__ == "__main__":
    main()
