"""End-to-end check of the piecewise-monotone lateral profiles on highD RG1.

For a real scenario the ego state is rebuilt at many progress values exactly
as VP builds repaired states, the STL monitor evaluates in_same_lane and
cut_in, and the estimates must (a) match the monitor at partition points and
(b) enclose the monitor robustness on reachable intervals.  The ego shape
lanelet assignment used by the estimator is checked against the monitor's.
"""

import contextlib
import io
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

B = pytest.importorskip("batch_test_vp_repairer_rg1")

from commonroad.scenario.state import CustomState  # noqa: E402
from crmonitor.common.vehicle import CurvilinearStateManager  # noqa: E402
from crrepairer.repairer.vp.optimization import VPOptimization  # noqa: E402

CASES = [("DEU_LocationALower-34_37_T-1", 9), ("DEU_LocationALower-30_46_T-1", 16)]


def _builder(scenario_id, ego_id):
    with contextlib.redirect_stdout(io.StringIO()):
        config = B.build_config(scenario_id, ego_id)
        monitor = B.STLRuleMonitor(config)
        repairer = B.VPTrajectoryRepairer(monitor, B.retrieve_ego_vehicle(config), config)
        repairer._prepare_semantic_lateral_geometry()
    return repairer._semantic_lateral_preprocessed_builder


@pytest.mark.parametrize("scenario_id,ego_id", CASES)
def test_piecewise_profiles_match_and_enclose_monitor(scenario_id, ego_id):
    builder = _builder(scenario_id, ego_id)
    assert builder is not None
    world, ego = builder.world, builder.ego
    other = world.vehicle_by_id(builder.other_id)
    clcs = builder.trajectory_clcs
    cut_in = builder._repairer_evaluators("cut_in")[0]
    same_lane = builder._repairer_evaluators("in_same_lane")[0]
    rng = np.random.default_rng(0)
    steps = sorted(t for t in other.lanelet_assignment if t in ego.states_cr)
    for time_step in steps[:: max(1, len(steps) // 2)]:
        profiles = builder._lateral_profiles(builder.other_id, time_step)
        cut_profile = builder._cut_in_profile(cut_in, builder.other_id, time_step)
        assert profiles is not None and cut_profile is not None
        geometry = profiles["geometry"]
        saved = (ego.states_cr[time_step], set(ego.lanelet_assignment[time_step]), ego.ccosy_cache)

        def monitor_at(s):
            position = np.asarray(clcs.convert_to_cartesian_coords(s, 0.0))
            heading = VPOptimization._orientation_from_trajectory_clcs(clcs, s)
            ego.states_cr[time_step] = CustomState(
                time_step=time_step, position=position, orientation=heading,
                velocity=saved[0].velocity,
            )
            lanelets = set(
                world.road_network.lanelet_network.find_lanelet_by_shape(
                    ego.shape.rotate_translate_local(position, heading)
                )
            )
            ego.lanelet_assignment[time_step] = lanelets
            ego.ccosy_cache = CurvilinearStateManager(world.road_network)
            for vehicle in (ego, other):
                vehicle.predicate_cache.cache[time_step] = defaultdict()
            rho_same = same_lane.evaluate_robustness(world, time_step, [ego.id, other.id])
            for vehicle in (ego, other):
                vehicle.predicate_cache.cache[time_step] = defaultdict()
            rho_cut = cut_in.evaluate_robustness(world, time_step, [other.id, ego.id])
            return rho_same, rho_cut, frozenset(lanelets)

        try:
            for index in rng.choice(len(geometry.grid), 25, replace=False):
                rho_same, rho_cut, lanelets = monitor_at(geometry.grid[index])
                assert lanelets == geometry.point_sets[index]
                estimate = profiles["same"].points[index]
                if not (math.isinf(estimate) and math.isinf(rho_same)):
                    assert same_lane._scale_lat_dist(estimate) == pytest.approx(rho_same, abs=1e-9)
                assert cut_profile.points[index] == pytest.approx(rho_cut, abs=1e-9)
            for _ in range(3):
                lower = rng.uniform(geometry.grid[0], geometry.grid[-1] - 1.0)
                upper = min(geometry.grid[-1], lower + rng.uniform(0.2, 12.0))
                same_range = profiles["same_query"].query(lower, upper)
                cut_range = cut_profile.query(lower, upper)
                for s in np.linspace(lower, upper, 40):
                    rho_same, rho_cut, _ = monitor_at(s)
                    assert same_lane._scale_lat_dist(same_range[0]) - 1e-9 <= rho_same
                    assert rho_same <= same_lane._scale_lat_dist(same_range[1]) + 1e-9
                    assert cut_range[0] - 1e-9 <= rho_cut <= cut_range[1] + 1e-9
        finally:
            ego.states_cr[time_step] = saved[0]
            ego.lanelet_assignment[time_step] = saved[1]
            ego.ccosy_cache = saved[2]
            for vehicle in (ego, other):
                vehicle.predicate_cache.cache[time_step] = defaultdict()
