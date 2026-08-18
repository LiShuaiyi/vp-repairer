from types import SimpleNamespace

import pytest

from crrepairer.repairer.vp.constraints import (
    UnsupportedVPCandidateError,
    VPConstraintExtraction,
)


class ActiveInterval:
    count = 1

    @staticmethod
    def contains(_time_step):
        return True


def make_extractor(rule, proposition):
    extractor = object.__new__(VPConstraintExtraction)
    extractor.config = SimpleNamespace(repair=SimpleNamespace(rules=[rule]))
    extractor._sel_prop = [proposition]
    extractor._last_extraction_debug = []
    return extractor


def test_in1_non_stop_line_candidate_is_rejected():
    prop = SimpleNamespace(name="some_fixed_intersection_fact", alphabet="a")
    extractor = make_extractor("R_IN1", prop)

    with pytest.raises(UnsupportedVPCandidateError, match="non-stop-line"):
        extractor._validate_intersection_candidate_support(
            temporal_steps={id(prop): ActiveInterval()},
            conflict_trajectory_interval=None,
        )

    assert extractor._last_extraction_debug == [
        {
            "proposition": prop.name,
            "kind": "unsupported_in1_non_stop_line",
        }
    ]


def test_negative_conflict_candidate_without_geometry_is_rejected():
    prop = SimpleNamespace(
        name="in_intersection_conflict_area__0_1",
        alphabet="~a",
    )
    extractor = make_extractor("R_IN3_hand_draft", prop)

    with pytest.raises(UnsupportedVPCandidateError, match="geometry is unavailable"):
        extractor._validate_intersection_candidate_support(
            temporal_steps={id(prop): ActiveInterval()},
            conflict_trajectory_interval=None,
        )

    assert extractor._last_extraction_debug == [
        {
            "proposition": prop.name,
            "kind": "conflict_geometry_unavailable",
        }
    ]


def test_supported_intersection_candidates_pass_validation():
    in1_prop = SimpleNamespace(name="behind_stop_line", alphabet="~a")
    in1 = make_extractor("R_IN1", in1_prop)
    in1._validate_intersection_candidate_support(
        temporal_steps={id(in1_prop): ActiveInterval()},
        conflict_trajectory_interval=None,
    )

    conflict_prop = SimpleNamespace(
        name="in_intersection_conflict_area__0_1",
        alphabet="~b",
    )
    intersection = make_extractor("R_IN4", conflict_prop)
    intersection._validate_intersection_candidate_support(
        temporal_steps={id(conflict_prop): ActiveInterval()},
        conflict_trajectory_interval=(1.0, 2.0, "monitor"),
    )


def test_monitor_geometry_fallback_enables_required_mode_and_restores_it():
    extractor = object.__new__(VPConstraintExtraction)
    extractor._use_monitor_conflict_geometry = False
    observed_modes = []

    def monitor_geometry(**_kwargs):
        observed_modes.append(extractor._use_monitor_conflict_geometry)
        return 1.0, 2.0

    extractor._monitor_conflict_interval_on_trajectory = monitor_geometry

    interval = extractor._monitor_conflict_interval_fallback()

    assert interval == (1.0, 2.0)
    assert observed_modes == [True]
    assert extractor._use_monitor_conflict_geometry is False


def test_unsupported_rg_predicate_uses_candidate_rejection_error():
    prop = SimpleNamespace(name="cut_in__1_0", alphabet="a")
    extractor = object.__new__(VPConstraintExtraction)
    extractor._tc = 0
    extractor._sel_prop = [prop]
    extractor._extract_speed_limit_values = lambda: {
        "lane": None,
        "type": None,
        "fov": None,
        "brake": None,
    }
    extractor._temporal_constraint_steps = lambda _states: {
        id(prop): ActiveInterval()
    }
    states = [
        SimpleNamespace(time_step=0, position=(0.0, 0.0), velocity=5.0),
        SimpleNamespace(time_step=1, position=(1.0, 0.0), velocity=5.0),
    ]
    clcs = SimpleNamespace(
        convert_to_curvilinear_coords=lambda x, _y: (x, 0.0)
    )

    with pytest.raises(UnsupportedVPCandidateError, match="RG predicate"):
        extractor._extract_interstate_constraints_manually(states, clcs)
