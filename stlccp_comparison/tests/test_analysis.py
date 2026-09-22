import pytest

from stlccp_comparison.analyze import paired_method_summary, stlccp_summary


def test_summary_uses_stlccp_metric_names():
    key = ("scenario", 1, "R_G3")
    rows = stlccp_summary(
        {key: {"success": True, "feasible": True, "time": 1.0}},
        {key: {"success": True, "time": 0.1}},
    )
    assert rows[0]["stlccp_success"] == 1
    assert rows[0]["stlccp_over_vp_geomean"] == pytest.approx(10.0)
    assert "micp_success" not in rows[0]


def test_paired_method_summary_counts_outcomes_and_runtime():
    keys = [("a", 1, "R_G3"), ("b", 2, "R_G3"), ("c", 3, "R_G3")]
    stlccp = {
        keys[0]: {"success": True, "feasible": True, "time": 1.0},
        keys[1]: {"success": True, "feasible": True, "time": 2.0},
        keys[2]: {"success": False, "feasible": False, "time": 4.0},
    }
    micp = {
        keys[0]: {"success": True, "feasible": True, "time": 2.0},
        keys[1]: {"success": False, "feasible": True, "time": 2.0},
        keys[2]: {"success": True, "feasible": True, "time": 2.0},
    }
    row = paired_method_summary(stlccp, micp)[0]
    assert row["both_success"] == 1
    assert row["stlccp_only_success"] == 1
    assert row["micp_only_success"] == 1
    assert row["neither_success"] == 0
    assert row["stlccp_over_micp_geomean"] == pytest.approx(1.0)
    assert row["stlccp_over_micp_geomean_both_success"] == pytest.approx(0.5)
