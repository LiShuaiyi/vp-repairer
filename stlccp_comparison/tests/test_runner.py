from types import SimpleNamespace

import pytest

import stlccp_comparison.runner as runner


def test_rg2_missing_predecessor_uses_none_fallback(monkeypatch):
    def missing(*_args):
        raise ValueError("R_G2 has no preceding vehicle at the original trigger")

    monkeypatch.setattr(runner, "select_fixed_other_id", missing)
    assert runner.comparison_fixed_other_id(SimpleNamespace(), "R_G2", 1) is None


def test_missing_predecessor_is_not_suppressed_for_other_rules(monkeypatch):
    def missing(*_args):
        raise ValueError("missing related vehicle")

    monkeypatch.setattr(runner, "select_fixed_other_id", missing)
    with pytest.raises(ValueError, match="missing related vehicle"):
        runner.comparison_fixed_other_id(SimpleNamespace(), "R_G1", 1)


def test_rg2_uses_full_scene_validation():
    assert runner.validation_other_id("R_G2", 7) is None
    assert runner.validation_other_id("R_G1", 7) == 7
