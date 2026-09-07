"""Single source of truth for the comparison cohorts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VP_RESULT_ROOT = REPO_ROOT / "evaluation/config/vp_temporal_full"

GROUPS = {
    "rg1": {
        "dataset": "highd",
        "rule": "R_G1",
        "input": VP_RESULT_ROOT / "vp_repairer_rg1_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    },
    "rg1_mona": {
        "dataset": "highd",
        "rule": "R_G1",
        "input": VP_RESULT_ROOT / "vp_repairer_rg1_mona_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/mona/scenarios"),
    },
    "rg2": {
        "dataset": "highd",
        "rule": "R_G2",
        "input": VP_RESULT_ROOT / "vp_repairer_rg2_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    },
    "rg3": {
        "dataset": "highd",
        "rule": "R_G3",
        "input": VP_RESULT_ROOT / "vp_repairer_rg3_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    },
    "rg1_rg3": {
        "dataset": "highd",
        "rule": "R_G1_R_G3",
        "input": VP_RESULT_ROOT / "vp_repairer_rg1_rg3_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/highD-repair"),
    },
    "in1": {
        "dataset": "ind",
        "rule": "R_IN1",
        "input": VP_RESULT_ROOT / "vp_repairer_in1_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024"),
    },
    # This is the current primary VP R_IN3 setup recorded by the project.  The
    # complete formula remains available as the optional in3_full cohort.
    "in3": {
        "dataset": "ind",
        "rule": "R_IN3_hand_draft",
        "input": VP_RESULT_ROOT / "vp_repairer_in3_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024"),
    },
    "in3_full": {
        "dataset": "ind",
        "rule": "R_IN3",
        "input": VP_RESULT_ROOT / "vp_repairer_in3_full_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024"),
    },
    "in4": {
        "dataset": "ind",
        "rule": "R_IN4",
        "input": VP_RESULT_ROOT / "vp_repairer_in4_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024"),
    },
    "in5": {
        "dataset": "ind",
        "rule": "R_IN5",
        "input": VP_RESULT_ROOT / "vp_repairer_in5_batch_result_updated.csv",
        "scenario_dir": Path("/data_linux/Lab/highD-cr-scenarios/ind_scenarios_2024"),
    },
}

# All distinct rules handled by VP. MONA is an extra RG1 dataset; in3_full is
# an optional formula variant, so neither is repeated in the default run.
DEFAULT_GROUPS = ("rg1", "rg2", "rg3", "rg1_rg3", "in1", "in3", "in4", "in5")


def parse_groups(value: str):
    groups = DEFAULT_GROUPS if value == "all" else tuple(x.strip() for x in value.split(",") if x.strip())
    unknown = sorted(set(groups) - GROUPS.keys())
    if unknown:
        raise ValueError(f"Unknown groups: {', '.join(unknown)}")
    return groups

