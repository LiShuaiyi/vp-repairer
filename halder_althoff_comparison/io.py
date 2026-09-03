"""JSON input/output helpers for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .environment import Environment
from .model import LatticeConfig
from .rules import build_rulebook


def load_problem(path: str | Path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        data: Dict[str, Any] = json.load(stream)
    lattice = LatticeConfig.from_dict(data["lattice"])
    environment = Environment(data, lattice.horizon_steps + 1, lattice.dt)
    parameters = {
        name: dict(values)
        for name, values in data.get("rule_parameters", {}).items()
    }
    if "R_IN1" in data["rule_order"]:
        in1 = parameters.setdefault("R_IN1", {})
        if "stop_line_s" not in in1:
            stop_lines = data.get("stop_lines", ())
            if len(stop_lines) != 1:
                raise ValueError(
                    "R_IN1 needs rule_parameters.R_IN1.stop_line_s or exactly "
                    "one value in stop_lines"
                )
            value = stop_lines[0]
            in1["stop_line_s"] = (
                float(value["s"]) if isinstance(value, dict) else float(value)
            )
    rules = build_rulebook(data["rule_order"], parameters, lattice)
    initial = data["initial_state"]
    return lattice, environment, rules, float(initial["s"]), float(initial["v"]), data


def write_result(path: str | Path, result, metadata=None) -> None:
    payload = result.as_dict()
    if metadata:
        payload["metadata"] = metadata
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
