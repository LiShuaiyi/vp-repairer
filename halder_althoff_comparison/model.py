"""Lattice and trajectory data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class State:
    k: int
    s: float
    v: float
    a: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"k": self.k, "s": self.s, "v": self.v, "a": self.a}


@dataclass(frozen=True)
class LatticeConfig:
    dt: float
    ds: float
    dv: float
    horizon_steps: int
    s_min: float
    s_max: float
    v_min: float
    v_max: float
    a_min: float
    a_max: float
    position_tolerance: float | None = None
    max_expansions: int = 1_000_000

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "LatticeConfig":
        config = cls(**value)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dt <= 0 or self.ds <= 0 or self.dv <= 0:
            raise ValueError("dt, ds and dv must be positive")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be at least one")
        if self.s_min >= self.s_max or self.v_min > self.v_max:
            raise ValueError("invalid state bounds")
        if self.a_min > self.a_max:
            raise ValueError("invalid acceleration bounds")
        if self.max_expansions < 1:
            raise ValueError("max_expansions must be positive")

    @property
    def transition_tolerance(self) -> float:
        if self.position_tolerance is not None:
            return float(self.position_tolerance)
        # Paper-faithful default: only connect lattice nodes that satisfy the
        # constant-acceleration model (up to floating-point error).
        return 1e-7
