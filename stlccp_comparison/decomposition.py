"""Structure-aware STL robustness decomposition from Takayama et al.

The existing comparison rules are ``stlpy`` trees.  A linear predicate
``a.T @ y - b >= 0`` has reversed robustness ``b - a.T @ y``.  Conjunctions
become maxima and are decomposed epigraphically, while disjunctions become
minima and remain as the only constraints that CCP has to majorize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from stlpy.STL import LinearPredicate, NonlinearPredicate


@dataclass(frozen=True)
class AffineSignal:
    """Reversed robustness ``offset + coefficients @ y[:, time]``."""

    coefficients: np.ndarray
    offset: float
    time: int
    name: str | None = None

    def evaluate(self, y: np.ndarray, auxiliary: np.ndarray) -> float:
        del auxiliary
        return float(self.offset + self.coefficients @ y[:, self.time])


@dataclass(frozen=True)
class Auxiliary:
    """Reference to an epigraph variable."""

    index: int

    def evaluate(self, y: np.ndarray, auxiliary: np.ndarray) -> float:
        del y
        return float(auxiliary[self.index])


Branch = Union[AffineSignal, Auxiliary]


@dataclass(frozen=True)
class ConvexBound:
    """A convex/affine child robustness constrained by an epigraph."""

    child: Branch
    parent: int


@dataclass(frozen=True)
class DisjunctiveBound:
    """A concave smooth-min constraint handled by one CCP slack variable."""

    children: tuple[Branch, ...]
    parent: int
    leaf_count: int


@dataclass(frozen=True)
class AuxiliaryDefinition:
    """Formula/time pair used to initialize an auxiliary variable."""

    index: int
    formula: object
    time: int


@dataclass(frozen=True)
class STLDecomposition:
    root: int
    num_auxiliary: int
    convex_bounds: tuple[ConvexBound, ...]
    disjunctive_bounds: tuple[DisjunctiveBound, ...]
    auxiliary_definitions: tuple[AuxiliaryDefinition, ...]
    num_predicate_occurrences: int

    def initialize_auxiliary(self, y: np.ndarray) -> np.ndarray:
        values = np.zeros(self.num_auxiliary, dtype=float)
        for definition in self.auxiliary_definitions:
            values[definition.index] = reversed_robustness(
                definition.formula, y, definition.time
            )
        return values


def _is_predicate(formula) -> bool:
    return isinstance(formula, (LinearPredicate, NonlinearPredicate))


def _leaf_count(formula) -> int:
    if _is_predicate(formula):
        return 1
    return sum(_leaf_count(child) for child in formula.subformula_list)


def _linear_leaf(predicate: LinearPredicate, time: int, horizon: int) -> AffineSignal:
    if not 0 <= time < horizon:
        raise IndexError(
            f"STL predicate time {time} outside horizon [0, {horizon - 1}]"
        )
    coefficients = -np.asarray(predicate.a, dtype=float).reshape(-1)
    offset = float(np.asarray(predicate.b, dtype=float).reshape(-1)[0])
    return AffineSignal(coefficients, offset, time, getattr(predicate, "name", None))


def reversed_robustness(formula, y: np.ndarray, time: int = 0) -> float:
    """Evaluate the exact, unsmoothed reversed robustness recursively."""

    if isinstance(formula, LinearPredicate):
        leaf = _linear_leaf(formula, time, y.shape[1])
        return leaf.evaluate(y, np.empty(0))
    if isinstance(formula, NonlinearPredicate):
        raise TypeError("STLCCP comparison currently supports linear predicates only")
    values = [
        reversed_robustness(child, y, time + int(child_time))
        for child, child_time in zip(formula.subformula_list, formula.timesteps)
    ]
    if not values:
        raise ValueError("STL operator has no children")
    if formula.combination_type == "and":
        return float(max(values))
    if formula.combination_type == "or":
        return float(min(values))
    raise ValueError(f"Unsupported STL combination: {formula.combination_type}")


class _Builder:
    def __init__(self, formula, horizon: int):
        self.formula = formula
        self.horizon = int(horizon)
        self.convex: list[ConvexBound] = []
        self.disjunctive: list[DisjunctiveBound] = []
        self.definitions: list[AuxiliaryDefinition] = []
        self._next_auxiliary = 0

    def new_auxiliary(self, formula, time: int) -> int:
        index = self._next_auxiliary
        self._next_auxiliary += 1
        self.definitions.append(AuxiliaryDefinition(index, formula, time))
        return index

    @staticmethod
    def children(formula, time: int):
        return [
            (child, time + int(child_time))
            for child, child_time in zip(
                formula.subformula_list, formula.timesteps
            )
        ]

    def flattened(self, formula, time: int, kind: str):
        result = []
        for child, child_time in self.children(formula, time):
            if not _is_predicate(child) and child.combination_type == kind:
                result.extend(self.flattened(child, child_time, kind))
            else:
                result.append((child, child_time))
        return result

    def branch(self, formula, time: int) -> Branch:
        if isinstance(formula, LinearPredicate):
            return _linear_leaf(formula, time, self.horizon)
        if isinstance(formula, NonlinearPredicate):
            raise TypeError("STLCCP comparison currently supports linear predicates only")
        auxiliary = self.new_auxiliary(formula, time)
        self.emit(formula, time, auxiliary)
        return Auxiliary(auxiliary)

    def emit(self, formula, time: int, parent: int) -> None:
        if _is_predicate(formula):
            self.convex.append(ConvexBound(self.branch(formula, time), parent))
            return
        kind = formula.combination_type
        if kind == "and":
            # max(children) <= parent iff every child <= parent.  Adjacent
            # conjunctions can share the same epigraph without auxiliaries.
            for child, child_time in self.flattened(formula, time, "and"):
                if _is_predicate(child):
                    value = self.branch(child, child_time)
                    self.convex.append(ConvexBound(value, parent))
                else:
                    self.emit(child, child_time, parent)
            return
        if kind == "or":
            flattened = self.flattened(formula, time, "or")
            branches = tuple(self.branch(child, child_time) for child, child_time in flattened)
            if not branches:
                raise ValueError("STL disjunction has no children")
            self.disjunctive.append(
                DisjunctiveBound(branches, parent, _leaf_count(formula))
            )
            return
        raise ValueError(f"Unsupported STL combination: {kind}")

    def build(self) -> STLDecomposition:
        root = self.new_auxiliary(self.formula, 0)
        self.emit(self.formula, 0, root)
        return STLDecomposition(
            root=root,
            num_auxiliary=self._next_auxiliary,
            convex_bounds=tuple(self.convex),
            disjunctive_bounds=tuple(self.disjunctive),
            auxiliary_definitions=tuple(self.definitions),
            num_predicate_occurrences=_leaf_count(self.formula),
        )


def decompose_formula(formula, horizon: int) -> STLDecomposition:
    """Apply the paper's structure-aware epigraphic decomposition."""

    return _Builder(formula, horizon).build()
