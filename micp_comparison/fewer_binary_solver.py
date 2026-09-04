"""Gurobi implementation of the Kurtz--Lin fewer-binary STL encoding.

Unlike :class:`stlpy.solvers.GurobiMICPSolver`, predicate leaves stay
continuous.  Binary variables are introduced only for disjunctions, using the
logarithmic SOS1 formulation from Kurtz and Lin (2022).
"""

from __future__ import annotations

import math

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from stlpy.STL import LinearPredicate, NonlinearPredicate
from stlpy.solvers import GurobiMICPSolver


class FewerBinaryGurobiSolver(GurobiMICPSolver):
    """Drop-in ``stlpy`` solver with logarithmic SOS1 disjunctions."""

    def _flat_children(self, formula, time_offset=0):
        """Flatten adjacent equal operators while preserving time offsets."""
        result = []
        for child, child_time in zip(formula.subformula_list, formula.timesteps):
            offset = time_offset + int(child_time)
            if (
                not isinstance(child, (LinearPredicate, NonlinearPredicate))
                and child.combination_type == formula.combination_type
            ):
                result.extend(self._flat_children(child, offset))
            else:
                result.append((child, offset))
        return result

    def _add_logarithmic_sos1(self, size, name):
        """Return an SOS1 vector encoded with ``ceil(log2(size))`` binaries."""
        if size < 2:
            raise ValueError("SOS1 needs at least two entries")
        weights = self.model.addMVar(
            size, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"{name}_lambda"
        )
        self.model.addConstr(weights.sum() == 1.0, name=f"{name}_sum")
        bits = self.model.addMVar(
            math.ceil(math.log2(size)), vtype=GRB.BINARY, name=f"{name}_bits"
        )
        for bit_index in range(bits.shape[0]):
            with_bit = [i for i in range(size) if (i >> bit_index) & 1]
            without_bit = [i for i in range(size) if not ((i >> bit_index) & 1)]
            if with_bit:
                self.model.addConstr(
                    gp.quicksum(weights[i] for i in with_bit) <= bits[bit_index]
                )
            if without_bit:
                self.model.addConstr(
                    gp.quicksum(weights[i] for i in without_bit)
                    <= 1.0 - bits[bit_index]
                )
        return weights

    def AddSubformulaConstraints(self, formula, z, t):
        if isinstance(formula, LinearPredicate):
            if not 0 <= t < self.T:
                raise IndexError(f"STL predicate time {t} outside horizon [0,{self.T - 1}]")
            self.model.addConstr(
                formula.a.T @ self.y[:, t]
                - formula.b
                + (1.0 - z) * self.M
                >= self.rho
            )
            return
        if isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        children = self._flat_children(formula)
        if formula.combination_type == "and":
            z_children = self.model.addMVar(
                len(children), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name="and_z",
            )
            self.model.addConstr(z <= z_children)
        elif formula.combination_type == "or":
            # lambda[0] represents the inactive parent (1-z); one of the
            # remaining entries selects the active disjunct when z=1.
            weights = self._add_logarithmic_sos1(len(children) + 1, "or")
            self.model.addConstr(weights[0] == 1.0 - z)
            z_children = weights[1:]
        else:
            raise ValueError(f"Unsupported STL combination: {formula.combination_type}")

        for index, (child, child_time) in enumerate(children):
            self.AddSubformulaConstraints(child, z_children[index], t + child_time)

    def Solve(self):
        """Return any feasible incumbent, including at a configured time limit."""
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        self.model.optimize()
        if self.model.SolCount > 0:
            return self.x.X, self.u.X, self.rho.X[0], self.model.Runtime
        return None, None, -np.inf, self.model.Runtime
