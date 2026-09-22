"""Modern CVXPY implementation of STLCCP and tree-weighted penalty CCP."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from scipy.special import logsumexp

from .decomposition import (
    AffineSignal,
    Auxiliary,
    Branch,
    STLDecomposition,
    reversed_robustness,
)


@dataclass
class STLCCPResult:
    x: np.ndarray | None
    u: np.ndarray | None
    robustness: float
    status: str
    converged: bool
    iterations: int
    max_slack: float | None
    solve_wall_time: float
    solver_time: float
    objective: float | None
    phase_history: list[dict] = field(default_factory=list)


def smooth_min_and_gradient(values, mode: str, k: float):
    """Return a smooth minimum and its gradient, evaluated stably."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        raise ValueError("smooth minimum needs at least one value")
    if mode == "true_min":
        gradient = np.zeros_like(values)
        gradient[int(np.argmin(values))] = 1.0
        return float(np.min(values)), gradient
    if mode not in {"lse", "mellowmin"}:
        raise ValueError(f"Unsupported smoothing mode: {mode}")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("Smoothing parameter k must be positive and finite")
    logits = -float(k) * values
    normalizer = logsumexp(logits)
    gradient = np.exp(logits - normalizer)
    result = -normalizer / float(k)
    if mode == "mellowmin":
        result += math.log(len(values)) / float(k)
    return float(result), gradient


class STLCCPSolver:
    """Solve a linear-system STL problem using sequential convex QPs.

    Formula disjunctions are the only relaxed/non-convex constraints.  Their
    smooth reversed robustness is concave, so its tangent is a global upper
    bound and yields the convex majorization required by CCP.
    """

    def __init__(
        self,
        formula,
        decomposition: STLDecomposition,
        system,
        x0,
        horizon: int,
        *,
        Q=None,
        R=None,
        state_min=None,
        state_max=None,
        control_min=None,
        control_max=None,
        chart_min=None,
        chart_max=None,
        robustness_margin: float = 0.01,
        slack_tolerance: float = 1e-5,
    ):
        started = time.perf_counter()
        self.formula = formula
        self.decomposition = decomposition
        self.system = system
        self.horizon = int(horizon)
        self.x0 = np.asarray(x0, dtype=float).reshape(system.n)
        self.Q = np.zeros((system.n, system.n)) if Q is None else np.asarray(Q, dtype=float)
        self.R = np.zeros((system.m, system.m)) if R is None else np.asarray(R, dtype=float)
        self.robustness_margin = float(robustness_margin)
        self.slack_tolerance = float(slack_tolerance)

        if self.horizon <= 0:
            raise ValueError("horizon must contain at least one sample")
        if decomposition.num_auxiliary <= 0:
            raise ValueError("decomposition must contain a root auxiliary")

        self.x = cp.Variable((system.n, self.horizon), name="x")
        self.u = cp.Variable((system.m, self.horizon), name="u")
        self.auxiliary = cp.Variable(decomposition.num_auxiliary, name="rho_aux")
        self.slack = (
            cp.Variable(len(decomposition.disjunctive_bounds), nonneg=True, name="ccp_slack")
            if decomposition.disjunctive_bounds else None
        )
        self.tau = cp.Parameter(nonneg=True, value=0.005, name="tau")
        self.y = system.C @ self.x + system.D @ self.u

        constraints = [self.x[:, 0] == self.x0]
        for step in range(self.horizon - 1):
            constraints.append(
                self.x[:, step + 1]
                == system.A @ self.x[:, step] + system.B @ self.u[:, step]
            )

        self._add_box_constraints(constraints, self.x, state_min, state_max)
        self._add_box_constraints(constraints, self.u, control_min, control_max)
        if chart_min is not None and chart_max is not None:
            chart_min = np.asarray(chart_min, dtype=float).reshape(-1)
            chart_max = np.asarray(chart_max, dtype=float).reshape(-1)
            for index in range(min(2, len(chart_min), len(chart_max))):
                if math.isfinite(chart_min[index]):
                    constraints.append(self.x[index, :] >= chart_min[index])
                if math.isfinite(chart_max[index]):
                    constraints.append(self.x[index, :] <= chart_max[index])

        for bound in decomposition.convex_bounds:
            constraints.append(
                self._branch_expression(bound.child)
                <= self.auxiliary[bound.parent]
            )

        self._linearizations = []
        for index, bound in enumerate(decomposition.disjunctive_bounds):
            gradient = cp.Parameter(len(bound.children), nonneg=True, name=f"gradient_{index}")
            intercept = cp.Parameter(name=f"intercept_{index}")
            expressions = cp.hstack([
                self._branch_expression(child) for child in bound.children
            ])
            tangent = intercept + cp.sum(cp.multiply(gradient, expressions))
            constraints.append(
                tangent <= self.auxiliary[bound.parent] + self.slack[index]
            )
            self._linearizations.append((gradient, intercept, bound))

        # Theorem 6 requires at least -N_or*s_c to retain satisfaction with
        # residual slack.  The experiment independently asks for a robustness
        # margin.  Enforce the stronger of those two thresholds.  Adding them
        # would impose an unintended extra margin that grows with formula size
        # (and made large temporal trees artificially infeasible).
        threshold = -max(
            self.robustness_margin,
            len(decomposition.disjunctive_bounds) * self.slack_tolerance,
        )
        constraints.append(self.auxiliary[decomposition.root] <= threshold)

        quadratic = 0
        for step in range(self.horizon):
            quadratic += cp.quad_form(self.x[:, step], self.Q)
            quadratic += cp.quad_form(self.u[:, step], self.R)
        objective = self.auxiliary[decomposition.root] + quadratic
        if self.slack is not None:
            weights = np.asarray(
                [bound.leaf_count for bound in decomposition.disjunctive_bounds],
                dtype=float,
            )
            objective += self.tau * (weights @ self.slack)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self.problem.is_dcp():
            raise ValueError("Constructed STLCCP subproblem is not convex")
        self.model_setup_time = time.perf_counter() - started

    @staticmethod
    def _add_box_constraints(constraints, variable, lower, upper):
        if lower is not None:
            for index, value in enumerate(np.asarray(lower, dtype=float).reshape(-1)):
                if math.isfinite(value):
                    constraints.append(variable[index, :] >= value)
        if upper is not None:
            for index, value in enumerate(np.asarray(upper, dtype=float).reshape(-1)):
                if math.isfinite(value):
                    constraints.append(variable[index, :] <= value)

    def _branch_expression(self, branch: Branch):
        if isinstance(branch, AffineSignal):
            return branch.offset + branch.coefficients @ self.y[:, branch.time]
        if isinstance(branch, Auxiliary):
            return self.auxiliary[branch.index]
        raise TypeError(f"Unknown decomposed branch: {type(branch)!r}")

    @staticmethod
    def _branch_value(branch: Branch, y, auxiliary):
        return branch.evaluate(y, auxiliary)

    def _output(self, x, u):
        return self.system.C @ x + self.system.D @ u

    def _initialize(self, initial_x=None, initial_u=None):
        if initial_u is None:
            initial_u = np.zeros((self.system.m, self.horizon), dtype=float)
        else:
            initial_u = np.asarray(initial_u, dtype=float).reshape(
                self.system.m, self.horizon
            )
        if initial_x is None:
            initial_x = np.zeros((self.system.n, self.horizon), dtype=float)
            initial_x[:, 0] = self.x0
            for step in range(self.horizon - 1):
                initial_x[:, step + 1] = (
                    self.system.A @ initial_x[:, step]
                    + self.system.B @ initial_u[:, step]
                )
        else:
            initial_x = np.asarray(initial_x, dtype=float).reshape(
                self.system.n, self.horizon
            )
            initial_x[:, 0] = self.x0
        y = self._output(initial_x, initial_u)
        auxiliary = self.decomposition.initialize_auxiliary(y)
        self.x.value = initial_x
        self.u.value = initial_u
        self.auxiliary.value = auxiliary
        if self.slack is not None:
            self.slack.value = np.zeros(len(self.decomposition.disjunctive_bounds))
        return initial_x, initial_u, auxiliary

    def _update_linearizations(self, x, u, auxiliary, mode, k):
        y = self._output(x, u)
        for gradient_parameter, intercept_parameter, bound in self._linearizations:
            values = np.asarray([
                self._branch_value(child, y, auxiliary)
                for child in bound.children
            ])
            smooth_value, gradient = smooth_min_and_gradient(values, mode, k)
            gradient_parameter.value = gradient
            intercept_parameter.value = smooth_value - float(gradient @ values)

    def _original_objective(self, x, u, auxiliary):
        value = float(auxiliary[self.decomposition.root])
        for step in range(self.horizon):
            value += float(x[:, step] @ self.Q @ x[:, step])
            value += float(u[:, step] @ self.R @ u[:, step])
        return value

    @property
    def size_metrics(self):
        metrics = self.problem.size_metrics
        return {
            "num_variables": int(metrics.num_scalar_variables),
            "num_constraints": int(
                metrics.num_scalar_eq_constr + metrics.num_scalar_leq_constr
            ),
            "num_equalities": int(metrics.num_scalar_eq_constr),
            "num_inequalities": int(metrics.num_scalar_leq_constr),
        }

    def solve(
        self,
        *,
        phases=("lse", "mellowmin"),
        lse_k=10.0,
        mellowmin_k=1000.0,
        initial_x=None,
        initial_u=None,
        tau0=5e-3,
        tau_max=1e3,
        tau_rate=2.0,
        max_iterations=25,
        cost_tolerance=1e-2,
        time_limit=None,
        threads=None,
        verbose=False,
    ) -> STLCCPResult:
        phases = tuple(phases)
        if not phases:
            raise ValueError("At least one STLCCP phase is required")
        if not self.decomposition.disjunctive_bounds:
            # With no min/disjunctive node the reformulated problem is already
            # one convex QP; smoothing and CCP iterations are unnecessary.
            phases = ("convex",)
        x_value, u_value, auxiliary_value = self._initialize(initial_x, initial_u)
        started = time.perf_counter()
        solver_time = 0.0
        total_iterations = 0
        phase_history = []
        status = "not_solved"
        converged = False

        for phase in phases:
            k = lse_k if phase == "lse" else mellowmin_k
            if phase == "true_min":
                k = 1.0
            tau = float(tau0)
            previous_original = None
            previous_penalized = None
            phase_converged = False
            for phase_iteration in range(1, int(max_iterations) + 1):
                elapsed = time.perf_counter() - started
                if time_limit is not None and elapsed >= time_limit:
                    status = "time_limit"
                    break
                self._update_linearizations(
                    x_value, u_value, auxiliary_value, phase, k
                )
                self.tau.value = tau
                options = {}
                if time_limit is not None:
                    options["TimeLimit"] = max(1e-3, float(time_limit) - elapsed)
                if threads is not None:
                    options["Threads"] = int(threads)
                try:
                    self.problem.solve(
                        solver=cp.GUROBI,
                        warm_start=True,
                        verbose=bool(verbose),
                        reoptimize=True,
                        **options,
                    )
                except cp.error.SolverError as exc:
                    status = f"solver_error: {exc}"
                    break
                status = str(self.problem.status)
                total_iterations += 1
                if self.problem.solver_stats.solve_time is not None:
                    solver_time += float(self.problem.solver_stats.solve_time)
                unbounded_statuses = {
                    cp.UNBOUNDED,
                    cp.UNBOUNDED_INACCURATE,
                    getattr(cp, "INFEASIBLE_OR_UNBOUNDED", "infeasible_or_unbounded"),
                }
                if self.problem.status in unbounded_statuses and tau < tau_max:
                    # With a very small formula, the paper's initial penalty
                    # tau0 can cost less than the improvement obtained by
                    # lowering the root auxiliary and increasing a slack.
                    # Penalty CCP already prescribes increasing tau; retry the
                    # same majorization instead of abandoning the phase.
                    phase_history.append({
                        "phase": phase,
                        "iteration": phase_iteration,
                        "tau": tau,
                        "status": status,
                        "penalty_retry": True,
                        "solver_time": self.problem.solver_stats.solve_time,
                    })
                    tau = min(float(tau_max), tau * float(tau_rate))
                    continue
                acceptable = {
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                    getattr(cp, "USER_LIMIT", "user_limit"),
                }
                if self.problem.status not in acceptable:
                    break
                if any(value.value is None for value in (self.x, self.u, self.auxiliary)):
                    status = "no_incumbent"
                    break
                x_value = np.asarray(self.x.value, dtype=float)
                u_value = np.asarray(self.u.value, dtype=float)
                auxiliary_value = np.asarray(self.auxiliary.value, dtype=float)
                max_slack = (
                    0.0
                    if self.slack is None
                    else float(np.max(np.asarray(self.slack.value, dtype=float)))
                )
                original = self._original_objective(
                    x_value, u_value, auxiliary_value
                )
                penalized = (
                    None if self.problem.value is None else float(self.problem.value)
                )
                original_change = (
                    math.inf
                    if previous_original is None
                    else abs(original - previous_original)
                )
                penalized_change = (
                    math.inf
                    if previous_penalized is None or penalized is None
                    else abs(penalized - previous_penalized)
                )
                phase_history.append({
                    "phase": phase,
                    "iteration": phase_iteration,
                    "tau": tau,
                    "status": status,
                    "objective": original,
                    "penalized_objective": penalized,
                    "max_slack": max_slack,
                    "original_change": (
                        None if not math.isfinite(original_change) else original_change
                    ),
                    "penalized_change": (
                        None if not math.isfinite(penalized_change) else penalized_change
                    ),
                    "solver_time": self.problem.solver_stats.solve_time,
                })
                if self.slack is None or (
                    max_slack <= self.slack_tolerance
                    and original_change <= cost_tolerance
                    and penalized_change <= cost_tolerance
                ):
                    phase_converged = True
                    break
                previous_original = original
                previous_penalized = penalized
                tau = min(float(tau_max), tau * float(tau_rate))
            else:
                status = "iteration_limit"

            phase_history.append({
                "phase": phase,
                "summary": True,
                "converged": phase_converged,
            })
            converged = phase_converged
            if status in {"time_limit", "no_incumbent"} or status.startswith("solver_error"):
                break
            # A valid iterate can warm-start the next smoothing phase even if
            # this phase hit its iteration cap.

        wall_time = time.perf_counter() - started
        has_candidate = self.x.value is not None and self.u.value is not None
        if not has_candidate:
            return STLCCPResult(
                None, None, -math.inf, status, False, total_iterations, None,
                wall_time, solver_time, None, phase_history,
            )
        x_value = np.asarray(self.x.value, dtype=float)
        u_value = np.asarray(self.u.value, dtype=float)
        auxiliary_value = np.asarray(self.auxiliary.value, dtype=float)
        y_value = self._output(x_value, u_value)
        exact_reversed = reversed_robustness(self.formula, y_value, 0)
        max_slack = (
            0.0
            if self.slack is None
            else float(np.max(np.asarray(self.slack.value, dtype=float)))
        )
        return STLCCPResult(
            x=x_value,
            u=u_value,
            robustness=-exact_reversed,
            status=status,
            converged=converged,
            iterations=total_iterations,
            max_slack=max_slack,
            solve_wall_time=wall_time,
            solver_time=solver_time,
            objective=(None if self.problem.value is None else float(self.problem.value)),
            phase_history=phase_history,
        )
