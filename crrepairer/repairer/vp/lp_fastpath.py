"""Lower-overhead, result-identical ``linprog(method="highs")`` for VP planning.

Every VP iteration solves small LPs with ``scipy.optimize.linprog``.  Besides
the HiGHS solve itself, each call validates and copies its inputs, converts the
constraint matrices to COO, stacks them and converts them to CSC, and parses
the solver options.  For the cached acceleration template the matrices are the
same objects in every call, so that conversion is repeated work.

``highs_linprog`` reproduces the exact sequence of SciPy 1.15.3
(``linprog`` -> ``_clean_inputs`` -> ``_linprog_highs`` -> ``_check_result``):
the same cleaned arrays, the same CSC matrix, the same HiGHS options and the
same result fields.  The CSC matrix of a sparse input pair is computed once
per pair of matrix objects (entries keep the objects; a hit requires the same
objects).  Inputs outside the shapes used by VP planning, non-finite inputs,
other SciPy versions, or ``CRREPAIR_VP_LP_FASTPATH=0`` call ``linprog``
directly, so errors and results are unchanged in every case.
"""

import os

import numpy as np
import scipy
from scipy.optimize import OptimizeResult, linprog

FASTPATH_ENV = "CRREPAIR_VP_LP_FASTPATH"
_SUPPORTED_SCIPY = "1.15.3"

try:  # SciPy-private helpers; the fast path is used only for the version above.
    from scipy.sparse import csc_matrix, issparse, vstack
    from scipy.optimize._linprog_util import (
        _check_result,
        _format_A_constraints,
        _format_b_constraints,
    )
    from scipy.optimize._linprog_highs import (
        _highs_to_scipy_status_message,
        _highs_wrapper,
        _replace_inf,
    )
    from scipy.optimize._highspy._core import (
        HighsDebugLevel,
        HighsModelStatus,
        ObjSense,
        simplex_constants as s_c,
    )

    _AVAILABLE = scipy.__version__ == _SUPPORTED_SCIPY
except Exception:  # pragma: no cover - other SciPy layouts
    _AVAILABLE = False

_CSC_CACHE = {}
_CSC_CACHE_SIZE = 32


def _enabled():
    return _AVAILABLE and os.environ.get(FASTPATH_ENV, "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds):
    return linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method="highs",
    )


def _highs_options():
    """Options built by ``_linprog_highs`` for ``linprog(method="highs")``."""
    return {
        "presolve": True,
        "sense": ObjSense.kMinimize,
        "solver": None,
        "time_limit": None,
        "highs_debug_level": HighsDebugLevel.kHighsDebugLevelNone,
        "dual_feasibility_tolerance": None,
        "ipm_optimality_tolerance": None,
        "log_to_console": False,
        "mip_max_nodes": None,
        "output_flag": False,
        "primal_feasibility_tolerance": None,
        "simplex_dual_edge_weight_strategy": None,
        "simplex_strategy": s_c.SimplexStrategy.kSimplexStrategyDual,
        "ipm_iteration_limit": None,
        "simplex_iteration_limit": None,
        "mip_rel_gap": None,
    }


def highs_linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Return ``linprog(..., method="highs")`` with less per-call overhead."""
    if not _enabled() or not isinstance(bounds, list):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # ---- _clean_inputs (identical conversions; invalid input -> linprog) ----
    try:
        c_clean = np.array(c, dtype=np.float64, copy=True).squeeze()
    except ValueError:
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if c_clean.size == 1:
        c_clean = c_clean.reshape(-1)
    n_x = len(c_clean)
    if n_x == 0 or c_clean.ndim != 1 or not np.isfinite(c_clean).all():
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    sparse = issparse(A_eq) or issparse(A_ub)
    if sparse and (
        (A_ub is not None and not issparse(A_ub))
        or (A_eq is not None and not issparse(A_eq))
    ):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    try:
        b_ub_clean = _format_b_constraints(b_ub)
        b_eq_clean = _format_b_constraints(b_eq)
    except (TypeError, ValueError):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    A = None
    if sparse:
        # The template matrices are immutable; their cleaned/stacked CSC form
        # (validated once) is cached per pair of matrix objects.
        entry = _CSC_CACHE.get((id(A_ub), id(A_eq)))
        if entry is not None and entry[0] is A_ub and entry[1] is A_eq:
            A, n_ub_rows, n_eq_rows = entry[2], entry[3], entry[4]
    if A is None:
        try:
            A_ub_clean = _format_A_constraints(A_ub, n_x, sparse_lhs=sparse)
            A_eq_clean = _format_A_constraints(A_eq, n_x, sparse_lhs=sparse)
        except (TypeError, ValueError):
            return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
        for matrix in (A_ub_clean, A_eq_clean):
            values = matrix.data if sparse else matrix
            if (
                len(matrix.shape) != 2
                or matrix.shape[1] != n_x
                or not np.isfinite(values).all()
            ):
                return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
        n_ub_rows, n_eq_rows = A_ub_clean.shape[0], A_eq_clean.shape[0]
        if sparse:
            A = csc_matrix(vstack((A_ub_clean, A_eq_clean)))
            if len(_CSC_CACHE) >= _CSC_CACHE_SIZE:
                _CSC_CACHE.clear()
            _CSC_CACHE[(id(A_ub), id(A_eq))] = (A_ub, A_eq, A, n_ub_rows, n_eq_rows)
        else:
            A = csc_matrix(np.vstack((A_ub_clean, A_eq_clean)))
    if (
        b_ub_clean.shape != (n_ub_rows,)
        or b_eq_clean.shape != (n_eq_rows,)
        or not np.isfinite(b_ub_clean).all()
        or not np.isfinite(b_eq_clean).all()
    ):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if len(bounds) != n_x or n_x <= 2:
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    try:
        bounds_clean = np.atleast_2d(np.array(bounds, dtype=float))
    except (TypeError, ValueError):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if bounds_clean.shape != (n_x, 2):
        return _fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
    bounds_clean[np.isnan(bounds_clean[:, 0]), 0] = -np.inf
    bounds_clean[np.isnan(bounds_clean[:, 1]), 1] = np.inf

    # ---- _linprog_highs ----
    lb, ub = bounds_clean.T.copy()
    with np.errstate(invalid="ignore"):
        lhs_ub = -np.ones_like(b_ub_clean) * np.inf
    lhs = np.concatenate((lhs_ub, b_eq_clean))
    rhs = np.concatenate((b_ub_clean, b_eq_clean))
    rhs = _replace_inf(rhs)
    lhs = _replace_inf(lhs)
    lb = _replace_inf(lb)
    ub = _replace_inf(ub)
    integrality = np.empty(0)
    res = _highs_wrapper(
        c_clean, A.indptr, A.indices, A.data, lhs, rhs, lb, ub,
        integrality.astype(np.uint8), _highs_options(),
    )
    if "slack" in res:
        slack = res["slack"]
        con = np.array(slack[len(b_ub_clean):])
        slack = np.array(slack[:len(b_ub_clean)])
    else:
        slack, con = None, None
    if "lambda" in res:
        lamda = res["lambda"]
        marg_ineqlin = np.array(lamda[:len(b_ub_clean)])
        marg_eqlin = np.array(lamda[len(b_ub_clean):])
        marg_upper = np.array(res["marg_bnds"][1, :])
        marg_lower = np.array(res["marg_bnds"][0, :])
    else:
        marg_ineqlin, marg_eqlin = None, None
        marg_upper, marg_lower = None, None
    highs_status = res.get("status", None)
    highs_message = res.get("message", None)
    status, message = _highs_to_scipy_status_message(highs_status, highs_message)
    x = res["x"]
    sol = {
        "x": x,
        "slack": slack,
        "con": con,
        "ineqlin": OptimizeResult({"residual": slack, "marginals": marg_ineqlin}),
        "eqlin": OptimizeResult({"residual": con, "marginals": marg_eqlin}),
        "lower": OptimizeResult({
            "residual": None if x is None else x - lb,
            "marginals": marg_lower,
        }),
        "upper": OptimizeResult({
            "residual": None if x is None else ub - x,
            "marginals": marg_upper,
        }),
        "fun": res.get("fun"),
        "status": status,
        "success": res["status"] == HighsModelStatus.kOptimal,
        "message": message,
        "nit": res.get("simplex_nit", 0) or res.get("ipm_nit", 0),
        "crossover_nit": res.get("crossover_nit"),
    }
    if np.any(x) and integrality is not None:
        sol.update({
            "mip_node_count": res.get("mip_node_count", 0),
            "mip_dual_bound": res.get("mip_dual_bound", 0.0),
            "mip_gap": res.get("mip_gap", 0.0),
        })

    # ---- linprog ----
    sol["status"], sol["message"] = _check_result(
        sol["x"], sol["fun"], sol["status"], sol["slack"], sol["con"],
        bounds_clean, 1e-9, sol["message"], None,
    )
    sol["success"] = sol["status"] == 0
    return OptimizeResult(sol)
