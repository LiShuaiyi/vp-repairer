# STLCCP comparison baseline

This directory implements the method from:

> Y. Takayama, K. Hashimoto, and T. Ohtsuka, “STLCCP: Efficient Convex
> Optimization-Based Framework for Signal Temporal Logic Specifications,”
> IEEE Transactions on Automatic Control, 2025.
> DOI: [10.1109/TAC.2025.3555949](https://doi.org/10.1109/TAC.2025.3555949),
> [arXiv:2305.09441](https://arxiv.org/abs/2305.09441).

It is an isolated comparison implementation: it does not modify `crrepairer/`
or the MICP baseline.  It reuses `micp_comparison.rules` so STLCCP and MICP
receive the same linear STL tree, two-dimensional CLCS dynamics, initial state,
state/control bounds, projection chart, fixed target vehicle and final traffic
rule monitor.

For ego-centric `R_G2`, the optimization witness is fixed when available, but
final validation is full-scene and existential, matching the saved MICP
baseline. If no predecessor exists at the recorded trigger, both comparisons
use the Lin2025 implementation's `other=None` / `not_abrupt` fallback.

## Implementation

- `decomposition.py` implements the paper's Algorithm 1.  Conjunction/always
  (`max` in reversed robustness) is decomposed epigraphically.  Each flattened
  disjunction/eventually (`min`) becomes one concave constraint.
- `solver.py` implements tree-weighted penalty CCP.  At every iteration, only
  the concave smooth-min constraints are replaced by their global affine upper
  bounds.  The resulting problem is a convex QP solved through CVXPY/Gurobi.
- `runner.py` provides the CommonRoad batch interface and exact monitor
  validation.
- `analyze.py` produces the same paired VP statistics as the MICP experiment.

The implementation targets the repository's current CVXPY version.  It does
not depend on the authors' CVXPY 1.1.19 fork, replace CVXPY internals, or use
the old external DCCP package.  Stable mellowmin values and gradients are
computed numerically and injected into a parameterized QP.

## Smoothing modes

- `lse`: log-sum-exp smooth minimum with `k=10`, matching the main numerical
  experiments in the paper.  It is not a sound over-approximation.
- `mellowmin`: sound mellowmin, default `k=1000`.
- `lse_mellowmin`: solve LSE first and use its result to warm-start mellowmin.
  This is the default for the traffic-rule comparison.
- `true_min`: one active subgradient of the exact minimum, mainly for
  diagnostics.

For residual CCP slack tolerance `s_c`, Theorem 6 requires a reversed-
robustness threshold of at most `-N_or * s_c`.  The implementation enforces
the stronger of that threshold and the configured robustness margin; it does
not add the two and thereby introduce a formula-size-dependent extra margin.
`encoded_compliant` separately checks the exact unsmoothed robustness.  Every
returned trajectory is also checked by the original `STLRuleMonitor`;
`success=True` only means that this final monitor accepts it.

## Run one smoke case

```bash
/data_linux/conda-envs/repair-autoware/bin/python \
  -m stlccp_comparison.runner \
  --dataset highd \
  --input evaluation/config/vp_temporal_full/vp_repairer_rg1_batch_result_updated.csv \
  --scenario-dir scenarios/experiment_scenarios/highd \
  --output /tmp/stlccp_rg1.csv \
  --rule R_G1 --limit 1 --threads 1 --quiet
```

Use `--gurobi-license` when the default license does not match the host.  The
runner first checks `~/gurobi.lic`, then the repository Docker WLS license.

For the paper's timing-only LSE variant:

```bash
python -m stlccp_comparison.runner ... --smoothing lse --lse-k 10
```

Run the established 704-case population by cohort:

```bash
/data_linux/conda-envs/repair-autoware/bin/python \
  -m stlccp_comparison.regression \
  --output-dir /tmp/stlccp-full \
  --workers 2 --threads 1 --time-limit 60
```

The optional full-semantics `in3` manifest is not part of those 704 cases; run
it explicitly with `--cohorts in3`.  `run_manifest.json` records all options
and output files.

The checked-in result and interpretation of the 704-case run are in
[`RESULTS.md`](RESULTS.md).  Machine-readable outputs are under
`results/latest_full/`.

For initial-value sensitivity experiments, use `--initialization random`,
`--seed`, and `--repeat`.  The default `trajectory` initialization projects the
recorded ego trajectory into the optimizer's CLCS; `rollout` uses zero input.

## Output and timing

The common fields have the same meaning as in `micp_comparison.runner`:

- `solver_feasible`: the final QP produced a candidate;
- `encoded_compliant`: exact unsmoothed robustness meets the configured margin;
- `monitor_compliant` / `success`: the original monitor accepts the rebuilt
  CommonRoad trajectory;
- `core_total_time`: specification, decomposition, CVXPY model setup and all
  CCP/QP iterations; monitor validation is separately reported;
- `qp_solver_time`: sum of solver-reported QP time, excluding CVXPY overhead;
- `ccp_iterations`, `max_final_slack`, and `phase_history`: convergence details.

Use the existing paired-analysis protocol:

```bash
python -m stlccp_comparison.analyze \
  --results /path/to/stlccp-results.csv \
  --vp evaluation/config/vp_temporal_full/vp_repairer_rg1_batch_result_updated.csv \
  --output /tmp/stlccp-summary.json
```

## Tests

```bash
/data_linux/conda-envs/repair-autoware/bin/python \
  -m pytest stlccp_comparison/tests -q
```

The optimization test is skipped when the Gurobi host license is unavailable;
all decomposition and smoothing tests remain dependency-local.
