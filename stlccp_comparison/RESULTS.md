# STLCCP 707-case comparison result

Run date: 2026-09-22; 11 IN rows re-run on 2026-11-25 for regenerated
scenario windows.

This run uses the established MICP population: the same 707 unique
`(scenario_id, ego_id, rule)` cases, Lin2025 rule semantics, full 2-D CLCS
dynamics, fixed surrounding-vehicle trajectories, and final
`STLRuleMonitor` validation. STLCCP uses LSE (`k=10`) followed by sound
mellowmin (`k=1000`). Gurobi uses one thread, a 30-second limit per QP, at
most 25 CCP iterations, and four cohort workers. This is one deterministic
initialization run, not a multi-seed study.

`success` means that the rebuilt CommonRoad trajectory passes the external
traffic-rule monitor. It does not merely mean that CVXPY/Gurobi returned a
point or that the comparison encoding accepted the point.

## Main result

| Rule | n | STLCCP feasible | STLCCP encoded | STLCCP success | STLCCP median core s | MICP success | MICP paired median core s | STLCCP/MICP geomean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R_G1 | 193 | 193 | 192 | 190 | 0.439 | 191 | 0.814 | 0.555 |
| R_G1_R_G3 | 87 | 87 | 87 | 85 | 1.022 | 84 | 1.688 | 0.623 |
| R_G2 | 69 | 60 | 43 | 16 | 0.507 | 16 | 1.149 | 0.466 |
| R_G3 | 100 | 100 | 99 | 100 | 0.314 | 100 | 0.861 | 0.470 |
| R_IN1 | 91 | 82 | 70 | 35 | 0.482 | 76 | 0.439 | 1.039 |
| R_IN3_hand_draft | 68 | 61 | 39 | 29 | 2.233 | 28 | 1.410 | 1.303 |
| R_IN4 | 49 | 37 | 29 | 15 | 1.327 | 12 | 1.385 | 0.906 |
| R_IN5 | 50 | 44 | 44 | 23 | 1.193 | 22 | 1.398 | 0.756 |
| **ALL** | **707** | **664** | **603** | **493 (69.7%)** | **0.641** | **529 (74.8%)** | **0.890** | **0.671** |

The runtime ratio is computed case-by-case over cases timed by both methods;
values below one favor STLCCP. Across all timed cases the ratio is 0.671.
Restricting it to the 480 cases repaired by both methods gives 0.586. This run
therefore shows a clear speed advantage over MICP, but not an overall success
advantage.

The paired success table is:

- both succeed: 480;
- STLCCP only: 13;
- MICP only: 49;
- neither: 165.

STLCCP's Wilson 95% interval for overall monitor success is 66.2%--73.0%; the
MICP point result is 74.8%. The largest accuracy loss is R_IN1 (35 versus 76).
STLCCP is tied or slightly ahead on R_G1_R_G3, R_G2, IN3, IN4, and IN5, though
small differences from a single run should not be over-interpreted.

## Generated-sampling subset

All 70 XML scenarios under `scenarios/experiment_scenarios/generated_sampling/`
are included in the 707-case table above. They were originally aggregated into
their corresponding IN rules; the following table reports that new-scenario
subset separately.

| Rule | XML/cases | Feasible | Encoded | Success | Success rate | Mean core s | Median core s |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_IN1 | 14 | 9 | 9 | 9 | 64.29% | 0.278 | 0.181 |
| R_IN3_hand_draft | 19 | 14 | 13 | 12 | 63.16% | 1.506 | 1.510 |
| R_IN4 | 22 | 12 | 12 | 2 | 9.09% | 1.251 | 1.254 |
| R_IN5 | 15 | 15 | 15 | 10 | 66.67% | 1.309 | 0.829 |
| **ALL** | **70** | **50** | **49** | **33** | **47.14%** | **1.138** | **0.854** |

The exact path comparison is 70/70 with no missing XML. Filtered raw rows and
the machine-readable subset summary are retained under
`results/generated_sampling/`.

## R_G2 alignment fix

An initial run reported 0/69 for R_G2. That was an implementation-comparison
error, not a valid algorithm result: STLCCP rejected 25 cases with no fixed
predecessor at the recorded trigger and validated the remaining cases with a
fixed-vehicle monitor, whereas the saved MICP baseline uses the Lin2025
`other=None` (`not_abrupt`) fallback and full-scene existential validation.

After aligning those two behaviors, STLCCP succeeds on 16/69 R_G2 cases,
exactly the same 16 case keys as MICP. STLCCP returns a candidate on 60/69
versus MICP's 49/69. The pre-fix CSV is retained under
`results/latest_full/diagnostics/rg2_pre_validation_alignment.csv`.

The remaining R_G2 failures are still meaningful. The fixed-witness Lin2025
optimization formula is not equivalent to the complete quantified monitor;
17 cases hit the CCP iteration limit and some encoding-valid candidates remain
monitor-invalid. The external monitor, rather than encoded robustness, is
therefore the success oracle.

## Diagnostics

Across all rules, 603 candidates satisfy exact unsmoothed robustness of the
comparison formula and 493 pass the external monitor. There are 125
encoded-pass/monitor-fail cases and 15 encoded-fail/monitor-pass cases. The gap
arises where the comparison uses fixed witnesses, polygonal or linearized
predicates, and reconstructed lane assignments.

Terminal solver states are 593 `optimal`, 57 `iteration_limit`, 39
`infeasible`, 14 Gurobi/CVXPY solver errors, and 4 setup errors. The errors are
kept as failed cases rather than dropped.

VP succeeds on 704/707 cases and has a 0.0117-second median core time. On cases
where both STLCCP and VP succeed, STLCCP is 50.5x slower by geometric mean.
STLCCP's useful performance comparison in this repository is therefore mainly
against the mixed-integer formulation.

## Artifacts

- `results/latest_full/all_stlccp_results.csv`: all 707 raw case records;
- `results/latest_full/summary.json`: per-rule STLCCP-vs-VP, MICP-vs-VP, and
  direct paired STLCCP-vs-MICP statistics;
- `results/latest_full/run_manifest.json`: run options and cohort counts;
- individual cohort CSVs in the same directory.
- `results/generated_sampling/all_generated_sampling_results.csv`: the 70
  generated-sampling rows extracted from the full result;
- `results/generated_sampling/summary.json`: per-rule statistics for that
  subset.

The final test suite completed with 18 passing tests using the host Gurobi
academic license.
